#include <iostream>
#include <fstream>
#include <chrono>
#include <ratio>
#include <iomanip>
#include <random>
#include <thread>
#include <functional>
#include <CL/sycl.hpp>

using namespace cl::sycl;

void matrix_print(buffer<float, 2>& buf) {
	host_accessor m{ buf ,read_only };
	auto range = m.get_range();
	for (int i = 0; i < range[0]; i++) {
		for (int j = 0; j < range[1]; j++) {
			std::cout << std::setw(16) << m[i][j];
		}
		std::cout << std::endl;
	}
}

void matrix_copy(buffer<float, 2>& to, buffer<float, 2>& from) {
	host_accessor ms{ from ,read_only };
	host_accessor md{ to ,write_only };
	assert(ms.get_range() == md.get_range());
	auto range = ms.get_range();
	for (int i = 0; i < range[0]; i++) {
		for (int j = 0; j < range[1]; j++) {
			md[i][j] = ms[i][j];
		}
	}
}

void matrix_init(buffer<float, 2>& buf) {
	host_accessor m{ buf ,read_write };

	static std::default_random_engine generator(1337);
	static std::uniform_real_distribution<float> distribution(-1.0, 1.0);

	int n = m.get_range()[0];
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < i; j++)
			m[i][j] = 0;
		m[i][i] = 1.0;
		for (int j = i + 1; j < n; j++)
			m[i][j] = distribution(generator);
	}
	for (int k = 0; k < n; k++) {
		for (int i = k + 1; i < n; i++) {
			for (int j = 0; j < n; j++) {
				m[i][j] += m[k][j];
			}
		}
	}
}

void gauss_normal(buffer<float, 2>& buf, queue& q) {
	host_accessor m{ buf ,read_write };
	int n = m.get_range()[0];
	for (int k = 0; k < n; k++) {
		for (int j = k + 1; j < n; j++) {
			m[k][j] = m[k][j] / m[k][k];
		}
		m[k][k] = 1;
		for (int i = k + 1; i < n; i++) {
			for (int j = k + 1; j < n; j++) {
				m[i][j] = m[i][j] - m[i][k] * m[k][j];
			}
			m[i][k] = 0;
		}
	}
}

void gauss_oneapi(buffer<float, 2>& buf, queue& q) {

	int n = buf.get_range()[0];
	for (int k = 0; k < n; k++) {

		q.submit([&](handler& h) {
			accessor m{ buf, h, read_write };
			h.parallel_for(range(n - k), [=](auto idx) {
				int j = k + idx;
				m[k][j] = m[k][j] / m[k][k];
				});
			});

		q.submit([&](handler& h) {
			accessor m{ buf, h, read_write };
			h.parallel_for(range(n - (k + 1), n - (k + 1)), [=](auto idx) {
				int i = k + 1 + idx.get_id(0);
				int j = k + 1 + idx.get_id(1);
				m[i][j] = m[i][j] - m[i][k] * m[k][j];
				});
			});

		q.submit([&](handler& h) {
			accessor m{ buf, h, read_write };
			h.parallel_for(range(n - (k + 1)), [=](auto idx) {
				int i = k + 1 + idx;
				m[i][k] = 0;
				});
			});
	}
	q.wait();
}

using gauss_func = std::function<void(buffer<float, 2>&, queue&)>;

double test(int n, const gauss_func& gauss, int times, queue& q) {
	buffer<float, 2> buf(range(n, n));

	// warm up
	matrix_init(buf);
	gauss(buf, q);

	std::chrono::duration<double, std::milli> elapsed{};
	for (int i = 0; i < times; i++) {
		matrix_init(buf);
		auto start = std::chrono::high_resolution_clock::now();
		gauss(buf, q);
		auto end = std::chrono::high_resolution_clock::now();
		elapsed += end - start;
	}
	return elapsed.count() / times;
}

void normal_test_output(const std::vector<gauss_func>& gauss_funcs,
	const std::vector<std::string>& names,
	int times,
	int begin,
	int end,
	queue& q) {
	for (auto& name : names) {
		std::cout << name << ",";
	}
	std::cout << std::endl;
	for (int n = begin; n <= end; n += 100) {
		std::cout << n << ",";
		for (auto& func : gauss_funcs) {
			std::cout << test(n, func, times, q) << ",";
		}
		std::cout << std::endl;
	}
}

int main() {

	queue q;
	device my_device = q.get_device();
	std::cout << "Device: " << my_device.get_info<info::device::name>() << std::endl;

	std::vector<gauss_func> gauss_funcs = {
			gauss_normal,
			gauss_oneapi,
	};

	std::vector<std::string> names = {
			"normal",
			"oneapi",
	};

	const int times = 5;
	const int begin = 100;
	const int end = 100;

	normal_test_output(gauss_funcs, names, times, begin, end, q);
}
