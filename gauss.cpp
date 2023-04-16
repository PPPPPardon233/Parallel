#include <iostream>
#include <fstream>
#include <iomanip>
#include <time.h>
#include <sys/time.h>
#include <cmath>
#include <arm_neon.h>
#include <immintrin.h>
#include "NEON_2_SSE.h"

#define N 1024
#define ele_t float

using namespace std;

ele_t new_mat[N][N] __attribute__((aligned(64)));
ele_t mat[N][N];

void LU(ele_t mat[N][N], int n){
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            new_mat[i][j] = mat[i][j];

    for (int i = 0; i < n; i++){
        for (int j = i + 1; j < n; j++){
            if (new_mat[i][i] == 0)
                continue;
            ele_t div = new_mat[j][i] / new_mat[i][i];
            for (int k = i; k < n; k++)
                new_mat[j][k] -= new_mat[i][k] * div;
        }
    }
}

void LU_simd(ele_t mat[N][N], int n){
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            new_mat[i][j] = mat[i][j];

    for (int i = 0; i < n; i++){
        for (int j = i + 1; j < n; j++){
            if (new_mat[i][i] == 0)
                continue;
            ele_t div = new_mat[j][i] / new_mat[i][i];
            float32x4_t div4 = vmovq_n_f32(div);
            float32x4_t mat_j;
            float32x4_t mat_i;
            float32x4_t res;
            for (int k = i; k < n; k += 4){
                mat_j = vld1q_f32(new_mat[j] + k);
                mat_i = vld1q_f32(new_mat[i] + k);
                res = vmlsq_f32(mat_j, div4, mat_i);
                vst1q_f32(new_mat[j] + k, res);
            }
        }
    }
}

void LU_simd_Aligned(ele_t mat[N][N], int n){
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            new_mat[i][j] = mat[i][j];

    for (int i = 0; i < n; i++){
        for (int j = i + 1; j < n; j++){
            if (new_mat[i][i] == 0)
                continue;
            ele_t div = new_mat[j][i] / new_mat[i][i];
            float32x4_t div4 = vmovq_n_f32(div);
            float32x4_t mat_j;
            float32x4_t mat_i;
            float32x4_t res;
            for (int k = i / 4 * 4; k < n; k += 4){
                mat_j = vld1q_f32(new_mat[j] + k);
                mat_i = vld1q_f32(new_mat[i] + k);
                res = vmlsq_f32(mat_j, div4, mat_i);
                vst1q_f32(new_mat[j] + k, res);
            }
        }
    }
}

void LU_sse_fma(ele_t mat[N][N], int n){
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            new_mat[i][j] = mat[i][j];

    for (int i = 0; i < n; i++){
        for (int j = i + 1; j < n; j++){
            if (new_mat[i][i] == 0)
                continue;
            ele_t div = new_mat[j][i] / new_mat[i][i];
            float32x4_t div4 = vmovq_n_f32(div);
            float32x4_t mat_j;
            float32x4_t mat_i;
            float32x4_t res;
            for (int k = i / 4 * 4; k < n; k += 4){
                mat_j = vld1q_f32(new_mat[j] + k);
                mat_i = vld1q_f32(new_mat[i] + k);
                res = _mm_fnmadd_ps(mat_i, div4, mat_j);
                vst1q_f32(new_mat[j] + k, res);
            }
        }
    }
}

void LU_avx(ele_t mat[N][N], int n){
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            new_mat[i][j] = mat[i][j];

    for (int i = 0; i < n; i++){
        for (int j = i + 1; j < n; j++){
            if (new_mat[i][i] == 0)
                continue;
            ele_t div = new_mat[j][i] / new_mat[i][i];
            __m256 div8 = _mm256_set1_ps(div);
            __m256 mat_j;
            __m256 mat_i;
            __m256 res;
            for (int k = i; k < n; k += 8){
                mat_j = _mm256_loadu_ps(new_mat[j] + k);
                mat_i = _mm256_loadu_ps(new_mat[i] + k);
                res = _mm256_fnmadd_ps(mat_i, div8, mat_j);
                _mm256_storeu_ps(new_mat[j] + k, res);
            }
        }
    }
}

void LU_avx_aligned(ele_t mat[N][N], int n){
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            new_mat[i][j] = mat[i][j];

    for (int i = 0; i < n; i++){
        for (int j = i + 1; j < n; j++){
            if (new_mat[i][i] == 0)
                continue;
            ele_t div = new_mat[j][i] / new_mat[i][i];
            __m256 div8 = _mm256_set1_ps(div);
            __m256 mat_j;
            __m256 mat_i;
            __m256 res;
            for (int k = i / 8 * 8; k < n; k += 8){
                mat_j = _mm256_load_ps(new_mat[j] + k);
                mat_i = _mm256_load_ps(new_mat[i] + k);
                res = _mm256_fnmadd_ps(mat_i, div8, mat_j);
                _mm256_store_ps(new_mat[j] + k, res);
            }
        }
    }
}

void LU_avx512(ele_t mat[N][N], int n){
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            new_mat[i][j] = mat[i][j];

    for (int i = 0; i < n; i++){
        for (int j = i + 1; j < n; j++){
            if (new_mat[i][i] == 0)
                continue;
            ele_t div = new_mat[j][i] / new_mat[i][i];
            __m512 div16 = _mm512_set1_ps(div);
            __m512 mat_j;
            __m512 mat_i;
            __m512 res;
            for (int k = i; k < n; k += 16){
                mat_j = _mm512_loadu_ps(new_mat[j] + k);
                mat_i = _mm512_loadu_ps(new_mat[i] + k);
                res = _mm512_fnmadd_ps(mat_i, div16, mat_j);
                _mm512_storeu_ps(new_mat[j] + k, res);
            }
        }
    }
}

void LU_avx512_aligned(ele_t mat[N][N], int n){
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            new_mat[i][j] = mat[i][j];

    for (int i = 0; i < n; i++){
        for (int j = i + 1; j < n; j++){
            if (new_mat[i][i] == 0)
                continue;
            ele_t div = new_mat[j][i] / new_mat[i][i];
            __m512 div16 = _mm512_set1_ps(div);
            __m512 mat_j;
            __m512 mat_i;
            __m512 res;
            for (int k = i / 16 * 16; k < n; k += 16){
                mat_j = _mm512_load_ps(new_mat[j] + k);
                mat_i = _mm512_load_ps(new_mat[i] + k);
                res = _mm512_fnmadd_ps(mat_i, div16, mat_j);
                _mm512_store_ps(new_mat[j] + k, res);
            }
        }
    }
}


int main(int argc, char** argv) {
    char** temp = argv;
    temp=temp++;
    ifstream data("gauss.dat", ios::in | ios::binary);
    data.read((char *)mat, N * N * sizeof(ele_t));
    data.close();
    timespec start, end;
    double time_used = 0;
    clock_gettime(CLOCK_REALTIME, &start);
    
    if (!strcmp(*(temp), "LU"))                 LU(mat, N);;
    if (!strcmp(*(temp), "LU_simd"))            LU_simd(mat, N);
    if (!strcmp(*(temp), "LU_simd_Aligned"))    LU_simd_Aligned(mat, N);
    if (!strcmp(*(temp), "LU_sse_fma"))         LU_sse_fma(mat, N);
    if (!strcmp(*(temp), "LU_avx"))             LU_avx(mat, N);
    if (!strcmp(*(temp), "LU_avx_aligned"))     LU_avx_aligned(mat, N);
    if (!strcmp(*(temp), "LU_avx512_aligned"))  LU_avx512_aligned(mat, N);
    if (!strcmp(*(temp), "LU_avx512"))          LU_avx512(mat, N);

    clock_gettime(CLOCK_REALTIME, &end);
    time_used = double(end.tv_nsec - start.tv_nsec);
    cout << time_used << endl;
    return 0;
}