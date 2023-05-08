#include <iostream>
#include <pthread.h>
#include <semaphore.h>
#include <immintrin.h>
#include <sys/time.h>
#include <omp.h>
using namespace std;

const int N = 2000, threadCount = 4;
sem_t semMain, semWorkstart[threadCount], semWorkend[threadCount];
float a[N][N];

void init(){
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			a[i][j] = float(rand()) / 10;
}

void OpenMP_SSE_Algorithm(){
    __m128 t0, t1, t2, t3;
    int i,j,k;
    float temp1[4],temp2[4];
#pragma omp parallel num_threads(threadCount),private(t0,t1,t2,t3,i,j,k,temp1,temp2)
    for (k = 0; k < N; k++){
        temp1[0]=temp1[1]=temp1[2]=temp1[3]=a[k][k];
        t0 = _mm_loadu_ps(temp1);
#pragma omp single
{
        for (j = k + 1; j + 3 < N; j += 4){
            t1 = _mm_loadu_ps(a[k] + j);
            t2 = _mm_div_ps(t1, t0);
            _mm_storeu_ps(a[k] + j, t2);
        }
        for (; j < N; j++)
            a[k][j] /= a[k][k];
        a[k][k] = 1.0;
}

#pragma omp for
        for (i = k + 1; i < N; i++){
            temp2[0]=temp2[1]=temp2[2]=temp2[3]=a[k][k];
            t0 = _mm_loadu_ps(temp2);
            for (j = k + 1; j + 3 < N; j += 4){
                t1 = _mm_loadu_ps(a[k] + j);
                t2 = _mm_loadu_ps(a[i] + j);
                t3 = _mm_mul_ps(t0, t1);
                t2 = _mm_sub_ps(t2, t3);
                _mm_storeu_ps(a[i] + j, t2);
            }
            for (; j < N; j++)
                a[i][j] -= a[i][k] * a[k][j];
            a[i][k] = 0.0;
        }
    }
}

void OpenMP_Algorithm(){
    int i,j,k;
#pragma omp parallel num_threads(threadCount),private(i,j,k)
    for (k = 0; k < N; k++){
#pragma omp single
{
	   for (j = k + 1; j < N; j++)
	       a[k][j] /= a[k][k];
	   a[k][k] = 1.0;
}
#pragma omp for
	   for (i = k + 1; i < N; i++) {
            for (j = k + 1; j < N; j++)
       		a[i][j] -= a[i][k] * a[k][j];
		 a[i][k] = 0;
	   }
    }
}

int main(int argc, char** argv){
    char** temp = argv;
    temp=temp++;
    timeval *start=new timeval();
    timeval *stop=new timeval();
    double durationTime=0.0;
    init();
    gettimeofday(start,NULL);

    if (!strcmp(*(temp), "OpenMP_SSE_Algorithm"))            OpenMP_SSE_Algorithm();
    if (!strcmp(*(temp), "OpenMP_Algorithm"))                 OpenMPAlgorithm();

    gettimeofday(stop,NULL);
    durationTime =stop->tv_sec*1000+double(stop->tv_usec)/1000-start->tv_sec*1000-double(start->tv_usec)/1000;
    cout  << double(durationTime) << endl;
    return 0;   
}