#include <iostream>
#include <sys/time.h>
#include <arm_neon.h>
#include <pthread.h>
#include <semaphore.h>
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

void OpenMP_NEON_Algorithm(){
    float32x4_t t0, t1, t2, t3;
    int i, j, k;
#pragma omp parallel num_threads(threadCount),private(i,j,k,t0,t1,t2,t3)
    for (k = 0; k < N; k++){
        t0 = vld1q_dup_f32(a[k] + k);
#pragma omp single
{
	  for (j = k + 1; j + 3 < N; j += 4){
     	      t1 = vld1q_f32(a[k] + j);
		 t2 = vdivq_f32(t1, t0);
		 vst1q_f32(a[k] + j, t2);
	  }
	  for (; j < N; j++)
		 a[k][j] /= a[k][k];
	  a[k][k] = 1.0;
}
#pragma omp for
	  for (i = k + 1; i < N; i++){
	      t0 = vld1q_dup_f32(a[i] + k);
		 for (j = k + 1; j + 3 < N; j += 4){
			t1 = vld1q_f32(a[k] + j);
			t2 = vld1q_f32(a[i] + j);
			t3 = vmulq_f32(t0, t1);
			t2 = vsubq_f32(t2, t3);
			vst1q_f32(a[i] + j, t2);
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

    if (!strcmp(*(temp), "OpenMP_NEON_Algorithm"))            OpenMP_NEON_Algorithm();
    if (!strcmp(*(temp), "OpenMP_Algorithm"))                 OpenMPAlgorithm();

    gettimeofday(stop,NULL);
    durationTime =stop->tv_sec*1000+double(stop->tv_usec)/1000-start->tv_sec*1000-double(start->tv_usec)/1000;
    cout  << double(durationTime) << endl;
    return 0;   
}