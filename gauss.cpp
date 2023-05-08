#include <pthread.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <time.h>
#include <sys/time.h>
#include <cmath>
#include <string.h>
#include <arm_neon.h>
#include <immintrin.h>
#include "NEON_2_SSE.h"

#define N 4096
#define NUM_THREADS 20
#define ele_t float

using namespace std;

ele_t new_mat[N][N] __attribute__((aligned(64)));
ele_t mat[N][N];

void LU(ele_t mat[N][N], int n){
    memcpy(new_mat, mat, sizeof(ele_t) * N * N);
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
    ele_t new_mat[N][N];
    memcpy(new_mat, mat, sizeof(ele_t) * N * N);
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

struct LU_data
{
    int th;
    pthread_mutex_t finished = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t startNext = PTHREAD_MUTEX_INITIALIZER;
    ele_t (*mat)[N][N];
    int n;
    int i, begin, nLines; // 当前行、开始消去行、结束消去行
};

void *subthread_LU(void *_params){
    LU_data *params = (LU_data *)_params;
    int i = params->i;
    int n = params->n;
    float32x4_t mat_j, mat_i, div4;
    for (int j = params->begin; j < params->begin + params->nLines; j++){
        if ((*params->mat)[i][i] == 0)
            continue;
        ele_t div = (*params->mat)[j][i] / (*params->mat)[i][i];
        div4 = vmovq_n_f32(div);
        for (int k = i / 4 * 4; k < n; k += 4){
            mat_j = vld1q_f32((*params->mat)[j] + k);
            mat_i = vld1q_f32((*params->mat)[i] + k);
            vst1q_f32((*params->mat)[j] + k, vmlsq_f32(mat_j, div4, mat_i));
        }
    }
}


void LU_pthread(ele_t mat[N][N], int n){
    memcpy(new_mat, mat, sizeof(ele_t) * N * N);
    pthread_t threads[NUM_THREADS];
    LU_data attr[NUM_THREADS];

    for (int i = 0; i < n; i++){
        int nLines = (n - i - 1) / NUM_THREADS;
        if (nLines > 31){
            for (int th = 0; th < NUM_THREADS; th++){
                attr[th].th = th;
                attr[th].mat = &new_mat;
                attr[th].n = n;
                attr[th].i = i;
                attr[th].nLines = nLines;
                attr[th].begin = i + 1 + th * nLines;
                int err = pthread_create(&threads[th], NULL, subthread_LU, (void *)&attr[th]);
                if (err) exit(-1);
            }
            for (int j = i + 1 + NUM_THREADS * ((n - i - 1) / NUM_THREADS); j < n; j++){
                if (new_mat[i][i] == 0)
                    continue;
                ele_t div = new_mat[j][i] / new_mat[i][i];
                for (int k = i; k < n; k++)
                    new_mat[j][k] -= new_mat[i][k] * div;
            }
            for (int th = 0; th < NUM_THREADS; th++)
                pthread_join(threads[th], NULL);
        }
        else{
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
}

void *subthread_static_LU(void *_params){
    LU_data *params = (LU_data *)_params;
    int i = params->i;
    int n = params->n;
    float32x4_t mat_j, mat_i, div4;
    while (true){
        pthread_mutex_lock(&(params->startNext));
        i = params->i;
        n = params->n;
        for (int j = params->begin; j < params->begin + params->nLines; j++){
            if ((*params->mat)[i][i] == 0)
                continue;
            ele_t div = (*params->mat)[j][i] / (*params->mat)[i][i];
            div4 = vmovq_n_f32(div);
            for (int k = i / 4 * 4; k < n; k += 4){
                mat_j = vld1q_f32((*params->mat)[j] + k);
                mat_i = vld1q_f32((*params->mat)[i] + k);
                vst1q_f32((*params->mat)[j] + k, vmlsq_f32(mat_j, div4, mat_i));
            }
        }
        pthread_mutex_unlock(&(params->finished));
    }
}

void LU_static_thread(ele_t mat[N][N], int n){
    memcpy(new_mat, mat, sizeof(ele_t) * N * N);
    pthread_t threads[NUM_THREADS];
    LU_data attr[NUM_THREADS];

    for (int th = 0; th < NUM_THREADS; th++){
        pthread_mutex_lock(&(attr[th].startNext));
        pthread_mutex_lock(&(attr[th].finished));
        int err = pthread_create(&threads[th], NULL, subthread_static_LU, (void *)&attr[th]);
        if (err) exit(-1);
    }

    for (int i = 0; i < n; i++){
        int nLines = (n - i - 1) / NUM_THREADS;
        for (int th = 0; th < NUM_THREADS; th++){
            attr[th].th = th;
            attr[th].mat = &new_mat;
            attr[th].n = n;
            attr[th].i = i;
            attr[th].nLines = nLines;
            attr[th].begin = i + 1 + th * nLines;
            pthread_mutex_unlock(&(attr[th].startNext));
        }
        for (int j = i + 1 + NUM_THREADS * ((n - i - 1) / NUM_THREADS); j < n; j++){
            if (new_mat[i][i] == 0)
                continue;
            ele_t div = new_mat[j][i] / new_mat[i][i];
            for (int k = i; k < n; k++)
                new_mat[j][k] -= new_mat[i][k] * div;
        }
        for (int th = 0; th < NUM_THREADS; th++)
            pthread_mutex_lock(&(attr[th].finished));
    }
}

int main(int argc, char** argv){
    char** temp = argv;
    temp=temp++;
    ifstream data("gauss.dat", ios::in | ios::binary);
    data.read((char *)mat, N * N * sizeof(ele_t));
    data.close();
    timespec start, end;
    double time_used = 0;
    clock_gettime(CLOCK_REALTIME, &start);
    
    if (!strcmp(*(temp), "LU"))                 LU(mat, N);
    if (!strcmp(*(temp), "LU_simd"))     	     LU_simd(mat, N);
    if (!strcmp(*(temp), "LU_pthread"))  	     LU_pthread(mat, N);
    if (!strcmp(*(temp), "LU_static_thread"))   LU_static_thread(mat, N);

    clock_gettime(CLOCK_REALTIME, &end);
    time_used = double(end.tv_nsec - start.tv_nsec);
    cout << time_used << endl;
    return 0;
}

