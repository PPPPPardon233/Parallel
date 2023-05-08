#include <iostream>
#include <fstream>
#include <cstring>
#include <immintrin.h>
#include <pthread.h>
#include <semaphore.h>
#include <algorithm>
#include <windows.h>
#include <sys/time.h>
using namespace std;

struct threadParm
{
	int ID, pos;
};

/*
 *采用位图存储,消元子和消元行的个数之和
 *矩阵列数
 *线程数量
 *消元子
 *被消元行
 *Rpos[i]表示首项为i的消元行在R中第Rpos行
 *Epos[i]表示E第i行首项的位置
*/


const int maxN = 30000;
const int maxM = 23500;
const int threadCount = 3;
int R[maxN][maxM / 32 + 1];
int E[maxN][maxM / 32 + 1];
int Rpos[maxM];
int Epos[maxN];
int sumR, sumE, Enum;
int workCount;
int turn;
pthread_t handle[threadCount];
threadParm tp[threadCount];
sem_t semMain, semWorkstart[threadCount],test;

void setBit(int& a, int pos, int flag){
	if (flag)
		a |= (1 << pos);
	else
		a &= ~(1 << pos);
}

int getBit(int& a, int pos){
	return (a >> pos) & 1;
}

void RInput(){
	ifstream infile("消元子.txt");
	int temp;
	bool newLine = true;
	for (int i = 0; i < maxM; i++)
		Rpos[i] = -1;
	while (infile >> temp){
		if (newLine){
			Rpos[temp] = sumR;
			newLine = false;
		}
		int pos1, pos2;
		pos1 = temp / 32;
		pos2 = temp % 32;
		setBit(R[sumR][pos1], pos2, 1);
		infile.get();
		if (infile.peek() == '\n'){
			infile.get();
			sumR++;
			newLine = true;
		}
	}
	infile.close();
}

void EInput(){
	int temp;
	bool newLine = true;
	ifstream infile("被消元行.txt");
	while (infile >> temp){
		if (newLine){
			Epos[sumE] = temp;
			newLine = false;
		}
		int pos1, pos2;
		pos1 = temp / 32;
		pos2 = temp % 32;
		setBit(E[sumE][pos1], pos2, 1);
		infile.get();
		if (infile.peek() == '\n'){
			infile.get();
			sumE++;
			newLine = true;
		}
	}
	Enum = sumE;
	infile.close();
}

void* PthreadFunction(void* parm){
	__m128i t0, t1;
	threadParm* p = (threadParm*)parm;
	int id = p->ID;
	while (1){
		sem_wait(&semWorkstart[id]);
		if (turn >= Enum)
			return NULL;
		int workPerThread = workCount / threadCount;
		int s = workPerThread * id * 4, e;
		if (id == threadCount - 1) e = workCount * 4;
		else e = workPerThread * (id + 1) * 4;
		for (int j = e - 4; j >= s; j -= 4){
			t0 = _mm_loadu_si128((__m128i*)(E[turn] + j));
			t1 = _mm_loadu_si128((__m128i*)(R[Rpos[Epos[turn]]] + j));
			t1 = _mm_xor_si128(t0, t1);
			_mm_storeu_si128((__m128i*)(E[turn] + j), t1);
		}
		int newpos = -1;
		for (int j = e - 1; j >= s; j--){
			if (E[turn][j] == 0)continue;
			for (int k = 31; k >= 0; k--)
				if (getBit(E[turn][j], k)){
					newpos = 32 * j + k;
					break;
				}
			if (newpos != -1)break;
		}
		p->pos = newpos;
		sem_post(&semMain);

	}
}

void PthreadParallelEliminate(){
	sem_init(&semMain, 0, 0);
	for (int id = 0; id < threadCount; id++)
		sem_init(&semWorkstart[id], 0, 0);
    	sem_init(&test, 0, 0);
	for (int id = 0; id < threadCount; id++){
		tp[id].ID = id;
		pthread_create(&handle[id], NULL, PthreadFunction, &tp[id]);
	}
	for (turn = 0; turn < Enum; turn++){
		while ((E[turn][0] || Epos[turn] > 31) && sumE > 0){
			if (Rpos[Epos[turn]] != -1){
				int d = Epos[turn] / 32;
				int newEpos = -1;
				d += (4 - d % 4);
				workCount = d / 4;
				for (int id = 0; id < threadCount; id++)
					sem_post(&semWorkstart[id]);
				for (int id = 0; id < threadCount; id++)
					sem_wait(&semMain);
				for (int id = 0; id < threadCount; id++)
					newEpos = max(newEpos, tp[id].pos);
				Epos[turn] = newEpos;
			}
			else{
				Rpos[Epos[turn]] = sumR;
				memcpy(R[sumR], E[turn], sizeof(R[sumR]));
				sumR++; sumE--;
				break;
			}
		}
	}
	for (int id = 0; id < threadCount; id++)
		sem_post(&semWorkstart[id]);
	for (int id = 0; id < threadCount; id++)
		pthread_join(handle[id], NULL);
	sem_destroy(&semMain);
	for (int id = 0; id < threadCount; id++)
		sem_destroy(&semWorkstart[id]);
}

int main(){
    timeval *start=new timeval();
    timeval *stop=new timeval();
    double durationTime=0.0;
    RInput();
    EInput();
    gettimeofday(start,NULL);
    PthreadParallelEliminate();
    gettimeofday(stop,NULL);
    durationTime =stop->tv_sec*1000+double(stop->tv_usec)/1000-start->tv_sec*1000-double(start->tv_usec)/1000;
    cout << double(durationTime) << endl;
    return 0;
}