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

void OpenMP_Algorithm(){
    int newEpos,pos;
#pragma omp parallel num_threads(threadCount),private(newEpos),shared(pos)
    for (int i = 0; i < Enum; i++){
        while ((E[i][0] || Epos[i] > 31) && sumE > 0){
            if (Rpos[Epos[i]] != -1){
                int d = Epos[i] / 32, newEpos = -1;
#pragma omp for
                for (int j = d; j >= 0; j--){
                    E[i][j] ^= R[Rpos[Epos[i]]][j];
			    if (newEpos == -1 && E[i][j] != 0){
                        for (int k = 31; k >= 0; k--)
				      if (getBit(E[i][j], k)){
				          newEpos = 32 * j + k;
					     pos=max(newEpos,pos);
					     break;
				      }
			    }
		     }
		     Epos[i] = pos;
	      }
	      else{
#pragma omp barrier
#pragma omp single
{
                Rpos[Epos[i]] = sumR;
		     memcpy(R[sumR], E[i], sizeof(R[sumR]));//E[i]升级为消元子
		     sumR++; sumE--;
                break;
}
	       }
	   }
    }
}

int main(){
    timeval *start=new timeval();
    timeval *stop=new timeval();
    double durationTime=0.0;
    RInput();
    EInput();
    gettimeofday(start,NULL);
    OpenMP_Algorithm();
    gettimeofday(stop,NULL);
    durationTime =stop->tv_sec*1000+double(stop->tv_usec)/1000-start->tv_sec*1000-double(start->tv_usec)/1000;
    cout << double(durationTime) << endl;
    return 0;
}
