#ifndef INPUTINIT_H
#define INPUTINIT_H
#include "Samples.h"
//#include "Config.h"   未用到
#include "hardware.h"
#include "cuMatrixVector.h"


void initTestdata(vector<vector<int> > &testX,
		vector<vector<int> > &testY,
		int nGram
		);

void initTraindata(vector<vector<int> > &trainX,
		vector<vector<int> > &trainY,
		int nGram);

void Data2GPU(vector<vector<int> > &trainX,
		vector<vector<int> > &trainY,
		vector<vector<int> > &testX,
		vector<vector<int> > &testY,
		int nGram
		);

void init_acti0(cuMatrixVector& acti_0,
		cuMatrix& sampleY,
		int nGram,
		int  batch_size);

__global__ void set_acti0_kernel(double** acti0,
		int* src,
		int* dev_ran,
		int cols,
		int ngram);

__global__ void set_sampleY_kernel(double* sampleY,
		int* src,
		int* dev_ran,
		int cols,
		int ngram);

void set_groundtruth(cuMatrixVector& gt, cuMatrix& sampleY);

__global__ void set_gt_kernel(float** gt_, float* y, int rows, int cols);


void  getDataMat(cuMatrixVector &sampleX, int off, int bs, int n,
		bool flag,int nGram);
void get_res_array(cuMatrix src, int *res, int offset) ;
void set_label(int* label, int size,bool flag);


#endif

