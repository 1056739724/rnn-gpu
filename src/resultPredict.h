#ifndef RESULTPREDICT_H
#define RESULTPREDICT_H
#include <vector>
#include <string>
#include <iostream>
//#include "Config.h"
#include "cuMatrix.h"
#include "cuMatrixVector.h"
#include "cuMath.h"
#include "InputInit.h"
void testNetwork(const std::vector<std::vector<int> > &,
		std::vector<std::vector<int> > &,
		vector<HiddenLayer> &Hiddenlayers,
		Smr &SMR,
		 std::vector<string> &re_wordmap,
		 int flag,int nGram);

void predict(cuMatrixVector &sampleX, vector<HiddenLayer> &Hiddenlayers,
		Smr &SMR,int* output,int offset);
#endif
