#ifndef COSTGRADIENT_H
#define COSTGRADIENT_H


#include "cuMatrixVector.h"
//#include "Config.h"
//#include "cuMath.h"
#include "InputInit.h"
void cuda_getNetworkCost(cuMatrixVector &acti_0, cuMatrix &sampleY,
		vector<Hl> &Hiddenlayers, Smr &SMR);
#endif
