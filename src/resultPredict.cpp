#include "resultPredict.h"

void testNetwork(const std::vector<std::vector<int> > &x,
		std::vector<std::vector<int> > &y,
		vector<HiddenLayer> &Hiddenlayers,
		Smr &SMR,
		 std::vector<string> &re_wordmap,
		 int flag,
		 int nGram)
{
	int size =x.size();
	int wn =re_wordmap.size();
	int* res = new int[size];
	int* truth = new int[size];
	int offset;
	int batch_size = 50;
	int batch_amount = size / batch_size;
	for (int i = 0; i < batch_amount; i++)
	{
		offset = i * batch_size;
		cuMatrixVector sampleX;
		getDataMat(sampleX, offset, batch_size, wn,flag,nGram);

		predict(sampleX, Hiddenlayers, SMR, res, offset);
	}
	offset = batch_amount * batch_size;
	if (size % batch_size)
	{
		batch_size = size % batch_size;
		cuMatrixVector sampleX;
		getDataMat(sampleX, offset, batch_size, wn, size,nGram);
		predict(sampleX, Hiddenlayers, SMR, res, offset);
	}
	set_label(truth, size,flag);
	int error = 0;
	for (int i = 0; i < size; i++)
	{
		if (truth[i] != res[i])
		{
			error++;
		}
	}
	float rate = (size - error)/(float)size;
	printf("total num : %d, correct : %d , correct rate: %f \n",size,size-error,rate);
	delete [] res;
	delete [] truth;
}
void predict(cuMatrixVector &sampleX,
		vector<HiddenLayer> &Hiddenlayers,
		Smr &SMR,
		int* output,
		int offset)
{
	int T = sampleX.size();
	int mid = (int) (T / 2.0);
	int HiddenNum =Hiddenlayers.size();
	std::vector<vector<cuMatrix> > acti_l(HiddenNum + 1);
	std::vector<vector<cuMatrix> > acti_r(HiddenNum + 1);
	for (int i = 0; i < T; i++)
	{
		cuMatrix* ptr = sampleX[i];
		acti_l[0].push_back(*ptr);
		acti_r[0].push_back(*ptr);
	}
	for (int i = 1; i <= HiddenNum; ++i)
	{
		//time forward
		for (int j = 0; j < T; j++)
		{
			acti_l[i].push_back(
					cuMatrix(Hiddenlayers[i - 1].cuda_U_l.rows(),
							acti_l[i - 1][0].cols()));
			acti_r[i].push_back(
					cuMatrix(Hiddenlayers[i - 1].cuda_U_r.rows(),
							acti_r[i - 1][0].cols()));
		}
		for (int j = 0; j < T; ++j)
		{
			cuMatrix tmpacti = Hiddenlayers[i - 1].cuda_U_l * acti_l[i - 1][j];
			if (j > 0)
				tmpacti = tmpacti + Hiddenlayers[i - 1].cuda_W_l * acti_l[i][j - 1];
			if (i > 1)
				tmpacti = tmpacti + Hiddenlayers[i - 1].cuda_U_l * acti_r[i - 1][j];
			tmpacti = ReLU(tmpacti);
			if (hiddenConfig[0].DropoutRate < 1.0)
			{
				tmpacti = tmpacti.Mul(hiddenConfig[0].DropoutRate);
			}
			tmpacti.copyTo(acti_l[i][j]);
		}
		//time backward
		for (int j = T - 1; j >= 0; --j)
		{
			cuMatrix tmpacti = Hiddenlayers[i - 1].cuda_U_r * acti_r[i - 1][j];
			if (j < T - 1)
				tmpacti = tmpacti + Hiddenlayers[i - 1].cuda_W_r * acti_r[i][j + 1];
			if (i > 1)
				tmpacti = tmpacti + Hiddenlayers[i - 1].cuda_U_r * acti_l[i - 1][j];
			tmpacti = ReLU(tmpacti);
			if (hiddenConfig[0].DropoutRate < 1.0)
			{
				tmpacti = tmpacti.Mul(hiddenConfig[0].DropoutRate);
			}
			tmpacti.copyTo(acti_r[i][j]);
		}
	}
    cuMatrix M = SMR.cuda_W_l * acti_l[acti_l.size() - 1][mid];
    M = M + SMR.cuda_W_r * acti_r[acti_r.size() - 1][mid];
	get_res_array(M, output, offset);
}
