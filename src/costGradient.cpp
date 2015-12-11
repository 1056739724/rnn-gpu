#include "costGradient.h"
#include "general_settings.h"
#include "cuMatrix.h"
#include "cuMath.h"

void cuda_getNetworkCost(cuMatrixVector &acti_0, //里面存放5个矩阵，每个矩阵1255*50,存放着50个随机子句子的word对应的int
		cuMatrix &sampleY,//5*50，存放着50个子句子对应的标签，对应的int值，是真实的分类标签
		vector<Hl> &Hiddenlayers,
		Smr &SMR)
{
	int T = acti_0.size();//5
	//nSamples为了得到50，用sampleY得到吧
//	int nSamples = acti_0[0]->cols();
	int nSamples =sampleY.cols();//50
	//HiddenNum：隐藏层层数
	int HiddenNum =Hiddenlayers.size();

	vector<vector<cuMatrix> > acti_l(HiddenNum + 1);
	vector<vector<cuMatrix> > acti_r(HiddenNum + 1);
	vector<vector<cuMatrix> > nonlin_l(HiddenNum);
	vector<vector<cuMatrix> > nonlin_r(HiddenNum);
//	vector<cuMatrixVector > bernoulli_l;
//	vector<cuMatrixVector > bernoulli_r;

	for (int i = 0; i < T; i++)
	{
		cuMatrix* ptr = acti_0[i];//每个元素都是0.000000
		acti_l[0].push_back(*ptr);
		acti_r[0].push_back(*ptr);
	}

	//hiddenlayer forward;
	for (int i = 1; i <= HiddenNum; i++) //1
	{
		for (int j = 0; j < T; j++) //5
		{
			//初始全是0
			acti_l[i].push_back(cuMatrix(Hiddenlayers[i - 1].cuda_U_l.rows(),
					acti_l[i - 1][0].cols()));//512*50
			acti_r[i].push_back(cuMatrix(Hiddenlayers[i - 1].cuda_U_l.rows(),
					acti_l[i - 1][0].cols()));
			nonlin_l[i-1].push_back(cuMatrix(Hiddenlayers[i - 1].cuda_U_l.rows(),
					acti_l[i - 1][0].cols()));
			nonlin_r[i-1].push_back(cuMatrix(Hiddenlayers[i - 1].cuda_U_l.rows(),
					acti_l[i - 1][0].cols()));
		}

//		bernoulli_l.push_back(tmp_bl);
//		bernoulli_r.push_back(tmp_br);

// time forward
		for (int j = 0; j < T; j++)
		{
			cuMatrix tmpacti = Hiddenlayers[i - 1].cuda_U_l * (acti_l[i - 1][j]);
//			tmpacti.printMat();
			if (j > 0)
			{
				tmpacti = Hiddenlayers[i - 1].cuda_W_l * (acti_l[i][j - 1])
						+ tmpacti;
			}
			if (i > 1)
			{
				tmpacti = Hiddenlayers[i - 1].cuda_U_l * (acti_r[i - 1][j])
						+ tmpacti;
			}
			//数据拷贝，把当前设备的dev数据，也就是data的dev拷贝到传过来的这个设备的data的dev
			tmpacti.copyTo(nonlin_l[i - 1][j]);
			tmpacti = ReLU(tmpacti);//线性修正，矩阵中元素大于0的保留，小于0的全部设置为0
			if (hiddenConfig[0].DropoutRate < 1.0)
			{
			}
			else
			{
				tmpacti.copyTo(acti_l[i][j]);
			}

		}

//time backwoard
		for (int j = T - 1; j >= 0; j--)
		{
			cuMatrix tmpacti = Hiddenlayers[i - 1].cuda_U_r * (acti_r[i - 1][j]);

			if (j < T - 1)
				tmpacti = Hiddenlayers[i - 1].cuda_W_r * (acti_r[i][j + 1])+ tmpacti;
			if (i > 1)
				tmpacti = Hiddenlayers[i - 1].cuda_U_r * (acti_l[i - 1][j])+ tmpacti;
			tmpacti.copyTo(nonlin_r[i - 1][j]);
			tmpacti = ReLU(tmpacti);//线性修真，矩阵大于0的部分保存，小于0的设置成0
			if (hiddenConfig[0].DropoutRate < 1.0)
			{
			} else
			{
				tmpacti.copyTo(acti_r[i][j]);
			}
		}
	}

// softmax layer forward
	vector<cuMatrix> p;
	vector<cuMatrix> groundTruth;
	for (int i = 0; i < T; i++)
	{
		//acti_l.size() 2
		cuMatrix M = SMR.cuda_W_l * (acti_l[acti_l.size() - 1][i]);
		M = SMR.cuda_W_r * (acti_r[acti_r.size() - 1][i]) + M;
		M = M - reduceMax(M);
		M = Exp(M);
		M = M / reduceSum(M);
		p.push_back(M);

	}

    //把类别放入5个向量中
	cuMatrixVector groundTruth_tmp;
	for (int i = 0; i < T; i++)//5个矩阵
	{
		cuMatrix* tmp = new cuMatrix(softmaxConfig.NumClasses, nSamples);
		groundTruth_tmp.push_back(tmp);
	}
	groundTruth_tmp.toGpu();
	set_groundtruth(groundTruth_tmp, sampleY);
	for (int i = 0; i < T; i++)//5
	{
		cuMatrix* ptr = groundTruth_tmp[i];
		groundTruth.push_back(*ptr);
	}

//cost function
	float j1 = 0.0f;
	float j2 = 0.0f;
	float j3 = 0.0f;
	float j4 = 0.0f;
	for (int i = 0; i < T; i++)
	{
		cuMatrix cumat = groundTruth[i].Mul(Log(p[i]));
		float tmpj = cumat.getSum();
		j1 -= tmpj;
	}

	j1 /= nSamples;
	j2 = Pow(SMR.cuda_W_l, 2.0f).getSum();
	j2 += Pow(SMR.cuda_W_r, 2.0f).getSum();
	j2 = j2 * softmaxConfig.WeightDecay/ 2;

	for (int i = 0; i < Hiddenlayers.size(); i++)
	{
		j3 += Pow(Hiddenlayers[i].cuda_W_l, 2).getSum();
		j3 += Pow(Hiddenlayers[i].cuda_W_r, 2).getSum();
		j3 = j3 *hiddenConfig[0].WeightDecay / 2;
	}
	for (int i = 0; i < Hiddenlayers.size(); i++)
	{
		j4 += Pow(Hiddenlayers[i].cuda_U_l, 2).getSum();
		j4 += Pow(Hiddenlayers[i].cuda_U_r, 2).getSum();
		j4 = j4 * hiddenConfig[0].WeightDecay/ 2;
	}
	SMR.cost = j1 + j2 + j3 + j4;

	 cout<<"j1 = "<<j1<<", j2 = "<<j2<<", j3 = "<<j3<<", j4 = "<<j4<<", Cost = "<<SMR.cost<<endl;

// SMR backward
	vector<cuMatrix> dis;
	vector<cuMatrix> dis2;
	for (int i = 0; i < T; i++)
	{
		cuMatrix tmpdis = groundTruth[i] - p[i];
		cuMatrix tmpdis2 = Pow(tmpdis, 2);
		dis.push_back(tmpdis);
		dis2.push_back(tmpdis2);
	}
	//Smr t-forward
	cuMatrix Swl_tmp(dis[0].rows(), acti_l[acti_l.size() - 1][0].rows());
	for (int i = 0; i < T; i++)
	{
		Swl_tmp = Swl_tmp - dis[i] * acti_l[acti_l.size() - 1][i].t();
	}
	Swl_tmp = Swl_tmp / nSamples;
	SMR.cuda_W_lgrad = Swl_tmp + SMR.cuda_W_l * softmaxConfig.WeightDecay;
	cuMatrix Swld2_tmp(dis2[0].rows(), acti_l[acti_l.size() - 1][0].rows());
	for (int i = 0; i < T; i++)
	{
		Swld2_tmp = Swld2_tmp
				+ dis2[i] * Pow(acti_l[acti_l.size() - 1][i].t(), 2);
	}

	Swld2_tmp = Swld2_tmp / nSamples;
	SMR.cuda_W_ld2 = Swld2_tmp + softmaxConfig.WeightDecay;
	//Smr t-borward
	cuMatrix Swr_tmp(dis[0].rows(), acti_r[acti_r.size() - 1][0].rows());
	for (int i = 0; i < T; i++)
	{
		Swr_tmp = Swr_tmp - dis[i] * acti_r[acti_r.size() - 1][i].t();
	}
	Swr_tmp = Swr_tmp / nSamples;
	SMR.cuda_W_rgrad = Swr_tmp + SMR.cuda_W_r * softmaxConfig.WeightDecay;
	cuMatrix Swrd2_tmp(dis2[0].rows(), acti_r[acti_r.size() - 1][0].rows());
	for (int i = 0; i < T; i++)
	{
		Swrd2_tmp = Swrd2_tmp
				+ dis2[i] * Pow(acti_r[acti_r.size() - 1][i].t(), 2);
	}
	Swrd2_tmp = Swrd2_tmp / nSamples;
	SMR.cuda_W_rd2 = Swrd2_tmp + softmaxConfig.WeightDecay;


//BPTT for last hidden
	vector<cuMatrixVector> delta_l(acti_l.size());
	vector<cuMatrixVector> delta_ld2(acti_l.size());
	vector<cuMatrixVector> delta_r(acti_r.size());
	vector<cuMatrixVector> delta_rd2(acti_r.size());
	for (int i = 0; i < delta_l.size(); i++)
	{
		for (int j = 0; j < T; j++)
		{
			delta_l[i].push_back(new cuMatrix(SMR.cuda_W_l.cols(), dis[0].cols()));
			delta_ld2[i].push_back(new cuMatrix(SMR.cuda_W_l.cols(), dis[0].cols()));
			delta_r[i].push_back(new cuMatrix(SMR.cuda_W_r.cols(), dis[0].cols()));
			delta_rd2[i].push_back(new cuMatrix(SMR.cuda_W_r.cols(), dis[0].cols()));
		}
	}

	//time forward
	for (int i = T - 1; i >= 0; i--)
	{
		cuMatrix tmp = SMR.cuda_W_l.t() * dis[i] * (-1.0f);
		cuMatrix tmp2 = Pow(SMR.cuda_W_l.t(), 2) * dis2[i];
		if (i < T - 1)
		{
			tmp = tmp
					+ Hiddenlayers[Hiddenlayers.size() - 1].cuda_W_l.t()
							* (*delta_l[delta_l.size() - 1][i + 1]);
			tmp2 = tmp2
					+ Pow(Hiddenlayers[Hiddenlayers.size() - 1].cuda_W_l.t(), 2)
							* (*delta_ld2[delta_ld2.size() - 1][i + 1]);
		}
		tmp.copyTo(*delta_l[delta_l.size() - 1][i]);
		tmp2.copyTo(*delta_ld2[delta_ld2.size() - 1][i]);
		*delta_l[delta_l.size() - 1][i] = delta_l[delta_l.size() - 1][i]->Mul(
				dReLU(nonlin_l[nonlin_l.size() - 1][i]));
		*delta_ld2[delta_ld2.size() - 1][i] =
				delta_ld2[delta_ld2.size() - 1][i]->Mul(
						Pow(dReLU(nonlin_l[nonlin_l.size() - 1][i]), 2.0));
		if (hiddenConfig[Hiddenlayers.size() - 1].WeightDecay< 1.0)
		{
		}
	}

	//time backward
	for (int i = 0; i < T; i++)
	{
		cuMatrix tmp = SMR.cuda_W_r.t() * dis[i] * (-1.0f);
		cuMatrix tmp2 = Pow(SMR.cuda_W_r.t(), 2) * dis2[i];
		if (i > 0)
		{
			tmp = tmp
					+ Hiddenlayers[Hiddenlayers.size() - 1].cuda_W_r.t()
							* (*delta_r[delta_r.size() - 1][i - 1]);
			tmp2 = tmp2
					+ Pow(Hiddenlayers[Hiddenlayers.size() - 1].cuda_W_r.t(), 2)
							* (*delta_rd2[delta_rd2.size() - 1][i - 1]);
		}
		tmp.copyTo(*delta_r[delta_r.size() - 1][i]);
		tmp2.copyTo(*delta_rd2[delta_rd2.size() - 1][i]);
		*delta_r[delta_r.size() - 1][i] = delta_r[delta_r.size() - 1][i]->Mul(
				dReLU(nonlin_r[nonlin_r.size() - 1][i]));
		*delta_rd2[delta_rd2.size() - 1][i] =
				delta_rd2[delta_rd2.size() - 1][i]->Mul(
						Pow(dReLU(nonlin_r[nonlin_r.size() - 1][i]), 2.0));
		if (hiddenConfig[0].WeightDecay< 1.0)
		{
		}
	}
//*************  hidden layers **********************
//****************************************************
//****************************************************
//****************************************************
//****************************************************
//***************************************************

	for (int i = HiddenNum - 1; i >= 0; i--)
	{
		// forward part.
		cuMatrix tmp;
		cuMatrix tmp2;
		if (i == 0)
		{
			tmp = *delta_l[i + 1][0] * acti_l[i][0].t();
			tmp2 = *delta_ld2[i + 1][0] * Pow(acti_l[i][0].t(), 2.0f);
			for (int j = 1; j < T; ++j) {
				tmp = tmp + *delta_l[i + 1][j] * acti_l[i][j].t();
				tmp2 = tmp2
						+ *delta_ld2[i + 1][j] * Pow(acti_l[i][j].t(), 2.0f);
			}
		}
		else
		{
			tmp = *delta_l[i + 1][0] * (acti_l[i][0].t() + acti_r[i][0].t());
			tmp2 =
					*delta_ld2[i + 1][0]
							* (Pow(acti_l[i][0].t(), 2.0)
									+ Pow(acti_r[i][0].t(), 2.0));
			for (int j = 1; j < T; ++j) {
				tmp = tmp
						+ *delta_l[i + 1][j]
								* (acti_l[i][j].t() + acti_r[i][j].t());
				tmp2 = tmp2
						+ *delta_ld2[i + 1][j]
								* (Pow(acti_l[i][j].t(), 2.0)
										+ Pow(acti_r[i][j].t(), 2.0));
			}
		}

		Hiddenlayers[i].cuda_U_lgrad =
				tmp / nSamples
						+ Hiddenlayers[i].cuda_U_l
								* hiddenConfig[0].WeightDecay;
		Hiddenlayers[i].cuda_U_ld2 = tmp2 / nSamples
				+hiddenConfig[0].WeightDecay;
		tmp = *delta_l[i + 1][T - 1] * acti_l[i + 1][T - 2].t();
		tmp2 = *delta_ld2[i + 1][T - 1] * Pow(acti_l[i + 1][T - 2].t(), 2.0);
		for (int j = T - 2; j > 0; j--)
		{
			tmp = tmp + *delta_l[i + 1][j] * acti_l[i + 1][j - 1].t();
			tmp2 = tmp2
					+ *delta_ld2[i + 1][j]
							* Pow(acti_l[i + 1][j - 1].t(), 2.0);
		}
		Hiddenlayers[i].cuda_W_lgrad =
				tmp / nSamples
						+ Hiddenlayers[i].cuda_W_l
								* hiddenConfig[0].WeightDecay;
		Hiddenlayers[i].cuda_W_ld2 = tmp2 / nSamples
				+ hiddenConfig[0].WeightDecay;
// backward part.

		if (i == 0)
		{
			tmp = *delta_r[i + 1][0] * acti_r[i][0].t();
			tmp2 = *delta_rd2[i + 1][0] * Pow(acti_r[i][0].t(), 2.0f);
			for (int j = 1; j < T; ++j)
			{
				tmp = tmp + *delta_r[i + 1][j] * acti_r[i][j].t();
				tmp2 = tmp2
						+ *delta_rd2[i + 1][j] * Pow(acti_r[i][j].t(), 2.0f);
			}
		}
		else
		{
			tmp = *delta_r[i + 1][0] * (acti_l[i][0].t() + acti_r[i][0].t());
			tmp2 =
					*delta_rd2[i + 1][0]
							* (Pow(acti_l[i][0].t(), 2.0)
									+ Pow(acti_r[i][0].t(), 2.0));
			for (int j = 1; j < T; ++j) {
				tmp = tmp
						+ *delta_r[i + 1][j]
								* (acti_l[i][j].t() + acti_r[i][j].t());
				tmp2 = tmp2
						+ *delta_rd2[i + 1][j]
								* (Pow(acti_l[i][j].t(), 2.0)
										+ Pow(acti_r[i][j].t(), 2.0));
			}
		}

		Hiddenlayers[i].cuda_U_rgrad =
				tmp / nSamples
						+ Hiddenlayers[i].cuda_U_r
								* hiddenConfig[0].WeightDecay;
		Hiddenlayers[i].cuda_U_rd2 = tmp2 / nSamples
				+hiddenConfig[0].WeightDecay;
		tmp = *delta_r[i + 1][0] * acti_r[i + 1][1].t();
		tmp2 = *delta_rd2[i + 1][0] * Pow(acti_l[i + 1][1].t(), 2.0);
		for (int j = 1; j < T - 1; j++)
		{
			tmp = tmp + *delta_r[i + 1][j] * acti_r[i + 1][j + 1].t();
			tmp2 = tmp2
					+ *delta_rd2[i + 1][j]
							* Pow(acti_r[i + 1][j + 1].t(), 2.0);
		}

		Hiddenlayers[i].cuda_W_rgrad =
				tmp / nSamples
						+ Hiddenlayers[i].cuda_W_r
								* hiddenConfig[0].WeightDecay;
		Hiddenlayers[i].cuda_W_rd2 = tmp2 / nSamples
				+ hiddenConfig[0].WeightDecay;
	}

}
