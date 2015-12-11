#include "weight_init.h"

using namespace cv;
using namespace std;

//  weightRandomInit(tpntw, word_vec_len, hiddenConfig[0].NumHiddenNeurons);
void weightRandomInit(Hl &ntw, int inputsize, int hiddensize)//hiddensize是512，inputsize是1255,是不同的键值对个数
{
	const float epsilon = 0.12;
	Mat tmp_ran = Mat::ones(hiddensize, inputsize, CV_32FC1);
	randu(tmp_ran, Scalar(-1.0), Scalar(1.0));
	ntw.U_l = tmp_ran * epsilon;
	ntw.cuda_U_l = cuMatrix((float*) ntw.U_l.data, hiddensize,inputsize);

	ntw.U_lgrad =  Mat::zeros(hiddensize, inputsize, CV_64FC1);
	ntw.U_ld2 = Mat::zeros(hiddensize, inputsize, CV_64FC1);
	ntw.cuda_U_lgrad = cuMatrix(hiddensize, inputsize);
	ntw.cuda_U_ld2 = cuMatrix(hiddensize, inputsize);
	ntw.lr_U = lrate_w;

	Mat tmp_ran1 = Mat::ones(hiddensize, hiddensize, CV_32FC1);
	randu(tmp_ran1, Scalar(-1.0), Scalar(1.0));
	ntw.W_l = tmp_ran1 * epsilon;
	ntw.cuda_W_l = cuMatrix((float*) ntw.W_l.data, hiddensize,hiddensize);

	ntw.W_lgrad = Mat::zeros(hiddensize, hiddensize, CV_64FC1);
	ntw.W_ld2 =  Mat::zeros(hiddensize, hiddensize, CV_64FC1);
	ntw.cuda_W_lgrad = cuMatrix(hiddensize, hiddensize);
	ntw.cuda_W_ld2 = cuMatrix(hiddensize, hiddensize);
	ntw.lr_W = lrate_w;

	Mat tmp_ran2 = Mat::ones(hiddensize, inputsize, CV_32FC1);
	randu(tmp_ran2, Scalar(-1.0), Scalar(1.0));
	ntw.U_r = tmp_ran2 * epsilon;
	ntw.cuda_U_r = cuMatrix((float*) ntw.U_r.data, hiddensize,inputsize);

	ntw.U_rgrad = Mat::zeros(hiddensize, inputsize, CV_64FC1);
	ntw.U_rd2 = Mat::zeros(hiddensize, inputsize, CV_64FC1);
	ntw.cuda_U_rgrad = cuMatrix(hiddensize, inputsize);
	ntw.cuda_U_rd2 = cuMatrix(hiddensize, inputsize);

	Mat tmp_ran3 = Mat::ones(hiddensize, hiddensize, CV_32FC1);
	randu(tmp_ran3, Scalar(-1.0), Scalar(1.0));
	ntw.W_r = tmp_ran3 * epsilon;
	ntw.cuda_W_r = cuMatrix((float*) ntw.W_r.data, hiddensize,hiddensize);

	ntw.W_rgrad =Mat::zeros(hiddensize, hiddensize, CV_64FC1);
	ntw.W_rd2 =Mat::zeros(hiddensize, hiddensize, CV_64FC1);
	ntw.cuda_W_rgrad = cuMatrix(hiddensize, hiddensize);
	ntw.cuda_W_rd2 = cuMatrix(hiddensize, hiddensize);


////    double epsilon = 0.12;
//	 float epsilon = 0.12f;
//    // weight between hidden layer with previous layer
//    ntw.U_l = Mat::ones(hiddensize, inputsize, CV_64FC1);//hiddensize是row，inputsize是col
//    randu(ntw.U_l, Scalar(-1.0), Scalar(1.0));//你可以通过用randu()函数产生的随机值来填充矩阵。另外两个参数应该是上下限
//    ntw.U_l = ntw.U_l * epsilon;
//    ntw.cuda_U_l= cuMatrix((float*) ntw.U_l.data,hiddensize,inputsize);
//
//    ntw.U_lgrad = Mat::zeros(hiddensize, inputsize, CV_64FC1);//hiddensize行数，inputsize列数，CV_64FC1：创建的矩阵的类型
//    ntw.cuda_U_lgrad= cuMatrix(hiddensize,inputsize);
//
//    ntw.U_ld2 = Mat::zeros(ntw.U_l.size(), CV_64FC1);//ntw.U_l.size()：nat 有row，有col
//    ntw.cuda_U_ld2= cuMatrix(hiddensize,inputsize);
//
//    ntw.lr_U = lrate_w;//3e-3
//    //// weight between current time t with previous time t-1
//    ntw.W_l = Mat::ones(hiddensize, hiddensize, CV_64FC1);
//    randu(ntw.W_l, Scalar(-1.0), Scalar(1.0));
//    ntw.W_l = ntw.W_l * epsilon;
//    ntw.cuda_W_l= cuMatrix((float*) ntw.W_l.data,hiddensize,hiddensize);
//
//    ntw.W_lgrad = Mat::zeros(hiddensize, hiddensize, CV_64FC1);
//    ntw.cuda_W_lgrad= cuMatrix(hiddensize,hiddensize);
//
//    ntw.W_ld2 = Mat::zeros(ntw.W_l.size(), CV_64FC1);
//    ntw.cuda_W_ld2= cuMatrix(hiddensize,hiddensize);
//    ntw.lr_W = lrate_w;//0.003
//
//    ntw.U_r = Mat::ones(hiddensize, inputsize, CV_64FC1);
//    randu(ntw.U_r, Scalar(-1.0), Scalar(1.0));
//    ntw.U_r = ntw.U_r * epsilon;
//    ntw.cuda_U_r= cuMatrix(hiddensize,inputsize);
//
//    ntw.U_rgrad = Mat::zeros(hiddensize, inputsize, CV_64FC1);
//    ntw.cuda_U_rgrad= cuMatrix(hiddensize,inputsize);
//
//    ntw.U_rd2 = Mat::zeros(ntw.U_r.size(), CV_64FC1);
//    ntw.cuda_U_rd2= cuMatrix(hiddensize,inputsize);
//    ntw.lr_U = lrate_w;//0.003
//
//    ntw.W_r = Mat::ones(hiddensize, hiddensize, CV_64FC1);
//    randu(ntw.W_r, Scalar(-1.0), Scalar(1.0));
//    ntw.W_r = ntw.W_r * epsilon;
//    ntw.cuda_W_r= cuMatrix((float*) ntw.W_r.data,hiddensize,hiddensize);
//
//    ntw.W_rgrad = Mat::zeros(hiddensize, hiddensize, CV_64FC1);
//    ntw.cuda_W_rgrad= cuMatrix(hiddensize,hiddensize);
//
//    ntw.W_rd2 = Mat::zeros(ntw.W_r.size(), CV_64FC1);
//    ntw.cuda_W_rd2= cuMatrix(hiddensize,hiddensize);
//    ntw.lr_W = lrate_w;
}

//有文件时，nclasses：11；nfeatures：1255
void weightRandomInit(Smr &smr, int nclasses, int nfeatures)
{
		const float epsilon = 0.12;
		Mat tmp_ran = Mat::ones(nclasses, nfeatures, CV_32FC1);
		randu(tmp_ran, Scalar(-1.0), Scalar(1.0));
		smr.W_l = tmp_ran * epsilon;
		smr.cuda_W_l = cuMatrix((float*) smr.W_l.data, nclasses, nfeatures);
		smr.W_lgrad = Mat::zeros(nclasses, nfeatures, CV_64FC1);
		smr.W_ld2 = Mat::zeros(nclasses, nfeatures, CV_64FC1);
		smr.cuda_W_lgrad = cuMatrix(nclasses, nfeatures);
		smr.cuda_W_ld2 = cuMatrix(nclasses, nfeatures);

		Mat tmp_ran1 = Mat::ones(nclasses, nfeatures, CV_32FC1);
		randu(tmp_ran1, Scalar(-1.0), Scalar(1.0));
		smr.W_r = tmp_ran1 * epsilon;
		smr.cuda_W_r = cuMatrix((float*) smr.W_r.data, nclasses, nfeatures);
		smr.W_rgrad = Mat::zeros(nclasses, nfeatures, CV_64FC1);
		smr.W_rd2 = Mat::zeros(nclasses, nfeatures, CV_64FC1);
		smr.cuda_W_rgrad = cuMatrix(nclasses, nfeatures);
		smr.cuda_W_rd2 = cuMatrix(nclasses, nfeatures);
		smr.cost = 0.0;
		smr.lr_W = lrate_w;//3e-3


//    double epsilon = 0.12;
//	 float epsilon = 0.12f;
//    smr.W_l = Mat::ones(nclasses, nfeatures, CV_64FC1);
//    randu(smr.W_l, Scalar(-1.0), Scalar(1.0));
//    smr.W_l = smr.W_l * epsilon;
//    smr.cuda_W_l= cuMatrix((float*) smr.W_l.data,nclasses,nfeatures);
//
//    smr.W_lgrad = Mat::zeros(nclasses, nfeatures, CV_64FC1);
//    smr.cuda_W_lgrad= cuMatrix(nclasses,nfeatures);
//
//    smr.W_ld2 = Mat::zeros(smr.W_l.size(), CV_64FC1);
//    smr.cuda_W_ld2= cuMatrix(nclasses,nfeatures);
//
//    smr.W_r = Mat::ones(nclasses, nfeatures, CV_64FC1);
//    randu(smr.W_r, Scalar(-1.0), Scalar(1.0));
//    smr.W_r = smr.W_r * epsilon;
//    smr.cuda_W_r= cuMatrix((float*) smr.W_r.data,nclasses,nfeatures);
//
//    smr.W_rgrad = Mat::zeros(nclasses, nfeatures, CV_64FC1);
//    smr.cuda_W_rgrad= cuMatrix(nclasses,nfeatures);
//
//    smr.W_rd2 = Mat::zeros(smr.W_r.size(), CV_64FC1);
//    smr.cuda_W_rd2= cuMatrix(nclasses,nfeatures);
//
//    smr.cost = 0.0;
//    smr.lr_W = lrate_w;//3e-3
}

void rnnInitPrarms(std::vector<Hl> &HiddenLayers, Smr &smr)
{
    // Init Hidden layers
	// hiddenConfig.size():1
	if(hiddenConfig.size() > 0)//隐藏层信息，读配置文件读出来的
	{
		Hl tpntw;//隐含层结构体
		//hiddenConfig[0].NumHiddenNeurons,512个神经元,word_vec_len是1255,是不同的键值对
        weightRandomInit(tpntw, word_vec_len, hiddenConfig[0].NumHiddenNeurons);
        HiddenLayers.push_back(tpntw);
        cout<<".... hiddenConfig.size() "<<hiddenConfig.size()<<endl;
        for(int i = 1; i < hiddenConfig.size(); i++)//1
        {
            Hl tpntw2;
            weightRandomInit(tpntw2, hiddenConfig[i - 1].NumHiddenNeurons, hiddenConfig[i].NumHiddenNeurons);
            HiddenLayers.push_back(tpntw2);
        }
    }
    // Init Softmax layer
    if(hiddenConfig.size() == 0)//没有配置文件
    {
    	//smr：softMax层，softmaxConfig.NumClasses：11，word_vec_len是1255
        weightRandomInit(smr, softmaxConfig.NumClasses, word_vec_len);
    }
    else
    {
    	//smr：softMax层，softmaxConfig.NumClasses：11，word_vec_len是1255；
    	//hiddenConfig[hiddenConfig.size() - 1].NumHiddenNeurons ：512个神经元
        weightRandomInit(smr, softmaxConfig.NumClasses, hiddenConfig[hiddenConfig.size() - 1].NumHiddenNeurons);
    }
}


