#include "train_network.h"
#include "cuMatrixVector.h"
#include "InputInit.h"
#include "costGradient.h"
#include "cuMatrix.h"
#include "resultPredict.h"

using namespace cv;
using namespace std;

//trainNetwork(trainX, trainY, HiddenLayers, smr, testX, testY, re_wordmap);
void trainNetwork(const std::vector<std::vector<int> > &x,//就是上面799束，将其每一束的5个组成一个句子，共8852组,范围：0~1254
		std::vector<std::vector<int> > &y,//训练集，8852组;范围：0~10
		std::vector<Hl> &HiddenLayers, //隐含层
		Smr &smr,//softmax层
        const std::vector<std::vector<int> > &tx,//5个词组成一个句子，共2249组
        std::vector<std::vector<int> > &ty,//共2249组
        std::vector<string> &re_wordmap)//re_wordmap存放原始数据中不同的word,最后添加___UNDEFINED___，___PADDING___,1255
{
    if (is_gradient_checking)//false
    {
        batch_size = 2;
        std::vector<Mat> sampleX;
        Mat sampleY = Mat::zeros(nGram, batch_size, CV_64FC1);
        getSample(x, sampleX, y, sampleY, re_wordmap);
        for(int i = 0; i < hiddenConfig.size(); i++)
        {
            gradientChecking_HiddenLayer(HiddenLayers, smr, sampleX, sampleY, i);   
        }
        gradientChecking_SoftmaxLayer(HiddenLayers, smr, sampleX, sampleY);
    }
    else
    {
        cout<<"****************************************************************************"<<endl
            <<"**                       TRAINING NETWORK......                             "<<endl
            <<"****************************************************************************"<<endl<<endl;

        // velocity(速率) vectors.
        Mat v_smr_W_l = Mat::zeros(smr.W_l.size(), CV_64FC1);//size()是Size(cols, rows)矩阵大小：11*512
        Mat smrW_ld2 = Mat::zeros(smr.W_l.size(), CV_64FC1);//11*512
        Mat v_smr_W_r = Mat::zeros(smr.W_l.size(), CV_64FC1);//11*512
        Mat smrW_rd2 = Mat::zeros(smr.W_l.size(), CV_64FC1);//11*512
        cuMatrix cuda_v_smr_W_l(smr.W_l.rows, smr.W_l.cols);
        cuMatrix cuda_smrW_ld2(smr.W_l.rows, smr.W_l.cols);
        cuMatrix cuda_v_smr_W_r(smr.W_l.rows, smr.W_l.cols);
        cuMatrix cuda_smrW_rd2(smr.W_l.rows, smr.W_l.cols);

        std::vector<Mat> v_hl_W_l;
        std::vector<Mat> hlW_ld2;
        std::vector<Mat> v_hl_U_l;
        std::vector<Mat> hlU_ld2;
        std::vector<Mat> v_hl_W_r;
        std::vector<Mat> hlW_rd2;
        std::vector<Mat> v_hl_U_r;
        std::vector<Mat> hlU_rd2;
        vector<cuMatrix> cuda_v_hl_W_l;
        vector<cuMatrix> cuda_hlW_ld2;
        vector<cuMatrix> cuda_v_hl_U_l;
        vector<cuMatrix> cuda_hlU_ld2;
        vector<cuMatrix> cuda_v_hl_W_r;
        vector<cuMatrix> cuda_hlW_rd2;
        vector<cuMatrix> cuda_v_hl_U_r;
        vector<cuMatrix> cuda_hlU_rd2;

        for(int i = 0; i < HiddenLayers.size(); ++i)//1
        {
        	//HiddenLayers[i].W_l:t与t-1时刻的权重
        	//HiddenLayers[i].U_l:隐藏层与前一层的权重
            Mat tmpW = Mat::zeros(HiddenLayers[i].W_l.size(), CV_64FC1);//512*512
            Mat tmpU = Mat::zeros(HiddenLayers[i].U_l.size(), CV_64FC1);//512*1255
            cuMatrix cuda_tmpW(HiddenLayers[i].cuda_W_l.rows(), HiddenLayers[i].cuda_W_l.cols());
            cuMatrix cuda_tmpU(HiddenLayers[i].cuda_U_l.rows(), HiddenLayers[i].cuda_U_l.cols());

            v_hl_W_l.push_back(tmpW);
            v_hl_U_l.push_back(tmpU);
            hlW_ld2.push_back(tmpW);
            hlU_ld2.push_back(tmpU);
            v_hl_W_r.push_back(tmpW);
            v_hl_U_r.push_back(tmpU);
            hlW_rd2.push_back(tmpW);
            hlU_rd2.push_back(tmpU);
            cuda_v_hl_W_l.push_back(cuda_tmpW);
            cuda_v_hl_U_l.push_back(cuda_tmpU);
            cuda_hlW_ld2.push_back(cuda_tmpW);
            cuda_hlU_ld2.push_back(cuda_tmpU);
            cuda_v_hl_W_r.push_back(cuda_tmpW);
            cuda_v_hl_U_r.push_back(cuda_tmpU);
            cuda_hlW_rd2.push_back(cuda_tmpW);
            cuda_hlU_rd2.push_back(cuda_tmpU);
        }

 //       double Momentum_w = 0.5;
 //       double Momentum_u = 0.5;
  //      double Momentum_d2 = 0.5;
        float Momentum_w = 0.5;
        float Momentum_u = 0.5;
        float Momentum_d2 = 0.5;
        Mat lr_W;
        Mat lr_U;
        cuMatrix cuda_lr_W;
        cuMatrix cuda_lr_U;
//        double mu = 1e-2;//0.02,不知道干什么的
        float mu = 1e-2;//0.02,不知道干什么的
        //printf("training_epochs:%d\niter_per_epo:%d\n",training_epochs,iter_per_epo);
        //30 100;
        //printf("batch_size:%d\n",batch_size);//批，随机选50个子句子
        //50
        for(int epo = 1; epo <= training_epochs; epo++)//30个时间,epoch指的是一个特定的时间
        {
            for(int k = 0; k <= iter_per_epo * epo; k++)//迭代的次数；iter_per_epo:100;
            {
                if(k > 30)
                {
                	Momentum_w = 0.95; Momentum_u = 0.95; Momentum_d2 = 0.90;
                }
                std::vector<Mat> sampleX;

                cuMatrixVector cuda_sampleX;//里面存放5个矩阵，每个矩阵1255**50
                for (int i = 0; i < x[0].size(); i++)//5
                {
                	cuda_sampleX.push_back(new cuMatrix(re_wordmap.size(),batch_size));//期盼是1255*50
                }
                cuda_sampleX.toGpu();//数据上传到gpu

                Mat sampleY = Mat::zeros(nGram, batch_size, CV_64FC1); //nGram sub-sentence len   5*50
                cuMatrix cuda_sampleY(nGram, batch_size);

                //x：训练集子句子的8852组，子句子的长度是5；
                //sampleX：vector<Mat>；范围：0~1254，原始数据第一列和从0开始的int值
                //y：子句子对应的元数据的标签label，范围：0~10,由原始数据第二列对应int值得到
                //sampleY：mat,行数是nGram 5,宽度是batch_size 50，每次随机抽取50个子句子
                //re_wordmap存放原始数据中不同的word,最后添加___UNDEFINED___，___PADDING___,1255
 //             getSample(x, sampleX, y, sampleY, re_wordmap);
                init_acti0(cuda_sampleX,cuda_sampleY, nGram,sampleY.cols);
                //sampleX:原始样本，也就是第一列对应的int值
                //sampleY：5*50;存放随机的50个子句子，每个句子对应的5行的第二列对应的int值,因为总共11类，故矩阵中值为0-10
//                getNetworkCost(sampleX, sampleY, HiddenLayers, smr);
                cuda_getNetworkCost(cuda_sampleX, cuda_sampleY, HiddenLayers, smr);


//                // softmax update
//                smrW_ld2 = Momentum_d2 * smrW_ld2 + (1.0 - Momentum_d2) * smr.W_ld2;
//                lr_W = smr.lr_W / (smrW_ld2 + mu);
//                v_smr_W_l = v_smr_W_l * Momentum_w + (1.0 - Momentum_w) * smr.W_lgrad.mul(lr_W);
//                smr.W_l -= v_smr_W_l;
//
//                smrW_rd2 = Momentum_d2 * smrW_rd2 + (1.0 - Momentum_d2) * smr.W_rd2;
//                lr_W = smr.lr_W / (smrW_rd2 + mu);
//                v_smr_W_r = v_smr_W_r * Momentum_w + (1.0 - Momentum_w) * smr.W_rgrad.mul(lr_W);
//                smr.W_r -= v_smr_W_r;
//
//                // hidden layer update
//                for(int i = 0; i < HiddenLayers.size(); i++)//1
//                {
//                    hlW_ld2[i] = Momentum_d2 * hlW_ld2[i] + (1.0 - Momentum_d2) * HiddenLayers[i].W_ld2;
//                    hlU_ld2[i] = Momentum_d2 * hlU_ld2[i] + (1.0 - Momentum_d2) * HiddenLayers[i].U_ld2;
//                    lr_W = HiddenLayers[i].lr_W / (hlW_ld2[i] + mu);
//                    lr_U = HiddenLayers[i].lr_U / (hlU_ld2[i] + mu);
//                    v_hl_W_l[i] = v_hl_W_l[i] * Momentum_w + (1.0 - Momentum_w) * HiddenLayers[i].W_lgrad.mul(lr_W);
//                    v_hl_U_l[i] = v_hl_U_l[i] * Momentum_u + (1.0 - Momentum_u) * HiddenLayers[i].U_lgrad.mul(lr_U);
//                    HiddenLayers[i].W_l -= v_hl_W_l[i];//v_hl_W_l[i]得到的应该是学习速率*偏导数
//                    HiddenLayers[i].U_l -= v_hl_U_l[i];
//
//                    hlW_rd2[i] = Momentum_d2 * hlW_rd2[i] + (1.0 - Momentum_d2) * HiddenLayers[i].W_rd2;
//                    hlU_rd2[i] = Momentum_d2 * hlU_rd2[i] + (1.0 - Momentum_d2) * HiddenLayers[i].U_rd2;
//                    lr_W = HiddenLayers[i].lr_W / (hlW_rd2[i] + mu);
//                    lr_U = HiddenLayers[i].lr_U / (hlU_rd2[i] + mu);
//                    v_hl_W_r[i] = v_hl_W_r[i] * Momentum_w + (1.0 - Momentum_w) * HiddenLayers[i].W_rgrad.mul(lr_W);
//                    v_hl_U_r[i] = v_hl_U_r[i] * Momentum_u + (1.0 - Momentum_u) * HiddenLayers[i].U_rgrad.mul(lr_U);
//                    HiddenLayers[i].W_r -= v_hl_W_r[i];
//                    HiddenLayers[i].U_r -= v_hl_U_r[i];
//                }
//                sampleX.clear();
//                std::vector<Mat>().swap(sampleX);

                cuda_smrW_ld2 = cuda_smrW_ld2 * Momentum_d2 + smr.cuda_W_ld2 * (1.0 - Momentum_d2);
                cuda_lr_W = smr.lr_W / (cuda_smrW_ld2 + mu);

                cuda_v_smr_W_l = cuda_v_smr_W_l * Momentum_w+ smr.cuda_W_lgrad.Mul(cuda_lr_W) * (1.0 - Momentum_w);
                smr.cuda_W_l = smr.cuda_W_l - cuda_v_smr_W_l;

                cuda_smrW_rd2 = cuda_smrW_rd2 * Momentum_d2 + smr.cuda_W_rd2 * (1.0 - Momentum_d2);
                cuda_lr_W = smr.lr_W / (cuda_smrW_rd2 + mu);

                cuda_v_smr_W_r = cuda_v_smr_W_r * Momentum_w+ smr.cuda_W_rgrad.Mul(cuda_lr_W) * (1.0 - Momentum_w);
                smr.cuda_W_r = smr.cuda_W_r - cuda_v_smr_W_r;
                // hidden layer update

                for (int i = 0; i < HiddenLayers.size(); i++)
                {

                	cuda_hlW_ld2[i] = cuda_hlW_ld2[i] * Momentum_d2+ HiddenLayers[i].cuda_W_ld2 * (1.0 - Momentum_d2);
                	cuda_hlU_ld2[i] = cuda_hlU_ld2[i] * Momentum_d2+ HiddenLayers[i].cuda_U_ld2 * (1.0 - Momentum_d2);
                	cuda_lr_W = HiddenLayers[i].lr_W / (cuda_hlW_ld2[i] + mu);
                	cuda_lr_U = HiddenLayers[i].lr_U / (cuda_hlU_ld2[i] + mu);
                	cuda_v_hl_W_l[i] = cuda_v_hl_W_l[i] * Momentum_w+ HiddenLayers[i].cuda_W_lgrad.Mul(cuda_lr_W)* (1.0 - Momentum_w);
                	cuda_v_hl_U_l[i] = cuda_v_hl_U_l[i] * Momentum_u+ HiddenLayers[i].cuda_U_lgrad.Mul(cuda_lr_U)
                								* (1.0 - Momentum_u);
                	HiddenLayers[i].cuda_W_l = HiddenLayers[i].cuda_W_l - cuda_v_hl_W_l[i];
                	HiddenLayers[i].cuda_U_l = HiddenLayers[i].cuda_U_l - cuda_v_hl_U_l[i];

                	cuda_hlW_rd2[i] = cuda_hlW_rd2[i] * Momentum_d2+ HiddenLayers[i].cuda_W_rd2 * (1.0 - Momentum_d2);
                	cuda_hlU_rd2[i] = cuda_hlU_rd2[i] * Momentum_d2+ HiddenLayers[i].cuda_U_rd2 * (1.0 - Momentum_d2);

                	cuda_lr_W = HiddenLayers[i].lr_W / (cuda_hlW_rd2[i] + mu);
                	cuda_lr_U = HiddenLayers[i].lr_U / (cuda_hlU_rd2[i] + mu);
                	cuda_v_hl_W_r[i] = cuda_v_hl_W_r[i] * Momentum_w+ HiddenLayers[i].cuda_W_rgrad.Mul(cuda_lr_W)
                								* (1.0 - Momentum_w);
                	cuda_v_hl_U_r[i] = cuda_v_hl_U_r[i] * Momentum_u+ HiddenLayers[i].cuda_U_rgrad.Mul(cuda_lr_U)
                								* (1.0 - Momentum_u);
                	HiddenLayers[i].cuda_W_r = HiddenLayers[i].cuda_W_r - cuda_v_hl_W_r[i];
                	HiddenLayers[i].cuda_U_r = HiddenLayers[i].cuda_U_r - cuda_v_hl_U_r[i];
                }


            }
            
            if(!is_gradient_checking)
            {
//               cout<<"Test training data: "<<endl;;
//                testNetwork(x, y, HiddenLayers, smr, re_wordmap);
//                cout<<"Test testing data: "<<endl;;
//                testNetwork(tx, ty, HiddenLayers, smr, re_wordmap);

            	cout << "Test training data: " << endl;
            	testNetwork(x, y,HiddenLayers, smr, re_wordmap,1,nGram);
            	cout << "Test test data: " << endl;
            	testNetwork(tx, ty,HiddenLayers, smr, re_wordmap,0,nGram);
            }
        }//30时间循环结束
        v_smr_W_l.release();
        v_hl_W_l.clear();
        //使用这种方法的前提是vector从前存储了大量数据，比如10000000，经过各种处理后，
        //现在只有100条，那么向清空原来数据所占有的空间，就可以通过这种交换技术swap技法
        //就是通过交换函数swap（），使得vector离开其自身的作用域，从而强制释放vector所占的内存空间。
        std::vector<Mat>().swap(v_hl_W_l);
        v_hl_U_l.clear();
        std::vector<Mat>().swap(v_hl_U_l);
        hlW_ld2.clear();
        std::vector<Mat>().swap(hlW_ld2);
        hlU_ld2.clear();
        std::vector<Mat>().swap(hlU_ld2);
        v_smr_W_r.release();
        v_hl_W_r.clear();
        std::vector<Mat>().swap(v_hl_W_r);
        v_hl_U_r.clear();
        std::vector<Mat>().swap(v_hl_U_r);
        hlW_rd2.clear();
        std::vector<Mat>().swap(hlW_rd2);
        hlU_rd2.clear();
        std::vector<Mat>().swap(hlU_rd2);
        
    }
}




