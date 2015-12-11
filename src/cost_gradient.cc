#include "cost_gradient.h"

using namespace cv;
using namespace std;

//y：mat,行数是nGram 5,宽度是batch_size 50，每次随机抽取50个子句子
//getNetworkCost(sampleX, sampleY, HiddenLayers, smr);
void getNetworkCost(std::vector<Mat> &x,//里面存放5个矩阵，每个矩阵1255*50,存放着50个随机子句子的word对应的int
		Mat &y,//5*50，存放着50个子句子对应的标签，对应的int值，是真实的分类标签
		std::vector<Hl> &hLayers,
		Smr &smr)
{
    int T = x.size();//5
    int nSamples = x[0].cols;//50
    Mat tmp, tmp2;
    // hidden layer forward(向前)
    std::vector<std::vector<Mat> > nonlin_l;//这个存放从左往右隐藏层结果（线性修正之前）
    std::vector<std::vector<Mat> > acti_l;//这个存放从左往右隐藏层结果（线性修正之后的，也就是经过了激活函数）
    std::vector<std::vector<Mat> > bernoulli_l;//使用Dropout防止过度拟合时，正向数据存放的位置

    std::vector<std::vector<Mat> > nonlin_r;//这个存放从右往左隐藏层结果（线性修正之前）
    std::vector<std::vector<Mat> > acti_r;//这个存放从右往左隐藏层结果（线性修正之后的，也就是经过了激活函数）
    std::vector<std::vector<Mat> > bernoulli_r;//使用Dropout防止过度拟合时，反向数据存放的位置

    std::vector<Mat> tmp_vec;
    acti_l.push_back(tmp_vec);
    acti_r.push_back(tmp_vec); 
    //T = 5 , sampleX里面5个矩阵。
    for(int i = 0; i < T; ++i)//5
    {
        acti_l[0].push_back(x[i]);//原始数据，下面将作为输入数据
        acti_r[0].push_back(x[i]);
        tmp_vec.push_back(tmp);
    }
    // hiddenConfig[0].WeightDecay:0.000001
    for(int i = 1; i <= hiddenConfig.size(); ++i)//1   这个属于在隐藏层之间正向和反向传播
    {
        // for each hidden layer
        acti_l.push_back(tmp_vec);
        nonlin_l.push_back(tmp_vec);
        bernoulli_l.push_back(tmp_vec);

        acti_r.push_back(tmp_vec);
        nonlin_r.push_back(tmp_vec);
        bernoulli_r.push_back(tmp_vec);

        // from left to right（正向）
        for(int j = 0; j < T; ++j)//5
        {
            // for each time slot

        	//acti_r[0][1,2,3,4,5] pushed from sampleX,row = 1255, col = 50;
        	//acti_l[0][1,2,3,4,5] pushed from sampleX,row = 1255, col = 50;
        	//hLayers[0].U_l 512 x 1255，隐藏层与上一层的权重。acti_l[0][0] 1255 x 50，将这个作为输入感觉是50个随机子句子的第j行一起输入
        	Mat tmpacti = hLayers[i - 1].U_l * acti_l[i - 1][j];//U_l是隐藏层和前一层的权值，acti_l[i - 1][j]是输入
        	//hLayers[i - 1].W_l:512*512；hLayers[i - 1].U_l:512*1255；acti_r[i - 1][j]:1255*50
            if(j > 0)//下一个时间
            {
            	//hLayers[i - 1].W_l：是当前时刻与前一时刻t-1的权重
            	tmpacti += hLayers[i - 1].W_l * acti_l[i][j - 1];
            }

            if(i > 1)//目前只有一层隐藏层，这一句执行不到
            {
            	tmpacti += hLayers[i - 1].U_l * acti_r[i - 1][j];
            }

            //tmpacti:512*50：每一次输出的矩阵，下面存储没有经过激活函数的输出结果
            tmpacti.copyTo(nonlin_l[i - 1][j]);//tmpacti矩阵复制到nonlin_l[i - 1][j]矩阵
            tmpacti = ReLUa(tmpacti); //保留大于0 的元素,小于0的全为0，这是他的激活函数，叫线性修正
            //http://www.cnblogs.com/tornadomeet/p/3258122.html
            if(hiddenConfig[i - 1].DropoutRate < 1.0)//hiddenConfig[0].DropoutRate :1
            {
                Mat bnl = getBernoulliMatrix(tmpacti.rows, tmpacti.cols, hiddenConfig[i - 1].DropoutRate);
                tmp = tmpacti.mul(bnl);
                tmp.copyTo(acti_l[i][j]);
                bnl.copyTo(bernoulli_l[i - 1][j]);
            }
            else
            {
            	tmpacti.copyTo(acti_l[i][j]);//存放经过激活函数之后的输出结果
            }
        }

        // from right to left
        for(int j = T - 1; j >= 0; --j)//4,3,2,1,0
        {
            // for each time slot
        	//acti_r[0][1,2,3,4,5] pushed from sampleX,row = 1255, col = 50;
        	//hLayers[i - 1].U_r:512*1255; acti_r[i - 1][j]:1255*50
        	//hLayers[i - 1].U_r：隐藏层与前一层权值
            Mat tmpacti = hLayers[i - 1].U_r * acti_r[i - 1][j];
            if(j < T - 1)
            {
            	//hLayers[i - 1].W_r：是当前时刻与前一时刻t-1的权重，反向传导
            	tmpacti += hLayers[i - 1].W_r * acti_r[i][j + 1];//hLayers[i - 1].W_r:512*512
            }
            if(i > 1)//不会执行，因为只有一层隐藏层
            {
            	tmpacti += hLayers[i - 1].U_r * acti_l[i - 1][j];
            }
            tmpacti.copyTo(nonlin_r[i - 1][j]);//矩阵拷贝，存放激活前的结果
            tmpacti = ReLUa(tmpacti); //保留大于0 的元素,小于0的全为0，这是他的激活函数，叫线性修正
            if(hiddenConfig[i - 1].DropoutRate < 1.0)
            {
                Mat bnl = getBernoulliMatrix(tmpacti.rows, tmpacti.cols, hiddenConfig[i - 1].DropoutRate);
                tmp = tmpacti.mul(bnl);
                tmp.copyTo(acti_r[i][j]);
                bnl.copyTo(bernoulli_r[i - 1][j]);
            }
            else
            {
            	tmpacti.copyTo(acti_r[i][j]);//存放激活后的结果
            }
        }
    }

    // softmax layer forward（向前）
    // printf("smr.W_l = %d x %d\n",smr.W_l.rows,smr.W_l.cols);
    // printf("acti_l[acti_l.size() - 1][i] = %d x %d\n",acti_l[acti_l.size() - 1][0].rows,acti_l[acti_l.size() - 1][0].cols);
    // smr.W_l 11 x 512, acti_l[acti_l.size()-1][i] 512 x 50
    // 每列一个样本。
    std::vector<Mat> p;//存放softmax假设函数的结果
    for(int i = 0; i < T; ++i)//5
    {
    	//acti_l[1].size():5; acti_l[1][i]存放的是隐藏层的正向输出，元素只有大于0和等于0；smr.W_l正向权重:11*512
    	//acti_l[1][0]:512*50
        Mat M = smr.W_l * acti_l[acti_l.size() - 1][i];//acti_l.size():2; acti_l[acti_l.size() - 1] outputs of hiddenlayers
        M += smr.W_r * acti_r[acti_r.size() - 1][i];//smr.W_r:11*512,acti_r.size():1
        //M:11*50
        //repeat 函数是将第一个参数扩展为与M相同大小的矩阵,这个reduce求最大值感觉就是假设函数在求概率
        //reduce(M, 0, CV_REDUCE_MAX):得到矩阵每一列的最大值
        M -= repeat(reduce(M, 0, CV_REDUCE_MAX), M.rows, 1);//reduce(M, 0, CV_REDUCE_MAX)的结果是得到矩阵1*50； max element in col
        //上面这句是用M矩阵的每列各个元素减去该列元素的最大值
        M = exp(M);//e的M次方，就是每个元素的M次方
        //reduce(M, 0, CV_REDUCE_SUM)：统计每列之和

        //M:11*50; reduce(M, 0, CV_REDUCE_SUM):1*50
        //repeat(reduce(M, 0, CV_REDUCE_SUM), M.rows, 1):11*50
        //M中各个元素 除以 该列的元素之和
        Mat tmpp = divide(M, repeat(reduce(M, 0, CV_REDUCE_SUM), M.rows, 1));//tmpp:11*50,divide:逐个元素相除
       //刚测试了一下，tmpp的每列概率之和是1，这就是假设函数实现了归一化
        p.push_back(tmpp);
    }

    std::vector<Mat> groundTruth;//存放样本的真实类别
    for(int i = 0; i < T; ++i)//5
    {
        Mat tmpgroundTruth = Mat::zeros(softmaxConfig.NumClasses, nSamples, CV_64FC1);//11*50,11是类别，50是样本数
        for(int j = 0; j < nSamples; j++)//50
        {
        	//y.ATD(i, j):就是取y的i行j列值记作m，再给矩阵tmpgroundTruth的m行j列赋值为1
        	//y:5*50,直接存放原数据第二列对应的类别值，0到10,11类
        	//5*50，存放着50个子句子对应的标签，对应的int值，是真实的分类标签
            tmpgroundTruth.ATD(y.ATD(i, j), j) = 1.0;
        }
        groundTruth.push_back(tmpgroundTruth);//tmpgroundTruth:11*50
    }

//    double J1 = 0.0; // softmax的代价函数
    float J1 = 0.0; // softmax的代价函数
    for(int i = 0; i < T; i++)//5
    {
        J1 +=  - sum1(groundTruth[i].mul(log(p[i])));//mul乘法
    }
    J1 /= nSamples;//50
    //smr.W_l：11*512；  smr.W_r：11*512； softmaxConfig.WeightDecay:1e-06(权值衰减的那个lambda); pow(smr.W_l, 2.0):计算smr.W_l的2次密
    //J2到J4，就是那个权重衰减也就是规则化项，就是反向传导算法，整体代价函数第二项
//    double J2 = (sum1(pow(smr.W_l, 2.0)) + sum1(pow(smr.W_r, 2.0))) * softmaxConfig.WeightDecay / 2;
//    double J3 = 0.0;
 //   double J4 = 0.0;
    float J2 = (sum1(pow(smr.W_l, 2.0)) + sum1(pow(smr.W_r, 2.0))) * softmaxConfig.WeightDecay / 2;
    float J3 = 0.0;
    float J4 = 0.0;
    //hLayers[0].W_l:512 x 512; hLayers[0].W_r:512 x 512
    //.hiddenConfig[hl].WeightDecay..1e-06
    for(int hl = 0; hl < hLayers.size(); hl++)//1
    {
    	//hLayers[hl].W_l:当前时刻t与前一时刻t-1的权重
        J3 += (sum1(pow(hLayers[hl].W_l, 2.0)) + sum1(pow(hLayers[hl].W_r, 2.0))) * hiddenConfig[hl].WeightDecay / 2;
    }
    for(int hl = 0; hl < hLayers.size(); hl++)
    {
    	//hLayers[hl].U_l:隐藏层与前一层的权重
        J4 += (sum1(pow(hLayers[hl].U_l, 2.0)) + sum1(pow(hLayers[hl].U_r, 2.0))) * hiddenConfig[hl].WeightDecay / 2;
    }
    smr.cost = J1 + J2 + J3 + J4;
    if(!is_gradient_checking)
    {
        cout<<", J1 = "<<J1<<", J2 = "<<J2<<", J3 = "<<J3<<", J4 = "<<J4<<", Cost = "<<smr.cost<<endl;
    }

    //softmax layer backward(反向)，这边应该是求偏导数吧
    //groundTruth[0]: 11 x 50，真实类别
    //acti_l[acti_l.size() - 1][0]: 512 x 50; acti_l.size():2;隐藏层正向输出
    //acti_l[acti_l.size() - 1] outputs of hiddenlayers，进行了线性修正
    //acti_l里面存的是一列是一个样本的结果，所以要转成行，使之一行一个样本输入
    //下面几行是softmax回归的代价函数加上权重衰减之后的导数，在第五页最上面那个公式
    tmp = - (groundTruth[0] - p[0]) * acti_l[acti_l.size() - 1][0].t();//该方法通过矩阵表达式（matrix expression）实现矩阵的转置
    for(int i = 1; i < T; ++i)
    {
        tmp += - (groundTruth[i] - p[i]) * acti_l[acti_l.size() - 1][i].t();
    }
    //后面那部分是权重衰减求完偏导数之后的，在反向传导算法的第二页
    //hLayers[i - 1].W_l：是当前时刻与前一时刻t-1的权重
    smr.W_lgrad =  tmp / nSamples + softmaxConfig.WeightDecay * smr.W_l;//softmaxConfig.WeightDecay:1e-06(权值衰减的那个lambda)

    //********  ???  *********  用smr.W_ld2 调整learning rate?
    tmp = pow((groundTruth[0] - p[0]), 2.0) * pow(acti_l[acti_l.size() - 1][0].t(), 2.0);
    for(int i = 1; i < T; ++i)
    {
        tmp += pow((groundTruth[i] - p[i]), 2.0) * pow(acti_l[acti_l.size() - 1][i].t(), 2.0);
    }
    //这个跟上面相比少了w，相当于整体代价函数 \textstyle J(W,b) 的偏导数，是因为权重衰减是作用于W，而不是b。
    smr.W_ld2 = tmp / nSamples + softmaxConfig.WeightDecay;

    tmp = - (groundTruth[0] - p[0]) * acti_r[acti_r.size() - 1][0].t();
    for(int i = 1; i < T; ++i)
    {
        tmp += - (groundTruth[i] - p[i]) * acti_r[acti_r.size() - 1][i].t();
    }
    smr.W_rgrad =  tmp / nSamples + softmaxConfig.WeightDecay * smr.W_r;

    tmp = pow((groundTruth[0] - p[0]), 2.0) * pow(acti_r[acti_r.size() - 1][0].t(), 2.0);
    for(int i = 1; i < T; ++i)
    {
        tmp += pow((groundTruth[i] - p[i]), 2.0) * pow(acti_r[acti_r.size() - 1][i].t(), 2.0);
    }
    smr.W_rd2 = tmp / nSamples + softmaxConfig.WeightDecay;

    // hidden layer backward(向后)，acti_l.size()：2
    std::vector<std::vector<Mat> > delta_l(acti_l.size());//acti_l：前向传导时每一次的残差
    std::vector<std::vector<Mat> > delta_ld2(acti_l.size());
    std::vector<std::vector<Mat> > delta_r(acti_r.size());//acti_r：反向传导时每一次的残差
    std::vector<std::vector<Mat> > delta_rd2(acti_r.size());
    for(int i = 0; i < delta_l.size(); i++)//2
    {
        delta_l[i].clear();
        delta_ld2[i].clear();
        delta_r[i].clear();
        delta_rd2[i].clear();
        Mat tmpmat;
        for(int j = 0; j < T; j++)//5
        {
            delta_l[i].push_back(tmpmat);
            delta_ld2[i].push_back(tmpmat);
            delta_r[i].push_back(tmpmat);
            delta_rd2[i].push_back(tmpmat);
        }
    }

    // Last hidden layer
    // Do BPTT backward pass for the forward hidden layer，
    for(int i = T - 1; i >= 0; i--)//4,3,2,1,0，正向，求残差
    {
    	//下面应该就是应用的反向传导算法第二页公式2，输出层的残差，为了下面计算隐藏层残差
    	//hLayers[i - 1].W_l：是当前时刻与前一时刻t-1的权重；delta_l.size()：2
        tmp = -smr.W_l.t() * (groundTruth[i] - p[i]); // 512 x 50
        tmp2 = pow(smr.W_l.t(), 2.0) * pow((groundTruth[i] - p[i]), 2.0);
        if(i < T - 1)//加上上一个时间
        {
            tmp += hLayers[hLayers.size() - 1].W_l.t() * delta_l[delta_l.size() - 1][i + 1];
            tmp2 += pow(hLayers[hLayers.size() - 1].W_l.t(), 2.0) * delta_ld2[delta_ld2.size() - 1][i + 1];
        }
        tmp.copyTo(delta_l[delta_l.size() - 1][i]);
        tmp2.copyTo(delta_ld2[delta_ld2.size() - 1][i]);
        //nonlin_l是正向是隐藏层的结果存放的位置，这个结果没有经过线性修正，分别存于nonlin_l[0][0],nonlin_l[0][1]....nonlin_l[0][4]中
        //nonlin_l.size():1    dReLU:使得矩阵中元素，大于0，置为1，其他的置0  delta_l.size()：2
        //下面应用的公式应该是反向传导算法的第三页4
        delta_l[delta_l.size() - 1][i] = delta_l[delta_l.size() - 1][i].mul(dReLUa(nonlin_l[nonlin_l.size() - 1][i]));
        delta_ld2[delta_ld2.size() - 1][i] = delta_ld2[delta_ld2.size() - 1][i].mul(pow(dReLUa(nonlin_l[nonlin_l.size() - 1][i]), 2.0));
//        printf("hiddenConfig[hiddenConfig.size() - 1].DropoutRate:%lf\n",hiddenConfig[hiddenConfig.size() - 1].DropoutRate);
        if(hiddenConfig[hiddenConfig.size() - 1].DropoutRate < 1.0)//没执行
        {
            delta_l[delta_l.size() - 1][i] = delta_l[delta_l.size() -1][i].mul(bernoulli_l[bernoulli_l.size() - 1][i]);
            delta_ld2[delta_ld2.size() - 1][i] = delta_ld2[delta_ld2.size() -1][i].mul(pow(bernoulli_l[bernoulli_l.size() - 1][i], 2.0));
        } 
    }


    // Do BPTT backward pass for the backward hidden layer
    for(int i = 0; i < T; i++)//5，反向，求残差
    {
        tmp = -smr.W_r.t() * (groundTruth[i] - p[i]);
        tmp2 = pow(smr.W_r.t(), 2.0) * pow((groundTruth[i] - p[i]), 2.0);
        if(i > 0)
        {
            tmp += hLayers[hLayers.size() - 1].W_r.t() * delta_r[delta_r.size() - 1][i - 1];
            tmp2 += pow(hLayers[hLayers.size() - 1].W_r.t(), 2.0) * delta_rd2[delta_rd2.size() - 1][i - 1];
        }
        //delta_r.size():2   delta_rd2.size()：2
        tmp.copyTo(delta_r[delta_r.size() - 1][i]);//拷贝到delta_r[1][0]到delta_r[1][4]
        tmp2.copyTo(delta_rd2[delta_rd2.size() - 1][i]);//拷贝到delta_rd2[1][0]到delta_rd2[1][4]
        delta_r[delta_r.size() - 1][i] = delta_r[delta_r.size() - 1][i].mul(dReLUa(nonlin_r[nonlin_r.size() - 1][i]));
        delta_rd2[delta_rd2.size() - 1][i] = delta_rd2[delta_rd2.size() - 1][i].mul(pow(dReLUa(nonlin_r[nonlin_r.size() - 1][i]), 2.0));

        if(hiddenConfig[hiddenConfig.size() - 1].DropoutRate < 1.0)
        {
            delta_r[delta_r.size() - 1][i] = delta_r[delta_r.size() -1][i].mul(bernoulli_r[bernoulli_r.size() - 1][i]);
            delta_rd2[delta_rd2.size() - 1][i] = delta_rd2[delta_rd2.size() -1][i].mul(pow(bernoulli_r[bernoulli_r.size() - 1][i], 2.0));
        } 
    }

    // hidden layers
 //   for(int i = delta_l.size() - 2; i > 0; --i)//delta_l.size():2，所以没执行
 //   {
 //       // Do BPTT backward pass for the forward hidden layer
 //       for(int j = T - 1; j >= 0; --j)
 //       {
 //           tmp = hLayers[i].U_l.t() * delta_l[i + 1][j];
//            tmp2 = pow(hLayers[i].U_l.t(), 2.0) * delta_ld2[i + 1][j];
  //          if(j < T - 1)
  //          {
  //              tmp += hLayers[i - 1].W_l.t() * delta_l[i][j + 1];
 //               tmp2 += pow(hLayers[i - 1].W_l.t(), 2.0) * delta_ld2[i][j + 1];
 //           }
 //           tmp += hLayers[i].U_r.t() * delta_r[i + 1][j];
 //           tmp2 += pow(hLayers[i].U_r.t(), 2.0) * delta_rd2[i + 1][j];
  //          tmp.copyTo(delta_l[i][j]);
  //          tmp2.copyTo(delta_ld2[i][j]);
  //          delta_l[i][j] = delta_l[i][j].mul(dReLU(nonlin_l[i - 1][j]));
 //           delta_ld2[i][j] = delta_ld2[i][j].mul(pow(dReLU(nonlin_l[i - 1][j]), 2.0));

  //          if(hiddenConfig[i - 1].DropoutRate < 1.0)
  //          {
  //              delta_l[i][j] = delta_l[i][j].mul(bernoulli_l[i - 1][j]);
  //              delta_ld2[i][j] = delta_ld2[i][j].mul(pow(bernoulli_l[i - 1][j], 2.0));
 //           }
 //       }


//        // Do BPTT backward pass for the backward hidden layer
//        for(int j = 0; j < T; ++j)
//        {
 //           tmp = hLayers[i].U_r.t() * delta_r[i + 1][j];
 //           tmp2 = pow(hLayers[i].U_r.t(), 2.0) * delta_rd2[i + 1][j];
 //           if(j > 0)
 //           {
 //               tmp += hLayers[i - 1].W_r.t() * delta_r[i][j - 1];
//                tmp2 += pow(hLayers[i - 1].W_r.t(), 2.0) * delta_rd2[i][j - 1];
//            }
//            tmp += hLayers[i].U_l.t() * delta_l[i + 1][j];
//            tmp2 += pow(hLayers[i].U_l.t(), 2.0) * delta_ld2[i + 1][j];
 //           tmp.copyTo(delta_r[i][j]);
//            tmp2.copyTo(delta_rd2[i][j]);
//            delta_r[i][j] = delta_r[i][j].mul(dReLU(nonlin_r[i - 1][j]));
//            delta_rd2[i][j] = delta_rd2[i][j].mul(pow(dReLU(nonlin_r[i - 1][j]), 2.0));
 //           if(hiddenConfig[i - 1].DropoutRate < 1.0)
//            {
//                delta_r[i][j] = delta_r[i][j].mul(bernoulli_r[i - 1][j]);
//                delta_rd2[i][j] = delta_rd2[i][j].mul(pow(bernoulli_r[i - 1][j], 2.0));
//            }
//        }
//    }


    //求偏导数
    for(int i = hiddenConfig.size() - 1; i >= 0; i--)//hiddenConfig.size()：1，故这个for循环只执行一次
    {
        // forward part.（向前）
        if(i == 0)//执行这个分支
        {
        	//delta_l[1][0]，[1][1]....[1][4]
        	//delta_l[1][0]：存放正向时的偏导数；acti_l[i][0]：正向时经过激活函数的结果
        	//下面公式是反向传导算法第三页第四个公式
            tmp = delta_l[i + 1][0] * acti_l[i][0].t();
            tmp2 = delta_ld2[i + 1][0] * pow(acti_l[i][0].t(), 2.0);
            for(int j = 1; j < T; ++j)//5
            {
                tmp += delta_l[i + 1][j] * acti_l[i][j].t();
                tmp2 += delta_ld2[i + 1][j] * pow(acti_l[i][j].t(), 2.0);
            }
        }
        else
        {
            tmp = delta_l[i + 1][0] * (acti_l[i][0].t() + acti_r[i][0].t());
            tmp2 = delta_ld2[i + 1][0] * (pow(acti_l[i][0].t(), 2.0) + pow(acti_r[i][0].t(), 2.0));
            for(int j = 1; j < T; ++j)
            {
                tmp += delta_l[i + 1][j] * (acti_l[i][j].t() + acti_r[i][j].t());
                tmp2 += delta_ld2[i + 1][j] * (pow(acti_l[i][j].t(), 2.0) + pow(acti_r[i][j].t(), 2.0));
            }
        }
        //反向传导算法整体代价函数对权重求偏导数，在反向传导算法的第二页上面第一个公式
        hLayers[i].U_lgrad = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].U_l;
        hLayers[i].U_ld2 = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta_l[i + 1][T - 1] * acti_l[i + 1][T - 2].t();//这边是本层上一个时间，故第二位不一样
        tmp2 = delta_ld2[i + 1][T - 1] * pow(acti_l[i + 1][T - 2].t(), 2.0);
        for(int j = T - 2; j > 0; j--)
        {
            tmp += delta_l[i + 1][j] * acti_l[i + 1][j - 1].t();
            tmp2 += delta_ld2[i + 1][j] * pow(acti_l[i + 1][j - 1].t(), 2.0);
        }
        hLayers[i].W_lgrad = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].W_l;
        hLayers[i].W_ld2 = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        // backward part.（向后）
        if(i == 0)//执行
        {
        	//delta_r[1][0]到delta_r[1][4]：残差
            tmp = delta_r[i + 1][0] * acti_r[i][0].t();
            tmp2 = delta_rd2[i + 1][0] * pow(acti_r[i][0].t(), 2.0);
            for(int j = 1; j < T; ++j)
            {
                tmp += delta_r[i + 1][j] * acti_r[i][j].t();
                tmp2 += delta_rd2[i + 1][j] * pow(acti_r[i][j].t(), 2.0);
            }
        }
        else
        {
            tmp = delta_r[i + 1][0] * (acti_l[i][0].t() + acti_r[i][0].t());
            tmp2 = delta_rd2[i + 1][0] * (pow(acti_l[i][0].t(), 2.0) + pow(acti_r[i][0].t(), 2.0));
            for(int j = 1; j < T; ++j)
            {
                tmp += delta_r[i + 1][j] * (acti_l[i][j].t() + acti_r[i][j].t());
                tmp2 += delta_rd2[i + 1][j] * (pow(acti_l[i][j].t(), 2.0) + pow(acti_r[i][j].t(), 2.0));
            }
        }

        hLayers[i].U_rgrad = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].U_r;
        hLayers[i].U_rd2 = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta_r[i + 1][0] * acti_r[i + 1][1].t();
        tmp2 = delta_rd2[i + 1][0] * pow(acti_r[i + 1][1].t(), 2.0);
        for(int j = 1; j < T - 1; j++)
        {
            tmp += delta_r[i + 1][j] * acti_r[i + 1][j + 1].t();
            tmp2 += delta_rd2[i + 1][j] * pow(acti_r[i + 1][j + 1].t(), 2.0);
        }
        hLayers[i].W_rgrad = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].W_r;
        hLayers[i].W_rd2 = tmp2 / nSamples + hiddenConfig[i].WeightDecay;
    }
    // destructor
    acti_l.clear();
    std::vector<std::vector<Mat> >().swap(acti_l);
    nonlin_l.clear();
    std::vector<std::vector<Mat> >().swap(nonlin_l);
    delta_l.clear();
    std::vector<std::vector<Mat> >().swap(delta_l);
    delta_ld2.clear();
    std::vector<std::vector<Mat> >().swap(delta_ld2);
    bernoulli_l.clear();
    std::vector<std::vector<Mat> >().swap(bernoulli_l);
    acti_r.clear();
    std::vector<std::vector<Mat> >().swap(acti_r);
    nonlin_r.clear();
    std::vector<std::vector<Mat> >().swap(nonlin_r);
    delta_r.clear();
    std::vector<std::vector<Mat> >().swap(delta_r);
    delta_rd2.clear();
    std::vector<std::vector<Mat> >().swap(delta_rd2);
    bernoulli_r.clear();
    std::vector<std::vector<Mat> >().swap(bernoulli_r);
    p.clear();
    std::vector<Mat>().swap(p);
    groundTruth.clear();
    std::vector<Mat>().swap(groundTruth);
}

