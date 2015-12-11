#include "InputInit.h"


__global__ void set_sampleY_kernel(float* sampleY,//5*50训练集标签
		int* src, //训练集标签数据
		int* dev_ran,//0到49的打乱次序
		int cols, //50
		int ngram) //5
{
	int tid = threadIdx.x;//0——4
	int bid = blockIdx.x;//0-——49
	sampleY[tid * cols + bid] = src[dev_ran[bid] * ngram + tid];
}


__global__ void set_acti0_kernel(float** acti0,
		//src：存放原始trainx：vector<vector<int>>，即原始数据第一列对应的从0开始到1254的一个值
		int* src,//已经上传到gpu上的训练集
		int* dev_ran,//0到49的打乱次序
		int cols,//50
		int ngram)
{//5个线程，50个线程块
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	float *p = acti0[tid];//第tid个mat的地址，acti0原本有5个矩阵，每个矩阵1255*50
	//src的维数有trainX.size()个，每一维5个int大小
	int n = src[dev_ran[bid] * ngram + tid];//dev_ran[bid] * ngram + tid：每次得到的是每个子句子的每一行
	//n是0到1254的一个值   因为矩阵转为指针都是按行存放的故下面式子正确
	p[n * cols + bid] = 1;//p指向某一个矩阵，故p[0]、p[1]....p[1255]，这是给矩阵第n*cols行第bid列赋值
}


void init_acti0(cuMatrixVector& acti_0,//vector<mat>，里面存放5个矩阵，每个矩阵1255*50
		cuMatrix& sampleY,//mat 5*50  存放50个子句子，每个子句子5行的类别
		int nGram,//每个子句子5行
		int batch_Size)//随机取50个子句子
{
	int batch_size = batch_Size;//50
	int ngram = nGram;//5

	int *dev_ran = NULL;//存放打乱之后的数字，且在gpu上面
	Samples::instance()->randproductor_init();//初始化randproductor：int*  存放从0到train_num（训练集子句子个数）
	cudaError_t cudaStat = cudaMalloc((void**) &dev_ran, batch_size * sizeof(int));
	if (cudaStat != cudaSuccess)
	{
		printf("init_acti0 failed\n");
		exit(0);
	}

	checkCudaErrors(cudaMemcpy(dev_ran,
			        Samples::instance()->get_rand(1),//打乱数据之后
					batch_size * sizeof(int),//拷贝大小50个数
					cudaMemcpyHostToDevice));
	//内建dim3类型,是一个三维数组，可以用于指定启动的线程块的数量：定义grid和block的组织方法。
	dim3 block = dim3(batch_size);//50
	dim3 thread = dim3(ngram);//5
    //<<<>>>中第一个参数表示设备在执行核函数时使用的并行线程块的数量
	//第二个参数表示cuda运行时在每个线程块中创建的线程数量
	set_acti0_kernel<<<block, thread>>>(acti_0.get_devPoint(),//已经在gpu上面
			Samples::instance()->get_trainX(),//gpu上的训练集
			dev_ran,//50个打乱的子句子的序号
			batch_size, //50
			ngram);//5

    //涉及到多个内核函数运行的时候就需要了，内核函数中各个线程在运行的时候不是同步的，所以在计算完成的时候一般需要同步一下；
	checkCudaErrors(cudaDeviceSynchronize());//同步
	getLastCudaError("set_acti0_kernel-2");

	set_sampleY_kernel<<<block, thread>>>(sampleY.getDev(),
			Samples::instance()->get_trainY(), //gpu上的训练集标签
			dev_ran, //0到49的打乱次序
			batch_size, //50
			ngram);//5
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("set_sampleY_kernel-2");
	checkCudaErrors(cudaFree(dev_ran));
}


__global__ void set_gt_kernel(float** gt_, //将要存放类别的向量
		float* y,
		int rows, //5
		int cols)//50
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	assert(tid < rows && bid < cols);
	float* p = gt_[tid];//gt_中有5个矩阵，每个矩阵11*50
	int i = y[tid * cols + bid];//这个i的值应该是类别
	assert(i < 12);
	p[i * cols + bid] = 1.0;
}

//把真实分类类别放到gt中，gt已经在gpu上面
void set_groundtruth(cuMatrixVector& gt, cuMatrix& sampleY)
{
	//sampleY:5*50
	dim3 block = dim3(sampleY.cols());//50
	dim3 thread = dim3(sampleY.rows());//5
	set_gt_kernel<<<block, thread>>>(gt.get_devPoint(), sampleY.getDev(),
			sampleY.rows(), sampleY.cols());
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("set_groundtruth ");
}


void initTestdata(vector<vector<int> > &testX,
		vector<vector<int> > &testY,
		int nGram)
{
	//为两个指针分配空间
	int *host_X = (int *) malloc(sizeof(int) * testX.size() * nGram);
	int *host_Y = (int *) malloc(sizeof(int) * testY.size() * nGram);

	for (int i = 0; i < testX.size(); i++)//测试集子句子的个数，每个子句子5行
	{
		//每个子句子的第一行开始，5行大小，拷贝进去，再拷下一个子句子的5行
		memcpy(host_X + i * nGram, &testX[i][0], sizeof(int) * nGram);
	}
	for (int i = 0; i < testY.size(); i++)
	{
		memcpy(host_Y + i * nGram, &testY[i][0], sizeof(int) * nGram);
	}
	//数据上传至gpu上
	Samples::instance()->testX2gpu(host_X, sizeof(int) * testX.size() * nGram);
	Samples::instance()->testY2gpu(host_Y, sizeof(int) * testY.size() * nGram);
	free (host_X);
	free (host_Y);
}


void initTraindata(vector<vector<int> > &trainX,
		vector<vector<int> > &trainY,
		int nGram)
{
	//trainX占用空间sizeof(int) * trainX.size() * nGram
	int *host_X = (int *) malloc(sizeof(int) * trainX.size() * nGram);
	int *host_Y = (int *) malloc(sizeof(int) * trainY.size() * nGram);
	//数据拷贝
	for (int i = 0; i < trainX.size(); i++)
	{
		//void*memcpy(void*dest, const void*src,unsigned int count);
		memcpy(host_X + i * nGram, &trainX[i][0], sizeof(int) * nGram);
	}
	for (int i = 0; i < trainY.size(); i++)
	{
		memcpy(host_Y + i *nGram, &trainY[i][0], sizeof(int) * nGram);
	}
	//训练集数据上传到gpu
	Samples::instance()->trainX2gpu(host_X,sizeof(int) * trainX.size() * nGram);
	Samples::instance()->trainY2gpu(host_Y,sizeof(int) * trainY.size() * nGram);
	free (host_X);
	free (host_Y);
}


void Data2GPU(vector<vector<int> > &trainX,//原始数据第一列对应的int值
		vector<vector<int> > &trainY,//原始数据第二列得到的int值
		vector<vector<int> > &testX,
		vector<vector<int> > &testY,
		int nGram)
{
	initTraindata(trainX,trainY,nGram);
	initTestdata(testX,testY,nGram);

}


__global__ void getDataMat_kernel(float** sampleX, int* src, int off, int cols,
		int ngram)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	float *p = sampleX[tid];
	int n = src[(off + bid) * ngram + tid];
	p[n * cols + bid] = 1.0;
}

void getDataMat(cuMatrixVector &sampleX, int off, int bs, int n,
		bool flag,int ngram)
{
		int n_gram = ngram;
		for (int i = 0; i < 5; i++)
		{
			sampleX.push_back(new cuMatrix(n, bs));
		}
		sampleX.toGpu();
		dim3 thread = dim3(n_gram);
		dim3 block = dim3(bs);
		if (flag)
		{
			getDataMat_kernel<<<block, thread>>>(sampleX.get_devPoint(),
					Samples::instance()->get_trainX(), off, bs, n_gram);
		}
		else
		{
			getDataMat_kernel<<<block, thread>>>(sampleX.get_devPoint(),
					Samples::instance()->get_testX(), off, bs, n_gram);
		}
		checkCudaErrors(cudaDeviceSynchronize());
		getLastCudaError("getDataMat_kernel ");

}

__global__ void get_res_array_kernel(float* src, int* dev_res, int rows,
		int cols) {
	int bid = blockIdx.x;
	float max = src[bid];
	dev_res[bid] = 0;
	for (int i = 1; i < rows; i++) {
		if (max < src[i * cols + bid]) {
			max = src[i * cols + bid];
			dev_res[bid] = i;
		}
	}
}

void get_res_array(cuMatrix src, int *res, int offset)
{
	int *dev_res;
	checkCudaErrors(cudaMalloc((void** )&dev_res, sizeof(int) * src.cols()));
	get_res_array_kernel<<<src.cols(), 1>>>(src.getDev(), dev_res, src.rows(),
			src.cols());
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("get_res_array ");
	checkCudaErrors(
			cudaMemcpy(res + offset, dev_res, sizeof(int) * src.cols(),
					cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(dev_res));
}

__global__ void set_label_kernel(int* dst, int *src, int num, int threadnum,
		int mid) {
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int off = bid * threadnum + tid;
	if (off < num) {
		dst[off] = src[off * (mid * 2 + 1) + mid];
	}
}

void set_label(int* label, int size,bool flag)
{
	int *dev_label;
	int mid =5/ 2;
	int num = size;
	checkCudaErrors(cudaMalloc((void** )&dev_label, sizeof(int) * num));
	int threadnum = Devices::instance()->max_ThreadsPerBlock() > num ? num : Devices::instance()->max_ThreadsPerBlock();
	int blocknum = num / threadnum + 1;
	dim3 blocks(blocknum);
	dim3 threads(threadnum);
	if (flag) {
		set_label_kernel<<<blocks, threads>>>(dev_label,
				Samples::instance()->get_trainY(), num, threadnum, mid);
		checkCudaErrors(cudaDeviceSynchronize());
		getLastCudaError("set_label");
	} else {
		set_label_kernel<<<blocks, threads>>>(dev_label,
				Samples::instance()->get_testY(), num, threadnum, mid);
		checkCudaErrors(cudaDeviceSynchronize());
		getLastCudaError("set_label");
	}
	checkCudaErrors(
			cudaMemcpy(label, dev_label, sizeof(int) * num,
					cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(dev_label));
	getLastCudaError("set_label2");
}

