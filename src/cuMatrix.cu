#include "cuMatrix.h"
#include "hardware.h"


static int MAX_THREADNUM = Devices::instance()->max_ThreadsPerBlock();

//cublasHandle_t类型是一个不透明指针类型结构支持CUBLAS库内容。
//CUBLAS库内容必须通过cublasCreate()初始化，
//返回的处理结果必须经过所有后续库函数调用，最后必须使用cublasDestroy()摧毁其中内容
cublasHandle_t& getHandle()
{
	static cublasHandle_t handle = NULL;
	if (handle == NULL)
	{
		cublasStatus_t status;
		status = cublasCreate(&handle);
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			printf("init: CUBLAS initialization failed\n");
			exit(0);
		}
	}
	return handle;
}

//两个矩阵相加核函数
__global__ void add_kernel(float* dev_x,
		float* dev_y,
		float* dev_z,//存放结果
		int cols,
		int maxt)//每个线程块线程的最大数量
{
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols)
	{
		dev_z[x * cols + y] = dev_x[x * cols + y] + dev_y[x * cols + y];
		y += maxt;
	}
}

//矩阵加上一个float类型的数
__global__ void add_kernel(float* dev_x,
		float y_,
		float* dev_z,//存放结果
		int cols,
		int maxt)//每个线程块线程的最大数量
{
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols)
	{
		dev_z[x * cols + y] = dev_x[x * cols + y] + y_;
		y += maxt;
	}
}

cuMatrix cuMatrix::operator +(cuMatrix cumat)//两个矩阵相加，+号运算符重载
{
	if (!size)
	{
		if (cumat.data->getDev() == NULL)
		{
			printf("cuMatrix error : both matrix are empty.\n");
			exit(0);
		}
		cuMatrix res = cumat;
		return res;
	}
	assert(cumat.rows() == rows() && cumat.cols() == cols());
	assert(data->getDev() != NULL && cumat.data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	cuMatrix res(rows(), cols());
	dim3 blocks=dim3(rows());
	dim3 threads=dim3(threadnum);
	add_kernel<<<blocks,threads>>>(data->getDev(),
			cumat.data->getDev(), res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("pre-element add cuMatrix + cuMatrix");
	return res;
}

cuMatrix cuMatrix::operator +(float i)
{
	assert(data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	cuMatrix res(rows(), cols());
	add_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(), i,
			res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("pre-element add cuMatrix + float");
	return res;
}

//矩阵与矩阵相减
__global__ void dec_kernel(float* dev_x,
		float* dev_y,
		float* dev_z, //存放运算结果
		int cols,//列
		int maxt)//每块最大线程数量
{
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols)
	{
		dev_z[x * cols + y] = dev_x[x * cols + y] - dev_y[x * cols + y];
		y += maxt;
	}
}

//矩阵减去一个数
__global__ void dec_kernel(float* dev_x,
		float y_,
		float* dev_z,
		int cols,
		int maxt)
{
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols)
	{
		dev_z[x * cols + y] = dev_x[x * cols + y] - y_;
		y += maxt;
	}
}

cuMatrix cuMatrix::operator -(cuMatrix cumat)//两个矩阵相减
{
	if (!size)
	{
		if (cumat.data->getDev() == NULL)
		{
			printf("cuMatrix error : both matrix are empty.\n");
			exit(0);
		}
		cuMatrix res = cumat * -1.0f;
		return res;
	}
	assert(cumat.rows() == rows() && cumat.cols() == cols());
	assert(data->getDev() != NULL && cumat.data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	cuMatrix res(rows(), cols());
	dim3 blocks=dim3(rows());
	dim3 threads=dim3(threadnum);
	dec_kernel<<<blocks,threads>>>(data->getDev(),
			cumat.data->getDev(), res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("pre-element add cuMatrix - cuMatrix");
	return res;
}

cuMatrix cuMatrix::operator -(float i)
{
	assert(data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	cuMatrix res(rows(), cols());
	dec_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(), i,
			res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("pre-element add cuMatrix - float");
	return res;
}

__global__ void mul_kernel(float* dev_x,
		float* dev_y,
		float* dev_z,
		int cols,
		int maxt)
{
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = dev_x[x * cols + y] * dev_y[x * cols + y];
		y += maxt;
	}
}

__global__ void mul_kernel(float* dev_x,
		float y_,
		float* dev_z,
		int cols,
		int maxt)
{
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols)
	{
		dev_z[x * cols + y] = dev_x[x * cols + y] * y_;
		y += maxt;
	}
}

//矩阵相乘，对应元素相乘
cuMatrix cuMatrix::Mul(cuMatrix cumat)
{
	assert(cumat.rows() == rows() && cumat.cols() == cols());
	assert(data->getDev() != NULL && cumat.data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	cuMatrix res(rows(), cols());
	dim3 blocks=dim3(rows());
	dim3 threads=dim3(threadnum);
	mul_kernel<<<blocks, threads>>>(data->getDev(),
			cumat.data->getDev(), res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("pre-element add cuMatrix * cuMatrix");
	return res;
}

//矩阵中每个元素乘以一个数
cuMatrix cuMatrix::operator *(float i)
{
	assert(data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	cuMatrix res(rows(), cols());
	mul_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(), i,
			res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("pre-element add cuMatrix * float");
	return res;
}
//res = this * cumat

//res = this * cumat
cuMatrix cuMatrix::operator *(cuMatrix cumat)
{
	//assert宏的原型定义在<assert.h>中，其作用是如果它的条件返回错误，则终止程序执行，原型定义：
	//assert的作用是现计算表达式 expression ，如果其值为假（即为0），
	//那么它先向stderr打印一条出错信息，然后通过调用 abort 来终止程序运行
	//http://www.cnblogs.com/ggzss/archive/2011/08/18/2145017.html
	assert(cols() == cumat.rows());//1255
	cuMatrix res(rows(), cumat.cols());
	float alpha = 1.0;
	float beta = 0.0;
	cublasStatus_t stat;
	stat = cublasSgemm(getHandle(),
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			cumat.cols(),
			rows(),
			cumat.rows(),
			&alpha,
			cumat.getDev(),
			cumat.cols(),
			getDev(),
			cols(),
			&beta,
			res.getDev(),
			res.cols());
	cudaStreamSynchronize(0);
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		printf("cuMatrix::Mul() error\n");
		exit(0);
	}
	return res;
}

__global__ void div_kernel(float* dev_x,
		float* dev_y,
		float* dev_z,//存放结果
		int cols,
		int maxt)
{
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols)
	{
		if (dev_y[x * cols + y] != 0)
		{
			dev_z[x * cols + y] = dev_x[x * cols + y] / dev_y[x * cols + y];
		}
		y += maxt;
	}
}

__global__ void div_kernel(float* dev_x,
		float y_,//矩阵中每个元素将要除以y_
		float* dev_z, //存放结果
		int cols,
		int maxt)
{
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols)
	{
		if (y_ != 0)
		{
			dev_z[x * cols + y] = dev_x[x * cols + y] / y_;
		}
		y += maxt;
	}
}

//应该是对应元素相除
cuMatrix cuMatrix::operator /(cuMatrix cumat)
{
	assert(cumat.rows() == rows() && cumat.cols() == cols());
	assert(data->getDev() != NULL && cumat.data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	cuMatrix res(rows(), cols());
	dim3 blocks=dim3(rows());
	dim3 threads=dim3(threadnum);
	div_kernel<<<blocks, threads>>>(data->getDev(),
			cumat.data->getDev(), res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("pre-element add cuMatrix / cuMatrix");
	return res;
}

//矩阵中每个元素除以i
cuMatrix cuMatrix::operator /(float i)
{
	assert(data->getDev() != NULL);
	assert(i != 0);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	cuMatrix res(rows(), cols());
	dim3 blocks=dim3(rows());
	dim3 threads=dim3(threadnum);
	div_kernel<<<blocks,threads>>>(data->getDev(), i,
			res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("pre-element add cuMatrix / float");
	return res;
}

__global__ void t_kernel(float* dev_src, float* dev_res, int res_r, int res_c,
		int maxt)
{
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < res_c)
	{
		dev_res[x * res_c + y] = dev_src[y * res_r + x];
		y += maxt;
	}
}

cuMatrix cuMatrix::t()
{
	assert(cols() != 0 && rows() != 0);
	cuMatrix res(cols(), rows());
	int threadnum = MAX_THREADNUM > res.cols() ? res.cols() : MAX_THREADNUM;
	t_kernel<<<dim3(res.rows()), dim3(threadnum)>>>(data->getDev(),
			res.data->getDev(), res.rows(), res.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("pre-element add cuMatrix / float");
	return res;
}

__global__ void Div_kernel(float x_,
		float* dev_y,
		float* dev_z,
		int cols,
		int maxt)
{
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols)
	{
		if (dev_y[x * cols + y] != 0)
		{
			dev_z[x * cols + y] = x_ / dev_y[x * cols + y];
		}
		y += maxt;
	}
}
cuMatrix operator /(float x, cuMatrix cumat)
{
	cuMatrix res(cumat.rows(), cumat.cols());
	int threadnum = MAX_THREADNUM > cumat.cols() ? cumat.cols() : MAX_THREADNUM;
	Div_kernel<<<dim3(cumat.rows()), dim3(threadnum)>>>(x, cumat.getDev(),
			res.getDev(), cumat.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("ElementDiv double matrix matrix ");
	return res;
}


