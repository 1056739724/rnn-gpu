#include "cuMath.h"
static int MAX_THREADNUM = Devices::instance()->max_ThreadsPerBlock();
__global__ void ReLU_kernel(float* src,
		float* dst,//目的
		int rows,//行数
		int cols,//列数
		int maxt)//每个线程块允许的最大线程数
{
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols)
	{
		assert(x * cols + y < rows * cols);
		if (src[x * cols + y] <= 0)
		{
			dst[x * cols + y] = 0;
		}
		else
		{
			dst[x * cols + y] = src[x * cols + y];
		}
		y += maxt;
	}
}

cuMatrix ReLU(cuMatrix& cumat)//线性修正
{
	cuMatrix res(cumat.rows(), cumat.cols());//大小不变的矩阵
	int threadnum = MAX_THREADNUM > cumat.cols() ? cumat.cols() : MAX_THREADNUM;
	dim3 blocks=dim3(cumat.rows());
	dim3 threads=dim3(threadnum);
	ReLU_kernel<<<blocks, threads>>>(cumat.getDev(),
			res.getDev(), cumat.rows(), cumat.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("ReLU ");
	return res;
}

__global__ void dReLU_kernel(float* src,
		float* dst,
		int cols,
		int maxt)
{
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols)
	{
		if (src[x * cols + y] <= 0)
		{
			dst[x * cols + y] = 0;
		}
		else
		{
			dst[x * cols + y] = 1;
		}
		y += maxt;
	}
}

cuMatrix dReLU(cuMatrix& cumat)
{
	cuMatrix res(cumat.rows(), cumat.cols());
	int threadnum = MAX_THREADNUM > cumat.cols() ? cumat.cols() : MAX_THREADNUM;
	dim3 blocks=dim3(cumat.rows());
	dim3 threads=dim3(threadnum);
	dReLU_kernel<<<blocks,threads>>>(cumat.getDev(),
			res.getDev(), cumat.cols(), MAX_THREADNUM);
	getLastCudaError("dReLU ");
	return res;
}

__global__ void reduce_max_kernel(float* dev_x,
		float* dev_y,//存放结果
        int rows,//行
		int cols, //列
		int maxt)//每个线程块的最大线程数
{
	int tid = threadIdx.x;//因为只有一个线程块
	while (tid < cols)
	{
		float max = (float) LONG_MIN;//长整形最小值
		for (int i = 0; i < rows; i++)
		{
			max = max > dev_x[i * cols + tid] ? max : dev_x[i * cols + tid];
		}
		for (int i = 0; i < rows; i++)
		{
			dev_y[i * cols + tid] = max;
		}
		tid += maxt;
	}
}

cuMatrix reduceMax(cuMatrix src)
{
	cuMatrix res(src.rows(), src.cols());
	int threadnum = MAX_THREADNUM > src.cols() ? src.cols() : MAX_THREADNUM;
	dim3 blocks=dim3(1);
	dim3 threads=dim3(threadnum);
	reduce_max_kernel<<<blocks,threads>>>(src.getDev(), res.getDev(),
			src.rows(), src.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("reduce_max");
	return res;
}

//  share memory?
__global__ void reduce_sum_kernel(float* dev_x,
		float* dev_y,
		int rows,
		int cols,
		int maxt)
{
	int tidx = blockIdx.x;
	int tidy = threadIdx.x;
	float sum = 0;//每列之和
	while (tidy < cols)
	{
		for (int i = 0; i < rows; i++)
		{
			sum += dev_x[i * cols + tidy];
		}
		dev_y[tidx * cols + tidy] = sum;
		tidy += maxt;
	}
}


cuMatrix reduceSum(cuMatrix src)
{
	cuMatrix res(src.rows(), src.cols());
	int threadnum = MAX_THREADNUM > src.cols() ? src.cols() : MAX_THREADNUM;
	dim3 blocks=dim3(src.rows());
	dim3 threads=dim3(threadnum);
	reduce_sum_kernel<<<blocks, threads>>>(src.getDev(),
			res.getDev(), src.rows(), src.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("reduce_sum");
	return res;
}

//行做block
__global__ void log_kernel(float* dev_x,
		float* dev_y,
		int cols,
		int maxt)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	while (tid < cols)
	{
		dev_y[bid * cols + tid] = log(dev_x[bid * cols + tid]);
		tid += maxt;
	}
}

//矩阵中每个值取log
cuMatrix Log(cuMatrix src)
{
	cuMatrix res(src.rows(), src.cols());
	int threadnum = MAX_THREADNUM > src.cols() ? src.cols() : MAX_THREADNUM;
	dim3 blocks=dim3(src.rows());
	dim3 threads=dim3(threadnum);
	log_kernel<<<blocks, threads>>>(src.getDev(),
			res.getDev(), src.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("ElementLog");

	return res;
}

//每个元素取指数
__global__ void exp_kernel(float* dev_x,
		float* dev_y,
		int cols,
		int maxt)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	while (tid < cols)
	{
		//取每个数e为底的指数
		dev_y[bid * cols + tid] = exp(dev_x[bid * cols + tid]);
		tid += maxt;
	}
}

//e为底的指数次方
cuMatrix Exp(cuMatrix src)
{
	cuMatrix res(src.rows(), src.cols());
	int threadnum = MAX_THREADNUM > src.cols() ? src.cols() : MAX_THREADNUM;
	dim3 blocks=dim3(src.rows());
	dim3 threads=dim3(threadnum);
	exp_kernel<<<blocks,threads>>>(src.getDev(),
			res.getDev(),
			src.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("ElementExp");
	return res;
}

__global__ void Pow_kernel(float* dev_x,
		float* dev_y,
		float* dev_z,//存放结果
		int cols,
		int maxt)
{
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols)
	{
		dev_z[x * cols + y] = pow(dev_x[x * cols + y], dev_y[x * cols + y]);
		y += maxt;
	}
}

//x矩阵中每个元素y次方  核函数
__global__ void Pow_kernel(float* dev_x,
		float y_,
		float* dev_z,
		int cols,
		int maxt)
{
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols)
	{
		dev_z[x * cols + y] = pow(dev_x[x * cols + y], y_);
		y += maxt;
	}
}

//x矩阵中每个元素y矩阵中对应元素次方
cuMatrix Pow(cuMatrix x,cuMatrix y)
{
	if (!(x.rows() == y.rows()))
	{
		printf("cuMatrix Pow(cuMatrix x,cuMatrix y) error: rows!\n");
		exit(0);
	}
	if (!(x.cols() == y.cols()))
	{
		printf("cuMatrix Pow(cuMatrix x,cuMatrix y) error: cols!\n");
		exit(0);
	}
	cuMatrix res(x.rows(), x.cols());
	int threadnum = MAX_THREADNUM > x.cols() ? x.cols() : MAX_THREADNUM;
	dim3 blocks=dim3(x.rows());
	dim3 threads=dim3(threadnum);
	Pow_kernel<<<blocks,threads>>>(x.getDev(),
			y.getDev(), res.getDev(), x.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("cuMatrix Pow(cuMatrix x,cuMatrix y)");
	return res;
}

//x矩阵中每个元素y次方
cuMatrix Pow(cuMatrix x,float y)
{
	int threadnum = MAX_THREADNUM > x.cols() ? x.cols() : MAX_THREADNUM;
	cuMatrix res(x.rows(), x.cols());
	dim3 blocks=dim3(x.rows());
	dim3 threads=dim3(threadnum);
	Pow_kernel<<<blocks,threads>>>(x.getDev(), y,
			res.getDev(), x.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("ElementPow matrix float matrix ");
	return res;
}
