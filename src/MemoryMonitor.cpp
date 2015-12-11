#include "MemoryMonitor.h"

//给dev在gpu上分配空间，且进行初始化全部
void MatData::Malloc__()
{
	assert(size);
//	host = (float*) malloc(size);
//	memset(host, 0, rows * cols * sizeof(float));
	cudaError_t cudaStat = cudaMalloc((void**) &dev, size);
	if (cudaStat != cudaSuccess)
	{
		printf("MatData::cudaMalloc() failed\n");
		exit(0);
	}
	cudaStat = cudaMemset(dev, 0, size);
	if (cudaStat != cudaSuccess)
	{
		printf("MatData::cudaMemset() failed\n");
		exit(0);
	}
}

//给dev在gpu上分配空间，且进行初始化全部
void MatData::Malloc()
{
	assert(dev == NULL && host == NULL);
	Malloc__();
}

//给host在cpu上分配空间，且进行初始化全部
void MatData::CpuMalloc()
{
	if (host == NULL && size != 0)
	{
		host = (float*) malloc(size);
		memset(host, 0, rows * cols * sizeof(float));
	}
}

//把gpu上面数据拷贝下来
void MatData::toCpu()
{
	CpuMalloc();
	assert(host != NULL && dev != NULL);
	cudaError_t cudaStat;
	cudaStat = cudaMemcpy(host, dev, size, cudaMemcpyDeviceToHost);
	if (cudaStat != cudaSuccess)
    {
		printf("MatData::toCPU data download failed\n");
		exit(0);
	}
}

//设置gpu上面的dev数据为src
void MatData::setGpu(float* src)
{
	assert(dev != NULL);
	cudaError_t cudaStat = cudaMemcpy(dev, src, size, cudaMemcpyHostToDevice);
	if (cudaStat != cudaSuccess)
	{
		printf("MatData::setGpu failed\n");
		exit(0);
	}
}
