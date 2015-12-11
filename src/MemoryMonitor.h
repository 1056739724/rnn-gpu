#ifndef MEMORYMONITOR_H
#define MEMORYMONITOR_H
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <assert.h>
#include <stdio.h>
#include <algorithm>
//#include "Config.h"
using namespace std;

class MatData
{
public:
	MatData(int r = 0, int c = 0)//构造函数，初始化私有成员变量
    {
		rows = r;
		cols = c;
		size = rows * cols * sizeof(float);
		host = NULL;
		if (size == 0)
		{
			dev = NULL;
		}
		else
		{
			Malloc__();//在gpu上对dev分配空间并且进行了初始化全部为0
		}
	}

	~MatData() //析构函数，释放指针，防止内存泄露
	{
		if (host != NULL)
			free(host);
		if (dev != NULL)
			cudaFree(dev);
	}

	float* getDev()//得到设备数据
	{
		assert(dev != NULL);
		return dev;
	}
	float* getHost()//得到cpu数据
	{
		if (host == NULL)
		{
			toCpu();//把gpu上面的数据拷贝到本地
		}
		return host;
	}

	void Malloc();//在gpu上对dev分配空间并且进行了初始化全部为0
	void toCpu();//把gpu上面数据拷贝到cpu上
	void setGpu(float* src);//设置gpu上面的dev数据为src

private:
	int rows;
	int cols;
	int size;
	float* host;
	float* dev;//设备上数据，分配在gpu上面
	void Malloc__();//在gpu上分配空间并且进行了初始化全部为0
	void CpuMalloc();//在cpu上分配空间并且进行了初始化全部为0
};

#endif
