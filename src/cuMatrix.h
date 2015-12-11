#ifndef CUMATRIX_H
#define CUMATRIX_H
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <memory>
#include "helper_cuda.h"

#include "MemoryMonitor.h"
//#include "hardware.h"

using namespace std;

class cuMatrix //整个文件就这一个类
{
public:
	cuMatrix(int r = 0, int c = 0)//构造函数
    {
		//给类的私有变量row，col，size赋值cuMatrix.h
		row = r;
		col = c;
		size = r * c * sizeof(float);//矩阵中的每个元素是float类型
		//std::make_shared<A>() 则是只执行一次内存申请，将数据和控制块的申请放到一起。
		//make_shared()函数可以接受最多10个参数，然后把它们传递给类型MatData的构造函数
		data = std::make_shared < MatData > (r, c);
	}

	cuMatrix(float *src, int r, int c)//构造函数
	{
		row = r;
		col = c;
		size = r * c * sizeof(float);
		data = make_shared < MatData > (r, c);
		data->setGpu(src);//设置gpu上面的dev数据为src
	}

	int rows()
	{
		return row;
	}

	int cols()
	{
		return col;
	}

	int sizes()
	{
		return size;
	}

	float* getDev()//得到gpu上数据
	{
		return data->getDev();
	}

	float* getHost()//得到cpu数据，float*,为空的时候把gpu上数据拷贝下来？？仅仅空的时候再拷贝下来？？
	{
		return data->getHost();
	}

	void printMat()//把cpu上数据float* host;打印出来
	{
		data->toCpu();//把gpu上面数据拷贝到cpu上
		for (int i = 0; i < rows(); i++)
		{
			for (int j = 0; j < cols(); j++)
			{
				printf("%f,",getHost()[i*cols() + j]);
			}
			printf("\n");
		}
	}

	float getSum()//计算cpu上数据float* host的总和
	{
		data->toCpu();
		float sum = 0;
		float *tmp = getHost();
		for (int i = 0; i < rows(); i++)
		{
			for (int j = 0; j < cols(); j++)
			{
				sum += tmp[i*cols() + j];
			}
		}
		return sum;
	}

    //数据拷贝，把当前设备的dev数据，也就是data的dev拷贝到传过来的这个设备的data的dev
	void copyTo(cuMatrix &dst)
	{
		//cols()：得到col  rows()得到row
		if(cols()!=dst.cols() || rows()!=dst.rows())
		{
			printf("cuMatrix::copyTo() size error\n");
			exit(0);
		}
		cudaError_t cudaStat;
		cudaStat = cudaMemcpy(dst.data->getDev(), data->getDev(), size, cudaMemcpyDeviceToDevice);
		if (cudaStat != cudaSuccess)
		{
			printf("cuMatrix::copyTo cudaMemcpy() failed\n");
			exit(0);
		}
	}

	cuMatrix t();
	cuMatrix Mul(cuMatrix cumat);      //per-element  mul
	cuMatrix operator +(cuMatrix cumat);
	cuMatrix operator +(float i);
	cuMatrix operator -(cuMatrix cumat);
	cuMatrix operator -(float i);
	cuMatrix operator *(cuMatrix cumat);//matrix mul
	cuMatrix operator *(float i);
	cuMatrix operator /(cuMatrix cumat);
	cuMatrix operator /(float i);
	friend cuMatrix operator /(float i,cuMatrix cumat);

	//shared_ptr是智能指针，作用有如同指针，但会记录有多少个shared_ptrs共同指向一个对象。
	//这便是所谓的引用计数（reference counting）。一旦最后一个这样的指针被销毁，
	//也就是一旦某个对象的引用计数变为0，这个对象会被自动删除。这在非环形数据结构中防止资源泄露很有帮助。
	shared_ptr<MatData> data;//MatData是MemoryMonitor.h中的类，这个文件中就这一个类

private:
	int row;
	int col;
	int size;
};
#endif
