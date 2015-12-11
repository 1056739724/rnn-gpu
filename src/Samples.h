//主要用于将训练集和测试集上传到

#ifndef SAMPLES_H
#define SAMPLES_H
//#include "Config.h"      暂未用到
#include <algorithm>
#include <stdio.h>
#include "general_settings.h"

class Samples
{
public:
	Samples() :
			randproductor(NULL),
			dev_trainX(NULL),
			dev_trainY(NULL),
			dev_testX(NULL),
			dev_testY(NULL),
			sizex(0),
			sizey(0),
			sizetx(0),
			sizety(0){}


	static Samples* instance()
	{

		static Samples* samples = new Samples();
		return samples;
	}


	void randproductor_init()//randproductor初始化
	{
		if (randproductor == NULL)//初始为空
		{
			//train_num：训练集的大小，子句子的个数，每一个子句子5行
			randproductor = (int *) malloc(train_num * sizeof(int));
			for (int i = 0; i < train_num; i++)
			{
				randproductor[i] = i;
			}
		}

	}

	int* &get_rand(bool x = 1)
	{
		if (randproductor == NULL)
		{
			printf("int* get_rand() error: randproductor = NULL, run Config::randproductor_init() first\n");
			exit(0);
		}
		else if (x)//1为真，执行
		{
			//打乱元素，random_shuffle()用来对一个元素序列进行重新排序（随机的）
			//第一个元素指向序列首元素的迭代器  第二个元素指向序列最后一个元素的下一个位置的迭代器
			/*用法1： char arr[] = {'a', 'b', 'c', 'd', 'e', 'f'};
                    std::random_shuffle(arr,arr+6);//迭代器

             用法2： vector<string> str;
             std::random_shuffle(str.begin(),str.end());//迭代器
			 *
			 */
			random_shuffle(randproductor, randproductor + train_num);
			return randproductor;
		}
		else
		{
			return randproductor;
		}
	}


	//测试集数据上传至gpu
	void testX2gpu(int *host_, //数据
			int size)//大小
	{
		//cudaMalloc:第一个参数事一个(void**)类型，故要&dev_testX取地址
		cudaError_t cudaStat = cudaMalloc((void**) &dev_testX, size);
		if (cudaStat != cudaSuccess)
		{
			printf("Samples::testX2gpu() failed\n");
			exit(0);
		}
		sizetx = size;//sizetx：测试集x占用内存大小
		cudaStat = cudaMemcpy(dev_testX, host_, size, cudaMemcpyHostToDevice);
		if (cudaStat != cudaSuccess)
		{
			printf("set_gpudata::toGPU data upload failed\n");
			exit(0);
		}
	}

	//测试集数据上传至gpu
	void testY2gpu(int *host_, int size)
	{
		cudaError_t cudaStat = cudaMalloc((void**) &dev_testY, size);
		if (cudaStat != cudaSuccess)
		{
			printf("Samples::testY2gpu() failed\n");
			exit(0);
		}
		sizety = size;
		cudaStat = cudaMemcpy(dev_testY, host_, size, cudaMemcpyHostToDevice);
		if (cudaStat != cudaSuccess)
		{
			printf("set_gpudata::toGPU data upload failed\n");
			exit(0);
		}
	}

	int* &get_testX()//得到测试集
	{
		return dev_testX;
	}
	int* &get_testY()//得到测试集标签
	{
		return dev_testY;
	}

	//训练集x上传
	void trainX2gpu(int *host_, int size)
	{
		cudaError_t cudaStat = cudaMalloc((void**) &dev_trainX, size);
		if (cudaStat != cudaSuccess)
		{
			printf("Samples::testY2gpu() failed\n");
			exit(0);
		}
		sizex = size;//训练集x大小
		cudaStat = cudaMemcpy(dev_trainX, host_, size, cudaMemcpyHostToDevice);
		if (cudaStat != cudaSuccess)
		{
			printf("set_gpudata::toGPU data upload failed\n");
			exit(0);
		}
	}

	//训练集y上传
	void trainY2gpu(int *host_, int size)
	{
		cudaError_t cudaStat = cudaMalloc((void**) &dev_trainY, size);
		if (cudaStat != cudaSuccess)
		{
			printf("Samples::testY2gpu() failed\n");
			exit(0);
		}
		sizey = size;//训练集y大小
		cudaStat = cudaMemcpy(dev_trainY, host_, size, cudaMemcpyHostToDevice);
		if (cudaStat != cudaSuccess)
		{
			printf("set_gpudata::toGPU data upload failed\n");
			exit(0);
		}
	}

	int* &get_trainX()//得到训练集
	{
		return dev_trainX;
	}

	int* &get_trainY()//得到训练集标签
	{
		return dev_trainY;
	}

	int get_sizetx()//得到测试集大小
	{
		return sizetx;
	}

	int get_sizety()//得到测试集标签大小
	{
		return sizety;
	}

	int get_sizex()//得到训练集大小
	{
		return sizex;
	}

	int get_sizey()//得到训练集标签大小
	{
		return sizey;
	}

private:
	int *randproductor;

	int *dev_trainX;
	int *dev_trainY;
	int *dev_testX;
	int *dev_testY;

	int sizex; //trainx
	int sizey;
	int sizetx;//testx
	int sizety;
};

#endif
