#ifndef CUMATRIXVECTOR_H
#define CUMATRIXVECTOR_H
#include <vector>
#include <stdio.h>
#include "cuMatrix.h"
#include <assert.h>
using namespace std;

class cuMatrixVector
{
public:
	cuMatrixVector()//构造函数
   {
		m_host = NULL;
		m_dev = NULL;
	}

	~cuMatrixVector()//析构函数，总觉得析构函数写的不对
	{
		if (m_host != NULL)
		{
			free(m_host);
		}
		if (m_dev != NULL)
		{
			cudaFree(m_dev);
		}
		for(int i = 0 ; i < m_vec.size() ; i++)
		{
			delete m_vec[i];
		}
		m_vec.clear();
	}

	cuMatrix*& operator[](size_t index)//重载[]运算符
	{
		if (index >= m_vec.size())
		{
			printf("cuMatrixVector operator[] error\n");
			exit(0);
		}
		return m_vec[index];
	}


	void toGpu()
	{
		cudaError_t cudaStat;
		//size是m_vec.size()* sizeof(float*)的原因是：
		//m_host是指向指针的指针，里面存放向量的地址，而向量中存放的也不是真实的向量，而是地址，真实向量的地址
		m_host = (float**) malloc(m_vec.size() * sizeof(float*));
		if (!m_host)
		{
			printf("cuMatrixVector malloc m_host fail\n");
			exit(0);
		}
		cudaStat = cudaMalloc((void**) &m_dev, m_vec.size() * sizeof(float*));//在cuda上分配空间
		if (cudaStat != cudaSuccess)
		{
			printf("cuMatrixVector cudaMalloc m_dev fail\n");
			exit(0);
		}
		for (int p = 0; p < (int) m_vec.size(); p++)//遍历向量，5
		{
			m_host[p] = m_vec[p]->getDev();//m_vec[p]是向量，getDev：得到每个向量gpu上面的数据
		}
		cudaStat = cudaMemcpy(m_dev, m_host, sizeof(float*) * m_vec.size(),cudaMemcpyHostToDevice);//拷贝到gpu上面
		if (cudaStat != cudaSuccess)
		{
			printf("cuMatrixVector::toGpu cudaMemcpy fail\n");
			exit(0);
		}
	}

	//矩阵向量添加矩阵
	void push_back(cuMatrix* p)
	{
		m_vec.push_back(p);
	}

	void clear()//向量清空
	{
		m_vec.clear();
	}

	size_t size()
	{
		return m_vec.size();
	}

	float** &get_host()//取得主机数据
	{
		assert(m_host != NULL);
		return m_host;
	}
	float** &get_devPoint()//取设备数据
	{
		assert(m_dev != NULL);
		return m_dev;
	}
private:
	vector<cuMatrix*> m_vec;
	float** m_host; //point 2 cuMatrix->Data->getDev()
	float** m_dev;
};
#endif
