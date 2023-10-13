#include "ImageBlender2d.cuh"
#include <random>
#include <time.h>
#include <iostream>
#include "cuda.h"
#include <cmath>
#include <algorithm>

#include "device_launch_parameters.h"

using namespace std;


// �̹��� ó�� ��� ��� : operation_time, total_time
pair<double, double> ImageBlender::Blend(double weight, DeviceType device_type)
{
	double operation_time, total_time;
	time_t operation_start, operation_end, total_start, total_end;
	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	if (device_type == CUDA && device_count != 0) // CUDA�� image blending �� ��
	{
		total_start = clock();
		int **src_img1_cuda, **src_img2_cuda, **dst_img_cuda;
		int row_byte_size = _rows * sizeof(int*);
		int col_byte_size = _cols * sizeof(int);

		// Device memory ���� �Ҵ�
		cudaMalloc((void **)&src_img1_cuda, row_byte_size);
		cudaMalloc((void **)&src_img2_cuda, row_byte_size);
		cudaMalloc((void **)&dst_img_cuda, row_byte_size);
		int **src_img1_cuda_row = new int*[_rows];
		int **src_img2_cuda_row = new int*[_rows];
		int **dst_img_cuda_row = new int*[_rows];
		for (int i = 0; i < _rows; i++)
		{
			cudaMalloc(src_img1_cuda_row + i, col_byte_size);
			cudaMalloc(src_img2_cuda_row + i, col_byte_size);
			cudaMalloc(dst_img_cuda_row + i, col_byte_size);
		}
		cudaMemcpy(src_img1_cuda, src_img1_cuda_row, row_byte_size, cudaMemcpyHostToDevice);
		cudaMemcpy(src_img2_cuda, src_img2_cuda_row, row_byte_size, cudaMemcpyHostToDevice);
		cudaMemcpy(dst_img_cuda, dst_img_cuda_row, row_byte_size, cudaMemcpyHostToDevice);

		// Host memory�� �ִ� ���� device memory�� ����
		for (int i = 0; i < _rows; i++)
		{
			cudaMemcpy(src_img1_cuda_row[i], _src_img1[i], col_byte_size, cudaMemcpyHostToDevice);
			cudaMemcpy(src_img2_cuda_row[i], _src_img2[i], col_byte_size, cudaMemcpyHostToDevice);
			cudaMemcpy(dst_img_cuda_row[i], _dst_img[i], col_byte_size, cudaMemcpyHostToDevice);
		}

		// GPU���� �����ϵ��� BlendByGpu Ŀ�� launch
		// �̶� size�� �°� 1024���� thread�� ������ �ִ� ����� ���� : <<< Parameter ��, Thread �� >>>
		operation_start = clock();
		dim3 threads_per_block(16, 16);
		dim3 num_blocks(1 + _rows / threads_per_block.x, 1 + _cols / threads_per_block.y);
		BlendByGpu << < num_blocks, threads_per_block >> > (src_img1_cuda, src_img2_cuda, dst_img_cuda, _rows, _cols, weight);
		cudaDeviceSynchronize(); // GPU ���� ��� ���� ������ ���� �ڵ� ���� ����
		operation_end = clock();

		// Device memory�� �ִ� ���� host memory�� ����
		for (int i = 0; i < _rows; i++)
		{
			cudaMemcpy(_dst_img[i], dst_img_cuda_row[i], col_byte_size, cudaMemcpyDeviceToHost);
		}
		// Device memory ����
		for (int i = 0; i < _rows; i++)
		{
			cudaFree(src_img1_cuda_row[i]);
			cudaFree(src_img2_cuda_row[i]);
			cudaFree(dst_img_cuda_row[i]);
		}
		cudaFree(src_img1_cuda);
		cudaFree(src_img2_cuda);
		cudaFree(dst_img_cuda);
		delete[] src_img1_cuda_row;
		delete[] src_img2_cuda_row;
		delete[] dst_img_cuda_row;
		total_end = clock();
	}
	else if (device_type == MP) // OpenMP�� image blending�� ��
	{
		total_start = clock();
		operation_start = clock();
		BlendByMp(_src_img1, _src_img2, _dst_img, _rows, _cols, weight);
		operation_end = clock();
		total_end = clock();
	}
	else // CPU�� image blending�� ��
	{
		total_start = clock();
		operation_start = clock();
		BlendByCpu(_src_img1, _src_img2, _dst_img, _rows, _cols, weight);
		operation_end = clock();
		total_end = clock();
	}
	operation_time = difftime(operation_end, operation_start);
	total_time = difftime(total_end, total_start);

	return pair<double, double>(operation_time, total_time);
}


// __global__ Ű���带 ���̸�, Host�� ���� ȣ��Ǿ� Device���� �۵��ȴ�.
__global__ void BlendByGpu(int**src_img1, int**src_img2, int**dst_img, int rows, int cols, double weight)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < rows && j < cols) {
		double weighted_element1 = (double)src_img1[i][j] * weight;
		double weighted_element2 = (double)src_img2[i][j] * (1.0 - weight);
		double val = round(weighted_element1 + weighted_element2);
		if (val < 0) {
			val = 0;
		}
		else if (val > 255) {
			val = 255;
		}
		dst_img[i][j] = (int)val;
	}
}
