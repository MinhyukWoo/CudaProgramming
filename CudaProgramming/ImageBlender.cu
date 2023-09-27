#include "ImageBlender.cuh"
#include <random>
#include <time.h>
#include <iostream>
#include "cuda.h"
#include <cmath>
#include <algorithm>

#include "device_launch_parameters.h"

using namespace std;


// 이미지 처리 계산 결과 : operation_time, total_time
pair<double, double> ImageBlender::Blend(double weight, DeviceType device_type)
{
	double operation_time, total_time;
	time_t operation_start, operation_end, total_start, total_end;

	if (device_type == CUDA) // CUDA로 image blending 할 때
	{
		total_start = clock();
		int * src_img_1_cuda, *src_img_2_cuda, *dst_img_cuda;
		int byte_size = _size * sizeof(int);

		// Device memory 동적 할당
		cudaMalloc(&src_img_1_cuda, byte_size);
		cudaMalloc(&src_img_2_cuda, byte_size);
		cudaMalloc(&dst_img_cuda, byte_size);

		// Host memory에 있는 값들 device memory로 복사
		cudaMemcpy(src_img_1_cuda, _src_img1, byte_size, cudaMemcpyHostToDevice);
		cudaMemcpy(src_img_2_cuda, _src_img2, byte_size, cudaMemcpyHostToDevice);

		// GPU에서 연산하도록 BlendByGpu 커널 launch
		// 이때 size에 맞게 1024개의 thread를 가지고 있는 블록을 만듦 : <<< Parameter 수, Thread 수 >>>
		operation_start = clock();
		BlendByGpu <<< _size / 1024 + 1, 1024 >>> (src_img_1_cuda, src_img_2_cuda, dst_img_cuda, _size, weight);
		cudaDeviceSynchronize(); // GPU 연산 모두 끝날 때까지 다음 코드 실행 막음
		operation_end = clock();

		// Device memory에 있는 값들 host memory에 복사
		cudaMemcpy(_dst_img, dst_img_cuda, byte_size, cudaMemcpyDeviceToHost);

		// Device memory 해제
		cudaFree(src_img_1_cuda);
		cudaFree(src_img_2_cuda);
		cudaFree(dst_img_cuda);
		total_end = clock();
	}
	else if (device_type == MP) // OpenMP로 image blending할 때
	{
		total_start = clock();
		operation_start = clock();
		BlendByMp(_src_img1, _src_img2, _dst_img, _size, weight);
		operation_end = clock();
		total_end = clock();
	}
	else // CPU로 image blending할 때
	{
		total_start = clock();
		operation_start = clock();
		BlendByCpu(_src_img1, _src_img2, _dst_img, _size, weight);
		operation_end = clock();
		total_end = clock();
	}
	operation_time = difftime(operation_end, operation_start);
	total_time = difftime(total_end, total_start);

	return pair<double, double>(operation_time, total_time);
}


// __global__ 키워드를 붙이면, Host를 통해 호출되어 Device에서 작동된다.
__global__ void BlendByGpu(int*src_img1, int*src_img2, int*dst_img, int size, double weight)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size) {
		double weighted_a = (double)src_img1[tid] * weight;
		double weighted_b = (double)src_img2[tid] * (1.0 - weight);
		double val = round(weighted_a + weighted_b);
		if (val < 0) {
			val = 0;
		}
		else if (val > 255) {
			val = 255;
		}
		dst_img[tid] = (int)val;
	}
}
