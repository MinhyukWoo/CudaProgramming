#include"test.hpp"
#include <curand_kernel.h>
#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<device_functions.h>
#include"cuda_runtime.h"
#include"device_launch_parameters.h"

__global__ void BadSum(int* dst, int *src, int size) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size) {
		int prevDst = *dst;
		*dst += src[tid];
		printf("%2d: %2d + %2d = %2d\n", tid, src[tid], prevDst, *dst);
	}
}



int GetBadSum(int * ptr, int size) {
	int* deviceSrc;
	cudaMalloc(&deviceSrc, size * sizeof(int));
	cudaMemcpy(deviceSrc, ptr, size * sizeof(int), cudaMemcpyHostToDevice);

	int * deviceDst, hostTmp = 0;
	cudaMalloc(&deviceDst, sizeof(int));
	cudaMemcpy(deviceDst, &hostTmp, sizeof(int), cudaMemcpyHostToDevice);

	BadSum << <1, 256 >> > (deviceDst, deviceSrc, size);

	int hostDst;
	cudaMemcpy(&hostDst, deviceDst, sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(deviceSrc);
	cudaFree(deviceDst);

	return hostDst;
}

__global__ void SumReduced(int* dst, int *src, int size) {
	__shared__ int sharedData[256];
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	sharedData[tid] = (id < size) ? src[id] : 0;
	__syncthreads();
	for (unsigned int stride = blockDim.x / 2; stride  > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			printf("%2d: %2d + %2d = %2d\n", tid, sharedData[tid], sharedData[tid + stride], sharedData[tid] + sharedData[tid + stride]);
			sharedData[tid] += sharedData[tid + stride];
		}
		__syncthreads();
		if (tid == 0) {
			printf("\n");
		}
	}
	if (tid == 0) {
		*dst = sharedData[0];
	}
}

int GetReducedSum(int * ptr, int size) {
	int* deviceSrc;
	cudaMalloc(&deviceSrc, size * sizeof(int));
	cudaMemcpy(deviceSrc, ptr, size * sizeof(int), cudaMemcpyHostToDevice);

	int * deviceDst, hostTmp = 0;
	cudaMalloc(&deviceDst, sizeof(int));
	cudaMemcpy(deviceDst, &hostTmp, sizeof(int), cudaMemcpyHostToDevice);

	SumReduced << <1, 16 >> > (deviceDst, deviceSrc, size);

	int hostDst;
	cudaMemcpy(&hostDst, deviceDst, sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(deviceSrc);
	cudaFree(deviceDst);
	return hostDst;
}