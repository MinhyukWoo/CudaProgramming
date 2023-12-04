#include"Reduce.hpp"
#include <curand_kernel.h>
#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<device_functions.h>
#include<cooperative_groups.h>
#include<cooperative_groups/reduce.h>

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

typedef void (*BinaryOperation)(int *, const int&, const int&);

__device__ void Plus(int * result, const int& lhs, const int& rhs) {
	*result = lhs + rhs;
}

__device__ void Max(int * result, const int& lhs, const int& rhs) {
	*result = (lhs > rhs) ? lhs : rhs;
}

__device__ void Min(int * result, const int& lhs, const int& rhs) {
	*result = (lhs > rhs) ? rhs : lhs;
}

__device__ BinaryOperation binaryOperations[] = { Plus, Max, Min };

__global__ void ReduceByKernel(int* dst, int *src, int size, E_BOPER binaryOperationIndex) {
	__shared__ int sharedData[256];
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	sharedData[tid] = (id < size) ? src[id] : 0;
	__syncthreads();
	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			int tmp = sharedData[tid];
			if (tid + stride < size) {
				binaryOperations[binaryOperationIndex](sharedData + tid, sharedData[tid], sharedData[tid + stride]);
				printf("%2d: (%2d, %2d) => %2d\n", tid, tmp, sharedData[tid + stride], sharedData[tid]);
			}
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

int Reduce(int * ptr, int size, E_BOPER index) {
	int* deviceSrc;
	cudaMalloc(&deviceSrc, size * sizeof(int));
	cudaMemcpy(deviceSrc, ptr, size * sizeof(int), cudaMemcpyHostToDevice);

	int * deviceDst, hostTmp = 0;
	cudaMalloc(&deviceDst, sizeof(int));
	cudaMemcpy(deviceDst, &hostTmp, sizeof(int), cudaMemcpyHostToDevice);
	ReduceByKernel << <1, 32 >> > (deviceDst, deviceSrc, size, index);

	int hostDst;
	cudaMemcpy(&hostDst, deviceDst, sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(deviceSrc);
	cudaFree(deviceDst);
	return hostDst;
}

__global__ void GetGLCM(
	unsigned int * comatrix, size_t comatrixColLength, WORD * image, size_t rowLength, size_t colLength, int deltaRow, int deltaCol
) {
	size_t rowIndex = blockIdx.x * blockDim.x + threadIdx.x;
	size_t colIndex = blockIdx.y * blockDim.y + threadIdx.y;
	if (!(rowIndex < rowLength && colIndex < colLength)){
		return;
	}
	if ((deltaRow > 0 && rowIndex + deltaRow >= rowLength)
		|| (deltaRow < 0 && rowIndex - deltaRow < 0)){
		return;
	}
	if ((deltaCol > 0 && colIndex + deltaCol >= colLength)
		|| (deltaCol < 0 && colIndex - deltaCol < 0)) {
		return;
	}

	size_t currIndex = rowIndex * colLength + colIndex;
	size_t nearIndex = (rowIndex + deltaRow) * colLength + (colIndex + deltaCol);
	size_t comatrixIndex = image[currIndex] * comatrixColLength + image[nearIndex];
	printf("%d, %d: %d, %d (%d)\n", rowIndex, colIndex, image[currIndex], image[nearIndex], comatrixIndex);

	atomicAdd(comatrix + comatrixIndex, (unsigned int)1);
}

unsigned int * GetComatrix(WORD * image, size_t rowLength, size_t colLength)
{
	size_t comtrixColLength = 9;
	size_t comatrixLength = comtrixColLength * comtrixColLength;

	unsigned int *hostComatrix = new unsigned int[comatrixLength] {0};
	unsigned int *deviceComatrix;
	cudaMalloc(&deviceComatrix, sizeof(unsigned int) * comatrixLength);
	cudaMemcpy(deviceComatrix, hostComatrix, sizeof(unsigned int) * comatrixLength, cudaMemcpyHostToDevice);
	WORD *deviceImage;
	cudaMalloc(&deviceImage, sizeof(WORD) * rowLength * colLength);
	cudaMemcpy(deviceImage, image, sizeof(WORD) * rowLength * colLength, cudaMemcpyHostToDevice);

	dim3 threads(16, 16);
	dim3 blocks(1 + rowLength / 16, 1 + colLength / 16);
	GetGLCM << <blocks, threads >> > (deviceComatrix, comtrixColLength, deviceImage, rowLength, colLength, 0, 1);

	cudaMemcpy(hostComatrix, deviceComatrix, sizeof(unsigned int) * comatrixLength, cudaMemcpyDeviceToHost);

	cudaFree(deviceComatrix);
	cudaFree(deviceImage);
	return hostComatrix;
}
