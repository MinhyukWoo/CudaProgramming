#include"GPUManager.cuh"
#include "device_launch_parameters.h"
#include<stdio.h>


__global__ void Init() {

}

GPUManager::GPUManager() : _threadSize(256) {
	Init << <1, 1 >> > ();
};

__global__ void ProcessOne1D(WORD *deviceDstPtr, WORD *deviceSrcPtr, size_t size, kernelOne1D_t kernelOne1D) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size) {
		kernelOne1D(deviceDstPtr, deviceSrcPtr, tid);
	}
}

void GPUManager::Process(WORD *dstPtr, WORD *srcPtr, size_t size, kernelOne1D_t kernelOne1D) {
	WORD *deviceSrcPtr, *deviceDstPtr;
	cudaMalloc(&deviceSrcPtr, sizeof(WORD) * size);
	cudaMemcpy(deviceSrcPtr, srcPtr, sizeof(WORD) * size, cudaMemcpyHostToDevice);
	cudaMalloc(&deviceDstPtr, sizeof(WORD) * size);
	ProcessOne1D << <1 + size / _threadSize, _threadSize >> > (deviceDstPtr, deviceSrcPtr, size, kernelOne1D);
	cudaMemcpy(dstPtr, deviceDstPtr, sizeof(WORD) * size, cudaMemcpyDeviceToHost);
	cudaFree(deviceSrcPtr);
	cudaFree(deviceDstPtr);
}


__global__ void ProcessTwo1D(WORD *deviceDstPtr, WORD *deviceSrcPtr1, WORD *deviceSrcPtr2, size_t size, kernelTwo1D_t kernelTwo1D) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size) {
		(*kernelTwo1D)(deviceDstPtr, deviceSrcPtr1, deviceSrcPtr2, tid);
	}
}


__device__ void addTwoWords_(WORD *dstPtr, WORD *srcPtr1, WORD *srcPtr2, size_t index) {
	dstPtr[index] = srcPtr1[index] < USHRT_MAX - srcPtr2[index] ? srcPtr1[index] + srcPtr2[index] : USHRT_MAX;
}

__device__ kernelTwo1D_t addTwoWords = addTwoWords_;


void GPUManager::Process(WORD *dstPtr,WORD *srcPtr1, WORD *srcPtr2, size_t size, kernelTwo1D_t kernelTwo1D) {
	WORD *deviceDstPtr, *deviceSrcPtr1, *deviceSrcPtr2;
	cudaMalloc(&deviceSrcPtr1, sizeof(WORD) * size);
	cudaMemcpy(deviceSrcPtr1, srcPtr1, sizeof(WORD) * size, cudaMemcpyHostToDevice);
	cudaMalloc(&deviceSrcPtr2, sizeof(WORD) * size);
	cudaMemcpy(deviceSrcPtr2, srcPtr2, sizeof(WORD) * size, cudaMemcpyHostToDevice);
	cudaMalloc(&deviceDstPtr, sizeof(WORD) * size);
	
	
	kernelTwo1D_t hostKernelTwo1D;
	cudaMemcpyFromSymbol(&hostKernelTwo1D, addTwoWords, sizeof(addTwoWords));
	
	ProcessTwo1D << <1 + size / _threadSize, _threadSize >> > (deviceDstPtr, deviceSrcPtr1, deviceSrcPtr2, size, hostKernelTwo1D);
	cudaDeviceSynchronize();
	cudaMemcpy(dstPtr, deviceDstPtr, sizeof(WORD) * size, cudaMemcpyDeviceToHost);
	cudaFree(deviceSrcPtr1);
	cudaFree(deviceSrcPtr2);
	cudaFree(deviceDstPtr);
}

