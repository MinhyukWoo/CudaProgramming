#pragma once
#include "GPUManager.cuh"
#include "GPUTypes.cuh"
#include "GPUEnum.cuh"
#include "GPUFunctions.cuh"
#include "device_launch_parameters.h"
#include <stdio.h>


__global__ void Init() {

}

GPUManager::GPUManager() : _threadSize(256) {
	Init << <1, 1 >> > ();
};

__global__ void ProcessTwo1D(WORD *deviceDstPtr, WORD *deviceSrcPtr1, WORD *deviceSrcPtr2, size_t size, kernelTwo1D_t kernelTwo1D) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size) {
		(*kernelTwo1D)(deviceDstPtr, deviceSrcPtr1, deviceSrcPtr2, tid);
	}
}


void GPUManager::Process(WORD *dstPtr,WORD *srcPtr1, WORD *srcPtr2, size_t size, E_CUDA_FUNC indexKernelsTwo1D) {
	WORD *deviceDstPtr, *deviceSrcPtr1, *deviceSrcPtr2;
	cudaMalloc(&deviceSrcPtr1, sizeof(WORD) * size);
	cudaMemcpy(deviceSrcPtr1, srcPtr1, sizeof(WORD) * size, cudaMemcpyHostToDevice);
	cudaMalloc(&deviceSrcPtr2, sizeof(WORD) * size);
	cudaMemcpy(deviceSrcPtr2, srcPtr2, sizeof(WORD) * size, cudaMemcpyHostToDevice);
	cudaMalloc(&deviceDstPtr, sizeof(WORD) * size);
	
	kernelTwo1D_t hostKernelTwo1D;
	auto error = cudaMemcpyFromSymbol(
		&hostKernelTwo1D, kernelsTwo1D[0], sizeof(kernelsTwo1D[0])
	);
	printf("%s\n", cudaGetErrorString(error));
	
	ProcessTwo1D << <1 + size / _threadSize, _threadSize >> > (deviceDstPtr, deviceSrcPtr1, deviceSrcPtr2, size, hostKernelTwo1D);
	cudaDeviceSynchronize();
	cudaMemcpy(dstPtr, deviceDstPtr, sizeof(WORD) * size, cudaMemcpyDeviceToHost);
	cudaFree(deviceSrcPtr1);
	cudaFree(deviceSrcPtr2);
	cudaFree(deviceDstPtr);
}

