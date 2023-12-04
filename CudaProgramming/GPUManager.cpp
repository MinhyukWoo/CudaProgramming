#pragma once
#include "GPUFunctions.hpp"
#include "GPUManager.hpp"
#include "GPUTypes.hpp"
#include "GPUEnum.cuh"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdexcept>


__global__ void Init() {
}

__global__ void ProcessTwo1D(WORD *deviceDstPtr, WORD *deviceSrcPtr1, WORD *deviceSrcPtr2, size_t size, E_CUDA_FUNC indexKernelTwo1D) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size) { // 주어진 크기이하인 인덱스만 사용
		kernelsTwo1D[indexKernelTwo1D](deviceDstPtr, deviceSrcPtr1, deviceSrcPtr2, tid); // 인덱스에 해당하는 함수 포인터 실행
	}
}

GPUManager::GPUManager() : _threadSize(256) {
	if (IsCUDAAvailable()) {
		Init << <1, 1 >> > (); // CUDA 초기 실행할 때 발생하는 오버헤드를 미리 발생 시킴
	}
};

void GPUManager::Process(WORD *dstPtr,WORD *srcPtr1, WORD *srcPtr2, size_t size, E_CUDA_FUNC indexKernelsTwo1D) {
	if(IsCUDAAvailable()) { // CUDA 장치가 있는지 검사
		if (indexKernelsTwo1D >= _E_CUDA_FUNC_END_ || indexKernelsTwo1D < 0) { // 유효한 인덱스 값인지 검사
			throw std::out_of_range("GPUManager.Process: Function's Index Error");
		}
		// Device 메모리 포인터 동적 할당
		WORD *deviceDstPtr, *deviceSrcPtr1, *deviceSrcPtr2;
		cudaMalloc(&deviceSrcPtr1, sizeof(WORD) * size);
		cudaMalloc(&deviceSrcPtr2, sizeof(WORD) * size);
		cudaMalloc(&deviceDstPtr, sizeof(WORD) * size);

		// Host 메모리에 있는 값을 Device 메모리에 복사
		cudaMemcpy(deviceSrcPtr1, srcPtr1, sizeof(WORD) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(deviceSrcPtr2, srcPtr2, sizeof(WORD) * size, cudaMemcpyHostToDevice);
		// 커널 실행
		ProcessTwo1D << <1 + size / _threadSize, _threadSize >> > (deviceDstPtr, deviceSrcPtr1, deviceSrcPtr2, size, indexKernelsTwo1D);
		// Device 메모리에 있는 값을 Host 메모리에 복사
		cudaMemcpy(dstPtr, deviceDstPtr, sizeof(WORD) * size, cudaMemcpyDeviceToHost);

		// Device 메모리 포인터 동적해제
		cudaFree(deviceSrcPtr1);
		cudaFree(deviceSrcPtr2);
		cudaFree(deviceDstPtr);
	}
}

bool GPUManager::IsCUDAAvailable()
{
	int device_count = 0;
	cudaGetDeviceCount(&device_count); // 실행 PC에 CUDA 장치가 있으면 1이상의 값
	return device_count != 0;
}

