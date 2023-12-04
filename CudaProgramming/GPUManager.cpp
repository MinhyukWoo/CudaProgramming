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
	if (tid < size) { // �־��� ũ�������� �ε����� ���
		kernelsTwo1D[indexKernelTwo1D](deviceDstPtr, deviceSrcPtr1, deviceSrcPtr2, tid); // �ε����� �ش��ϴ� �Լ� ������ ����
	}
}

GPUManager::GPUManager() : _threadSize(256) {
	if (IsCUDAAvailable()) {
		Init << <1, 1 >> > (); // CUDA �ʱ� ������ �� �߻��ϴ� ������带 �̸� �߻� ��Ŵ
	}
};

void GPUManager::Process(WORD *dstPtr,WORD *srcPtr1, WORD *srcPtr2, size_t size, E_CUDA_FUNC indexKernelsTwo1D) {
	if(IsCUDAAvailable()) { // CUDA ��ġ�� �ִ��� �˻�
		if (indexKernelsTwo1D >= _E_CUDA_FUNC_END_ || indexKernelsTwo1D < 0) { // ��ȿ�� �ε��� ������ �˻�
			throw std::out_of_range("GPUManager.Process: Function's Index Error");
		}
		// Device �޸� ������ ���� �Ҵ�
		WORD *deviceDstPtr, *deviceSrcPtr1, *deviceSrcPtr2;
		cudaMalloc(&deviceSrcPtr1, sizeof(WORD) * size);
		cudaMalloc(&deviceSrcPtr2, sizeof(WORD) * size);
		cudaMalloc(&deviceDstPtr, sizeof(WORD) * size);

		// Host �޸𸮿� �ִ� ���� Device �޸𸮿� ����
		cudaMemcpy(deviceSrcPtr1, srcPtr1, sizeof(WORD) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(deviceSrcPtr2, srcPtr2, sizeof(WORD) * size, cudaMemcpyHostToDevice);
		// Ŀ�� ����
		ProcessTwo1D << <1 + size / _threadSize, _threadSize >> > (deviceDstPtr, deviceSrcPtr1, deviceSrcPtr2, size, indexKernelsTwo1D);
		// Device �޸𸮿� �ִ� ���� Host �޸𸮿� ����
		cudaMemcpy(dstPtr, deviceDstPtr, sizeof(WORD) * size, cudaMemcpyDeviceToHost);

		// Device �޸� ������ ��������
		cudaFree(deviceSrcPtr1);
		cudaFree(deviceSrcPtr2);
		cudaFree(deviceDstPtr);
	}
}

bool GPUManager::IsCUDAAvailable()
{
	int device_count = 0;
	cudaGetDeviceCount(&device_count); // ���� PC�� CUDA ��ġ�� ������ 1�̻��� ��
	return device_count != 0;
}

