#include "VectorAdditionUsingStream.cuh"
#include "cuda.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <curand_kernel.h>
#include <time.h>

__global__ void set_element_random(int* devicePtr, curandState* deviceStates, int lengthData, int seed) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < lengthData) {
		curand_init(seed, idx, 0, &deviceStates[idx]);
		devicePtr[idx] = curand(&deviceStates[idx]) % 100 + 1;
	}
}

__global__ void tmp() {

}

VectorAdditionUsingStream::VectorAdditionUsingStream(int lengthData) {
	tmp << <1, 1 >> > ();
	time_t startTime, memcpyTime, synchronizeTime, endTime;
	startTime = clock();
	_lengthData = lengthData;
	int* devicePtr;
	curandState * deviceStates;
	int seed = (int)time(NULL);
	
	cudaMallocHost(&_data, sizeof(int) * lengthData);
	cudaMalloc(&devicePtr, sizeof(int) * lengthData);
	cudaMalloc(&deviceStates, sizeof(curandState) * lengthData);
	
	set_element_random << <1 + lengthData / 256, 256, 0 >> > (devicePtr, deviceStates, lengthData, seed);
	cudaMemcpy(_data, devicePtr, sizeof(int) * lengthData, cudaMemcpyDeviceToHost);
	memcpyTime = clock();

	cudaDeviceSynchronize();
	synchronizeTime = clock();

	cudaFree(&devicePtr);
	cudaFree(&deviceStates);
	endTime = clock();
	std::cout << "Default Stream에서 cudaMemcpy까지 걸린 시간 : " << memcpyTime - startTime << std::endl;
	std::cout << "Device Synchronzie까지 걸리는 시간 : " << synchronizeTime - startTime<< std::endl;
	std::cout << "소요 시간 : " << endTime - startTime << std::endl;
}

VectorAdditionUsingStream::VectorAdditionUsingStream(int lengthData, int lengthStream) {
	time_t startTime;
	startTime = clock();
	_lengthData = lengthData;
	int* devicePtr;
	curandState * deviceStates;
	int seed = (int)time(NULL);

	cudaMallocHost(&_data, sizeof(int) * lengthData);
	cudaMalloc(&devicePtr, sizeof(int) * lengthData);
	cudaMalloc(&deviceStates, sizeof(curandState) * lengthData);

	int subLength = lengthData / lengthStream;
	int rest = lengthData % subLength;
	int currentPosition = 0;
	cudaStream_t *streams = new cudaStream_t[lengthStream];
	std::cout << "각 Stream 별 cudaMemcpy까지 걸린 시간" << std::endl;
	for (int i = 0; i < lengthStream; i++) {
		cudaStreamCreate(&streams[i]);
		int currentLength = subLength;
		if (i == lengthStream-1) {
			currentLength += rest;
		}
		set_element_random << <1 + lengthData / 256, 256, 0, streams[i] >> > (devicePtr + currentPosition, deviceStates, currentLength, seed);
		cudaMemcpyAsync(_data + currentPosition, devicePtr + currentPosition, sizeof(int) * currentLength, cudaMemcpyDeviceToHost, streams[i]);
		std::cout << i << ":" << clock() - startTime << std::endl;
		currentPosition += subLength;
	}
	std::cout << "각 Stream 별 Synchronzie까지 걸리는 시간" << std::endl;
	for (int i = 0; i < lengthStream; i++) {
		cudaStreamSynchronize(streams[i]);
		cudaStreamDestroy(streams[i]);
		std::cout << i << ":" << clock() - startTime << std::endl;
	}

	cudaFree(&devicePtr);
	cudaFree(&deviceStates);
	time_t endTime;
	endTime = clock();
	std::cout << "소요 시간" << endTime - startTime << std::endl;
}


VectorAdditionUsingStream::~VectorAdditionUsingStream() {

}

void VectorAdditionUsingStream::Process() {
	
}

void VectorAdditionUsingStream::Print() {
	std::cout << "Array:" << std::endl;
	for (int i = 0; i < _lengthData; i++) {
		std::cout << _data[i] << " ";
	}
	std::cout << std::endl;
}