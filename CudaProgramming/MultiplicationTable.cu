#include "MultiplicationTable.cuh"
#include "cuda.h"
#include <stdio.h>
#include <time.h>

MultiplicationTable::MultiplicationTable() {

}

MultiplicationTable::~MultiplicationTable() {

}

__global__ void Calculate(int * arr) {
	arr[blockIdx.x * blockDim.x + threadIdx.x] = (blockIdx.x + 1) * (threadIdx.x + 1);
}

double MultiplicationTable::PrintTableByGpu(int end) {
	clock_t t_start, t_end;
	int * arr;
	cudaMallocManaged(&arr, end * end * sizeof(int));
	t_start = clock();
	Calculate <<< end, end >>> (arr);
	t_end = clock();
	cudaDeviceSynchronize();
	printf("GPU: %d * %d = %d\n", end, end, arr[end * end - 1]);
	cudaFree(arr);
	return t_end - t_start;
}

double MultiplicationTable::PrintTableByCpu(int end) {
	clock_t t_start, t_end;
	int * arr = new int[end * end];
	t_start = clock();
	for (int i = 0; i < end; i++) {
		for (int j = 0; j < end; j++) {
			arr[i * end + j] = (i + 1) * (j + 1);
		}
	}
	t_end = clock();
	printf("CPU: %d * %d = %d\n", end, end, arr[end * end - 1]);
	delete arr;
	return t_end - t_start;
}