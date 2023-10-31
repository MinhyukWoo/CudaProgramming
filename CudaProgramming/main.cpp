#include "MultiplicationTable.cuh"
#include <iostream>
#include <stdio.h>
#include <time.h>
#include "ImageBlender2d.cuh"
#include "VectorAdditionUsingStream.cuh"
#include "GPUManager.cuh"
#include<algorithm>

using namespace std;

// 구구단
void PrintMultiplicationTable(int end) {
	MultiplicationTable m_table;
	clock_t gpu_start, gpu_end, cpu_start, cpu_end;
	double t1, t2;

	gpu_start = clock();
	t1 = m_table.PrintTableByGpu(end);
	gpu_end = clock();

	cpu_start = clock();
	t2 = m_table.PrintTableByCpu(end);
	cpu_end = clock();
	
	cout << endl;
	cout << "1단부터 " << end << "단까지 CPU   총 소요시간: " << cpu_end - cpu_start << endl;
	cout << "1단부터 " << end << "단까지 CPU 연산 소요시간: " << t1 << endl;
	cout << "1단부터 " << end << "단까지 GPU   총 소요시간: " << gpu_end - gpu_start << endl;
	cout << "1단부터 " << end << "단까지 GPU 연산 소요시간: " << t2 << endl;
	cout << "====================" << endl;
}

// 구구단 Print
void PrintMultiplicationTable() {
	PrintMultiplicationTable(9);
	PrintMultiplicationTable(100);
	PrintMultiplicationTable(1000);
	char tmp[100];
	cin >> tmp;
}

// Image Blender
//void PrintImageBlendingResult(int size) {
//	ImageBlender image_blender(size);
//	const double weight = 0.5f;
//
//	pair<double, double> cpu_times = image_blender.Blend( weight, CPU );
//	pair<double, double> mp_times = image_blender.Blend( weight, MP );
//	pair<double, double> cuda_times = image_blender.Blend( weight, CUDA );
//	cout << "____________________" << endl;
//	cout << "실험환경: size=" << size << endl;
//	cout << " CPU: 연산 소요 시간(" << cpu_times.first << "ms), 총 소요 시간(" << cpu_times.second << "ms)" << endl;
//	cout << "  MP: 연산 소요 시간(" << mp_times.first << "ms), 총 소요 시간(" << mp_times.second << "ms)" << endl;
//	cout << "CUDA: 연산 소요 시간(" << cuda_times.first << "ms), 총 소요 시간(" << cuda_times.second << "ms)" << endl;
//	cout << "--------------------" << endl;
//}

void PrintImageBlendingResult(int rows, int cols) {
	ImageBlender image_blender(rows, cols);
	const double weight = 0.5f;

	pair<double, double> cpu_times = image_blender.Blend(weight, CPU);
	pair<double, double> mp_times = image_blender.Blend(weight, MP);
	pair<double, double> cuda_times = image_blender.Blend(weight, CUDA);
	cout << "____________________" << endl;
	cout << "실험환경: (" << rows << ", "<< cols << ")" << endl;
	cout << " CPU: 연산 소요 시간(" << cpu_times.first << "ms), 총 소요 시간(" << cpu_times.second << "ms)" << endl;
	cout << "  MP: 연산 소요 시간(" << mp_times.first << "ms), 총 소요 시간(" << mp_times.second << "ms)" << endl;
	cout << "CUDA: 연산 소요 시간(" << cuda_times.first << "ms), 총 소요 시간(" << cuda_times.second << "ms)" << endl;
	cout << "--------------------" << endl;
}

void PrintVectorAdditionUsingStream() {
	VectorAdditionUsingStream vectorAddition1(100000);
	VectorAdditionUsingStream vectorAddition2(100000, 10);
}

// ========================================

void AddByCpu(WORD *dstPtr, WORD *srcPtr1, WORD *srcPtr2, size_t index) {
	dstPtr[index] = srcPtr1[index] < USHRT_MAX - srcPtr2[index] ? srcPtr1[index] + srcPtr2[index] : USHRT_MAX;
}

void SubtractByCpu(WORD *dstPtr, WORD *srcPtr1, WORD *srcPtr2, size_t index) {
	dstPtr[index] = srcPtr1[index] >= srcPtr2[index] ? srcPtr1[index] - srcPtr2[index] : 0;
}

void MinByCpu(WORD *dstPtr, WORD *srcPtr1, WORD *srcPtr2, size_t index) {
	dstPtr[index] = std::min(srcPtr1[index], srcPtr2[index]);
};

void PrintWords(WORD* ptr1, WORD* ptr2, WORD* ptr3, size_t length) {
	size_t subLength = length > 10 ? 10 : length;
	printf("%3c  %5s %5s %5s\n", ' ', "src1", "src2", "dst");
	for (size_t i = 0; i < subLength; i++) {
		printf("%3d: %5d %5d %5d\n", i+1, ptr1[i], ptr2[i], ptr3[i]);
	}
	if (length > subLength) {
		printf("...\n");
		size_t i = length - 1;
		printf("%3d: %5d %5d %5d\n", i+1, ptr1[i], ptr2[i], ptr3[i]);
	}
	printf("\n");
}

void TestGPUManager(size_t length = 100) {
	WORD* ptr1 = new WORD[length];
	WORD* ptr2 = new WORD[length];
	WORD* ptr3 = new WORD[length];
	time_t start, end;
	GPUManager gpuManager;

	for (size_t i = 0; i < length; i++) {
		ptr1[i] = rand() % USHRT_MAX;
		ptr2[i] = rand() % USHRT_MAX;
	}

	// START:ADD
	printf("====================\n");
	printf("  %s\n", "ADD");
	printf("====================\n");
	// CPU
	start = clock();
	for (int i = 0; i < length; i++) {
		AddByCpu(ptr3, ptr1, ptr2, i);
	}
	end = clock();
	printf("CPU DONE in %dms\n", end - start);
	PrintWords(ptr1, ptr2, ptr3, length);

	// GPU
	start = clock();
	if (gpuManager.IsCUDAAvailable()) {
		gpuManager.Process(ptr3, ptr1, ptr2, length, E_CUDA_FUNC::CUDA_FUNC_ADD);
	}
	end = clock();
	printf("CUDA DONE in %lldms\n", end - start);
	PrintWords(ptr1, ptr2, ptr3, length);
	// DONE: ADD


	// START:SUBTRACT
	printf("====================\n");
	printf("  %s\n", "SUBTRACT");
	printf("====================\n");
	// CPU
	start = clock();
	for (int i = 0; i < length; i++) {
		SubtractByCpu(ptr3, ptr1, ptr2, i);
	}
	end = clock();
	printf("CPU DONE in %dms\n", end - start);
	PrintWords(ptr1, ptr2, ptr3, length);

	// GPU
	start = clock();
	if (gpuManager.IsCUDAAvailable()) {
		gpuManager.Process(ptr3, ptr1, ptr2, length, E_CUDA_FUNC::CUDA_FUNC_SUBTRACT);
	}
	end = clock();
	printf("CUDA DONE in %lldms\n", end - start);
	PrintWords(ptr1, ptr2, ptr3, length);
	// DONE: SUBTRACT


	// START:MIN
	printf("====================\n");
	printf("  %s\n", "MIN");
	printf("====================\n");
	// CPU
	start = clock();
	for (int i = 0; i < length; i++) {
		MinByCpu(ptr3, ptr1, ptr2, i);
	}
	end = clock();
	printf("CPU DONE in %dms\n", end - start);
	PrintWords(ptr1, ptr2, ptr3, length);

	// GPU
	start = clock();
	if (gpuManager.IsCUDAAvailable()) {
		gpuManager.Process(ptr3, ptr1, ptr2, length, E_CUDA_FUNC::CUDA_FUNC_MIN);
	}
	end = clock();
	printf("CUDA DONE in %lldms\n", end - start);
	PrintWords(ptr1, ptr2, ptr3, length);
	// DONE: MIN


	// START: Error Handling
	printf("====================\n");
	printf("  %s\n", "Error Handling");
	printf("====================\n");

	// GPU
	try
	{
		start = clock();
		if (gpuManager.IsCUDAAvailable()) {
			gpuManager.Process(ptr3, ptr1, ptr2, length, (E_CUDA_FUNC)100);
		}
		end = clock();
		printf("CUDA DONE in %lldms\n", end - start);
		PrintWords(ptr1, ptr2, ptr3, length);
	}
	catch (const std::exception& except)
	{
		cout << except.what() << endl;
	}
	// DONE: Error Handling
}

int main() {
	TestGPUManager(10000000);
	cout << "프로그램이 종료되었습니다." << endl;
	char tmp[100];
	cin >> tmp;
	return 0;
}