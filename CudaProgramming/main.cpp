#include "MultiplicationTable.cuh"
#include <iostream>
#include <stdio.h>
#include <time.h>
#include "ImageBlender2d.cuh"
#include "VectorAdditionUsingStream.cuh"
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

int main() {
	//for (int i = 1; i <= 10000; i *= 10) {
	//	PrintImageBlendingResult(i, i);
	//}
	PrintVectorAdditionUsingStream();
	cout << "프로그램이 종료되었습니다." << endl;
	char tmp[100];
	cin >> tmp;
	return 0;
}