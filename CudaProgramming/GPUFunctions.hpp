#pragma once
#include "GPUTypes.hpp"
#include "GPUEnum.cuh"
#include "math_functions.h"
#ifdef __cplusplus 
extern "C" {//<-- extern ����
#endif
	// using CUDA_FUNC_NONE
	__device__ void None(WORD *dstPtr, WORD *srcPtr1, WORD *srcPtr2, size_t index) {

	}

	// using CUDA_FUNC_ADD
	__device__ void Add(WORD *dstPtr, WORD *srcPtr1, WORD *srcPtr2, size_t index) {
		dstPtr[index] = srcPtr1[index] < USHRT_MAX - srcPtr2[index] ? srcPtr1[index] + srcPtr2[index] : USHRT_MAX;
	};

	// using CUDA_FUNC_SUBTRACT
	__device__ void Subtract(WORD *dstPtr, WORD *srcPtr1, WORD *srcPtr2, size_t index) {
		dstPtr[index] = srcPtr1[index] >= srcPtr2[index] ? srcPtr1[index] - srcPtr2[index] : 0;
	};

	// using CUDA_FUNC_MIN
	__device__ void Min(WORD *dstPtr, WORD *srcPtr1, WORD *srcPtr2, size_t index) {
		dstPtr[index] = min(srcPtr1[index], srcPtr2[index]);
	};

	// kernelsTwo1D ������ E_CUDA_FUNC ���� ������ �Ȱ��� �ϱ�
	/// CUDA������ ���漱���� �ȵǹǷ� kernelsTwo1D�� ���� �Ʒ��� ��ġ
	__device__ kernelTwo1D_t kernelsTwo1D[_E_CUDA_FUNC_END_] = {
		None			// CUDA_FUNC_NONE
		, Add			// CUDA_FUNC_ADD
		, Subtract		// CUDA_FUNC_SUBTRACT
		, Min			// CUDA_FUNC_MIN
		/* , �Լ��̸� */
	};
#ifdef __cplusplus 
}
#endif
