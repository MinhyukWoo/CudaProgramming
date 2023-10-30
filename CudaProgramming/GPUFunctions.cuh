#pragma once
#include <cuda_runtime.h>
#include "GPUTypes.cuh"
#ifdef __cplusplus 
extern "C" {//<-- extern ����
#endif
	// using CUDA_FUNC_ADDTWOWORDS
	__device__ void addTwoWords(WORD *dstPtr, WORD *srcPtr1, WORD *srcPtr2, size_t index) {
		dstPtr[index] = srcPtr1[index] < USHRT_MAX - srcPtr2[index] ? srcPtr1[index] + srcPtr2[index] : USHRT_MAX;
	};


	// kernelsTwo1D ������ E_CUDA_FUNC ���� ������ �Ȱ��� �ϱ�
	/// CUDA������ ���漱���� �ȵǹǷ� kernelsTwo1D�� ���� �Ʒ��� ��ġ
	__device__ kernelTwo1D_t kernelsTwo1D[_E_CUDA_FUNC_END_] = {
		 addTwoWords		// CUDA_FUNC_ADDTWOWORDS
		/* ,�Լ��̸� */
	};
#ifdef __cplusplus 
}
#endif
