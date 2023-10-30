#pragma once
#include <cuda_runtime.h>
#include "GPUTypes.cuh"
#ifdef __cplusplus 
extern "C" {//<-- extern 시작
#endif
	// using CUDA_FUNC_ADDTWOWORDS
	__device__ void addTwoWords(WORD *dstPtr, WORD *srcPtr1, WORD *srcPtr2, size_t index) {
		dstPtr[index] = srcPtr1[index] < USHRT_MAX - srcPtr2[index] ? srcPtr1[index] + srcPtr2[index] : USHRT_MAX;
	};


	// kernelsTwo1D 순서와 E_CUDA_FUNC 순서 무조건 똑같이 하기
	/// CUDA에서는 전방선언이 안되므로 kernelsTwo1D는 가장 아래에 배치
	__device__ kernelTwo1D_t kernelsTwo1D[_E_CUDA_FUNC_END_] = {
		 addTwoWords		// CUDA_FUNC_ADDTWOWORDS
		/* ,함수이름 */
	};
#ifdef __cplusplus 
}
#endif
