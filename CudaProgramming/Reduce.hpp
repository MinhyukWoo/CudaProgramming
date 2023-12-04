#pragma once
#include <cuda_runtime.h>
#include "GPUTypes.hpp"
#ifdef __cplusplus 
extern "C" {//<-- extern ½ÃÀÛ
#endif
	enum E_BOPER
	{
		PLUS = 0,
		MAX,
		MIN,
	};
	int GetBadSum(int * ptr, int size);
	int Reduce(int * ptr, int size, E_BOPER index);
	unsigned int* GetComatrix(WORD * image, size_t rowLength, size_t colLength);
#ifdef __cplusplus 
}
#endif