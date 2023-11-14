#pragma once
#include <cuda_runtime.h>
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


#ifdef __cplusplus 
}
#endif