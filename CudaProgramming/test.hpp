#pragma once
#include <cuda_runtime.h>
#ifdef __cplusplus 
extern "C" {//<-- extern ����
#endif
	int GetBadSum(int * ptr, int size);
	int GetReducedSum(int * ptr, int size);
#ifdef __cplusplus 
}
#endif