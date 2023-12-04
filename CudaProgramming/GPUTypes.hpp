#pragma once
#include <cuda_runtime.h>
#ifdef __cplusplus 
extern "C" {//<-- extern ½ÃÀÛ
#endif
	typedef unsigned short WORD;
	typedef void(*kernelOne1D_t)(WORD*, WORD*, size_t);
	typedef void(*kernelTwo1D_t)(WORD*, WORD*, WORD*, size_t);

#ifdef __cplusplus 
}
#endif