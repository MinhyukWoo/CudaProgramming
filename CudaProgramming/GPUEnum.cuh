#pragma once
#include <cuda_runtime.h>
#ifdef __cplusplus 
extern "C" {//<-- extern ½ÃÀÛ
#endif
	enum E_CUDA_FUNC
	{
		CUDA_FUNC_NONE = 0,
		CUDA_FUNC_ADD,
		CUDA_FUNC_SUBTRACT,
		CUDA_FUNC_MIN,

		_E_CUDA_FUNC_END_
	};

#ifdef __cplusplus 
}
#endif