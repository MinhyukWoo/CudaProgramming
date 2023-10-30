#pragma once
#include <cuda_runtime.h>
#include "GPUTypes.cuh"
#include "GPUEnum.cuh"

#ifdef __cplusplus 
extern "C" {//<-- extern ½ÃÀÛ
#endif
	class GPUManager {
	public:
		GPUManager();
		virtual ~GPUManager() {
		};
		void Process(WORD *dstPtr, WORD *srcPtr1, WORD *srcPtr2, size_t size, E_CUDA_FUNC indexKernelsTwo1D);
	private:
		int _threadSize;
	};


#ifdef __cplusplus 
}
#endif
