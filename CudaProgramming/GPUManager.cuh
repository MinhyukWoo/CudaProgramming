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
		bool IsCUDAAvailable();
	private:
		int _threadSize;
	};

	void AddKernelTwo1D();
#ifdef __cplusplus 
}
#endif
