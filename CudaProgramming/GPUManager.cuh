#pragma once
#include <cuda_runtime.h>
#ifdef __cplusplus 
extern "C" {//<-- extern ½ÃÀÛ
#endif
	typedef unsigned short WORD;
	typedef void(*kernelOne1D_t)(WORD*, WORD*, size_t);
	typedef void(*kernelTwo1D_t)(WORD*, WORD*, WORD*, size_t);
	class GPUManager {
	public:
		GPUManager();
		virtual ~GPUManager() {
		};
		void Process(WORD *dstPtr, WORD *srcPtr, size_t size, kernelOne1D_t kernel1D);
		void Process(WORD *dstPtr, WORD *srcPtr1, WORD *srcPtr2, size_t size, kernelTwo1D_t kernel1D);
	private:
		int _threadSize;
	};
#ifdef __cplusplus 
}
#endif