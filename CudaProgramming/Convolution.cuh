#pragma once
#include <cuda_runtime.h>
#ifdef __cplusplus 
extern "C" {//<-- extern ����
#endif
	class Convolution {
	public:
		Convolution();
		~Convolution();
		void Apply(int** image_arr);
		int ** kernel;
	};
#ifdef __cplusplus 
}
#endif

