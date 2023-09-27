#pragma once
#include <cuda_runtime.h>
#ifdef __cplusplus 
extern "C" {//<-- extern ����
#endif
	class MultiplicationTable {
	public:
		MultiplicationTable();
		~MultiplicationTable();
		double PrintTableByGpu(int end);
		double PrintTableByCpu(int end);
	};
#ifdef __cplusplus 
}
#endif