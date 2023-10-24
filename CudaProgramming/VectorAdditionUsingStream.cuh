#pragma once
#include <cuda_runtime.h>
#ifdef __cplusplus 
extern "C" {//<-- extern ½ÃÀÛ
#endif
	class VectorAdditionUsingStream {
	public:
		VectorAdditionUsingStream() : VectorAdditionUsingStream(100) {};
		VectorAdditionUsingStream(int lenghtData);
		VectorAdditionUsingStream(int lenghtData, int lengthStream);
		virtual ~VectorAdditionUsingStream();
		void Process();
		void Print();
	private:
		int * _data;
		int _lengthData;
	};
#ifdef __cplusplus 
}
#endif