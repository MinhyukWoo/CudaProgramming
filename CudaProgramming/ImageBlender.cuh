#pragma once
#include<vector>
#include <cuda_runtime.h>
#ifdef __cplusplus 
extern "C" {//<-- extern 시작
#endif
	enum DeviceType{CUDA, CPU, MP};
	class ImageBlender {
	public:
		ImageBlender() : ImageBlender(10) {
			
		}
		ImageBlender(int size);
		~ImageBlender();
		std::pair<double, double> Blend(double weight, DeviceType device_type);
		void Print();
	private:
		int _size;
		int * _src_img1;
		int * _src_img2;
		int * _dst_img;
	};
	void BlendByCpu(int*src_img1, int*src_img2, int*dst_img, int size, double weight);
	__global__ void BlendByGpu(int*src_img1, int*src_img2, int*dst_img, int size, double weight);
	void BlendByMp(int*src_img1, int*src_img2, int*dst_img, int size, double weight);
#ifdef __cplusplus 
}
#endif