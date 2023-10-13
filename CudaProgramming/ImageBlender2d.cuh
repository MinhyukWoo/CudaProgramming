#pragma once
#include <vector>
#include <cuda_runtime.h>

#ifdef __cplusplus 
extern "C" {//<-- extern 시작
#endif
	enum DeviceType
	{
		CUDA = 0,
		CPU,
		MP,
	};

	// 랜덤으로 이미지 만들어 줄 클래스
	class ImageBlender
	{
	public:
		ImageBlender() : ImageBlender(100, 100) {}
		ImageBlender(int rows, int cols);
		~ImageBlender();

		// 이미지 처리 계산 결과 : operation_time, total_time
		std::pair<double, double> Blend(double weight, DeviceType device_type);

		// 이미지 내부 값 전부 Print
		void Print();

	private:
		int _rows, _cols;		// image total size
		int ** _src_img1;
		int ** _src_img2;
		int ** _dst_img;
	};

	//(혜정) 클래스 내부 함수로 안만들고 글로벌로 선언한 이유는?
	// 각 Device Type 별로 연산 진행
	void BlendByCpu(int**src_img1, int**src_img2, int**dst_img, int rows, int cols, double weight);

	void BlendByMp(int**src_img1, int**src_img2, int**dst_img, int rows, int cols, double weight);

	// __global__ 키워드를 붙이면, Host를 통해 호출되어 Device에서 작동된다.
	/// cu에서 정의해야함
	__global__ void BlendByGpu(int**src_img1, int**src_img2, int**dst_img, int rows, int cols, double weight);

#ifdef __cplusplus 
}
#endif