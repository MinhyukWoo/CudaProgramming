#pragma once
#include <vector>
#include <cuda_runtime.h>

#ifdef __cplusplus 
extern "C" {//<-- extern ����
#endif
	enum DeviceType
	{
		CUDA = 0,
		CPU,
		MP,
	};

	// �������� �̹��� ����� �� Ŭ����
	class ImageBlender
	{
	public:
		ImageBlender() : ImageBlender(100, 100) {}
		ImageBlender(int rows, int cols);
		~ImageBlender();

		// �̹��� ó�� ��� ��� : operation_time, total_time
		std::pair<double, double> Blend(double weight, DeviceType device_type);

		// �̹��� ���� �� ���� Print
		void Print();

	private:
		int _rows, _cols;		// image total size
		int ** _src_img1;
		int ** _src_img2;
		int ** _dst_img;
	};

	//(����) Ŭ���� ���� �Լ��� �ȸ���� �۷ι��� ������ ������?
	// �� Device Type ���� ���� ����
	void BlendByCpu(int**src_img1, int**src_img2, int**dst_img, int rows, int cols, double weight);

	void BlendByMp(int**src_img1, int**src_img2, int**dst_img, int rows, int cols, double weight);

	// __global__ Ű���带 ���̸�, Host�� ���� ȣ��Ǿ� Device���� �۵��ȴ�.
	/// cu���� �����ؾ���
	__global__ void BlendByGpu(int**src_img1, int**src_img2, int**dst_img, int rows, int cols, double weight);

#ifdef __cplusplus 
}
#endif