#include"Convolution.cuh"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;

Convolution::Convolution()
{
	this->kernel = new int*[3];
	for (int i = 0; i < 3; i++) {
		this->kernel[i] = new int[3];
	}
}

Convolution::~Convolution() {
	for (int i = 0; i < 3; i++) {
		delete this->kernel[i];
	}
	delete this->kernel;
}

__global__ void convolution() {

}

void Convolution::Apply(int ** image_array) {
	
}


int** GetImageArray(const char * directory) {
	Mat img = imread(directory, IMREAD_COLOR);
	cvtColor(img, img, CV_RGBA2GRAY);

	int ** image_array = new int*[img.rows];
	for (int i = 0; i < img.rows; i++)
	{
		image_array[i] = new int[img.cols];
	}
	uchar* img_data = img.data;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			image_array[i][j] = img_data[i * img.cols + j];
		}
	}
	return image_array;
}


void PrintConvolution() {
	int ** image_array = GetImageArray("Lenna.png");
	for (int i = 0; i < sizeof(image_array) / sizeof(*image_array); i++)
	{
		delete image_array[i];
	}
	delete image_array;
}