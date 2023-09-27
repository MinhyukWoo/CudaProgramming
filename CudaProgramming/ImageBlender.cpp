#include"ImageBlender.cuh"
#include<random>
#include<time.h>
#include<iostream>
#include<cmath>
#include<algorithm>
#include <omp.h>
using namespace std;


ImageBlender::ImageBlender(int size) {
	time_t tmp_time;
	srand((unsigned int)time(&tmp_time));
	_size = size;
	_src_img1 = new int[size];
	_src_img2 = new int[size];
	_dst_img = new int[size];
	for (int i = 0; i < size; i++)
	{
		_src_img1[i] = rand() % 256;
		_src_img2[i] = rand() % 256;
		_dst_img[i] = 0;
	}
}
ImageBlender::~ImageBlender() {
	delete _src_img1;
	delete _src_img2;
	delete _dst_img;
}


void ImageBlender::Print() {
	cout << "src_img1: " << '[';
	for (int i = 0; i < _size; i++)
	{
		cout << _src_img1[i] << ' ';
	}
	cout << ']' << endl;

	cout << "src_img2: " << '[';
	for (int i = 0; i < _size; i++)
	{
		cout << _src_img2[i] << ' ';
	}
	cout << ']' << endl;

	cout << "dst_img: " << '[';
	for (int i = 0; i < _size; i++)
	{
		cout << _dst_img[i] << ' ';
	}
	cout << ']' << endl;
}


void BlendByCpu(int*src_img1, int*src_img2, int*dst_img, int size, double weight) {
	for (int i = 0; i < size; i++) {
		double weighted_a = (double)src_img1[i] * weight;
		double weighted_b = (double)src_img2[i] * (1.0 - weight);
		double val = round(weighted_a + weighted_b);
		if (val < 0) {
			val = 0;
		}
		else if (val > 255) {
			val = 255;
		}
		dst_img[i] = (int)val;
	}
}


void BlendByMp(int*src_img1, int*src_img2, int*dst_img, int size, double weight) {
	#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		double weighted_a = (double)src_img1[i] * weight;
		double weighted_b = (double)src_img2[i] * (1.0 - weight);
		double val = round(weighted_a + weighted_b);
		if (val < 0) {
			val = 0;
		}
		else if (val > 255) {
			val = 255;
		}
		dst_img[i] = (int)val;
	}
}