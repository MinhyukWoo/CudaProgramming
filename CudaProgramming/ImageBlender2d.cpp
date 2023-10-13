#include "ImageBlender2d.cuh"

#include <random>
#include <time.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <omp.h>
using namespace std;


ImageBlender::ImageBlender(int rows, int cols) {
	time_t tmp_time;
	srand((unsigned int)time(&tmp_time));
	_rows = rows;
	_cols = cols;
	_src_img1 = new int*[_rows];
	_src_img2 = new int*[_rows];
	_dst_img = new int*[_rows];
	for (int i = 0; i < _rows; i++)
	{
		_src_img1[i] = new int[_cols];
		_src_img2[i] = new int[_cols];
		_dst_img[i] = new int[_cols];
		for (int j = 0; j < _cols; j++)
		{
			_src_img1[i][j] = rand() % 256;
			_src_img2[i][j] = rand() % 256;
			_dst_img[i][j] = 0;
		}
	}
}


ImageBlender::~ImageBlender() {
	for (int i = 0; i < _rows; i++)
	{
		delete _src_img1[i];
		delete _src_img2[i];
		delete _dst_img[i];
	}
	delete _src_img1;
	delete _src_img2;
	delete _dst_img;
}


// 이미지 내부 값 전부 Print
void ImageBlender::Print() {
	cout << "src_img1: " << '[';
	for (int i = 0; i < _rows; i++)
	{
		for (int j = 0; j < _cols; j++)
		{
			cout << _src_img1[i][j] << ' ';
		}
	}
	cout << ']' << endl;

	cout << "src_img2: " << '[';
	for (int i = 0; i < _rows; i++)
	{
		for (int j = 0; j < _cols; j++)
		{
			cout << _src_img2[i][j] << ' ';
		}
	}
	cout << ']' << endl;

	cout << "dst_img: " << '[';
	for (int i = 0; i < _rows; i++)
	{
		for (int j = 0; j < _cols; j++)
		{
			cout << _dst_img[i][j] << ' ';
		}
	}
	cout << ']' << endl;
}


void BlendByCpu(int**src_img1, int**src_img2, int**dst_img, int rows, int cols, double weight) {
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			double weighted_element1 = (double)src_img1[i][j] * weight;
			double weighted_element2 = (double)src_img2[i][j] * (1.0 - weight);
			double val = round(weighted_element1 + weighted_element2);
			if (val < 0) {
				val = 0;
			}
			else if (val > 255) {
				val = 255;
			}
			dst_img[i][j] = (int)val;
		}
	}
}


void BlendByMp(int**src_img1, int**src_img2, int**dst_img, int rows, int cols, double weight) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			double weighted_element1 = (double)src_img1[i][j] * weight;
			double weighted_element2 = (double)src_img2[i][j] * (1.0 - weight);
			double val = round(weighted_element1 + weighted_element2);
			if (val < 0) {
				val = 0;
			}
			else if (val > 255) {
				val = 255;
			}
			dst_img[i][j] = (int)val;
		}
	}
}