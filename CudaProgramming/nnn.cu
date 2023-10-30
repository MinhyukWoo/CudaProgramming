#include "nnn.cuh"
#include "device_launch_parameters.h"
#include<stdio.h>


__device__ void addTwoWords_(WORD *dstPtr, WORD *srcPtr1, WORD *srcPtr2, size_t index) {
	dstPtr[index] = srcPtr1[index] < USHRT_MAX - srcPtr2[index] ? srcPtr1[index] + srcPtr2[index] : USHRT_MAX;
}
