#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cstdio>

__device__ unsigned short atomicAdd(unsigned short * address, unsigned short val) {
	size_t longAddressModulo = (size_t)address & 2;
	auto* baseAddress = (unsigned int *)((char*)address - longAddressModulo);
	unsigned int longVal = (unsigned int)val << (8 * longAddressModulo);
	unsigned int longOld = atomicAdd(baseAddress, longVal);

	unsigned int mask = 0x0000ffff << (8 * longAddressModulo);
	unsigned int maskedOld = longOld & mask;
	unsigned int overflow = (maskedOld + longVal) & ~mask;
	if (overflow) {
		atomicSub(baseAddress, overflow);
	}
	return (unsigned int)(maskedOld >> 8 * longAddressModulo);
}

__global__ void __Sum(unsigned short * src, unsigned short * dst, int length) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= length) {
		return;
	}
	atomicAdd(dst + 1, src[tid]);
	atomicAdd(dst + 0, 1);
}

void TestAtomicAdd() {
	unsigned short arr[] = { 65535, 20};
	int length = sizeof(arr) / sizeof(unsigned short);
	unsigned short *deviceSrc;
	unsigned short *deviceDst;
	unsigned short *out = new unsigned short[2]{ 0 };
	cudaMalloc(&deviceSrc, sizeof(unsigned short) * length);
	cudaMalloc(&deviceDst, sizeof(unsigned short) * 2);
	cudaMemcpy(deviceSrc, arr, sizeof(unsigned short) * length, cudaMemcpyHostToDevice);
	__Sum << <1, 16 >> > (deviceSrc, deviceDst, length);
	cudaMemcpy(out, deviceDst, sizeof(unsigned short) * 2, cudaMemcpyDeviceToHost);
	cudaFree(deviceSrc);
	cudaFree(deviceDst);
	printf("%d %d\n", out[0], out[1]);
}