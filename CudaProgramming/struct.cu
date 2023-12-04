#include<cuda_runtime.h>
#include"struct.hpp"
#include<cstdio>
#include<string>


__global__ void __test(ExampleStruct *exampleStruct) {
	printf("Device: %d %.3f %s\n", exampleStruct->a, exampleStruct->b, exampleStruct->c);
}

void TestStruct() {
	ExampleStruct *hostExampleStruct = new ExampleStruct();
	ExampleStruct *deviceExampleStruct;
	char *text;
	cudaMalloc(&deviceExampleStruct, sizeof(ExampleStruct));
	cudaMalloc(&text, strlen(hostExampleStruct->c));
	cudaMemcpy(
		deviceExampleStruct, hostExampleStruct, sizeof(ExampleStruct), cudaMemcpyHostToDevice
	);
	cudaMemcpy(text, hostExampleStruct->c, strlen(hostExampleStruct->c), cudaMemcpyHostToDevice);
	cudaMemcpy(&(deviceExampleStruct->c), &text, sizeof(text), cudaMemcpyHostToDevice);

	printf("  Host: %d %.3f %s\n", hostExampleStruct->a, hostExampleStruct->b, hostExampleStruct->c);
	__test << <1, 1 >> > (deviceExampleStruct);

	cudaFree(deviceExampleStruct);
	delete hostExampleStruct;
}