#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

static inline void check(cudaError_t err, const char* context) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << context << ": "
			<< cudaGetErrorString(err) << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

#define CHECK(x) check(x, #x)
__global__ void mykernel(float* r, const float* d, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i >= n || j >= n)
		return;
	float v = HUGE_VALF;
	for (int k = 0; k < n; ++k) {
		float x = d[n*j + k];
		float y = d[n*k + i];
		float z = x + y;
		v = min(v, z);
	}
	r[n*i + j] = v;
}

static inline int divup(int a, int b) {
	return (a + b - 1)/b;
}

static inline int roundup(int a, int b) {
	return divup(a, b) * b;
}

void step(float* r, const float* d, int n) {
	// Allocate memory & copy data to GPU
	float* dGPU = NULL;
	CHECK(cudaMalloc((void**)&dGPU, n * n * sizeof(float)));
	float* rGPU = NULL;
	CHECK(cudaMalloc((void**)&rGPU, n * n * sizeof(float)));
	CHECK(cudaMemcpy(dGPU, d, n * n * sizeof(float), cudaMemcpyHostToDevice));

	// Run kernel
	dim3 dimBlock(16, 16);
	dim3 dimGrid(divup(n, dimBlock.x), divup(n, dimBlock.y));
	mykernel<<<dimGrid, dimBlock>>>(rGPU, dGPU, n);
	CHECK(cudaGetLastError());

	// Copy data back to CPU & release memory
	CHECK(cudaMemcpy(r, rGPU, n * n * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK(cudaFree(dGPU));
	CHECK(cudaFree(rGPU));
}

int main() {
	int n = 5000;
	int seed = 0;

	std::mt19937 gen(seed);
	std::uniform_real_distribution<> dis(1.0, 8.0);

	std::vector<float> d(n*n);
	for (int i = 0; i < d.size(); i++)
		d[i] = dis(gen);
	for (int i = 0; i < n; i++)
		d[i*n + i] = 0;


	std::vector<float> r(n*n);
	auto t1 = std::chrono::high_resolution_clock::now();

	step(r.data(), d.data(), n);

	auto t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> duration_s = t2 - t1;
	std::cout << "Duration: " << duration_s.count() << std::endl;

	// for (int i = 0; i < n; ++i) {
	//     for (int j = 0; j < n; ++j) {
	//         std::cout << d[i*n + j] << " ";
	//     }
	//     std::cout << "\n";
	// }

	// for (int i = 0; i < n; ++i) {
	//     for (int j = 0; j < n; ++j) {
	//         std::cout << r[i*n + j] << " ";
	//     }
	//     std::cout << "\n";
	// }
}