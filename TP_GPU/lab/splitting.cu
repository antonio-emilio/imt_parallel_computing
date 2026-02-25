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

__host__ __device__ inline int p5 (int i)
{
	return i*i*i*i*i;
}

__host__ __device__ inline int value(int x)
{
	int a = 0;
	for (int i = 0; i < 30; ++i)
	{
		if (x & (1 << i))
		{
			a += p5(i+1);
		}
		else
		{
			a -= p5(i+1);
		}
	}
	return abs(a);
}


void display_final_result(int x)
{
	int a = 0;
	int b = 0;
	for (int i = 0; i < 30; ++i)
	{
		if (x & (1 << i))
		{
			a += p5(i+1);
		}
		else
		{
			b += p5(i+1);
		}
	}
	std::cout << "a: " << a << std::endl;
	std::cout << "b: " << b << std::endl;
	std::cout << "x: " << std::hex << x << std::endl;
}


__global__ void mykernel(int* r) {
	constexpr int iterations = 1 << 14;
	int x3 = blockIdx.x;
	int x2 = threadIdx.x;
	int best_x = 0;
	int best_v = value(best_x);
	for (int x1 = 0; x1 < iterations; ++x1)
	{
		int x = (x3 << 20) | (x2 << 14) | x1;
		int v = value(x);
		if (v < best_v)
		{
			best_x = x;
			best_v = v;
		}
	}
	r[(x3 << 6) | x2] = best_x;
}

int main() {
	// compute number of blocks and threads
	constexpr int blocks = 1 << 10;
	constexpr int threads = 1 << 6;

	// create pointer and allocate GPU memory
	int* rGPU = NULL;
	cudaMalloc((void**)&rGPU, blocks * threads * sizeof(int));

	// launch kernel
	// measure time before and after kernel launch to compute execution time
	auto start = std::chrono::high_resolution_clock::now();
	
	mykernel<<<blocks, threads>>>(rGPU);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Kernel execution time: " << elapsed.count() << " seconds" << std::endl;

	// allocate memory on CPU side
	std::vector<int> r(blocks * threads);

	// transfer values from GPU to CPU
	cudaMemcpy(r.data(), rGPU, blocks * threads * sizeof(int), cudaMemcpyDeviceToHost);

	// free allocated GPU memory
	cudaFree(rGPU);

	int best_x = 0;
	int best_v = value(best_x);

	// final step : determine best x amongs the ones that have been returned by
	// the GPU

	for (int i = 0; i < blocks * threads; ++i)
	{
		int v = value(r[i]);
		if (v < best_v)
		{
			best_x = r[i];
			best_v = v;
		}
	}
	display_final_result(best_x);
}
