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

__global__ void mykernel(float* r, const float* d, int n, int nn) {
    int ia = threadIdx.x;
    int ja = threadIdx.y;
    int ic = blockIdx.x;
    int jc = blockIdx.y;

    const float* t = d + nn * nn;

    __shared__ float xx[4][64];
    __shared__ float yy[4][64];

    float v[8][8];
    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            v[ib][jb] = HUGE_VALF;
        }
    }
    for (int ks = 0; ks < n; ks += 4) {
        int ija = ja * 8 + ia;
        int i = ic * 64 + ija;
        int j = jc * 64 + ija;
        for (int f = 0; f < 4; ++f) {
            int k = ks + f;
            xx[f][ija] = t[nn*k + i];
            yy[f][ija] = d[nn*k + j];
        }

        __syncthreads();

        #pragma unroll
        for (int f = 0; f < 4; ++f) {
            float y[8];
            for (int jb = 0; jb < 8; ++jb) {
                y[jb] = yy[f][jb * 8 + ja];
            }
            for (int ib = 0; ib < 8; ++ib) {
                float x = xx[f][ib * 8 + ia];
                for (int jb = 0; jb < 8; ++jb) {
                    v[ib][jb] = min(v[ib][jb], x + y[jb]);
                }
            }
        }

        __syncthreads();
    }
    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            int i = ic * 64 + ib * 8 + ia;
            int j = jc * 64 + jb * 8 + ja;
            if (i < n && j < n) {
                r[n*i + j] = v[ib][jb];
            }
        }
    }
}

__global__ void myppkernel(const float* r, float* d, int n, int nn) {
    int ja = threadIdx.x;
    int i = blockIdx.y;

    float* t = d + nn * nn;

    for (int jb = 0; jb < nn; jb += 64) {
        int j = jb + ja;
        float v = (i < n && j < n) ? r[n*i + j] : HUGE_VALF;
        d[nn*i + j] = v;
        t[nn*j + i] = v;
    }
}


static inline int divup(int a, int b) {
	return (a + b - 1)/b;
}

static inline int roundup(int a, int b) {
	return divup(a, b) * b;
}

void step(float* r, const float* d, int n) {
    int nn = roundup(n, 64);

    // Allocate memory & copy data to GPU
    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, 2 * nn * nn * sizeof(float)));
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, n * n * sizeof(float)));
    CHECK(cudaMemcpy(rGPU, d, n * n * sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel
    {
        dim3 dimBlock(64, 1);
        dim3 dimGrid(1, nn);
        myppkernel<<<dimGrid, dimBlock>>>(rGPU, dGPU, n, nn);
        CHECK(cudaGetLastError());
    }

    // Run kernel
    {
        dim3 dimBlock(8, 8);
        dim3 dimGrid(nn / 64, nn / 64);
        mykernel<<<dimGrid, dimBlock>>>(rGPU, dGPU, n, nn);
        CHECK(cudaGetLastError());
    }

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