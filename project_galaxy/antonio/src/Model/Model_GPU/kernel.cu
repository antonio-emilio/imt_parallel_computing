#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (2.0f)
__global__ void compute_acc(
	const float3 * __restrict__ positionsGPU,
	float3 * __restrict__ accelerationsGPU,
	const float * __restrict__ massesGPU,
	int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	const bool active = (i < static_cast<unsigned int>(n_particles));

	// Shared-memory tiling: each block loads one tile of j-particles once,
	// then all threads in the block reuse it for force accumulation.
	extern __shared__ unsigned char sharedRaw[];
	float3* shPositions = reinterpret_cast<float3*>(sharedRaw);
	float* shMasses = reinterpret_cast<float*>(shPositions + blockDim.x);
	
	float ax = 0.0f, ay = 0.0f, az = 0.0f;

	// Now each thread access positionsGPU[j] sequentially
	// Inactive threads in the last partial block still participate in
	// tile loads/barriers, but they do not read/write particle state.
	float xi = 0.0f;
	float yi = 0.0f;
	float zi = 0.0f;
	if (active)
	{
		xi = positionsGPU[i].x;
		yi = positionsGPU[i].y;
		zi = positionsGPU[i].z;
	}

	for (int tileStart = 0; tileStart < n_particles; tileStart += static_cast<int>(blockDim.x))
	{
		const int jGlobal = tileStart + static_cast<int>(threadIdx.x);

		if (jGlobal < n_particles)
		{
			shPositions[threadIdx.x] = positionsGPU[jGlobal];
			shMasses[threadIdx.x] = massesGPU[jGlobal];
		}
		else
		{
			shPositions[threadIdx.x] = make_float3(0.0f, 0.0f, 0.0f);
			shMasses[threadIdx.x] = 0.0f;
		}

		__syncthreads();

		const int tileSize = ((tileStart + static_cast<int>(blockDim.x)) < n_particles)
			? static_cast<int>(blockDim.x)
			: (n_particles - tileStart);

		if (active)
		{
			#pragma unroll 4
			for (int k = 0; k < tileSize; ++k)
			{
				const float diffx = shPositions[k].x - xi;
				const float diffy = shPositions[k].y - yi;
				const float diffz = shPositions[k].z - zi;

				// Branchless softening keeps warp execution uniform.
				const float d2 = fmaf(diffx, diffx, fmaf(diffy, diffy, diffz * diffz));
				const float inv = rsqrtf(fmaxf(d2, 1.0f));
				const float dij = 10.0f * inv * inv * inv;

				const float mass = shMasses[k];
				ax += diffx * dij * mass;
				ay += diffy * dij * mass;
				az += diffz * dij * mass;
			}
		}

		__syncthreads();

	}
	if (active)
	{
		accelerationsGPU[i].x = ax;
		accelerationsGPU[i].y = ay;
		accelerationsGPU[i].z = az;
	}


}

__global__ void maj_pos(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_particles)
	{
		return;
	}

	velocitiesGPU[i].x += accelerationsGPU[i].x * EPS;
	velocitiesGPU[i].y += accelerationsGPU[i].y * EPS;
	velocitiesGPU[i].z += accelerationsGPU[i].z * EPS;

	positionsGPU[i].x += velocitiesGPU[i].x * DIFF_T;
	positionsGPU[i].y += velocitiesGPU[i].y * DIFF_T;
	positionsGPU[i].z += velocitiesGPU[i].z * DIFF_T;

}

void update_position_cu(float3* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, float* massesGPU, int n_particles)
{
	int nthreads = 128;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;
	size_t shared_bytes = nthreads * (sizeof(float3) + sizeof(float));

	compute_acc<<<nblocks, nthreads, shared_bytes>>>(positionsGPU, accelerationsGPU, massesGPU, n_particles);
	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}

/*
__global__ void compute_acc(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, float* massesGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void maj_pos(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

}

void update_position_cu(float3* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, float* massesGPU, int n_particles)
{
	int nthreads = 128;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, massesGPU, n_particles);
	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}
*/

#endif // GALAX_MODEL_GPU
