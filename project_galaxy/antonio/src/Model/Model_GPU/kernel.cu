#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (2.0f)
__global__ void compute_acc(
	const float4 * __restrict__ bodiesGPU,
	float4 * __restrict__ accelerationsGPU,
	int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	const bool active = (i < static_cast<unsigned int>(n_particles));

	// Shared-memory tiling with float4 body layout (x, y, z, mass).
	extern __shared__ float4 shBodies[];
	
	float ax = 0.0f, ay = 0.0f, az = 0.0f;

	// Each thread keeps one i-particle in registers and scans all j-tiles.
	// Inactive threads in the last partial block still participate in
	// tile loads/barriers, but they do not read/write particle state.
	float xi = 0.0f;
	float yi = 0.0f;
	float zi = 0.0f;
	if (active)
	{
		xi = bodiesGPU[i].x;
		yi = bodiesGPU[i].y;
		zi = bodiesGPU[i].z;
	}

	for (int tileStart = 0; tileStart < n_particles; tileStart += static_cast<int>(blockDim.x))
	{
		const int jGlobal = tileStart + static_cast<int>(threadIdx.x);

		if (jGlobal < n_particles)
		{
			shBodies[threadIdx.x] = bodiesGPU[jGlobal];
		}
		else
		{
			shBodies[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
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
				const float4 bodyj = shBodies[k];
				const float diffx = bodyj.x - xi;
				const float diffy = bodyj.y - yi;
				const float diffz = bodyj.z - zi;

				// Branchless softening keeps warp execution uniform.
				const float d2 = fmaf(diffx, diffx, fmaf(diffy, diffy, diffz * diffz));
				const float inv = rsqrtf(fmaxf(d2, 1.0f));
				const float dij = 10.0f * inv * inv * inv;

				const float mass = bodyj.w;
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
		// w is padding/alignment for float4 acceleration storage.
		accelerationsGPU[i].w = 0.0f;
	}


}

__global__ void maj_pos(float4 * bodiesGPU, float4 * velocitiesGPU, float4 * accelerationsGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_particles)
	{
		return;
	}

	velocitiesGPU[i].x += accelerationsGPU[i].x * EPS;
	velocitiesGPU[i].y += accelerationsGPU[i].y * EPS;
	velocitiesGPU[i].z += accelerationsGPU[i].z * EPS;
	// w is alignment padding for float4 velocity storage.
	velocitiesGPU[i].w = 0.0f;

	// Keep mass in .w unchanged; only integrate x/y/z.
	bodiesGPU[i].x += velocitiesGPU[i].x * DIFF_T;
	bodiesGPU[i].y += velocitiesGPU[i].y * DIFF_T;
	bodiesGPU[i].z += velocitiesGPU[i].z * DIFF_T;

}

void update_position_cu(float4* bodiesGPU, float4* velocitiesGPU, float4* accelerationsGPU, int n_particles)
{
	int nthreads = 256;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;
	size_t shared_bytes = nthreads * sizeof(float4);

	compute_acc<<<nblocks, nthreads, shared_bytes>>>(bodiesGPU, accelerationsGPU, n_particles);
	maj_pos    <<<nblocks, nthreads>>>(bodiesGPU, velocitiesGPU, accelerationsGPU, n_particles);
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
