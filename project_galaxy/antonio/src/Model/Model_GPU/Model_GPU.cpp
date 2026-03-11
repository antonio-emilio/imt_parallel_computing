#ifdef GALAX_MODEL_GPU

#include <cmath>
#include <iostream>

#include "Model_GPU.hpp"
#include "kernel.cuh"

inline bool cuda_malloc(void ** devPtr, size_t size)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc(devPtr, size);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "error: unable to allocate buffer" << std::endl;
		return false;
	}
	return true;
}

inline bool cuda_memcpy(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(dst, src, count, kind);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "error: unable to copy buffer" << std::endl;
		return false;
	}
	return true;
}

void update_position_gpu(float3* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, float* massesGPU, int n_particles)
{
	update_position_cu(positionsGPU, velocitiesGPU, accelerationsGPU, massesGPU, n_particles);
	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		std::cout << "error: unable to synchronize threads" << std::endl;
}

/*
n_particles comes from Model.hpp
model_cpu:
	velocitiesx   (n_particles),
	accelerationsx(n_particles),

	->  particles.x[i] = initstate.positionsx[i];
		Initialize using the data from the Initstate object
	->  std::copy(initstate.velocitiesx.begin(), initstate.velocitiesx.end(), velocitiesx.begin());
		std::copy(1,2,3) - copy elements from one range to another. 
       	1 and 2 specify the range of elements to copy (the source), 
    	3 specifies the beginning of the destination range where the elements will be copied to
		need to be copy to avoid modify Initstate
		Starts with the initial velocities, but later it will modify them.
	*/		
Model_GPU
::Model_GPU(const Initstate& initstate, Particles& particles)
: Model(initstate, particles),
  positionsf3    (n_particles),
  velocitiesf3   (n_particles),
  accelerationsf3(n_particles)
{
	// init cuda
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
		std::cout << "error: unable to setup cuda device" << std::endl;

	for (int i = 0; i < n_particles; i++)
	{
		positionsf3[i].x     = initstate.positionsx [i];
		positionsf3[i].y     = initstate.positionsy [i];
		positionsf3[i].z     = initstate.positionsz [i];
		velocitiesf3[i].x    = initstate.velocitiesx[i];
		velocitiesf3[i].y    = initstate.velocitiesy[i];
		velocitiesf3[i].z    = initstate.velocitiesz[i];
		accelerationsf3[i].x = 0;
		accelerationsf3[i].y = 0;
		accelerationsf3[i].z = 0;
	}

	
	cuda_malloc((void**)&positionsGPU,     n_particles * sizeof(float3));
	cuda_malloc((void**)&velocitiesGPU,    n_particles * sizeof(float3));
	cuda_malloc((void**)&accelerationsGPU, n_particles * sizeof(float3));
	cuda_malloc((void**)&massesGPU,        n_particles * sizeof(float));

	cuda_memcpy(positionsGPU,     positionsf3.data()     , n_particles * sizeof(float3), cudaMemcpyHostToDevice);
	cuda_memcpy(velocitiesGPU,    velocitiesf3.data()    , n_particles * sizeof(float3), cudaMemcpyHostToDevice);
	cuda_memcpy(accelerationsGPU, accelerationsf3.data(), n_particles * sizeof(float3), cudaMemcpyHostToDevice);
	cuda_memcpy(massesGPU,        initstate.masses.data(), n_particles * sizeof(float), cudaMemcpyHostToDevice);
}

Model_GPU
::~Model_GPU()
{
	cudaFree((void**)&positionsGPU);
	cudaFree((void**)&velocitiesGPU);
	cudaFree((void**)&accelerationsGPU);
	cudaFree((void**)&massesGPU);
}

void Model_GPU
::step()
{
	/*
	cpu_fast: 
	->  std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
		std::fill(start_iterator, end_iterator, value);
		set all elements of a container to the same value.

	-> for bellow is the same, but was two fors
		->	const float diffx = particles.x[j] - particles.x[i];
			First was calculed the distance between particles in the positions i,j 
		->  float dij = diffx * diffx + diffy * diffy + diffz * diffz;
			than squared distance between the particles.
			is used because it avoids computing a slow square root initially
		->  if (dij < 1.0) dij = 10.0;
			Handle very small distances (softening)
			if distance  becomes very small → force becomes infinite → simulation explodes.
			code limits the force when particles are too close.
		->  else dij = std::sqrt(dij); dij = 10.0 / (dij * dij * dij);
			apply equation gravitational acceleration
		->  accelerationsx[i] += diffx * dij * initstate.masses[j];
			adds the acceleration caused by particle j to particle i.
	*/
	update_position_gpu(positionsGPU, velocitiesGPU, accelerationsGPU, massesGPU, n_particles);
	
	cuda_memcpy(positionsf3.data(), positionsGPU, n_particles * sizeof(float3), cudaMemcpyDeviceToHost);

	
	for (int i = 0; i < n_particles; i++)
	{
		particles.x[i] = positionsf3[i].x;
		particles.y[i] = positionsf3[i].y;
		particles.z[i] = positionsf3[i].z;
	}
	
}

#endif // GALAX_MODEL_GPU
