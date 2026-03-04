#ifdef GALAX_MODEL_GPU

#ifndef MODEL_GPU_HPP_
#define MODEL_GPU_HPP_

#include "../Model.hpp"

#include <cuda_runtime.h>
#include "kernel.cuh"

class Model_GPU : public Model
{
private:
	/*
	At model_CPU there isn't positions
	and velocities and accelerations are not float3 
	but 3 vectors of floats, one for each component.
	*/
	std::vector<float3> positionsf3    ;
	std::vector<float3> velocitiesf3   ;
	std::vector<float3> accelerationsf3;

	/*
	Also in model_cpu doesn't have this pointers
	Also doesn't have massesGPU
	*/
	float3* positionsGPU;
	float3* velocitiesGPU;
	float3* accelerationsGPU;
	float*  massesGPU;

public:
	Model_GPU(const Initstate& initstate, Particles& particles);

	/*
	In model_cpu the destructor is default, 
	and step = 0
	*/
	virtual ~Model_GPU();

	virtual void step();
};
#endif // MODEL_GPU_HPP_

#endif // GALAX_MODEL_GPU
