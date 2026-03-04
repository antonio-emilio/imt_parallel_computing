#include <cmath>

#include "Model_CPU.hpp"

Model_CPU
::Model_CPU(const Initstate& initstate, Particles& particles)
: Model(initstate, particles),
  velocitiesx   (n_particles),
  velocitiesy   (n_particles),
  velocitiesz   (n_particles),
  accelerationsx(n_particles),
  accelerationsy(n_particles),
  accelerationsz(n_particles)
{
    // Initialize using the data from the Initstate object
	for (int i = 0; i < n_particles; i++)
	{
		particles.x[i] = initstate.positionsx[i];
		particles.y[i] = initstate.positionsy[i];
		particles.z[i] = initstate.positionsz[i];
	}
    // std::copy - copy elements from one range to another. 
    // It takes three arguments: the first two specify the range of elements to copy (the source), 
    // and the third specifies the beginning of the destination range where the elements will be copied to.
    std::copy(initstate.velocitiesx.begin(), initstate.velocitiesx.end(), velocitiesx.begin());
    std::copy(initstate.velocitiesy.begin(), initstate.velocitiesy.end(), velocitiesy.begin());
    std::copy(initstate.velocitiesz.begin(), initstate.velocitiesz.end(), velocitiesz.begin());
}
