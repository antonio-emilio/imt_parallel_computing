#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>

namespace xs = xsimd;
using b_type = xs::batch<float, xs::avx2>;

Model_CPU_fast
::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
}

void Model_CPU_fast
::step()
{
	std::memset(accelerationsx.data(), 0, n_particles * sizeof(float));
	std::memset(accelerationsy.data(), 0, n_particles * sizeof(float));
	std::memset(accelerationsz.data(), 0, n_particles * sizeof(float));

	#pragma omp parallel for
	for (int i = 0; i < n_particles; i += b_type::size)
	{
		// Load_unaligned - load data from memory into SIMD registers without requiring specific alignment. This is useful when the data may not be aligned to the SIMD register size, but it can be slower than aligned loads if the data is not properly aligned in memory.
		const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
		const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
		const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
		
		b_type raccx_i = b_type(0.0f);
		b_type raccy_i = b_type(0.0f);
		b_type raccz_i = b_type(0.0f);

		for (int j = 0; j < n_particles; j++)
		{
			const b_type rdiffx = b_type(particles.x[j]) - rposx_i;
			const b_type rdiffy = b_type(particles.y[j]) - rposy_i;
			const b_type rdiffz = b_type(particles.z[j]) - rposz_i;

			b_type rdij = rdiffx*rdiffx + rdiffy*rdiffy + rdiffz*rdiffz;
			
			// SIMD version of the if-else statement
			rdij = xs::select(rdij < 1.0f, b_type(10.0f), b_type(10.0f) * xs::sqrt(rdij) / (rdij * rdij));
			
			b_type mass_factor = b_type(initstate.masses[j]);
			raccx_i += rdiffx * rdij * mass_factor;
			raccy_i += rdiffy * rdij * mass_factor;
			raccz_i += rdiffz * rdij * mass_factor;
		}

		raccx_i.store_unaligned(&accelerationsx[i]);
		raccy_i.store_unaligned(&accelerationsy[i]);
		raccz_i.store_unaligned(&accelerationsz[i]);
	}

	#pragma omp parallel for
	for (int i = 0; i < n_particles; i += b_type::size)
	{
		b_type vx = b_type::load_unaligned(&velocitiesx[i]);
		b_type vy = b_type::load_unaligned(&velocitiesy[i]);
		b_type vz = b_type::load_unaligned(&velocitiesz[i]);
		
		b_type ax = b_type::load_unaligned(&accelerationsx[i]);
		b_type ay = b_type::load_unaligned(&accelerationsy[i]);
		b_type az = b_type::load_unaligned(&accelerationsz[i]);
		
		b_type px = b_type::load_unaligned(&particles.x[i]);
		b_type py = b_type::load_unaligned(&particles.y[i]);
		b_type pz = b_type::load_unaligned(&particles.z[i]);
		
		vx += ax * b_type(2.0f);
		vy += ay * b_type(2.0f);
		vz += az * b_type(2.0f);
		
		px += vx * b_type(0.1f);
		py += vy * b_type(0.1f);
		pz += vz * b_type(0.1f);
		
		vx.store_unaligned(&velocitiesx[i]);
		vy.store_unaligned(&velocitiesy[i]);
		vz.store_unaligned(&velocitiesz[i]);
		
		px.store_unaligned(&particles.x[i]);
		py.store_unaligned(&particles.y[i]);
		pz.store_unaligned(&particles.z[i]);
	}

}

#endif // GALAX_MODEL_CPU_FAST
