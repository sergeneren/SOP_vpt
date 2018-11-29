#pragma once
#ifndef RAYH
#define RAYH


#include "vec3.h"

__device__ float vanDerCorput(curandState *local_rand_state, int base = 2) {

	int n = int(curand_uniform(local_rand_state) * 100);
	float rand = 0, denom = 1, invBase = 1.f / base;

	while (n) {

		denom *= base;
		rand += (n%base) / denom;
		n *= invBase;

	}
	return rand;
}

class ray
{
public:
	__device__ ray() {};
	__device__ ray(const vec3& a, const vec3& b, float ti = 0.0) { A = a; B = b; _time = ti; };
	__device__ vec3 origin() const	{ return A; }
	__device__ vec3 direction() const { return B;  }
	__device__ float time() const { return _time; }
	__device__ vec3 point_at_parameter(float t) const { return A + t * B; }
	
	vec3 A; 
	vec3 B; 
	float _time;
};
#endif // !RAYH

