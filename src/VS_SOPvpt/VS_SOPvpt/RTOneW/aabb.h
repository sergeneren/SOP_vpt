#pragma once

#ifndef AABBH
#define AABBH

#include "vec3.h"
#include "ray.h"

__host__ __device__ inline float ffmin(float a, float b) { return a < b ? a : b; }
__host__ __device__ inline float ffmax(float a, float b) { return a > b ? a : b; }



class aabb {

public:

	__device__ aabb() {};

	__device__ aabb(const vec3& a, const vec3&b) { _min = a; _max = b; }

	__device__ vec3 min() const { return _min; }
	__device__ vec3 max() const { return _max; }


	__device__ bool hit(const ray& r, float tmin, float tmax) const {

		for (int a = 0; a < 3; a++) {

			float invD = 1.0f / r.direction()[a];

			float t0 = (min()[a] - r.origin()[a]) * invD;
			float t1 = (max()[a] - r.origin()[a]) * invD;
			if (invD < 0.0f) std::swap(t0, t1);

			tmin = t0>tmin ? t0: tmin;
			tmax = t1<tmax ? t1: tmax;

			if (tmax < tmin) return false;
		}

		return true;
	}


	vec3 _min;
	vec3 _max;


};


__device__ aabb surrounding_box(aabb box0, aabb box1) {


	vec3 small(fmin(box0.min().x(), box1.min().x()),
		fmin(box0.min().y(), box1.min().y()),
		fmin(box0.min().y(), box1.min().y()));

	vec3 big(fmax(box0.max().x(), box1.max().x()),
		fmax(box0.max().y(), box1.max().y()),
		fmax(box0.max().y(), box1.max().y()));

	return aabb(small, big);
}



#endif // !AABBH
