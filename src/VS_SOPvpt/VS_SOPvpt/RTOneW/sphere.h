#pragma once

#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"



class sphere :
	public hitable
{
public:
	__device__ sphere() {};
	

	__device__ sphere(vec3 cen, float r, material *m) : center(cen), radius(r), mat_ptr(m) {};

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
	__device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;

	vec3 center; 
	float radius; 
	material *mat_ptr; 
};


__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {


	vec3 oc = r.origin() - center;

	float a = dot(r.direction(), r.direction());
	float b = dot(oc, r.direction());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b -  a*c;
	if (discriminant > 0) {
		float temp = (-b - sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			rec.mat_ptr = mat_ptr;
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			rec.mat_ptr = mat_ptr;
			return true;
		}
	}
	return false;
}

__device__ bool sphere::bounding_box(float t0, float t1, aabb& box) const {

	box = aabb(center - vec3(radius, radius, radius), center + vec3(radius, radius, radius));

	return true;
}









class moving_sphere : public hitable {


	__device__ moving_sphere() {};

	__device__ moving_sphere(vec3 cen0, vec3 cen1, float t0, float t1, float r, material *m) : center0(cen0), center1(cen1),
																					time0(t0), time1(t1),radius(r),mat_ptr(m) {};
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
	__device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;
	vec3 center0, center1;
	__device__ vec3 center(float time) const;
	float time0, time1, radius;
	material *mat_ptr;
	
};

__device__ vec3 moving_sphere::center(float time)const {

	return center0 + ((time - time0) / (time1 - time0))*(center1 - center0);
}


__device__ bool moving_sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {

	vec3 oc = r.origin() - center(r.time());
	float a = dot(r.direction(), r.direction());
	float b = dot(oc, r.direction());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - a * c;
	if (discriminant > 0) {
		float temp = (-b - sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center(r.time())) / radius;
			rec.mat_ptr = mat_ptr;
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center(r.time())) / radius;
			rec.mat_ptr = mat_ptr;
			return true;
		}
	}
	return false;

}




__device__ bool moving_sphere::bounding_box(float t0, float t1, aabb& box) const {

	aabb box0 = aabb(center0 - vec3(radius, radius, radius), center0 + vec3(radius, radius, radius));
	aabb box1 = aabb(center1 - vec3(radius, radius, radius), center1 + vec3(radius, radius, radius));

	box = surrounding_box(box0, box1);

	return true;
}




#endif // !SPHEREH