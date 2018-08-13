
/*! \file
*
*	Light Base class. All children must implement Le and Li   
*
*/

#include <GU/GU_Detail.h>
#include <SYS/SYS_Math.h>
#include <GU/GU_RayIntersect.h>

#include "Interaction.h"
#include "Ray.h"
#include "Sampler.h"

#ifndef __Light_h__
#define __Light_h__

namespace VPT {

	class Light {
	public:

		virtual ~Light() {};
		Light(int flags, const GU_Detail *gdp, int nSamples = 1);
		
		virtual UT_Vector3 Sample_Li(const Interaction &ref, const UT_Vector2 &u , UT_Vector3 *wi, float *pdf, VisibilityTester *vis) const = 0;
		virtual UT_Vector3 Power() const = 0; 
		virtual UT_Vector3 Le(const Ray &r ) const = 0; 
		virtual float Pdf_Li(const Interaction &ref, UT_Vector3 *wi) const = 0;
		virtual UT_Vector3 Sample_Le(const UT_Vector2 &u1, const UT_Vector2 &u2, float time, Ray *ray, UT_Vector3 *nLight, float *pdfPos, float *pdfDir) const = 0;
		virtual void Pdf_le(const Ray &ray, const UT_Vector3 &nLight, float *pdfPos, float *pdfDir) const = 0;
		
		//Public Data
		const int flags; 
		const int nSamples; 


	private:
	};


	class VisibilityTester {

	public:
		VisibilityTester() {}
		
		VisibilityTester(const Interaction &p0, const Interaction &p1) : p0(p0), p1(p1) {}
		const Interaction &P0() const { return p0; }
		const Interaction &P1() const { return p1; }
		bool Unoccluded(const GU_RayIntersect *isect) const; 
		UT_Vector3 Tr(const GU_Detail *gdp, Sampler &sampler);
	private:
		Interaction p0, p1; 
	};
	
	//TODO Area Light


}// namespace VPT
#endif // !__Light_h__