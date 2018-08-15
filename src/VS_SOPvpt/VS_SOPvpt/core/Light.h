
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
#include "Constants.h"

#ifndef __Light_h__
#define __Light_h__

namespace VPT {

	class Light {
	public:

		virtual ~Light() {};
		Light() {};
		
		/*
		virtual Spectrum Sample_Li(const Interaction &ref, const UT_Vector2 &u , UT_Vector3 *wi, float *pdf, VisibilityTester *vis) const = 0;
		virtual Spectrum Power() const = 0;
		virtual Spectrum Le(const Ray &r ) const = 0;
		virtual float Pdf_Li(const Interaction &ref, UT_Vector3 *wi) const = 0;
		virtual Spectrum Sample_Le(const UT_Vector2 &u1, const UT_Vector2 &u2, float time, Ray *ray, UT_Vector3 *nLight, float *pdfPos, float *pdfDir) const = 0;
		virtual void Pdf_le(const Ray &ray, const UT_Vector3 &nLight, float *pdfPos, float *pdfDir) const = 0;
		*/

		UT_Vector3F pos;
		UT_Vector3F color; 

		


	private:
	};


	Light createLight(UT_Vector3 pos, UT_Vector3 color) {

		return Light();

	}



	class VisibilityTester {

	public:
		VisibilityTester() {}
		
		VisibilityTester(const UT_Vector3 &p0, const UT_Vector3 &p1) : p0(p0), p1(p1) {}
		const UT_Vector3 &P0() const { return p0; }
		const UT_Vector3 &P1() const { return p1; }
		
		bool Unoccluded(const GU_RayIntersect *isect) const {
			
			GU_RayInfo hitinfo; 
			hitinfo.init(1e2F, 0.001F, GU_FIND_CLOSEST);
			isect->sendRay(p0, p1, hitinfo);
			return !hitinfo.myHitList->isEmpty();
		
		}
		
		//UT_Vector3 Tr(const GU_Detail *gdp, Sampler &sampler);
	private:
		UT_Vector3 p0, p1;
	};
	
	//TODO Area Light


}// namespace VPT
#endif // !__Light_h__