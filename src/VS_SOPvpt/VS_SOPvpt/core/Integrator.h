
/*! \file
*
*	Integrator base class. All children must implement their own render() 
*
*/


#include <GU/GU_Detail.h>
#include <SYS/SYS_Math.h>
#include <GU/GU_RayIntersect.h>
#include <UT/UT_Interrupt.h>

//Local
#include "Light.h"


#ifndef __Integrator_h__
#define __Integrator_h__

namespace VPT {

	class Integrator {

	public:

		Integrator() {};

		UT_Vector3F render(const UT_Vector3F orig, const UT_Vector3F P,
			const GU_Detail *prim_gdp,
			GU_RayIntersect *isect,
			UT_Array<Light> *lights, 
			size_t depth=5) {
			
			UT_Vector3F L(0, 0, 0); 

			GU_RayInfo hit_info;
			UT_Vector3F dir = P - orig; 
			dir.normalize();
			hit_info.init(1e2F, 0.001F, GU_FIND_CLOSEST);
			
			int numhit = isect->sendRay(orig, dir, hit_info);
			

			UT_Vector3F n = hit_info.myNml;
			UT_Vector3F p_isect = UT_Vector3(hit_info.myU, hit_info.myV, hit_info.myW); 
			UT_Vector3F wo = p_isect - P;
			wo.normalize();

			hit_info.reset();
			
			
			for (const auto &light : lights) {



			}

			if (numhit) {
				
				return n;

			}else{
				
				L.assign(0.0F, 0.573F, 0.9F);
				return L;
			}
		};
	
	protected:
		
		virtual  ~Integrator() {};

	};

}

#endif