
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
			GU_RayInfo hit_info,
			UT_Array<Light> &lights, 
			size_t depth=5) {
			
			UT_Vector3F L(0, 0, 0); 

			UT_Vector3F dir = P - orig; 
			dir.normalize();			
			int numhit = isect->sendRay(orig, dir, hit_info);

			UT_Vector3F n = hit_info.myNml;
			UT_Vector3F tmp = orig + dir * hit_info.myT;
			UT_Vector3F wo = tmp - P;
			wo.normalize();

			hit_info.reset();
			
			if (numhit) {

				UT_Vector3F light_amt(0,0,0), specular_color(0,0,0); 
				
				for (int i = 0; i < lights.size(); i++) {
					
					UT_Vector3F light_pos = lights[i].pos;
					UT_Vector3F light_dir = light_pos - P;
					light_dir.normalize();
					float light_dist = light_dir.dot(light_dir); // pow^2
					
					
					VisibilityTester vis(tmp, light_pos);
					
					GU_RayInfo shadow_info;
					shadow_info.init(1e2F, 0.001F, GU_FIND_CLOSEST);

					
					int occ = isect->sendRay(tmp, light_pos, shadow_info) > 0 ;
					
					float LdotN = SYSmax(0.f, n.dot(-light_dir)); 
					light_amt += occ * LdotN;
					UT_Vector3F ref_dir = reflect(-light_dir, n); 
					
					specular_color += -ref_dir.dot(dir); 
					//return UT_Vector3F(float(occ));
					return UT_Vector3F(float(occ));
				}
				
				

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