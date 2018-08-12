
#include <GU/GU_Detail.h>
#include <SYS/SYS_Math.h>

#ifndef __Light_h__
#define __Light_h__

namespace VPT {

	class Light {

	public:

		enum LIGHT_TYPE
		{
			point,
			directional,
			spot,
			area,
			HDR,
			mesh
		};

		UT_Vector3F pos;
		UT_Vector3F color;
		float intensity;
		size_t type = 0;

	};

}
#endif // !__Light_h__