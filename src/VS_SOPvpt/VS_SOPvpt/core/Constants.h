#pragma once
namespace VPT {


	class Camera;
	class CameraSample;
	class Integrator;
	class Ray; 
	class RayDifferential; 
	class Interaciton;
	class SampledSpectrum;
	class RGBSpectrum; 

	typedef RGBSpectrum Spectrum; 

	class SurfaceInteraction; 
	class MediumInteraction;
	class MediumInterface; 
	class Medium; 
	class BxDF; 
	class BSDF;
	class BRDF;
	class BTDF;
	class BSSRDF;
	class SeparableBSSRDF;
	class TabulatedBSSRDF;
	struct BSSRDFTable;
	class Light;
	class VisibilityTester;
	class AreaLight;
	struct Distribution1D;
	class Distribution2D;


#ifdef _MSC_VER
#define Maxfloat std::numeric_limits<float>::max()
#define Infinity std::numeric_limits<float>::infinity()
#else
	static constexpr float Maxfloat = std::numeric_limits<float>::max();
	static constexpr float Infinity = std::numeric_limits<float>::infinity();
#endif
#ifdef _MSC_VER
#define MachineEpsilon (std::numeric_limits<float>::epsilon() * 0.5)
#else
	static constexpr float MachineEpsilon =
		std::numeric_limits<float>::epsilon() * 0.5;
#endif

	static constexpr float ShadowEpsilon = 0.0001f;
	static constexpr float Pi = 3.14159265358979323846;
	static constexpr float InvPi = 0.31830988618379067154;
	static constexpr float Inv2Pi = 0.15915494309189533577;
	static constexpr float Inv4Pi = 0.07957747154594766788;
	static constexpr float PiOver2 = 1.57079632679489661923;
	static constexpr float PiOver4 = 0.78539816339744830961;
	static constexpr float Sqrt2 = 1.41421356237309504880;



	UT_Vector3F reflect(UT_Vector3F I, UT_Vector3F N) {

		return I - 2 * N.dot(I) * N;
	}
	

	UT_Vector3F refract(const UT_Vector3F &I, const UT_Vector3F &N, const float &ior)
	{
		
		float cosi = SYSclamp(N.dot(I), -1 , 1 ,0);
		float etai = 1, etat = ior;
		UT_Vector3F n = N;
		if (cosi < 0) { cosi = -cosi; }
		else { std::swap(etai, etat); n = -N; }
		float eta = etai / etat;
		float k = 1 - eta * eta * (1 - cosi * cosi);
		
		UT_Vector3F ret = eta * I + (eta * cosi - sqrtf(k)) * n;
		if (k < 0) return ret; 
		else return UT_Vector3F(0,0,0); 
	}




}