#include "renderHub.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "RTOneW/vec3.h"


class hitable;
class camera;


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
extern "C" void init_fb(vec3* fb, int nx, int ny);
extern "C" void rand_init(curandState *rand_state);
extern "C" void render_init(int nx, int ny, int tx, int ty, curandState *rand_state);
extern "C" void render_image(vec3 *fb, int nx, int ny, int tx, int ty, int ns, camera **cam, hitable **world, curandState *rand_state);
extern "C" void create_world(hitable **d_list, hitable **d_world, camera **d_camera,
	vec3 cam_pos, int sphere_num, vec3 *sphere_pos, vec3 *sphere_color,
	int nx, int ny, float fov, float aperture, curandState *rand_state);
extern "C" void free_world(hitable **d_list, hitable **d_world, camera **d_camera, int n);


bool rendering = true;
int spp;

#define convertUTF(val) UTF_to_vec3(val)
vec3 UTF_to_vec3(UT_Vector3F utf) {

	return vec3(utf.x(), utf.y(), utf.z());

}


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

void imageOutput(UT_Vector3F *pix, vec3 *fb, int s, int width, int height) {

	float s_inv = 1 / float(s);
	for (int row = height - 1; row >= 0; row--) {
		for (int col = 0; col < width; col++) {

			size_t pixel_index = row * width + col;
			pix[pixel_index] = UT_Vector3F(fb[pixel_index][0] * s_inv
				, fb[pixel_index][1] * s_inv
				, fb[pixel_index][2] * s_inv);
		}
	}

}


void render(UT_Vector3F *pix, UT_Vector3F cam_pos, int sphere_num, UT_Vector3F *sphere_pos, UT_Vector3F *sphere_color, int width, int height, int spp, float fov, float aperture, int b_size, int t_size) {

	int nx = width;
	int ny = height;
	int ns = spp;
	int tx = b_size;
	int ty = t_size;


	int num_pixels = nx * ny;
	size_t fb_size = num_pixels * sizeof(vec3);

	// allocate FB
	vec3 *fb;

	checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));
	init_fb(fb, nx, ny);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// allocate random state
	curandState *d_rand_state;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
	curandState *d_rand_state2;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

	// we need that 2nd random state to be initialized for the world creation
	rand_init(d_rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// make our world of hitables & the camera
	hitable **d_list;
	int num_hitables = sphere_num;
	checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables * sizeof(hitable *)));
	hitable **d_world;
	checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
	camera **d_camera;
	checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

	vec3 d_cam_pos = convertUTF(cam_pos);

	vec3 *h_sphere_pos = new vec3[num_hitables];
	vec3 *h_sphere_color = new vec3[num_hitables];

	for (size_t i = 0; i < num_hitables; i++)
	{
		h_sphere_pos[i] = convertUTF(sphere_pos[i]);
		h_sphere_color[i] = convertUTF(sphere_color[i]);
	}


	vec3 *d_sphere_pos;
	checkCudaErrors(cudaMalloc((void **)&d_sphere_pos, num_hitables * sizeof(vec3)));
	vec3 *d_sphere_color;
	checkCudaErrors(cudaMalloc((void **)&d_sphere_color, num_hitables * sizeof(vec3)));

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(d_sphere_pos, h_sphere_pos, num_hitables * sizeof(vec3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_sphere_color, h_sphere_color, num_hitables * sizeof(vec3), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	create_world(d_list, d_world, d_camera, d_cam_pos, num_hitables, d_sphere_pos, d_sphere_color, nx, ny, fov, aperture, d_rand_state2);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Render our buffer

	render_init(nx, ny, tx, ty, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	for (int s = 1; s <= spp; s++) {


		if (!rendering) break;

		render_image(fb, nx, ny, tx, ty, ns, d_camera, d_world, d_rand_state);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}


	imageOutput(pix, fb, spp, width, height);


	// clean up
	checkCudaErrors(cudaDeviceSynchronize());

	free_world(d_list, d_world, d_camera, num_hitables);
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_world));

	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(fb));
	cudaDeviceReset();


	rendering = true;

}

