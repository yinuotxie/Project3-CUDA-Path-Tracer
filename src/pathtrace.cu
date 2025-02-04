#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <OpenImageDenoise/oidn.hpp>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "main.h"


#define ERRORCHECK 1
#define PRINT_DEPTH 0

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

#define CACHE_FIRST_BOUNCE 1
#define SORT_RAY_BY_MATERIAL 1
#define USE_DOF 0

#if CACHE_FIRST_BOUNCE
#define USE_STRATIFIED_SAMPLING 0
#else
#define USE_STRATIFIED_SAMPLING 1
#endif


void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

// Predicate for stream compaction
struct isPathActive
{
	__host__ __device__ bool operator()(const int& stencil)
	{
		return stencil == 1;
	}
};

// sort the material 
struct materialSort
{
	__host__ __device__ bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b)
	{
		return a.materialId < b.materialId;
	}
};


__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;

static ShadeableIntersection* dev_intersections = NULL;
static PathSegment* dev_cache_paths = NULL;
static ShadeableIntersection* dev_first_bounce_intersections = NULL;
static int* dev_stencil = NULL;

// for bvh 
#if USE_BVH
static LinearBVHNode* dev_bvh = NULL;
#endif

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void pathtraceInit(Scene* scene) {
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));

	// cache the path and first bounce intersection
	cudaMalloc(&dev_cache_paths, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_first_bounce_intersections, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_stencil, pixelcount * sizeof(int));

#if USE_BVH
	cudaMalloc(&dev_bvh, scene->bvh.size() * sizeof(LinearBVHNode));
	cudaMemcpy(dev_bvh, scene->bvh.data(), scene->bvh.size() * sizeof(LinearBVHNode), cudaMemcpyHostToDevice);
#endif 
		
	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);

	cudaFree(dev_intersections);
	cudaFree(dev_first_bounce_intersections);
	cudaFree(dev_stencil);

	checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments,
	int sqrtSamples)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		// segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);

#if CACHE_FIRST_BOUNCE
		float jitterX = 0.0f, jitterY = 0.0f;
#else
		// Create stratified samples within the pixel
		// square root of the number of samples
		float invSqrtSamples = 1.0f / sqrtSamples;
		int sampleX = x % sqrtSamples;
		int sampleY = y % sqrtSamples;

		float jitterX = (sampleX + u01(rng)) * invSqrtSamples;
		float jitterY = (sampleY + u01(rng)) * invSqrtSamples;
#endif
		// Calculate point P on the image plane
		glm::vec3 pixelOffset = -cam.right * cam.pixelLength.x *
			((float)x - (float)cam.resolution.x * 0.5f + jitterX) -
			cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + jitterY);
		glm::vec3 P = cam.position + cam.view + pixelOffset;

#if USE_DOF
		// Calculate the focal point on the focal plane
		glm::vec3 toFocalPointDirection = glm::normalize(P - cam.position);
		float t = cam.focalDistance / glm::dot(toFocalPointDirection, cam.view);
		glm::vec3 focalPoint = cam.position + toFocalPointDirection * t;

		// Sample a point within the lens (aperture)
		float angle = u01(rng) * 2.0f * PI;
		float radius = sqrtf(u01(rng)) * cam.aperture * 0.5f;
		glm::vec3 lensSample = cam.position + cam.right * cos(angle) * radius + cam.up * sin(angle) * radius;

		segment.ray.origin = lensSample;
		segment.ray.direction = glm::normalize(focalPoint - lensSample);
#else
		segment.ray.origin = cam.position;
		segment.ray.direction = glm::normalize(P - cam.position);
#endif

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, ShadeableIntersection* intersections
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == TRIANGLE) {
				t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}

__global__ void computeIntersectionsBVH(
	int depth, int num_paths, PathSegment* pathSegments, Geom* geoms, 
	int geoms_size, ShadeableIntersection* intersections, LinearBVHNode* bvh, int bvh_size) {

	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths) {

		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		
		// BVH traversal
		glm::vec3 invDir = 1.0f / pathSegment.ray.direction;
		int dirIsNeg[3] = { invDir.x < 0.f, invDir.y < 0.f, invDir.z < 0.f };
		int currentNodeIdx = 0, toVisitOffset = 0;
		int nodesToVisit[128];

		// bfs
		while (true) {
			if (currentNodeIdx < bvh_size) {
				auto& bvhNode = bvh[currentNodeIdx];

				if (intersectBVHNode(pathSegment.ray, bvhNode, dirIsNeg, invDir)) {
					if (bvhNode.geomCount > 0) {   // hit the leaf node
						int start = bvhNode.geomIndex, end = start + bvhNode.geomCount;

						for (int i = start; i < end; ++i) {
							auto& geom = geoms[i];

							if (geom.type == CUBE)
							{
								t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
							}
							else if (geom.type == SPHERE)
							{
								t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
							}
							else if (geom.type == TRIANGLE) {
								t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
							}

							// Compute the minimum t from the intersection tests to determine what
							// scene geometry object was hit first.
							if (t > 0.0f && t_min > t)
							{
								t_min = t;
								hit_geom_index = i;
								intersect_point = tmp_intersect;
								normal = tmp_normal;
							}
						}	
						if (toVisitOffset == 0) break;
						currentNodeIdx = nodesToVisit[--toVisitOffset];
					}	
					else {
						if (dirIsNeg[bvhNode.axis]) {
							nodesToVisit[toVisitOffset++] = currentNodeIdx + 1;
							currentNodeIdx = bvhNode.rightChildOffset;
						}
						else {
							nodesToVisit[toVisitOffset++] = bvhNode.rightChildOffset;
							currentNodeIdx = currentNodeIdx + 1;
						}
					}
				}
				else {
					if (toVisitOffset == 0) break;
					currentNodeIdx = nodesToVisit[--toVisitOffset];
				}
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}

__global__ void shadeBSDFMaterial(
	int iter, int num_paths, ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments, Material* materials){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
			// Set up the RNG
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.diffuse;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].color *= (materialColor * material.emittance);
				pathSegments[idx].remainingBounces = 0;
			}
			else {
				glm::vec3 intersectPoint = getPointOnRay(pathSegments[idx].ray, intersection.t);
				scatterRay(pathSegments[idx], intersectPoint, intersection.surfaceNormal,
					material, rng);
			}
		} 
		else {
			// If there was no intersection, color the ray black and terminate the ray.
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
		}
	}
}


// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}


__global__ void computeStencil(int num_paths, PathSegment* paths, int* stencil) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx < num_paths) {
		stencil[idx] = (paths[idx].remainingBounces > 0) ? 1 : 0;
	}
}

void denoise(const int& pixelcount, const Camera& cam) {
	int width = cam.resolution.x, height = cam.resolution.y;

	glm::vec3* denoised_image = new glm::vec3[pixelcount];
	cudaMemcpy(denoised_image, dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	oidn::DeviceRef device = oidn::newDevice();
	device.commit();
	
	// create buffer
	oidn::BufferRef colorBuf = device.newBuffer(width * height * sizeof(glm::vec3));
	colorBuf.read(0, pixelcount * sizeof(glm::vec3), denoised_image);

	// create filter
	oidn::FilterRef filter = device.newFilter("RT");
	filter.setImage("color", colorBuf, oidn::Format::Float3, width, height);
	filter.setImage("output", colorBuf, oidn::Format::Float3, width, height);
	filter.set("hdr", true);
	filter.commit();

	colorBuf.write(0, pixelcount * sizeof(glm::vec3), denoised_image);
	cudaMemcpy(dev_image, denoised_image, pixelcount * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	filter.execute();

	
	// Check for any error
	const char* errorMessage;
	if (device.getError(errorMessage) != oidn::Error::None) {
		printf("OIDN Error: %s\n", errorMessage);
	}
}


/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
	// printf("Iter: %d\n", iter);
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;
	const int sqrtSamples = hst_scene->sqrtSamples;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	if (!CACHE_FIRST_BOUNCE || iter == 1) {
		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths,
			sqrtSamples);
		checkCUDAError("generate camera ray");

		// cache the paths
		cudaMemcpy(dev_cache_paths, dev_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
	}
	
	int depth = 0;

	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	bool iterationComplete = false;
	while (!iterationComplete) {
		// clean shading chunks
		// cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));

		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		if (CACHE_FIRST_BOUNCE && depth == 0) {
			if (iter == 1) {
#if USE_BVH
				computeIntersectionsBVH << <numblocksPathSegmentTracing, blockSize1d >> > (
					depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections,
					dev_bvh, hst_scene->bvh.size());
				checkCUDAError("BVH computer intersection failed!");
#else
				computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
					depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections);
				checkCUDAError("computer intersection failed!");

#endif
				// cache the first bounce
				cudaMemcpy(dev_first_bounce_intersections, dev_intersections,
					pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
			else {
				cudaMemcpy(dev_paths, dev_cache_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
				cudaMemcpy(dev_intersections, dev_first_bounce_intersections,
					pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
		}
		else {
#if USE_BVH
			computeIntersectionsBVH << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections,
				dev_bvh, hst_scene->bvh.size());
			checkCUDAError("BVH computer intersection failed!");
#else
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections);
			checkCUDAError("computer intersection failed!");
#endif
		}

		cudaDeviceSynchronize();
		depth++;

#if PRINT_DEPTH
		printf("Depth: %d\n", depth);
		printf("Num of Paths: %d\n", num_paths);
#endif

#if SORT_RAY_BY_MATERIAL
		// sort rays by material type 
		thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, materialSort());
#endif 

		// shade
		shadeBSDFMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter, num_paths, dev_intersections, dev_paths, dev_materials);
		checkCUDAError("shade BDFS material failed!");

		computeStencil << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_paths, dev_stencil);

		// stream compaction
		auto new_end = thrust::stable_partition(thrust::device, dev_paths, dev_paths + num_paths, dev_stencil, isPathActive());
		num_paths = new_end - dev_paths;


		if (num_paths <= 0 || depth >= traceDepth) {
			iterationComplete = true;
		}

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}


