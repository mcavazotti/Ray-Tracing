
#include <iostream>
#include <curand_kernel.h>
#include <float.h>
#include <time.h>

#include "cuda_utils.h"
#include "camera.h"
//#include "color.h"
#include "hittable_list.h"
#include "material.h"
#include "rtweekend.h"
#include "sphere.h"

__global__ void rand_init(curandState *randState) {
  if(threadIdx.x == 0 && blockIdx.x == 0)
    curand_init(1998, 0 ,0, randState);
}

__global__ void renderInit(int maxX, int maxY, curandState *randState) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;

  if((i >= maxX) || (j >= maxY)) return;

  int pixelIdx = j*maxX + i;

  curand_init(1998 + pixelIdx,0,0, &randState[pixelIdx]);
}

__device__ color get_color(const ray &r, hittable **world, int maxRecursionDepth, curandState *localRandState){
  ray currentRay = r;
  color currentAttenuation = color(1,1,1);

  for(int i = 0; i < maxRecursionDepth; i++){
    hit_record rec;
    if((*world)->hit(currentRay, 0.001f, FLT_MAX, rec)){
      ray scattered;
      color attenuation;
      if(rec.mat_ptr->scatter(currentRay,rec,attenuation, scattered, localRandState)) {
        currentAttenuation = currentAttenuation * attenuation;
        currentRay = scattered;
      }
      else return color(0,0,0);
    }
    else {
      vec3 unitDirection = unit_vector(currentRay.direction());
      float t = 0.5f*(unitDirection.y() +1.0f);
      color c = (1.0f-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);

      return currentAttenuation * c;
    }
  }

  return color(0,0,0);
}

__global__ void render(color *fb, int maxX, int maxY, int samples,int recursionDepth, camera **cam, hittable **world, curandState *randState) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if((i >= maxX) || (j >= maxY)) return;

  int pixelIdx = j*maxX + i;

  curandState localRandState = randState[pixelIdx];
  color col(0,0,0);

  for(int s = 0; s < samples; s++){
    float u = float(i + curand_uniform(&localRandState)) / float(maxX);
    float v = float(j + curand_uniform(&localRandState)) / float(maxY);
    ray r = (*cam)->get_ray(u,v, &localRandState);
    col += get_color(r,world,recursionDepth,&localRandState);
  }

  randState[pixelIdx] = localRandState;

  col /= float(samples);
  col[0] = sqrtf(col[0]);
  col[1] = sqrtf(col[1]);
  col[2] = sqrtf(col[2]);

  fb[pixelIdx] = col;
}

__global__ void createWorld(hittable **d_list, hittable **d_world, camera **d_camera, int imgX, int imgY, curandState *currentState) {
  if(threadIdx.x == 0 && blockIdx.x == 0){
    curandState localRandState = *currentState;

    d_list[0] = new sphere(point3(0, -1000,-1), 1000, new lambertian(color(0.5,0.5,0.5)));
    int i = 1;
    
    for(int a = -11; a < 11; a++){
      for(int b = -11; b < 11; b++){
        float chooseMat = random_float(&localRandState);
        point3 center(a+random_float(&localRandState), 0.2, b+random_float(&localRandState));

        if(chooseMat < 0.8f)
          d_list[i++] = new sphere(center, 0.2, new lambertian(color(random_float(&localRandState)*random_float(&localRandState),random_float(&localRandState)*random_float(&localRandState),random_float(&localRandState)*random_float(&localRandState))));
        else if (chooseMat <0.95f)
          d_list[i++] = new sphere(center, 0.2, new metal(color(0.5f*(1.0f*random_float(&localRandState)),0.5f*(1.0f*random_float(&localRandState)),0.5f*(1.0f*random_float(&localRandState))), 0.5f*random_float(&localRandState)));
        else
          d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
      }
    }

    d_list[i++] = new sphere(point3(0,1,0), 1, new dielectric(1.5));
    d_list[i++] = new sphere(point3(-4,1,0), 1, new lambertian(color(0.4,0.2,0.1)));
    d_list[i++] = new sphere(point3(4,1,0), 1, new metal(color(0.7,0.6,0.5),0));

    *currentState = localRandState;

    *d_world = new hittable_list(d_list, 22*22+1+3);
    //*d_world = new hittable_list(d_list, 1+3);

    point3 lookFrom(13,2,3);
    point3 lookAt(0,0,0);

    float distToFocus = (lookFrom - lookAt).length();
    float aperture = 0.1;
    *d_camera = new camera(lookFrom, lookAt, vec3(0,1,0), 30, float(imgX)/float(imgY), aperture, distToFocus);
  }
}

__global__ void freeWorld(hittable **d_list, hittable **d_world, camera **d_camera) {
    for(int i=0; i < 22*22+1+3; i++) {
    //for(int i=0; i < 1+3; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

int main(int argc, char *argv[]) {
  if( argc < 4){
        std::cerr << "Missing arguments." << std::endl
                << "Usage: ./rayTracer <w> <h> <samples>" << std::endl;
        exit(-1);
    }
    
    int imageWidth = atoi(argv[1]);
    int imageHeight = atoi(argv[2]);
    int samplesPerPixel = atoi(argv[3]);

    float aspectRatio = imageWidth / imageHeight;
    int maxDepth = 50;
    int threadX = BLOCK_X;
    int threadY = BLOCK_Y;

    std:: cerr << "Rendering a " << imageWidth << "x" << imageHeight << " image in "
                << threadX << "x" << threadY << " blocks.\n";

    int numPixels = imageHeight * imageWidth;
    size_t frameBufferSize = numPixels * sizeof(color);

    color *frameBuffer;
    checkCudaErrors(cudaMallocManaged((void **)&frameBuffer, frameBufferSize));

    curandState *d_randState;
    curandState *d_randState2;
    checkCudaErrors(cudaMalloc((void **)&d_randState, numPixels*sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void **)&d_randState2, sizeof(curandState)));

    rand_init<<<1,1>>>(d_randState2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    hittable **d_hittableList;
    int numHittables = 22*22+1+3;
    //int numHittables = 1+3;
    checkCudaErrors(cudaMalloc((void **)&d_hittableList, numHittables * sizeof(hittable *)));
    
    hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
    
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

    createWorld<<<1,1>>>(d_hittableList, d_world, d_camera, imageWidth, imageHeight, d_randState2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
  
    dim3 blocks(imageWidth/threadX +1, imageHeight/threadY + 1);
    dim3 threads(threadX,threadY);

    renderInit<<<blocks, threads>>>(imageWidth, imageHeight, d_randState);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    struct timespec start, stop;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    render<<<blocks, threads>>>(frameBuffer, imageWidth, imageHeight, samplesPerPixel, maxDepth, d_camera, d_world, d_randState);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC_RAW, &stop);
    double timer_milisecs =
      ((stop.tv_sec * 1000 * 1000 * 1000 + stop.tv_nsec) -
       (start.tv_sec * 1000 * 1000 * 1000 + start.tv_nsec))/(1000*1000) ;
    std::cerr << "Elapsed time " << timer_milisecs << "ms.\n";
    
    
    std::cout << "P3\n" << imageWidth << " " << imageHeight << "\n255\n";
    for (int j = imageHeight-1; j >= 0; j--) {
        for (int i = 0; i < imageWidth; i++) {
            size_t pixelIdx = j*imageWidth + i;
            int ir = int(255.99*frameBuffer[pixelIdx].x());
            int ig = int(255.99*frameBuffer[pixelIdx].y());
            int ib = int(255.99*frameBuffer[pixelIdx].z());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    freeWorld<<<1,1>>>(d_hittableList,d_world,d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_hittableList));
    checkCudaErrors(cudaFree(d_randState));
    checkCudaErrors(cudaFree(d_randState2));
    checkCudaErrors(cudaFree(frameBuffer));

    cudaDeviceReset();

    std::cerr << "\nDone.\n";
    return 0;
  }
  