#include <iostream>
#include <stdlib.h>
#include "cuda_utils.h"
#include "vec3.h"
#include "ray.h"

__device__ vec3 getColor(const ray& r){
    vec3 unitDirection = unit_vector(r.direction());
    float t = 0.5f*unitDirection.y() +1.0f;
    return (1.0f-t)*vec3(1,1,1) + t*vec3(0.5,0.7,1);
}

__global__ void render(vec3 *fb, int maxX, int maxY, vec3 lowerLeft, vec3 horizontal, vec3 vertical, vec3 origin) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= maxX) || (j >= maxY)) return;
    
    int pixelIdx = j*maxX + i;
    float u = float(i) / maxX;
    float v = float(j) / maxY;

    ray r(origin,lowerLeft+u*horizontal+v*vertical);
    fb[pixelIdx] = getColor(r);
    
    
     
}


int main(int argc, char const *argv[]){
    if( argc < 3){
        std::cerr << "Missing arguments." << std::endl
                << "Usage: ./rayTracer <w> <h>" << std::endl;
        exit(-1);
    }
    
    int imageWidth = atoi(argv[1]);
    int imageHeight = atoi(argv[2]);
    int threadX = BLOCK_X;
    int threadY = BLOCK_Y;

    std:: cerr << "Rendering a " << imageWidth << "x" << imageHeight << " image in "
                << threadX << "x" << threadY << " blocks.\n";

    // Allocate frame buffer
    int numPixels = imageHeight * imageWidth;
    size_t frameBufferSize = numPixels * sizeof(vec3);

    vec3 *frameBuffer;

    checkCudaErrors(cudaMallocManaged((void **)&frameBuffer, frameBufferSize));

    // Render
    dim3 blocks(imageWidth/threadX+1,imageHeight/threadY+1);
    dim3 threads(threadX,threadY);

    render<<<blocks,threads>>>(frameBuffer, imageWidth, imageHeight,
                                vec3(-2.0, -1.0, -1.0),
                                vec3(4.0, 0.0, 0.0),
                                vec3(0.0, 2.0, 0.0),
                                vec3(0.0, 0.0, 0.0));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "P3\n" << imageWidth << " " << imageHeight <<"\n255\n";
    for(int j = imageHeight-1; j >= 0; j--){
        for(int i = 0; i < imageWidth; i++){
            size_t pixelIdx = j*imageWidth + i;
            float r = frameBuffer[pixelIdx][0];
            float g = frameBuffer[pixelIdx][1];
            float b = frameBuffer[pixelIdx][2];

            int ir = int(255.99*r);
            int ig = int(255.99*g);
            int ib = int(255.99*b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    checkCudaErrors(cudaFree(frameBuffer));

    return 0;
}