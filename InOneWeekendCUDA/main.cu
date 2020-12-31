#include <iostream>
#include <stdlib.h>
#include "cuda_utils.h"
#include "vec3.h"

__global__ void render(vec3 *fb, int maxX, int maxY) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= maxX) || (j >= maxY)) return;
    
    int pixelIdx = j*maxX + i;
    fb[pixelIdx][0] = float(i) / maxX;
    fb[pixelIdx][1] = float(j) / maxY;
    fb[pixelIdx][2] = 0.2;
    
     
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

    render<<<blocks,threads>>>(frameBuffer, imageWidth, imageHeight);
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