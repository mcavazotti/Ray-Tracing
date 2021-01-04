#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func,
                const char *const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result)
              << "(" << cudaGetErrorName(result) << ") at " << file << ":" << line << " '" << func
              << "' \n";
    cudaDeviceReset();
    exit(99);
  }
}

#ifndef BLOCK_X
    #define BLOCK_X 8
#endif

#ifndef BLOCK_Y
    #define BLOCK_Y 8
#endif

#endif