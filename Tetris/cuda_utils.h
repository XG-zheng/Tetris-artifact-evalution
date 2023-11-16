#ifndef CUDA_UTILS_H_
#define CUDA_UTILS_H_

#include <cuda_runtime.h>
#include <cudnn.h>
#include <string>
#include <cstdio>

#define checkCudaErrors(call) do {                                \
    cudaError_t err = call;                                       \
    if (err != cudaSuccess) {                                     \
        printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__,   \
                cudaGetErrorString(err));                         \
        exit(EXIT_FAILURE);                                       \
    }                                                             \
} while (0)                                                       \


#define checkCUDNN(expression)                                    \
  {                                                               \
    cudnnStatus_t status = (expression);                          \
    if (status != CUDNN_STATUS_SUCCESS) {                         \
        printf("cuDNN error at %s %d: %s\n", __FILE__, __LINE__,  \
                cudnnGetErrorString(status));                     \
        exit(EXIT_FAILURE);                                       \
    }                                                             \
  }                                                               \

#define checkCUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

void CuDNNConv2d(float *h_input, float *h_filter, float *h_output,
                int batch_size, int img_h, int img_w,
                int in_channel, int out_channel, int kernel_size,
                int stride, int padding, cudaStream_t cuda_stream,
                std::string layout);

#endif  // CUDA_UTILS_H_
