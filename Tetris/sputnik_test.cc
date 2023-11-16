#include <cuda_runtime.h>
#include <cudnn.h>  
#include <random>
#include <iostream>
#include <string>
#include <fstream>
#include <assert.h>
#include <memory>
#include <cstring>

#include "spconv2d_kernel.cuh"
#include "tensor_utils.h"
#include "cuda_utils.h"

namespace sputnik {
    cudaError_t CudaSpmm(int m, int k, int n, int nonzeros,
        const int* __restrict__ row_indices,
        const float* __restrict__ values,
        const int* __restrict__ row_offsets,
        const int* __restrict__ column_indices,
        const float* __restrict__ dense_matrix,
        float* __restrict__ output_matrix, cudaStream_t stream);
}

void SortedRowSwizzle(int rows, const int *row_offsets, int *row_indices) {
  // Create our unsorted row indices.
  std::vector<int> swizzle_staging(rows);
  std::iota(swizzle_staging.begin(), swizzle_staging.end(), 0);

  // Argsort the row indices based on their length.
  std::sort(swizzle_staging.begin(), swizzle_staging.end(),
            [&row_offsets](int idx_a, int idx_b) {
              int length_a = row_offsets[idx_a + 1] - row_offsets[idx_a];
              int length_b = row_offsets[idx_b + 1] - row_offsets[idx_b];
              return length_a > length_b;
            });

  // Copy the ordered row indices to the output.
  std::memcpy(row_indices, swizzle_staging.data(), sizeof(int) * rows);
}

void ConvertFilterToCSR(int M, int K, float *filter,
                        int *row_indices, int *row_offsets,
                        int *column_indices, float *values ) {
  // row_indices, row_offsets, column_indices, values have already alloc mem in external.
  row_offsets[0] = 0;
  int nnz = 0;
  for (int i = 0; i < M; ++i) {
    row_offsets[i + 1] = row_offsets[i];
    for (int j = 0; j < K; ++j) {
      if (filter[i * K + j] != 0) {
        nnz += 1;
        int offset = row_offsets[i + 1];
        column_indices[offset] = j;
        values[offset] = filter[i * K + j];
        row_offsets[i + 1] += 1;
      }
    }
  }
  SortedRowSwizzle(M, row_offsets, row_indices);
  
  std::cout << "nnz = " << nnz << ", sparsity = " << (1.0 - nnz * 1.0 / (M * K * 1.0)) << "\n";
}

int GetMatrixNNZ(int M, int K, float *filter) {
  int nnz = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      if (filter[i * K + j] != 0) {
        nnz += 1;
      }
    }
  }
  return nnz;
}

float SputnikSpconv2d(float *h_input, float *h_filter, float *h_output,
                int batch_size, int img_h, int img_w,
                int in_channel, int out_channel, int kernel_size,
                int stride, int padding, cudaStream_t cuda_stream,
                std::string layout, bool timing_im2col) {
  int out_h = (img_h + padding * 2 - kernel_size) / stride + 1;
  int out_w = (img_w + padding * 2 - kernel_size) / stride + 1;
  int M = out_channel;
  int N = batch_size * out_h * out_w;
  int K = in_channel * kernel_size * kernel_size;
  int nnz = GetMatrixNNZ(M, K, h_filter);

  size_t input_byte_size = batch_size * in_channel * img_h * img_w * sizeof(float);
  size_t filter_byte_size = kernel_size * kernel_size * in_channel * out_channel * sizeof(float);
  size_t output_byte_size = batch_size * out_h * out_w * out_channel * sizeof(float);
  size_t im2col_byte_size = batch_size * out_h * out_w * in_channel * kernel_size * kernel_size * sizeof(float);
  size_t row_indices_byte_size = M * sizeof(int);
  size_t row_offsets_byte_size = (M + 1) * sizeof(int);
  size_t column_indices_byte_size = nnz * sizeof(int);
  size_t values_byte_size = nnz * sizeof(float);

  int *h_row_indices, *h_column_indices, *h_row_offsets;
  float *h_im2col, *h_values;

  h_im2col = (float *)malloc(im2col_byte_size);
  h_row_indices = (int *)malloc(row_indices_byte_size);
  h_row_offsets = (int *)malloc(row_offsets_byte_size);
  h_column_indices = (int *)malloc(column_indices_byte_size);
  h_values = (float *)malloc(values_byte_size);

  ConvertFilterToCSR(M, K, h_filter, h_row_indices,
                    h_row_offsets, h_column_indices, h_values);
  // for (int i = 0; i < M; ++i)
  //   std::cout << h_row_indices[i] << "\n";
  int *d_row_indices, *d_column_indices, *d_row_offsets;
  float *d_im2col, *d_values;
  float *d_input, *d_output;
  checkCudaErrors(cudaMalloc(&d_input, input_byte_size));
  checkCudaErrors(cudaMalloc(&d_output, output_byte_size));
  checkCudaErrors(cudaMalloc(&d_im2col, im2col_byte_size));

  // alloc device memory for csr
  checkCudaErrors(cudaMalloc(&d_row_indices, row_indices_byte_size));
  checkCudaErrors(cudaMalloc(&d_row_offsets, row_offsets_byte_size));
  checkCudaErrors(cudaMalloc(&d_column_indices, column_indices_byte_size));
  checkCudaErrors(cudaMalloc(&d_values, values_byte_size));

  // copy input to device
  checkCudaErrors(cudaMemcpy(d_input, h_input, input_byte_size, cudaMemcpyHostToDevice));

  // copy csr to device
  checkCudaErrors(cudaMemcpy(d_row_indices, h_row_indices, row_indices_byte_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_row_offsets, h_row_offsets, row_offsets_byte_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_column_indices, h_column_indices, column_indices_byte_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_values, h_values, values_byte_size, cudaMemcpyHostToDevice));

  cudnnHandle_t handle;
  checkCUDNN(cudnnCreate(&handle));
  checkCUDNN(cudnnSetStream(handle, cuda_stream));
 
  cudnnTensorDescriptor_t inputDesc, outputDesc;
  checkCUDNN(cudnnCreateTensorDescriptor(&inputDesc));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputDesc));
  cudnnFilterDescriptor_t filterDesc;
  checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
  cudnnConvolutionDescriptor_t convDesc;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
  if (layout == "NCHW") {
    checkCUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,  
                                          batch_size, in_channel, img_h, img_w));
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,  
                                          out_channel, in_channel, kernel_size, kernel_size));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding, padding, stride, stride, 1, 1,  
                                              CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    checkCUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
                                          batch_size, out_channel, out_h, out_w)); 
  } else if (layout == "NHWC") {
    checkCUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,  
                                          batch_size, in_channel, img_h, img_w));
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC,  
                                          out_channel, in_channel, kernel_size, kernel_size));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding, padding, stride, stride, 1, 1,  
                                              CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    checkCUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 
                                          batch_size, out_channel, out_h, out_w)); 
  }

  cudaDeviceSynchronize();

  // warm up
  for (int i = 0; i < WARM; ++i) {
    // the output of im2col : (in_channel * kernel_size * kernel_size, batch_size * OH * OW)
    cudnnIm2Col(handle, inputDesc, d_input, filterDesc, convDesc, d_im2col);
    // checkCudaErrors(cudaMemcpy(h_im2col, d_im2col, im2col_byte_size, cudaMemcpyDeviceToHost));
    sputnik::CudaSpmm(M, K, N, nnz, d_row_indices, d_values, d_row_offsets, d_column_indices,
              d_im2col, d_output, cuda_stream);
  }

  cudaDeviceSynchronize();

  float elapsedTime;
  cudaEvent_t start , end;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&end));


  checkCudaErrors(cudaEventRecord(start, cuda_stream));
  if (timing_im2col) {
    for (int i = 0; i < REPEAT; ++i) {
    // im2col
    cudnnIm2Col(handle, inputDesc, d_input, filterDesc, convDesc, d_im2col);
    // spmm
    sputnik::CudaSpmm(M, K, N, nnz, d_row_indices, d_values, d_row_offsets, d_column_indices,
              d_im2col, d_output, cuda_stream);
    }
  } else {

    for (int i = 0; i < REPEAT; ++i) {
    // spmm
    sputnik::CudaSpmm(M, K, N, nnz, d_row_indices, d_values, d_row_offsets, d_column_indices,
              d_im2col, d_output, cuda_stream);
    }
  
  }
  checkCudaErrors(cudaEventRecord(end , cuda_stream));
  checkCudaErrors(cudaEventSynchronize(end));
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime , start , end));

  std::cout<<"Time of SputinkSpmm :"<<elapsedTime/REPEAT <<"ms"<<"\n";
  std::cout<<"FLOPS of SputinkSpmm :"<< (long long)M * N * K * 2.0 * 1000/(elapsedTime/REPEAT)/1024/1024/1024/1024 << "T\n";
  
  checkCudaErrors(cudaMemcpy(h_output, d_output, output_byte_size, cudaMemcpyDeviceToHost));

  // tranpose it to NCHW for golden check
  std::vector<float> temp_out(out_channel * batch_size * out_h * out_w);
  for (int n = 0; n < batch_size; ++n)
    for (int c = 0; c < out_channel; ++c)
      for (int h = 0; h < out_h; ++h)
        for (int w = 0; w < out_w; ++w) {
          temp_out[n * out_channel * out_h * out_w + c * out_h * out_w + h * out_w + w] = 
            h_output[c * batch_size * out_h * out_w + n * out_h * out_w + h * out_w + w];
        }

  for (int i = 0; i < batch_size * out_channel * out_h * out_w; ++i)
    h_output[i] = temp_out[i];

  return elapsedTime;
}

int main(int argc,char *argv[]) {
  std::cout << "args : " <<argc;
  for (int i = 0; i < argc; ++i) {
      std::cout <<" " << argv[i];
  }
  std::cout << "\n\n";
  assert(argc <= 4);

  srand(time(0));
  cudaDeviceProp device_prop;
  int dev_id = 0;
  checkCudaErrors(cudaSetDevice(dev_id));
  checkCudaErrors(cudaGetDeviceProperties(&device_prop, dev_id));
  printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", dev_id,
          device_prop.name, device_prop.major, device_prop.minor);

  std::string layout = "NCHW";
  int batch_size = 4;
  if (argc >= 3) {
    batch_size = atoi(argv[2]);
  }

  bool timing_im2col = false;
  if (argc >= 4) {
    timing_im2col = atoi(argv[3]);
  }

  int in_channel, img_h, img_w; // 输入数据维度 
  int kernel_size, out_channel, stride, padding, groups; // 卷积核大小和输出通道数
  std::string weight_file = "./conv_weight_data/resnet50/ResNet50-magnitude-98-acc-57.90/module8_0_module3_1_conv2d_0_H_56_W_56_IC_64_OC_64_KS_3_Pad_1_S_1_G_1";
  // std::string weight_file = "./conv_weight_data/mobilenet_v1/mobilenet_v1-86-acc-70.1/conv2d_4_H_112_W_112_IC_32_OC_64_KS_1_Pad_0_S_1_G_1";
  if (argc >= 2) {
    weight_file = argv[1];
  }

  // initialize filter data and config
  float *h_filter = ReadConfigAndDataFromFile(weight_file, batch_size, layout, img_h, img_w,
                                              in_channel, out_channel, kernel_size, 
                                              stride, padding, groups);


  if (kernel_size == 1 && stride == 1) {
    timing_im2col = false;
  }

  int out_h = (img_h + padding * 2 - kernel_size) / stride + 1;
  int out_w = (img_w + padding * 2 - kernel_size) / stride + 1;

  // alloc memory
  size_t input_byte_size = (size_t)batch_size * in_channel * img_h * img_w * sizeof(float);
  size_t filter_byte_size = (size_t)kernel_size * kernel_size * in_channel * out_channel * sizeof(float);
  size_t output_byte_size = (size_t)batch_size * out_h * out_w * out_channel * sizeof(float);

  float *h_input = (float *)malloc(input_byte_size); 
  float *h_output = (float *)malloc(output_byte_size); // store the result of cudnn conv2d
  float *h_check = (float *)malloc(output_byte_size); // store the result of cpu conv2d
  float *h_sparse = (float *)malloc(output_byte_size); // store the result of sparse cpu conv2d
  GenRandTensor(h_input, batch_size * in_channel * img_h * img_w);

  // Create cuda stream
  cudaStream_t cuda_stream;
  checkCudaErrors(cudaStreamCreate(&cuda_stream));



  SputnikSpconv2d(h_input, h_filter, h_output, batch_size, img_h, img_w,
                  in_channel, out_channel, kernel_size, stride, padding,
                  cuda_stream, layout, timing_im2col);
  #ifndef BENCHMARK
    if (layout == "NHWC") {
      NHWC2NCHW(h_input, batch_size, in_channel, img_h, img_w);
    }

    HostConv2d(h_input, h_filter, h_check, batch_size, img_h, img_w,
                in_channel, out_channel, kernel_size, stride, padding);

    if (layout == "NHWC") {
      NCHW2NHWC(h_input, batch_size, in_channel, img_h, img_w);
    }
    if (!TensorEqual(h_output, h_check, batch_size * out_h * out_w * out_channel)) {
      std::cout << "Sputnik Error.\n";
    } else {
      std::cout << "Sputnik Pass.\n";
    }
  #endif
  return 0;
}
