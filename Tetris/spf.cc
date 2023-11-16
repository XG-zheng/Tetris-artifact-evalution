#include <cuda_runtime.h>
#include <cudnn.h>  
#include <random>
#include <iostream>
#include <string>
#include <fstream>
#include <assert.h>
#include <cstdlib>
#include <stdio.h>
#include <unordered_map>
#include <functional>
#include <vector>

#include "spconv2d_utils.h"
#include "spconv2d_kernel.cuh"
#include "tensor_utils.h"
#include "cuda_utils.h"
#include "ablation_kernel.cuh"



template<typename ValueType, typename OffsetType, typename PositionType>
void SparseConv2d_SPF(float *h_input, float *h_filter, float *h_output,
  int batch_size, int img_h, int img_w,
  int in_channel, int out_channel, int kernel_size,
  int stride, int padding, cudaStream_t cuda_stream,
  std::string layout, bool apply_reorder) {

  int out_h = (img_h + padding * 2 - kernel_size) / stride + 1;
  int out_w = (img_w + padding * 2 - kernel_size) / stride + 1;
  size_t input_byte_size = batch_size * in_channel * img_h * img_w * sizeof(ValueType);
  size_t output_byte_size = batch_size * out_h * out_w * out_channel * sizeof(ValueType);
  float *d_input, *d_output;
  checkCudaErrors(cudaMalloc(&d_input, input_byte_size));
  checkCudaErrors(cudaMalloc(&d_output, output_byte_size));
  NCHW2NHWC(h_input, batch_size, in_channel, img_h, img_w);
  checkCudaErrors(cudaMemcpy(d_input, h_input, input_byte_size, cudaMemcpyHostToDevice));
  NHWC2NCHW(h_input, batch_size, in_channel, img_h, img_w);

  OffsetType *d_stage_len, *d_offsets;
  ValueType *d_values;
  PositionType *d_position, *d_oc_permutation;
  SparseFilter<ValueType, OffsetType, PositionType> sparse_filter(h_filter, in_channel, out_channel,
      kernel_size, 128, true);
  d_offsets = sparse_filter.GetDeviceOffsets();
  d_stage_len = sparse_filter.GetDeviceStageLen();
  d_values = sparse_filter.GetDeviceValues();
  d_position = sparse_filter.GetDevicePosition();
  d_oc_permutation = sparse_filter.GetDeviceOCPermutation();
  

  cudaDeviceSynchronize();
  float elapsedTime;
  cudaEvent_t start, end;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&end));
    
  for (int i = 0; i < WARM; ++i) {
    SparseConv2d<ValueType, OffsetType, PositionType,
                        3, 1, 128, 2, 4>(
                          d_input, d_offsets, d_stage_len, d_oc_permutation, d_position, d_values, d_output,
                          batch_size, img_h, img_w, in_channel, out_channel,
                          kernel_size, stride, padding, cuda_stream
                        );
  }
  cudaDeviceSynchronize();
  checkCudaErrors(cudaEventRecord(start, cuda_stream));

  for (int i = 0; i < REPEAT; ++i) {
    SparseConv2d<ValueType, OffsetType, PositionType,
                        3, 1, 128, 2, 4>(
                          d_input, d_offsets, d_stage_len, d_oc_permutation, d_position, d_values, d_output,
                          batch_size, img_h, img_w, in_channel, out_channel,
                          kernel_size, stride, padding, cuda_stream
                        );
  }
  
  checkCudaErrors(cudaEventRecord(end, cuda_stream));
  checkCudaErrors(cudaEventSynchronize(start));
  checkCudaErrors(cudaEventSynchronize(end));
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime , start , end));
  elapsedTime /= REPEAT;
  std::cout<<"Time of Ablation :"<< elapsedTime <<"ms"<<"\n";
  std::cout<<"FLOPS of Ablation :"<< (long long)out_h * out_w * batch_size * out_channel * in_channel * kernel_size * kernel_size * 2 * 1000/(elapsedTime)/1024/1024/1024/1024 << "T\n";
  checkCudaErrors(cudaMemcpy(h_output, d_output, output_byte_size, cudaMemcpyDeviceToHost));

  // just for verifying with host output.
  if (layout == "NHWC") {
    NHWC2NCHW(h_output, batch_size, out_channel, out_h, out_w);
  }
  cudaFree(d_input);
  cudaFree(d_output);
}

int main(int argc,char *argv[]) {
  std::cout << "args : " <<argc;
  for (int i = 0; i < argc; ++i) {
      std::cout <<" " << argv[i];
  }
  std::cout << "\n\n";
  assert(argc <= 3);

  srand(time(0));
  cudaDeviceProp device_prop;
  int dev_id = 0;
  checkCudaErrors(cudaSetDevice(dev_id));
  checkCudaErrors(cudaGetDeviceProperties(&device_prop, dev_id));

  printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", dev_id,
          device_prop.name, device_prop.major, device_prop.minor);

  std::string layout = "NHWC";
  int batch_size = 4;
  if (argc >= 3) {
    batch_size = atoi(argv[2]);
  }

  bool apply_reorder = true;

  int in_channel, img_h, img_w; 
  int kernel_size, out_channel, stride, padding, groups; 

  std::string weight_file = "./conv_weight_data/vgg19/vgg19-92-acc-71.7/module8_0_module2_0_conv2d_3_H_56_W_56_IC_256_OC_256_KS_3_Pad_1_S_1_G_1";

  if (argc >= 2) {
    weight_file = argv[1];
  }

  // read filter and config from file
  float *h_filter = ReadConfigAndDataFromFile(weight_file, batch_size, layout, img_h, img_w,
                                              in_channel, out_channel, kernel_size, 
                                              stride, padding, groups);

  int out_h = (img_h + padding * 2 - kernel_size) / stride + 1;
  int out_w = (img_w + padding * 2 - kernel_size) / stride + 1;

  // alloc host memory
  size_t input_byte_size = (size_t)batch_size * in_channel * img_h * img_w * sizeof(float);
  size_t filter_byte_size = (size_t)kernel_size * kernel_size * in_channel * out_channel * sizeof(float);
  size_t output_byte_size = (size_t)batch_size * out_h * out_w * out_channel * sizeof(float);

  float *h_input = (float *)malloc(input_byte_size); 
  float *h_output = (float *)malloc(output_byte_size); // store the result of cudnn conv2d
  float *h_check = (float *)malloc(output_byte_size); // store the result of cpu conv2d
  float *h_sparse = (float *)malloc(output_byte_size); // store the result of sparse cpu conv2d

  GenRandTensor(h_input, batch_size * in_channel * img_h * img_w);

  // create cuda stream
  cudaStream_t cuda_stream;
  checkCudaErrors(cudaStreamCreate(&cuda_stream));

  SparseConv2d_SPF<float, int, int>(h_input, h_filter, h_sparse, batch_size, img_h, img_w,
              in_channel, out_channel, kernel_size, stride, padding, cuda_stream, layout, apply_reorder);

  #ifndef BENCHMARK
    std::cout << "HostConv2d begin\n";
    HostConv2d(h_input, h_filter, h_check, batch_size, img_h, img_w,
                in_channel, out_channel, kernel_size, stride, padding);
    std::cout << "HostConv2d end\n";
  
    if (!TensorEqual(h_sparse, h_check, batch_size * out_h * out_w * out_channel)) {
      std::cout << "Sparse Error.\n";
    } else {
      std::cout << "Sparse Pass.\n";
    }
  #endif

  // free memory
  free(h_input);
  free(h_filter);
  free(h_output);
  free(h_sparse);
  free(h_check);
  return 0; 
}
