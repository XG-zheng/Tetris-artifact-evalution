#include <cuda_runtime.h>
#include <cudnn.h>  
#include <random>
#include <iostream>
#include <string>
#include <fstream>
#include <assert.h>
#include <cstdlib>

#include "tensor_utils.h"
#include "cuda_utils.h"


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

  float sparsity = 0;
  if (argc >= 4) {
    sparsity = atof(argv[3]);
  }

  int in_channel, img_h, img_w; 
  int kernel_size, out_channel, stride, padding, groups; 
  std::string weight_file = "./conv_weight_data/vgg19/vgg19-92-acc-71.7/module8_2_module3_0_conv2d_0_H_28_W_28_IC_512_OC_512_KS_3_Pad_1_S_1_G_1";

  if (argc >= 2) {
    weight_file = argv[1];
  }


  // initialize filter data and config
  float *h_filter = ReadConfigAndDataFromFile(weight_file, batch_size, layout, img_h, img_w,
                                              in_channel, out_channel, kernel_size, 
                                              stride, padding, groups);


  int out_h = (img_h + padding * 2 - kernel_size) / stride + 1;
  int out_w = (img_w + padding * 2 - kernel_size) / stride + 1;

  // CPU内存分配和数据初始化
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

  // pruning in out_channel
  out_channel *= (1 - sparsity);
  if (out_channel == 0) out_channel = 1;

  CuDNNConv2d(h_input, h_filter, h_output, batch_size, img_h, img_w,
              in_channel, out_channel, kernel_size, stride, padding, cuda_stream, layout);

  #ifndef BENCHMARK
  HostConv2d(h_input, h_filter, h_check, batch_size, img_h, img_w,
              in_channel, out_channel, kernel_size, stride, padding);
  if (!TensorEqual(h_output, h_check, batch_size * out_h * out_w * out_channel)) {
    std::cout << "Cudnn Error.\n";
  } else {
    std::cout << "Cudnn Pass.\n";
  }
  #endif


  free(h_input);
  free(h_filter);
  free(h_output);
  free(h_sparse);
  free(h_check);
  return 0; 
}
