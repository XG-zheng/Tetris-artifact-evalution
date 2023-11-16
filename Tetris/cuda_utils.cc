#include "cuda_utils.h"

#include "tensor_utils.h"


void CuDNNConv2d(float *h_input, float *h_filter, float *h_output,
                int batch_size, int img_h, int img_w,
                int in_channel, int out_channel, int kernel_size,
                int stride, int padding, cudaStream_t cuda_stream,
                std::string layout) {

  int out_h = (img_h + padding * 2 - kernel_size) / stride + 1;
  int out_w = (img_w + padding * 2 - kernel_size) / stride + 1;
  size_t input_byte_size = batch_size * in_channel * img_h * img_w * sizeof(float);
  size_t filter_byte_size = kernel_size * kernel_size * in_channel * out_channel * sizeof(float);
  size_t output_byte_size = batch_size * out_h * out_w * out_channel * sizeof(float);

  // GPU内存分配和数据拷贝
  float *d_filter, *d_input, *d_output;
  checkCudaErrors(cudaMalloc(&d_input, input_byte_size));
  checkCudaErrors(cudaMalloc(&d_filter, filter_byte_size));
  checkCudaErrors(cudaMalloc(&d_output, output_byte_size));
  if (layout == "NCHW") {
    checkCudaErrors(cudaMemcpy(d_filter, h_filter, filter_byte_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_input, h_input, input_byte_size, cudaMemcpyHostToDevice));
  }
  else {
    KCRS2KRSC(h_filter, out_channel, in_channel, kernel_size, kernel_size);
    NCHW2NHWC(h_input, batch_size, in_channel, img_h, img_w);
    checkCudaErrors(cudaMemcpy(d_input, h_input, input_byte_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_filter, h_filter, filter_byte_size, cudaMemcpyHostToDevice));
    NHWC2NCHW(h_input, batch_size, in_channel, img_h, img_w);
    KRSC2KCRS(h_filter, out_channel, in_channel, kernel_size, kernel_size);
  }
  // 创建cuDNN句柄和描述子
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
  // 设置conv2d参数        
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

  cudnnConvolutionFwdAlgo_t conv_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  {
    int returnedAlgoCount = 0;
    cudnnConvolutionFwdAlgoPerf_t perfRes[20];
    checkCUDNN(cudnnFindConvolutionForwardAlgorithm(handle, inputDesc, filterDesc,
                                                    convDesc, outputDesc, 10, &returnedAlgoCount,
                                                    perfRes));
    float min_time = 1e9;
    for (int i = 0; i < returnedAlgoCount; i++) {
        if (perfRes[i].status == CUDNN_STATUS_SUCCESS) {  
            if(perfRes[i].time < min_time) {
              min_time = perfRes[i].time;
              conv_algo = perfRes[i].algo;
            }
        }
    }
  }
  size_t workspace_size;
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle, inputDesc, filterDesc,  
                                                    convDesc, outputDesc,  
                                                    conv_algo, 
                                                    &workspace_size)); 
  void *d_workspace;  
  cudaMalloc(&d_workspace, workspace_size);
  float alpha = 1.0, beta = 0.0;
  // warm up
  for (int i = 0; i < WARM; ++i) {
    cudnnConvolutionForward(handle, &alpha, inputDesc, d_input, filterDesc, d_filter, 
                                        convDesc, conv_algo, d_workspace,
                                        workspace_size, &beta, outputDesc, d_output);
  }
  cudaDeviceSynchronize();
  cudaEvent_t start, end;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&end));

  checkCudaErrors(cudaEventRecord(start, cuda_stream));
  for (int i = 0; i < REPEAT; ++i)
    cudnnConvolutionForward(handle, &alpha, inputDesc, d_input, filterDesc, d_filter, 
                                        convDesc, conv_algo, d_workspace,
                                        workspace_size, &beta, outputDesc, d_output);
  checkCudaErrors(cudaEventRecord(end, cuda_stream));
  checkCudaErrors(cudaEventSynchronize(start));
  checkCudaErrors(cudaEventSynchronize(end));
  float elapsedTime;
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime , start , end));

  std::cout<<"Time of cuDNN :"<<elapsedTime/REPEAT <<"ms"<<"\n";
  std::cout<<"FLOPS of cuDNN :"<< (long long)out_h * out_w * batch_size * out_channel * in_channel * kernel_size * kernel_size * 2 * 1000/(elapsedTime/REPEAT)/1024/1024/1024/1024 << "T\n";
  checkCudaErrors(cudaMemcpy(h_output, d_output, output_byte_size, cudaMemcpyDeviceToHost));

  if (layout == "NHWC") {
    NHWC2NCHW(h_output, batch_size, out_channel, out_h, out_w);
  }

  cudaFree(d_input);
  cudaFree(d_filter);
  cudaFree(d_output);
  cudaFree(d_workspace);
  cudnnDestroy(handle);
  cudnnDestroyTensorDescriptor(inputDesc);
  cudnnDestroyTensorDescriptor(outputDesc);
  cudnnDestroyFilterDescriptor(filterDesc);
  cudnnDestroyConvolutionDescriptor(convDesc);
}
