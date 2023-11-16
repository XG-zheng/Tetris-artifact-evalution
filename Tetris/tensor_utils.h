#ifndef TENSOR_UTILS_H_
#define TENSOR_UTILS_H_

#include <iostream>
#include <string>

// #define DEBUG

#define BENCHMARK
#define REPEAT 100
#define WARM 30

float RandFloat();

// check whether two float number are equal
bool FloatEqual(float x, float y);

bool TensorEqual(float *tensor_x, float *tensor_y, size_t len);


void GenRandTensor(float *tensor, size_t len);

void PrintConv2dConfig(int64_t nnz, int batch_size, std::string layout, int img_h, int img_w,
                       int in_channel, int out_channel,
                       int kernel_size, int stride, int padding, int groups);

float* ReadConfigAndDataFromFile(std::string weight_file, int batch_size, std::string layout, int &img_h, int &img_w,
                              int &in_channel, int &out_channel,
                              int &kernel_size, int &stride, int &padding, int &groups);

void NCHW2NHWC(float *src, int N, int C, int H, int W);

void NHWC2NCHW(float *src, int N, int C, int H, int W);

// K represents the number of output feature maps
// C is the number of input feature maps
// R is the number of rows per filter
// S is the number of columns per filter
void KCRS2KRSC(float *src, int K, int C, int R, int S);

void KRSC2KCRS(float *src, int K, int C, int R, int S);

void HostConv2d(float *input, float *filter, float *output,
                int batch_size, int img_h, int img_w,
                int in_channel, int out_channel,
                int kernel_size, int stride, int padding);

#endif  // TENSOR_UTILS_H_
