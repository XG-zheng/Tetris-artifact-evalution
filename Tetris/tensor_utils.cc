#include "tensor_utils.h"

#include <vector>
#include <fstream>
#include <assert.h>
#include <random>


float RandFloat() {
    #ifdef DEBUG
        return 1;
    #endif
    return rand()%RAND_MAX * 2.0/RAND_MAX - 1;
}

// check whether two float number are equal
bool FloatEqual(float x, float y) {
    return fabs(x - y) <= 1e-5*(fabs(x) + fabs(y))  + 1e-5;
}

bool TensorEqual(float *tensor_x, float *tensor_y, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    if (!FloatEqual(tensor_x[i], tensor_y[i])) {
      std::cout << "pos = " << i << ", value_x = " 
                << tensor_x[i] << ", value_y = " << tensor_y[i] << "\n";
      return false;
    }
    // else {
    //   std::cout << "yes = " << i << ", value_x = " 
    //             << tensor_x[i] << ", value_y = " << tensor_y[i] << "\n";
    // }
  }
  return true;
}


void GenRandTensor(float *tensor, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    tensor[i] = RandFloat();
  }
}

void PrintConv2dConfig(int64_t nnz, int batch_size, std::string layout, int img_h, int img_w,
                       int in_channel, int out_channel,
                       int kernel_size, int stride, int padding, int groups) {
  std::cout << "[Config] sparsity = " << (1.0 - nnz * 1.0 /(int64_t)(out_channel * in_channel * kernel_size * kernel_size))
            << ", nnz = " << nnz
            << ", batch_size = " << batch_size << ", img_h = " << img_h
            << ", img_w = " << img_w << ", in_channel = " << in_channel
            << ", out_channel = " << out_channel << ", kernel_size = " << kernel_size
            << ", padding = " << padding << ", stride = " << stride
            << "\n\n";
}

float* ReadConfigAndDataFromFile(std::string weight_file, int batch_size, std::string layout, int &img_h, int &img_w,
                              int &in_channel, int &out_channel,
                              int &kernel_size, int &stride, int &padding, int &groups) {
  int64_t nnz = 0;
  std::ifstream reader;
  reader.open(weight_file);
  if (!reader.is_open()) {
    std::cout << "Open file error.\n";
    exit(1);
  }

  reader >> img_h >> img_w >> in_channel >> out_channel >> kernel_size >> padding >> stride >> groups;
  size_t filter_byte_size = (size_t)kernel_size * kernel_size * in_channel * out_channel * sizeof(float);
  float *h_filter = (float *)malloc(filter_byte_size);

  for (int64_t i = 0; i < (int64_t)out_channel * in_channel * kernel_size * kernel_size; ++i) {
    reader >> h_filter[i];
    if (h_filter[i] != 0) {
      nnz += 1;
      #ifdef DEBUG
        h_filter[i] = 1.0;
      #endif
    }
  }

  PrintConv2dConfig(nnz, batch_size, layout, img_h, img_w, in_channel, out_channel, kernel_size, stride, padding, groups);
  return h_filter;
}


void NCHW2NHWC(float *src, int N, int C, int H, int W) {
  std::vector<float> dst_data(N * C * H * W);
  for (int n = 0; n < N; ++n) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        for (int c = 0; c < C; ++c) {
          dst_data[n * H * W * C + h * W * C + w * C + c] = src[n * H * W *C + c * H * W + h * W + w];
        }
      }
    }
  }
  for (int i = 0; i < N * H * W * C; ++i) src[i] = dst_data[i];
}

void NHWC2NCHW(float *src, int N, int C, int H, int W) {
  std::vector<float> dst_data(N * C * H * W);
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          dst_data[n * H * W *C + c * H * W + h * W + w] = src[n * H * W * C + h * W * C + w * C + c];
        }
      }
    }
  }
  for (int i = 0; i < N * H * W * C; ++i) src[i] = dst_data[i];
}

void KCRS2KRSC(float *src, int K, int C, int R, int S) {
  std::vector<float> dst_data(K * C * R * S);
  for (int k = 0; k < K; ++k)
    for (int r = 0; r < R; ++r)
      for (int s = 0; s < S; ++s) {
        for (int c = 0; c < C; ++c) {
          dst_data[k * R * S * C + r * S * C + s * C + c] = src[k * C * R * S + c * R *S + r * S  + s];
        }
      }
  for (int i = 0; i < K * R * S * C; ++i) src[i] = dst_data[i];
}

void KRSC2KCRS(float *src, int K, int C, int R, int S) {
  std::vector<float> dst_data(K * C * R * S);
  for (int k = 0; k < K; ++k)
    for (int c = 0; c < C; ++c) 
      for (int r = 0; r < R; ++r)
        for (int s = 0; s < S; ++s) {
          dst_data[k * C * R * S + c * R *S + r * S  + s] = src[k * R * S * C + r * S * C + s * C + c];
        }
  for (int i = 0; i < K * R * S * C; ++i) src[i] = dst_data[i];
}

void HostConv2d(float *input, float *filter, float *output,
                int batch_size, int img_h, int img_w,
                int in_channel, int out_channel,
                int kernel_size, int stride, int padding) {
  int out_h = (img_h + padding * 2 - kernel_size) / stride + 1;
  int out_w = (img_w + padding * 2 - kernel_size) / stride + 1;

#define output(n, c, h, w) output[(n) * out_channel * out_h * out_w + (c) *  out_h * out_w + (h) * out_w + (w)]
#define input(n, c, h, w) input[(n) * in_channel * img_h * img_w + (c) *  img_h * img_w + (h) * img_w + (w)]
#define filter(oc, ic, kh, kw) filter[(oc) * in_channel * kernel_size * kernel_size + (ic) * kernel_size * kernel_size + (kh) * kernel_size + (kw)]

  for (int n = 0; n < batch_size; ++n) {
    for (int oc = 0; oc < out_channel; ++oc) {
      for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
          output(n, oc, h, w) = 0.f;
          for (int ic = 0; ic < in_channel; ++ic) {
            for (int x = 0; x < kernel_size; ++x) {
              for (int y = 0; y < kernel_size; ++y) {
                if (h * stride + x < padding || h * stride + x >= img_h + padding) continue;
                if (w * stride + y < padding || w * stride + y >= img_w + padding) continue;
                output(n, oc, h, w) += input(n, ic, h * stride + x - padding, w * stride + y - padding) * filter(oc, ic, x, y);
              }
            }
          }
        }
      }
    }
  }

#undef output
#undef input
#undef filter

}
