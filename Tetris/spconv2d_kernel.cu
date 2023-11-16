#include "spconv2d_kernel.cuh"

#include <stdio.h>
#include <unordered_map>
#include <functional>
#include <vector>

#include "tensor_utils.h"

// #define TILING_SEARCH


template<typename ValueType, int KERNEL_SIZE, int STRIDE,
          int TILE_IC, int TILE_H, int TILE_W>
__device__ void __forceinline__ LoadImg2SharedMem(ValueType *shm, ValueType *input,
                                            int img_h, int img_w, int in_channel, int batch_id,
                                            int start_ic, int start_h, int start_w, 
                                            int padding) {
#define input(n, h, w, c) input[(n) * img_h * img_w * in_channel + (h) * img_w * in_channel + (w) * in_channel + (c)]

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int thread_id = threadIdx.x % WARP_SIZE;
  const int ALL_WARP_NUM = blockDim.x / WARP_SIZE;
  constexpr int INPUT_TILE_H = (TILE_H - 1) * STRIDE + KERNEL_SIZE;
  constexpr int INPUT_TILE_W = (TILE_W - 1) * STRIDE + KERNEL_SIZE;
  constexpr int TILE_SIZE = INPUT_TILE_H * INPUT_TILE_W;
  for (int j = warp_id; j < TILE_SIZE; j += ALL_WARP_NUM) {
    int h = j / INPUT_TILE_W;
    int w = j % INPUT_TILE_W;
    if (start_h + h >= padding  && start_h + h < img_h + padding &&
        start_w + w >= padding  && start_w + w < img_w + padding) {
      for (int ic = thread_id; ic < TILE_IC; ic += WARP_SIZE) {
        if (ic + start_ic < in_channel) {
          shm[h * INPUT_TILE_W * TILE_IC + w * TILE_IC + ic] = 
            input(batch_id, start_h + h - padding, start_w + w - padding, start_ic + ic);
        }
      }
    } else {
      for (int ic = thread_id; ic < TILE_IC; ic += WARP_SIZE) {
        shm[h * INPUT_TILE_W * TILE_IC + w * TILE_IC + ic] = 0.0;
      }
    }
  }
  __syncthreads();

#undef input

}


template<typename ValueType, typename OffsetType, typename PositionType,
         int KERNEL_SIZE, int STRIDE, int TILE_IC, int TILE_H, int TILE_W, int TILE_OC> 
__global__ void SparseConv2dKernel(
                          ValueType *input, OffsetType *offsets, OffsetType *stage_len,
                          PositionType *oc_permutation, PositionType *position, ValueType *values,
                          ValueType *output,
                          int batch_size, int img_h, int img_w,
                          int in_channel, int out_channel, int kernel_size,
                          int stride, int padding) {
#define output(n, h, w, c) output[(n) * out_h * out_w * out_channel + (h) * out_w * out_channel + (w) * out_channel + (c)]
#define input(n, h, w, c) input[(n) * img_h * img_w * in_channel + (h) * img_w * in_channel + (w) * in_channel + (c)]
         
  const int oc = threadIdx.x + blockIdx.y * TILE_OC;

  const int INPUT_TILE_W = (TILE_W - 1) * stride + kernel_size;

  const int out_h = (img_h + padding * 2 - kernel_size) / stride + 1;
  const int out_w = (img_w + padding * 2 - kernel_size) / stride + 1;
  const int H_TILE_NUM = (out_h + TILE_H - 1) / TILE_H;
  const int W_TILE_NUM = (out_w + TILE_W - 1) / TILE_W;

  const int STAGE_NUM_PER_IC = (in_channel - 1) / TILE_IC + 1;

  const int batch_id = blockIdx.x / (H_TILE_NUM * W_TILE_NUM);
  const int start_h = (blockIdx.x % (H_TILE_NUM * W_TILE_NUM)) / W_TILE_NUM * TILE_H;
  const int start_w = blockIdx.x % W_TILE_NUM * TILE_W;
  const int VEC_WIDTH = 4;

  __shared__ ValueType img_tile[TILE_IC * ((TILE_H - 1) * STRIDE + KERNEL_SIZE) * ((TILE_W - 1) * STRIDE + KERNEL_SIZE)];
  ValueType out_reg[TILE_H * TILE_W] = {0.};

  const int OUT_CHANNEL_GROUP_ID = (threadIdx.x + blockIdx.y * TILE_OC) / WARP_SIZE;
  const int OC_LANE_ID = threadIdx.x % WARP_SIZE;
  for (int i = 0; i < STAGE_NUM_PER_IC; i++) {
    LoadImg2SharedMem<ValueType, KERNEL_SIZE, STRIDE,
                TILE_IC, TILE_H, TILE_W>(img_tile, input,
      img_h, img_w, in_channel, batch_id,
      i * TILE_IC, start_h * STRIDE, start_w * STRIDE, padding);

    if (oc < out_channel) {
      int offset_id = OUT_CHANNEL_GROUP_ID * STAGE_NUM_PER_IC + i;
      int start_offset = offsets[offset_id];

      int nnz = offsets[offset_id + 1] - start_offset;

      int start = start_offset * WARP_SIZE;
    
      for (int k = 0; k < nnz; k += VEC_WIDTH) {
        int4 pos = *reinterpret_cast<int4*>(&position[start + k * WARP_SIZE + OC_LANE_ID * VEC_WIDTH]);
        float4 v = *reinterpret_cast<float4*>(&values[start + k * WARP_SIZE + OC_LANE_ID * VEC_WIDTH]);

        int y, x, ic;
        y = pos.x & ((1 << BIT_S) - 1);
        x = (pos.x >> BIT_S) & ((1 << BIT_R) - 1);
        ic = (pos.x >> (BIT_S + BIT_R));
        for (int h = 0; h < TILE_H; ++h) {
            for (int w = 0; w < TILE_W; ++w) {
              out_reg[h * TILE_W + w] += img_tile[(h * STRIDE + x) * INPUT_TILE_W * TILE_IC + (w * STRIDE + y) * TILE_IC + ic] * v.x; 
          }
        }

        y = pos.y & ((1 << BIT_S) - 1);
        x = (pos.y >> BIT_S) & ((1 << BIT_R) - 1);
        ic = (pos.y >> (BIT_S + BIT_R));
        for (int h = 0; h < TILE_H; ++h) {
            for (int w = 0; w < TILE_W; ++w) {
              out_reg[h * TILE_W + w] += img_tile[(h * STRIDE + x) * INPUT_TILE_W * TILE_IC + (w * STRIDE + y) * TILE_IC + ic] * v.y; 
          }
        }

        y = pos.z & ((1 << BIT_S) - 1);
        x = (pos.z >> BIT_S) & ((1 << BIT_R) - 1);
        ic = (pos.z >> (BIT_S + BIT_R));
        for (int h = 0; h < TILE_H; ++h) {
            for (int w = 0; w < TILE_W; ++w) {
              out_reg[h * TILE_W + w] += img_tile[(h * STRIDE + x) * INPUT_TILE_W * TILE_IC + (w * STRIDE + y) * TILE_IC + ic] * v.z; 
          }
        }

        y = pos.w & ((1 << BIT_S) - 1);
        x = (pos.w >> BIT_S) & ((1 << BIT_R) - 1);
        ic = (pos.w >> (BIT_S + BIT_R));
        for (int h = 0; h < TILE_H; ++h) {
            for (int w = 0; w < TILE_W; ++w) {
              out_reg[h * TILE_W + w] += img_tile[(h * STRIDE + x) * INPUT_TILE_W * TILE_IC + (w * STRIDE + y) * TILE_IC + ic] * v.w; 
          }
        }
      }
    }
    
    __syncthreads();
  }

  if (oc >= out_channel) {
    return;
  }

  int ori_oc = oc_permutation[oc];

  for (int h = 0; h < TILE_H; h++) {
    for (int w = 0; w < TILE_W; w++) {
      if (start_h + h < out_h && start_w + w < out_w) {
        output(batch_id, start_h + h, start_w + w, ori_oc) = out_reg[h * TILE_W + w];
      }
    }
  }

#undef output
#undef input
}


template<typename ValueType, typename OffsetType, typename PositionType,
         int KERNEL_SIZE, int STRIDE, int TILE_IC, int TILE_H, int TILE_W> 
void SparseConv2d(ValueType *input, OffsetType *offsets, OffsetType *stage_len,
                          PositionType *oc_permutation, PositionType *position, ValueType *values,
                          ValueType *output,
                          int batch_size, int img_h, int img_w,
                          int in_channel, int out_channel, int kernel_size,
                          int stride, int padding, cudaStream_t cuda_stream) {
  int out_h = (img_h + padding * 2 - kernel_size) / stride + 1;
  int out_w = (img_w + padding * 2 - kernel_size) / stride + 1;


  constexpr int TILE_OC = 128;
  dim3 grid_size(batch_size * ((out_h + TILE_H - 1) / TILE_H) * ((out_w + TILE_W - 1) / TILE_W),
                (out_channel + TILE_OC - 1) / TILE_OC);
  
  dim3 block_size(TILE_OC);

  SparseConv2dKernel<ValueType, OffsetType, PositionType,
            KERNEL_SIZE, STRIDE, TILE_IC, TILE_H, TILE_W, TILE_OC>
            <<<grid_size, block_size, 0, cuda_stream>>>(
              input, offsets, stage_len, oc_permutation, position, values, output,
              batch_size, img_h, img_w, in_channel, out_channel,
              kernel_size, stride, padding
  );
}


// template void CudaSparseConv2d<float, int, int>(float *h_input, float *h_filter, float *h_output,
//   int batch_size, int img_h, int img_w,
//   int in_channel, int out_channel, int kernel_size,
//   int stride, int padding, cudaStream_t cuda_stream,
//   std::string layout, bool apply_reorder);

#define TETRIS_INSTANTIATE_TILED_FLOAT(fn, KERNEL_SIZE, STRIDE, TILE_IC, TILE_H, TILE_W)          \
  template void fn<float, int, int, KERNEL_SIZE, STRIDE, TILE_IC, TILE_H, TILE_W>(                \
    float*, int*, int*, int*, int*, float*, float*,                                               \
    int, int, int, int, int, int, int, int, cudaStream_t);


TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 1, 1);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 1, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 1, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 1, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 1, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 1, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 1, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 1, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 1, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 1, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 1, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 1, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 1, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 1, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 2, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 2, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 2, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 2, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 2, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 2, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 2, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 2, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 2, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 2, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 2, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 2, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 2, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 3, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 3, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 3, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 3, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 3, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 3, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 3, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 3, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 3, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 3, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 3, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 3, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 4, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 4, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 4, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 4, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 4, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 4, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 4, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 4, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 4, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 4, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 4, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 5, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 5, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 5, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 5, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 5, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 5, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 5, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 5, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 5, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 5, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 6, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 6, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 6, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 6, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 6, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 6, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 6, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 6, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 6, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 7, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 7, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 7, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 7, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 7, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 7, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 7, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 7, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 8, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 8, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 8, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 8, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 8, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 8, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 8, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 9, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 9, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 9, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 9, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 9, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 9, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 10, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 10, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 10, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 10, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 10, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 11, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 32, 11, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 1, 1);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 1, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 1, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 1, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 1, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 1, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 1, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 1, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 1, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 1, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 1, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 1, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 1, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 1, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 2, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 2, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 2, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 2, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 2, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 2, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 2, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 2, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 2, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 2, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 2, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 2, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 2, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 3, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 3, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 3, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 3, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 3, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 3, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 3, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 3, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 3, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 3, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 3, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 3, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 4, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 4, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 4, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 4, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 4, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 4, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 4, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 4, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 4, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 4, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 4, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 5, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 5, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 5, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 5, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 5, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 5, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 5, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 6, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 6, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 6, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 6, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 6, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 7, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 64, 7, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 1, 1);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 1, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 1, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 1, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 1, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 1, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 1, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 1, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 1, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 1, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 1, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 1, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 1, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 1, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 2, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 2, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 2, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 2, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 2, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 2, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 2, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 2, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 2, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 3, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 3, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 3, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 3, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 3, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 4, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 4, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 4, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 256, 1, 1);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 256, 1, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 256, 1, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 256, 1, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 256, 1, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 256, 1, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 256, 2, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 256, 2, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 256, 2, 4);

TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 1, 1);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 1, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 1, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 1, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 1, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 1, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 1, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 1, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 1, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 1, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 1, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 1, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 1, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 1, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 2, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 2, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 2, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 2, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 2, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 2, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 2, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 2, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 2, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 2, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 2, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 2, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 2, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 3, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 3, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 3, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 3, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 3, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 3, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 3, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 3, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 3, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 3, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 3, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 4, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 4, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 4, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 4, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 4, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 4, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 4, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 5, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 5, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 5, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 5, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 32, 6, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 1, 1);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 1, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 1, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 1, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 1, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 1, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 1, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 1, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 1, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 1, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 1, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 1, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 1, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 1, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 2, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 2, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 2, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 2, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 2, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 2, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 2, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 2, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 3, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 3, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 3, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 3, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 64, 4, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 128, 1, 1);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 128, 1, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 128, 1, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 128, 1, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 128, 1, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 128, 1, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 128, 1, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 128, 2, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 128, 2, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 128, 2, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 256, 1, 1);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 256, 1, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 2, 256, 1, 3);

TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 1, 1);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 1, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 1, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 1, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 1, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 1, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 1, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 1, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 1, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 1, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 1, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 1, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 1, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 1, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 2, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 2, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 2, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 2, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 2, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 2, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 2, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 2, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 2, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 2, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 2, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 2, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 2, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 3, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 3, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 3, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 3, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 3, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 3, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 3, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 3, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 3, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 3, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 3, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 3, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 4, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 4, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 4, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 4, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 4, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 4, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 4, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 4, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 4, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 4, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 4, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 5, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 5, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 5, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 5, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 5, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 5, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 5, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 5, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 5, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 5, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 6, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 6, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 6, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 6, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 6, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 6, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 6, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 6, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 6, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 7, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 7, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 7, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 7, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 7, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 7, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 7, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 7, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 8, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 8, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 8, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 8, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 8, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 8, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 8, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 9, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 9, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 9, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 9, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 9, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 9, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 10, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 10, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 10, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 10, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 10, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 11, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 11, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 11, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 11, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 12, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 12, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 12, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 13, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 32, 13, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 1, 1);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 1, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 1, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 1, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 1, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 1, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 1, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 1, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 1, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 1, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 1, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 1, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 1, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 1, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 2, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 2, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 2, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 2, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 2, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 2, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 2, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 2, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 2, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 2, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 2, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 2, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 2, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 3, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 3, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 3, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 3, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 3, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 3, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 3, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 3, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 3, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 3, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 3, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 3, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 4, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 4, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 4, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 4, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 4, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 4, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 4, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 4, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 4, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 4, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 4, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 5, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 5, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 5, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 5, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 5, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 5, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 5, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 5, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 5, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 5, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 6, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 6, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 6, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 6, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 6, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 6, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 6, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 6, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 6, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 7, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 7, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 7, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 7, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 7, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 7, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 7, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 8, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 8, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 8, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 8, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 8, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 9, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 64, 9, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 1, 1);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 1, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 1, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 1, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 1, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 1, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 1, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 1, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 1, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 1, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 1, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 1, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 1, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 1, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 2, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 2, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 2, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 2, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 2, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 2, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 2, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 2, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 2, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 2, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 2, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 2, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 2, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 3, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 3, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 3, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 3, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 3, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 3, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 3, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 3, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 3, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 3, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 3, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 3, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 4, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 4, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 4, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 4, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 4, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 4, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 4, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 4, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 4, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 5, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 5, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 5, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 5, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 5, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 6, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 6, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 128, 6, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 1, 1);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 1, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 1, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 1, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 1, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 1, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 1, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 1, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 1, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 1, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 1, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 1, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 1, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 1, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 2, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 2, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 2, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 2, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 2, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 2, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 2, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 2, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 2, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 2, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 2, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 3, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 3, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 3, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 3, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 3, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 3, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 4, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 4, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 1, 256, 4, 6);

TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 1, 1);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 1, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 1, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 1, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 1, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 1, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 1, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 1, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 1, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 1, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 1, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 1, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 1, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 1, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 2, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 2, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 2, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 2, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 2, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 2, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 2, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 2, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 2, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 2, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 2, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 2, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 2, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 3, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 3, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 3, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 3, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 3, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 3, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 3, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 3, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 3, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 3, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 3, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 3, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 4, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 4, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 4, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 4, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 4, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 4, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 4, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 4, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 4, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 4, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 4, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 5, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 5, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 5, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 5, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 5, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 5, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 5, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 6, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 6, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 6, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 6, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 32, 7, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 1, 1);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 1, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 1, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 1, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 1, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 1, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 1, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 1, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 1, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 1, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 1, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 1, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 1, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 1, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 2, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 2, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 2, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 2, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 2, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 2, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 2, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 2, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 2, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 2, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 2, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 2, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 2, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 3, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 3, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 3, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 3, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 3, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 3, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 3, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 3, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 4, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 4, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 4, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 4, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 64, 5, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 128, 1, 1);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 128, 1, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 128, 1, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 128, 1, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 128, 1, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 128, 1, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 128, 1, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 128, 1, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 128, 1, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 128, 1, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 128, 1, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 128, 1, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 128, 1, 13);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 128, 1, 14);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 128, 2, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 128, 2, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 128, 2, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 128, 2, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 128, 2, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 128, 2, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 128, 2, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 128, 3, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 128, 3, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 128, 3, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 256, 1, 1);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 256, 1, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 256, 1, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 256, 1, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 256, 1, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 256, 1, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 256, 1, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 256, 1, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 256, 1, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 256, 1, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 256, 1, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 256, 1, 12);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 256, 2, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 256, 2, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 1, 2, 256, 2, 4);

TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 32, 1, 1);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 32, 1, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 32, 1, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 32, 1, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 32, 1, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 32, 1, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 32, 1, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 32, 1, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 32, 1, 9);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 32, 1, 10);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 32, 1, 11);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 32, 2, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 32, 2, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 32, 2, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 32, 2, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 32, 2, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 32, 2, 7);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 32, 2, 8);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 32, 3, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 32, 3, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 32, 3, 5);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 32, 3, 6);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 32, 4, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 64, 1, 1);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 64, 1, 2);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 64, 1, 3);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 64, 1, 4);
TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 7, 2, 64, 2, 2);
