#include "spconv2d_kernel.cuh"

template<typename ValueType, int KERNEL_SIZE, int STRIDE,
          int TILE_IC, int TILE_H, int TILE_W>
__device__ void __forceinline__ LoadImg2ShareMem(ValueType *shm, ValueType *input,
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
          shm[ic * TILE_SIZE + h * INPUT_TILE_W + w] = 
            input(batch_id, start_h + h - padding, start_w + w - padding, start_ic + ic);
        }
      }
    } else {
      for (int ic = thread_id; ic < TILE_IC; ic += WARP_SIZE) {
        shm[ic * TILE_SIZE + h * INPUT_TILE_W + w] = 0.0;
      }
    }
  }
  __syncthreads();

#undef input
}

template<typename ValueType, typename OffsetType, typename PositionType,
         int KERNEL_SIZE, int STRIDE, int TILE_IC, int TILE_H, int TILE_W, int TILE_OC>
__global__ void BaselineKernel(ValueType *input, OffsetType *offsets,
                          PositionType *position, ValueType *values,
                          ValueType *output,
                          int batch_size, int img_h, int img_w,
                          int in_channel, int out_channel, int kernel_size,
                          int stride, int padding) {
#define output(n, h, w, c) output[(n) * out_h * out_w * out_channel + (h) * out_w * out_channel + (w) * out_channel + (c)]
#define input(n, h, w, c) input[(n) * img_h * img_w * in_channel + (h) * img_w * in_channel + (w) * in_channel + (c)]
             
  const int oc = threadIdx.x + blockIdx.x * TILE_OC;
  if (oc >= out_channel) {
    return;
  }

  constexpr int INPUT_TILE_H = (TILE_H - 1) * STRIDE + KERNEL_SIZE;
  constexpr int INPUT_TILE_W = (TILE_W - 1) * STRIDE + KERNEL_SIZE;
  constexpr int TILE_SIZE = INPUT_TILE_H * INPUT_TILE_W;

  const int out_h = (img_h + padding * 2 - KERNEL_SIZE) / stride + 1;
  const int out_w = (img_w + padding * 2 - KERNEL_SIZE) / stride + 1;
  const int H_TILE_NUM = (out_h + TILE_H - 1) / TILE_H;
  const int W_TILE_NUM = (out_w + TILE_W - 1) / TILE_W;

  const int STAGE_NUM_PER_IC = (in_channel - 1) / TILE_IC + 1;

  const int batch_id = blockIdx.y / (H_TILE_NUM * W_TILE_NUM);
  const int start_h = (blockIdx.y % (H_TILE_NUM * W_TILE_NUM)) / W_TILE_NUM * TILE_H;
  const int start_w = blockIdx.y % W_TILE_NUM * TILE_W;

  __shared__ ValueType img_tile[TILE_IC * ((TILE_H - 1) * STRIDE + KERNEL_SIZE) * ((TILE_W - 1) * STRIDE + KERNEL_SIZE)];
  ValueType out_reg[TILE_H * TILE_W] = {0.};

  for (int i = 0; i < STAGE_NUM_PER_IC; i++) {
    int stage_id = STAGE_NUM_PER_IC * oc + i;
    int start_offset = offsets[stage_id], end_offset = offsets[stage_id + 1];
    LoadImg2ShareMem<ValueType, KERNEL_SIZE, STRIDE,
                  TILE_IC, TILE_H, TILE_W>(img_tile, input,
        img_h, img_w, in_channel, batch_id,
        i * TILE_IC, start_h * STRIDE, start_w * STRIDE, padding);

    for (int k = start_offset; k < end_offset; k++) {
      
      float v = values[k];
      unsigned int pos = position[k];
      unsigned int y = pos & ((1 << BIT_S) - 1);
      unsigned int x = (pos >> (BIT_S)) & ((1 << BIT_R) - 1);
      unsigned int ic = (pos >> (BIT_S + BIT_R));
      for (int h = 0; h < TILE_H; ++h) {
          for (int w = 0; w < TILE_W; ++w) {
            out_reg[h * TILE_W + w] += img_tile[ic * TILE_SIZE + (h * STRIDE + x) * INPUT_TILE_W + w * STRIDE + y] * v; 
        }
      }
    }
    __syncthreads();
  }
 
  for (int h = 0; h < TILE_H; h++) {
    for (int w = 0; w < TILE_W; w++) {
      if (start_h + h < out_h && start_w + w < out_w) {
        output(batch_id, start_h + h, start_w + w, oc) = out_reg[h * TILE_W + w];
      
      }
    }
  }

#undef output
#undef input
}

template<typename ValueType, typename OffsetType, typename PositionType,
         int KERNEL_SIZE, int STRIDE, int TILE_IC, int TILE_H, int TILE_W> 
void Baseline(ValueType *input, OffsetType *offsets,
                          PositionType *position, ValueType *values,
                          ValueType *output,
                          int batch_size, int img_h, int img_w,
                          int in_channel, int out_channel, int kernel_size,
                          int stride, int padding, cudaStream_t cuda_stream) {
  int out_h = (img_h + padding * 2 - KERNEL_SIZE) / stride + 1;
  int out_w = (img_w + padding * 2 - KERNEL_SIZE) / stride + 1;


  constexpr int TILE_OC = 64;
  dim3 grid_size((out_channel + TILE_OC - 1) / TILE_OC,
                 batch_size * ((out_h + TILE_H - 1) / TILE_H) * ((out_w + TILE_W - 1) / TILE_W));
  dim3 block_size(TILE_OC);
  BaselineKernel<ValueType, OffsetType, PositionType,
              KERNEL_SIZE, STRIDE, TILE_IC, TILE_H, TILE_W, TILE_OC>
            <<<grid_size, block_size, 0, cuda_stream>>>(
              input, offsets, position, values, output,
              batch_size, img_h, img_w, in_channel, out_channel,
              kernel_size, stride, padding
  );
}

template void Baseline<float, int, int, 3, 1, 128, 2, 4>(
    float*, int*, int*, float*, float*,
    int, int, int, int, int, int, int, int, cudaStream_t);

