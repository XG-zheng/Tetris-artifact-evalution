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

struct VecHashFunc {
  size_t operator()(const std::vector<int32_t> &k) const {
    std::string h = "";
    for (auto &it : k) {
      h += std::to_string(it) + "-";
    }
    return std::hash<std::string>()(h);
  }
};

template<typename ValueType, typename OffsetType, typename PositionType,
         int KERNEL_SIZE, int STRIDE, int TILE_IC, int TILE_H, int TILE_W> 
float TimingTetrisSparseConv2d(ValueType *input, OffsetType *offsets, OffsetType *stage_len,
                          PositionType * oc_permutation, PositionType *position, ValueType *values,
                          ValueType *output,
                          int batch_size, int img_h, int img_w,
                          int in_channel, int out_channel, int kernel_size,
                          int stride, int padding, cudaStream_t cuda_stream,
                          std::string layout, bool apply_reorder) {
  cudaDeviceSynchronize();
  float elapsedTime;
  cudaEvent_t start, end;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&end));
    
  for (int i = 0; i < WARM; ++i) {
    SparseConv2d<ValueType, OffsetType, PositionType,
                        KERNEL_SIZE, STRIDE, TILE_IC, TILE_H, TILE_W>(
                          input, offsets, stage_len, oc_permutation, position, values, output,
                          batch_size, img_h, img_w, in_channel, out_channel,
                          kernel_size, stride, padding, cuda_stream
                        );
  }
  cudaDeviceSynchronize();
  checkCudaErrors(cudaEventRecord(start, cuda_stream));

  for (int i = 0; i < REPEAT; ++i) {
    SparseConv2d<ValueType, OffsetType, PositionType,
                        KERNEL_SIZE, STRIDE, TILE_IC, TILE_H, TILE_W>(
                          input, offsets, stage_len, oc_permutation, position, values, output,
                          batch_size, img_h, img_w, in_channel, out_channel,
                          kernel_size, stride, padding, cuda_stream
                        );
  }
  
  checkCudaErrors(cudaEventRecord(end, cuda_stream));
  checkCudaErrors(cudaEventSynchronize(start));
  checkCudaErrors(cudaEventSynchronize(end));
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime , start , end));
  return elapsedTime/REPEAT;
}


template<typename ValueType, typename OffsetType, typename PositionType>
float GetTetrisSparseConv2dTime(ValueType *d_input, ValueType *h_filter,
                  ValueType *d_output,
                  int batch_size, int img_h, int img_w,
                  int in_channel, int out_channel, int kernel_size,
                  int stride, int padding, cudaStream_t cuda_stream,
                  std::string layout, bool apply_reorder
                  ) {

  using TimingFuncType = std::function<float(ValueType *, OffsetType *, OffsetType *,
    PositionType *, PositionType *, ValueType *,
    ValueType *,
    int, int, int,
    int, int, int,
    int, int, cudaStream_t,
    std::string, bool)>;


  static std::unordered_map<std::vector<int32_t>, TimingFuncType, VecHashFunc> funcs = {
{{1, 1, 128, 1, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 1, 7>},
{{1, 1, 64, 6, 14}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 64, 6, 14>},
{{1, 1, 128, 1, 14}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 1, 14>},
{{1, 2, 256, 1, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 2, 256, 1, 4>},
{{3, 1, 256, 1, 3}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 256, 1, 3>},
{{3, 1, 128, 1, 9}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 128, 1, 9>},
{{1, 1, 256, 1, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 1, 7>},
{{1, 1, 128, 2, 12}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 2, 12>},
{{1, 1, 128, 4, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 4, 4>},
{{1, 1, 128, 2, 9}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 2, 9>},
{{1, 1, 32, 2, 13}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 32, 2, 13>},
{{1, 1, 64, 4, 6}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 64, 4, 6>},
{{3, 1, 64, 1, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 64, 1, 7>},
{{3, 1, 32, 3, 10}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 32, 3, 10>},
{{1, 1, 256, 2, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 2, 7>},
{{1, 2, 256, 1, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 2, 256, 1, 7>},
{{3, 1, 64, 4, 6}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 64, 4, 6>},
{{3, 1, 64, 2, 10}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 64, 2, 10>},
{{3, 2, 32, 2, 9}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 2, 32, 2, 9>},
{{1, 1, 256, 1, 11}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 1, 11>},
{{3, 2, 256, 1, 2}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 2, 256, 1, 2>},
{{3, 1, 128, 2, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 128, 2, 8>},
{{1, 1, 32, 2, 6}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 32, 2, 6>},
{{3, 2, 32, 3, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 2, 32, 3, 8>},
{{3, 2, 32, 2, 5}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 2, 32, 2, 5>},
{{1, 1, 32, 4, 6}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 32, 4, 6>},
{{3, 1, 128, 1, 3}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 128, 1, 3>},
{{1, 2, 128, 1, 14}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 2, 128, 1, 14>},
{{3, 1, 64, 4, 14}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 64, 4, 14>},
{{3, 1, 64, 1, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 64, 1, 8>},
{{3, 1, 32, 2, 13}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 32, 2, 13>},
{{1, 1, 256, 3, 6}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 3, 6>},
{{1, 2, 128, 1, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 2, 128, 1, 7>},
{{3, 2, 64, 1, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 2, 64, 1, 8>},
{{1, 1, 128, 4, 6}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 4, 6>},
{{3, 1, 128, 3, 3}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 128, 3, 3>},
{{3, 1, 128, 1, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 128, 1, 7>},
{{1, 1, 128, 3, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 3, 8>},
{{1, 1, 128, 1, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 1, 8>},
{{1, 1, 32, 2, 10}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 32, 2, 10>},
{{3, 1, 64, 2, 14}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 64, 2, 14>},
{{1, 1, 32, 4, 5}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 32, 4, 5>},
{{1, 1, 128, 4, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 4, 7>},
{{3, 1, 32, 8, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 32, 8, 8>},
{{3, 1, 32, 2, 10}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 32, 2, 10>},
{{1, 2, 128, 1, 6}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 2, 128, 1, 6>},
{{3, 2, 32, 1, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 2, 32, 1, 7>},
{{1, 1, 64, 3, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 64, 3, 8>},
{{3, 2, 128, 2, 2}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 2, 128, 2, 2>},
{{1, 1, 64, 8, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 64, 8, 8>},
{{1, 1, 128, 2, 6}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 2, 6>},
{{1, 1, 256, 1, 5}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 1, 5>},
{{1, 1, 64, 1, 6}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 64, 1, 6>},
{{1, 1, 256, 1, 10}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 1, 10>},
{{1, 1, 32, 3, 9}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 32, 3, 9>},
{{3, 1, 64, 4, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 64, 4, 4>},
{{1, 1, 256, 1, 3}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 1, 3>},
{{3, 1, 64, 2, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 64, 2, 8>},
{{1, 1, 128, 2, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 2, 4>},
{{3, 1, 128, 1, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 128, 1, 8>},
{{3, 1, 128, 3, 6}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 128, 3, 6>},
{{3, 1, 32, 1, 2}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 32, 1, 2>},
{{1, 1, 32, 3, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 32, 3, 7>},
{{1, 1, 64, 3, 9}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 64, 3, 9>},
{{1, 1, 128, 2, 14}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 2, 14>},
{{3, 2, 64, 2, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 2, 64, 2, 4>},
{{3, 1, 32, 3, 9}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 32, 3, 9>},
{{3, 2, 256, 1, 3}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 2, 256, 1, 3>},
{{3, 2, 32, 3, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 2, 32, 3, 4>},
{{1, 1, 64, 2, 2}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 64, 2, 2>},
{{1, 1, 64, 1, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 64, 1, 7>},
{{1, 1, 256, 3, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 3, 8>},
{{3, 1, 32, 2, 9}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 32, 2, 9>},
{{1, 1, 64, 3, 3}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 64, 3, 3>},
{{1, 1, 256, 1, 6}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 1, 6>},
{{3, 1, 256, 1, 5}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 256, 1, 5>},
{{1, 1, 256, 2, 6}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 2, 6>},
{{1, 1, 256, 3, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 3, 7>},
{{3, 1, 64, 4, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 64, 4, 8>},
{{1, 1, 64, 1, 10}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 64, 1, 10>},
{{1, 1, 256, 1, 2}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 1, 2>},
{{3, 1, 128, 2, 9}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 128, 2, 9>},
{{1, 2, 128, 1, 10}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 2, 128, 1, 10>},
{{3, 1, 128, 2, 5}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 128, 2, 5>},
{{1, 1, 128, 1, 12}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 1, 12>},
{{1, 1, 256, 2, 9}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 2, 9>},
{{1, 1, 64, 1, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 64, 1, 8>},
{{1, 1, 256, 4, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 4, 4>},
{{3, 2, 32, 2, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 2, 32, 2, 4>},
{{3, 1, 128, 3, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 128, 3, 7>},
{{1, 1, 256, 1, 14}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 1, 14>},
{{3, 1, 128, 4, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 128, 4, 4>},
{{3, 1, 64, 4, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 64, 4, 7>},
{{1, 1, 256, 4, 6}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 4, 6>},
{{3, 1, 128, 1, 2}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 128, 1, 2>},
{{3, 1, 64, 2, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 64, 2, 7>},
{{1, 1, 256, 3, 3}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 3, 3>},
{{1, 1, 128, 2, 5}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 2, 5>},
{{3, 1, 64, 6, 10}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 64, 6, 10>},
{{3, 1, 32, 5, 10}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 32, 5, 10>},
{{1, 1, 64, 3, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 64, 3, 4>},
{{1, 1, 64, 4, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 64, 4, 4>},
{{1, 1, 256, 3, 5}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 3, 5>},
{{1, 1, 256, 1, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 1, 4>},
{{1, 1, 64, 3, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 64, 3, 7>},
{{3, 2, 64, 2, 3}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 2, 64, 2, 3>},
{{3, 1, 64, 3, 3}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 64, 3, 3>},
{{3, 1, 128, 1, 14}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 128, 1, 14>},
{{3, 1, 256, 1, 2}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 256, 1, 2>},
{{1, 2, 256, 1, 6}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 2, 256, 1, 6>},
{{1, 1, 256, 2, 2}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 2, 2>},
{{1, 1, 128, 4, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 4, 8>},
{{3, 2, 64, 2, 2}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 2, 64, 2, 2>},
{{1, 1, 32, 2, 9}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 32, 2, 9>},
{{3, 1, 64, 2, 5}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 64, 2, 5>},
{{3, 1, 64, 3, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 64, 3, 7>},
{{1, 1, 256, 2, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 2, 8>},
{{1, 1, 128, 3, 14}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 3, 14>},
{{3, 1, 32, 8, 10}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 32, 8, 10>},
{{3, 1, 128, 2, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 128, 2, 4>},
{{3, 2, 32, 4, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 2, 32, 4, 4>},
{{3, 2, 32, 2, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 2, 32, 2, 7>},
{{3, 1, 32, 4, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 32, 4, 4>},
{{3, 1, 256, 2, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 256, 2, 4>},
{{1, 1, 32, 1, 12}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 32, 1, 12>},
{{3, 2, 32, 2, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 2, 32, 2, 8>},
{{7, 2, 32, 2, 3}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 7, 2, 32, 2, 3>},
{{1, 1, 256, 2, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 2, 4>},
{{1, 2, 128, 1, 2}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 2, 128, 1, 2>},
{{3, 2, 64, 1, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 2, 64, 1, 4>},
{{3, 1, 64, 2, 9}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 64, 2, 9>},
{{1, 1, 32, 2, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 32, 2, 8>},
{{3, 2, 128, 1, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 2, 128, 1, 7>},
{{1, 1, 256, 2, 3}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 2, 3>},
{{1, 1, 64, 4, 5}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 64, 4, 5>},
{{7, 2, 32, 3, 3}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 7, 2, 32, 3, 3>},
{{1, 2, 256, 1, 2}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 2, 256, 1, 2>},
{{1, 1, 128, 3, 3}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 3, 3>},
{{7, 2, 32, 3, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 7, 2, 32, 3, 4>},
{{3, 1, 32, 3, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 32, 3, 7>},
{{1, 1, 128, 3, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 3, 4>},
{{1, 1, 128, 1, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 1, 4>},
{{3, 2, 128, 1, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 2, 128, 1, 4>},
{{1, 1, 64, 2, 6}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 64, 2, 6>},
{{1, 1, 128, 1, 10}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 1, 10>},
{{1, 1, 32, 1, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 32, 1, 8>},
{{3, 2, 128, 1, 2}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 2, 128, 1, 2>},
{{1, 1, 256, 4, 5}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 4, 5>},
{{3, 1, 64, 5, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 64, 5, 7>},
{{1, 1, 128, 2, 10}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 2, 10>},
{{1, 2, 256, 1, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 2, 256, 1, 8>},
{{3, 1, 32, 4, 6}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 32, 4, 6>},
{{1, 1, 64, 1, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 64, 1, 4>},
{{7, 2, 32, 2, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 7, 2, 32, 2, 4>},
{{1, 1, 256, 1, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 1, 8>},
{{1, 1, 64, 1, 11}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 64, 1, 11>},
{{3, 2, 128, 1, 6}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 2, 128, 1, 6>},
{{3, 1, 128, 2, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 128, 2, 7>},
{{3, 2, 64, 1, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 2, 64, 1, 7>},
{{3, 1, 128, 3, 5}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 128, 3, 5>},
{{1, 1, 64, 2, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 64, 2, 8>},
{{3, 1, 32, 2, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 32, 2, 4>},
{{3, 1, 32, 3, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 32, 3, 8>},
{{3, 1, 64, 2, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 64, 2, 4>},
{{3, 2, 32, 1, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 2, 32, 1, 8>},
{{1, 1, 128, 2, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 2, 8>},
{{1, 1, 256, 3, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 3, 4>},
{{1, 1, 128, 1, 6}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 1, 6>},
{{1, 1, 32, 2, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 32, 2, 4>},
{{1, 1, 64, 2, 4}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 64, 2, 4>},
{{1, 1, 128, 2, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 2, 7>},
{{3, 1, 128, 4, 6}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 128, 4, 6>},
{{1, 1, 128, 4, 5}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 4, 5>},
{{1, 1, 64, 1, 12}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 64, 1, 12>},
{{1, 1, 32, 1, 2}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 32, 1, 2>},
{{3, 2, 128, 2, 3}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 2, 128, 2, 3>},
{{3, 1, 64, 3, 14}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 64, 3, 14>},
{{3, 1, 256, 2, 3}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 256, 2, 3>},
{{1, 1, 128, 1, 3}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 1, 3>},
{{3, 1, 32, 6, 14}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 32, 6, 14>},
{{3, 1, 128, 1, 5}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 128, 1, 5>},
{{3, 1, 64, 3, 8}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 64, 3, 8>},
{{1, 1, 128, 3, 7}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 128, 3, 7>},
{{1, 1, 256, 2, 5}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 1, 1, 256, 2, 5>},
{{3, 1, 128, 2, 3}, TimingTetrisSparseConv2d<ValueType, OffsetType, PositionType, 3, 1, 128, 2, 3>},

  };
  OffsetType *d_stage_len, *d_offsets;
  ValueType *d_values;
  PositionType *d_position, *d_oc_permutation;

  float best_time = 1e18;
  std::vector<int32_t> best_config;
  TimingFuncType best_func;
  for (auto iter : funcs) {
    if (iter.first[0] != kernel_size || iter.first[1] != stride) {
      continue;
    }

    int tile_ic = iter.first[2];

    SparseFilter<ValueType, OffsetType, PositionType> sparse_filter(h_filter, in_channel, out_channel,
      kernel_size, tile_ic, apply_reorder);
    d_offsets = sparse_filter.GetDeviceOffsets();
    d_stage_len = sparse_filter.GetDeviceStageLen();
    d_values = sparse_filter.GetDeviceValues();
    d_position = sparse_filter.GetDevicePosition();
    d_oc_permutation = sparse_filter.GetDeviceOCPermutation();

    float cur_time = (iter.second)(
      d_input, d_offsets, d_stage_len, d_oc_permutation, d_position, d_values, d_output,
      batch_size, img_h, img_w, in_channel, out_channel,
      kernel_size, stride, padding, cuda_stream, layout, apply_reorder
    );

    if (cur_time < best_time) {
      best_time = cur_time;
      best_config = iter.first;
      best_func = iter.second;
    }
  }

  // best func
  int tile_ic = best_config[2];
  SparseFilter<ValueType, OffsetType, PositionType> sparse_filter(h_filter, in_channel, out_channel,
    kernel_size, tile_ic, apply_reorder);
  d_offsets = sparse_filter.GetDeviceOffsets();
  d_stage_len = sparse_filter.GetDeviceStageLen();
  d_values = sparse_filter.GetDeviceValues();
  d_position = sparse_filter.GetDevicePosition();
  d_oc_permutation = sparse_filter.GetDeviceOCPermutation();
  best_time = (best_func)(
    d_input, d_offsets, d_stage_len, d_oc_permutation, d_position, d_values, d_output,
      batch_size, img_h, img_w, in_channel, out_channel,
      kernel_size, stride, padding, cuda_stream, layout, apply_reorder
  );

  return best_time;
}


template<typename ValueType, typename OffsetType, typename PositionType>
void CudaSparseConv2d(float *h_input, float *h_filter, float *h_output,
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

  float tetris_time = GetTetrisSparseConv2dTime<ValueType, OffsetType, PositionType>
                            (d_input, h_filter, d_output,
                              batch_size, img_h, img_w,
                              in_channel, out_channel, kernel_size,
                              stride, padding, cuda_stream,
                              layout, apply_reorder);

  std::cout<<"Time of Spconv2d :"<< tetris_time <<"ms"<<"\n";
  std::cout<<"FLOPS of Spconv2d :"<< (long long)out_h * out_w * batch_size * out_channel * in_channel * kernel_size * kernel_size * 2 * 1000/(tetris_time)/1024/1024/1024/1024 << "T\n";
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

  CudaSparseConv2d<float, int, int>(h_input, h_filter, h_sparse, batch_size, img_h, img_w,
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
