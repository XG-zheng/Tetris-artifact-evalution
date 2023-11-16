#include <string>

#include "cuda_utils.h"
#include "spconv2d_utils.h"


template<typename ValueType, typename OffsetType, typename PositionType>
void CudaSparseConv2d(float *h_input, float *h_filter, float *h_output,
    int batch_size, int img_h, int img_w,
    int in_channel, int out_channel, int kernel_size,
    int stride, int padding, cudaStream_t cuda_stream,
    std::string layout, bool apply_reorder);

template<typename ValueType, typename OffsetType, typename PositionType,
        int KERNEL_SIZE, int STRIDE, int TILE_IC, int TILE_H, int TILE_W> 
void SparseConv2d(ValueType *input, OffsetType *offsets, OffsetType *stage_len,
                     PositionType *oc_permutation, PositionType *position, ValueType *values,
                     ValueType *output,
                     int batch_size, int img_h, int img_w,
                     int in_channel, int out_channel, int kernel_size,
                     int stride, int padding, cudaStream_t cuda_stream);
