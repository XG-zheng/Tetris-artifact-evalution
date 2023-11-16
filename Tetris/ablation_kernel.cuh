
template<typename ValueType, typename OffsetType, typename PositionType,
         int KERNEL_SIZE, int STRIDE, int TILE_IC, int TILE_H, int TILE_W> 
void Baseline(ValueType *input, OffsetType *offsets,
                          PositionType *position, ValueType *values,
                          ValueType *output,
                          int batch_size, int img_h, int img_w,
                          int in_channel, int out_channel, int kernel_size,
                          int stride, int padding, cudaStream_t cuda_stream);

