codebase_dir='../Tetris/'
if [ -n "$1" ]; then
  codebase_dir=$1
fi

rm -f ${codebase_dir}build/*.o
nvcc -c --generate-code=arch=compute_70,code=sm_70 -O3 ${codebase_dir}tensor_utils.cc -o ${codebase_dir}build/tensor_utils.o
nvcc -c --generate-code=arch=compute_70,code=sm_70 -O3 ${codebase_dir}cuda_utils.cc -o ${codebase_dir}build/cuda_utils.o
nvcc -c --generate-code=arch=compute_70,code=sm_70 -O3 ${codebase_dir}spconv2d_utils.cc -o ${codebase_dir}build/spconv2d_utils.o
nvcc -c --generate-code=arch=compute_70,code=sm_70 -O3 ${codebase_dir}ablation_kernel.cu -o ${codebase_dir}build/ablation_kernel.o
nvcc -c --generate-code=arch=compute_70,code=sm_70 -O3 ${codebase_dir}spconv2d_kernel.cu -o ${codebase_dir}build/spconv2d_kernel.o
nvcc -O3 --generate-code=arch=compute_70,code=sm_70 ${codebase_dir}baseline.cc -lcublas -lcudnn ${codebase_dir}build/cuda_utils.o ${codebase_dir}build/tensor_utils.o ${codebase_dir}build/spconv2d_utils.o  ${codebase_dir}build/ablation_kernel.o ${codebase_dir}build/spconv2d_kernel.o -o ${codebase_dir}build/baseline_test.o
nvcc -O3 --generate-code=arch=compute_70,code=sm_70 ${codebase_dir}spf.cc -lcublas -lcudnn ${codebase_dir}build/cuda_utils.o ${codebase_dir}build/tensor_utils.o ${codebase_dir}build/spconv2d_utils.o  ${codebase_dir}build/ablation_kernel.o ${codebase_dir}build/spconv2d_kernel.o -o ${codebase_dir}build/spf_test.o
nvcc -O3 --generate-code=arch=compute_70,code=sm_70 -DBANK_OPTIMIZE ${codebase_dir}spf.cc -lcublas -lcudnn ${codebase_dir}build/cuda_utils.o ${codebase_dir}build/tensor_utils.o ${codebase_dir}build/spconv2d_utils.o  ${codebase_dir}build/ablation_kernel.o ${codebase_dir}build/spconv2d_kernel.o -o ${codebase_dir}build/bank_sr_test.o
nvcc -O3 --generate-code=arch=compute_70,code=sm_70 -DBANK_OPTIMIZE -DREORDER_OUT_CHANNEL ${codebase_dir}spf.cc -lcublas -lcudnn ${codebase_dir}build/cuda_utils.o ${codebase_dir}build/tensor_utils.o ${codebase_dir}build/spconv2d_utils.o  ${codebase_dir}build/ablation_kernel.o ${codebase_dir}build/spconv2d_kernel.o -o ${codebase_dir}build/filter_gr_test.o
nvcc -O3 --generate-code=arch=compute_70,code=sm_70 -DBANK_OPTIMIZE -DREORDER_OUT_CHANNEL ${codebase_dir}spconv2d.cc -lcublas -lcudnn ${codebase_dir}build/cuda_utils.o ${codebase_dir}build/tensor_utils.o ${codebase_dir}build/spconv2d_utils.o  ${codebase_dir}build/ablation_kernel.o ${codebase_dir}build/spconv2d_kernel.o -o ${codebase_dir}build/para_test.o
nvcc -O3 --generate-code=arch=compute_70,code=sm_70 -DREORDER_OUT_CHANNEL ${codebase_dir}spf.cc -lcublas -lcudnn ${codebase_dir}build/cuda_utils.o ${codebase_dir}build/tensor_utils.o ${codebase_dir}build/spconv2d_utils.o  ${codebase_dir}build/ablation_kernel.o ${codebase_dir}build/spconv2d_kernel.o -o ${codebase_dir}build/wo_bank_sr_test.o
nvcc -O3 --generate-code=arch=compute_70,code=sm_70 -DBANK_OPTIMIZE -DREORDER_OUT_CHANNEL ${codebase_dir}spf.cc -lcublas -lcudnn ${codebase_dir}build/cuda_utils.o ${codebase_dir}build/tensor_utils.o ${codebase_dir}build/spconv2d_utils.o  ${codebase_dir}build/ablation_kernel.o ${codebase_dir}build/spconv2d_kernel.o -o ${codebase_dir}build/wi_bank_sr_test.o
