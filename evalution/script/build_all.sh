codebase_dir='../Tetris/'
if [ -n "$1" ]; then
  codebase_dir=$1
fi

rm -f ${codebase_dir}build/*.o
nvcc -c --generate-code=arch=compute_70,code=sm_70 -O3 ${codebase_dir}tensor_utils.cc -o ${codebase_dir}build/tensor_utils.o
nvcc -c --generate-code=arch=compute_70,code=sm_70 -O3 ${codebase_dir}cuda_utils.cc -o ${codebase_dir}build/cuda_utils.o
nvcc -O3 --generate-code=arch=compute_70,code=sm_70 -lcusparse -lcudnn ${codebase_dir}build/cuda_utils.o ${codebase_dir}build/tensor_utils.o ${codebase_dir}cusparse_test.cc -o ${codebase_dir}build/cusparse_test.o
nvcc -O3 --generate-code=arch=compute_70,code=sm_70 -lcublas -lcudnn ${codebase_dir}build/cuda_utils.o ${codebase_dir}build/tensor_utils.o ${codebase_dir}cudnn_test.cc -o ${codebase_dir}build/cudnn_test.o
nvcc -O3 --generate-code=arch=compute_70,code=sm_70 -L../sputnik/build/sputnik ${codebase_dir}sputnik_test.cc -lsputnik -lcublas -lcudnn ${codebase_dir}build/cuda_utils.o ${codebase_dir}build/tensor_utils.o -o ${codebase_dir}build/sputnik_test.o 
nvcc -c --generate-code=arch=compute_70,code=sm_70 -O3 ${codebase_dir}spconv2d_utils.cc -o ${codebase_dir}build/spconv2d_utils.o
nvcc -c --generate-code=arch=compute_70,code=sm_70 -O3 ${codebase_dir}spconv2d_kernel.cu -o ${codebase_dir}build/spconv2d_kernel.o
nvcc -O3 --generate-code=arch=compute_70,code=sm_70 ${codebase_dir}spconv2d.cc -lcublas -lcudnn ${codebase_dir}build/cuda_utils.o ${codebase_dir}build/tensor_utils.o ${codebase_dir}build/spconv2d_utils.o ${codebase_dir}build/spconv2d_kernel.o -o ${codebase_dir}build/spconv2d_test.o
