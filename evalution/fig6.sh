#!/bin/bash

bs='-1' # run all bs in [1, 4, 8, 16]
model='all' # run all model in ['mobilenet', 'yolov3', 'resnet50', 'vgg19']

if [ -n "$1" ]; then
    bs=$1
fi

if [ -n "$2" ]; then
    model=$2
fi



# build Sputnik/Tetris/cuDNN/cuSPARSE
echo '-- Build Sputnik/Tetris/cuDNN/cuSPARSE(about 3min)...'
./script/build_all.sh


# run Tetris/cuDNN/cuSPARSE/Sputnik, log store in ./log/other-log
echo '-- Run Tetris/cuDNN/cuSPARSE/Sputnik'
python3 conv2d_timing.py --batch_size $bs --model_name $model

# run taco ucf, log in ./log/UCF-log
echo '-- Run TACO-UCF'
./script/run_ucf.sh  $bs $model


# draw fig6, store in ./figure/fig6.png(pdf)
echo '-- Draw fig6'
python3 overall_speedup.py --batch_size $bs --model_name $model
