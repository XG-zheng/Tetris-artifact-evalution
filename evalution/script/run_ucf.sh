#!/bin/bash

base_log_dir=$(pwd)'/log/UCF-log'

ucf_code_base='../Unified-Convolution-Framework/benchmark/filter_sparse_convolution/gpu/'
cd ${ucf_code_base}

base_dataset_dir='./UCF-Dataset'

if [[ $1 == "-1" ]]
then
    batch_size=(1 4 8 16)
else
    batch_size=($1)
fi

if [[ $2 == "all" ]]
then
    models=('mobilenet' 'yolov3' 'resnet50' 'vgg19')
else
    models=($2)
fi

for batch in ${batch_size[@]}
do
    for model in ${models[@]}
    do
        model_dataset_dir=$base_dataset_dir'/'$model
        model_log_dir=$base_log_dir'/'$model
        echo $model_log_dir
        if [ ! -d $model_log_dir ]; then
            mkdir  -p $model_log_dir
        fi
        all_ckpts_log_path=$model_log_dir'/batch-'$batch'.log'
        rm -f $all_ckpts_log_path
        for ckpt in $(ls $model_dataset_dir)
        do
            model_cfg=$model_dataset_dir'/'$ckpt'/model.cfg'
            model_log_path=$model_log_dir'/'$ckpt'-batch-'$batch'.log'
            # echo $model_cfg
            i=0
            rm -f $model_log_path
            while read cfg; do
                ((i+=1));
                echo -n Layer c$i " / " >> $model_log_path;
                echo -n " " >> $model_log_path;
                cmd=${cfg/ 1/ $batch};
                echo $cmd >> $model_log_path;
                ./filter_sparse_2dconv_gpu $cmd 90 NMPQ NCHW UUUU MRSC UUUC bench c$i >> $model_log_path;
            done < $model_cfg
            cat $model_log_path >> $all_ckpts_log_path
            echo "#####################" >> $all_ckpts_log_path
        done
    done
done
