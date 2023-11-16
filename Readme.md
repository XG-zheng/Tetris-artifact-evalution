This is a repository for an artifact evaluation of "Tetris: Accelerating Sparse Convolution by Exploiting Memory Reuse on GPU".

## ENV
- V100-PCIE-32GB

## Build baseline

The evalution depend on `Unified-Convolution-Framework` and `sputnik`. To get all source code, run the following code:
```
git clone --recursive [this-repo-url]
```


### Sputnik
```
cd /Tetris-artifact-evalution/sputnik
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHS="70;75"
make -j12
export LD_LIBRARY_PATH=$(pwd)/sputnik/:$LD_LIBRARY_PATH
```

### TACO-UCF

- build TACO-UCF
```
cd /Tetris-artifact-evalution/Unified-Convolution-Framework
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCUDA=ON ..
make -j12
export LD_LIBRARY_PATH=$(pwd)/lib/:$LD_LIBRARY_PATH
```

- build sparse conv
```
cd /Tetris-artifact-evalution/Unified-Convolution-Framework/benchmark/filter_sparse_convolution/gpu/
make
```

### Install python dependency
```
cd /Tetris-artifact-evalution/evalution
python3 -m pip install -r requirement.txt 
```

## Test
First, copy the test dataset to TACO-UCF repo
```
cp -r /Tetris-artifact-evalution/UCF-Dataset  /Tetris-artifact-evalution/Unified-Convolution-Framework/benchmark/filter_sparse_convolution/gpu/
```

Run the following command to enter specific dir. 
```
cd /Tetris-artifact-evalution/evalution
mkdir ../Tetris/build
mkdir ./figure
```
To test specific config, run the following command:

`./fig6.sh BATCH_SIZE MODEL_NAME` 

BATCH\_SIZE in [1, 4, 8, 16]

MODEL\_NAME in [vgg19, mobilnet, yolov3, resnet50]

**Example0：** vgg19 with batch size 1

`./fig6.sh 1 vgg19`

**Example1:** test all models with all batch size 

`./fig6.sh`
