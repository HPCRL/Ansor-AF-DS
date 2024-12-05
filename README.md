# ICS24_Ansor_AF_DS

## Introduction

This repository contains the figures, tables data and source code in the paper [ICS'24: Accelerated Auto-Tuning of GPU Kernels for Tensor Computations](https://dl.acm.org/doi/10.1145/3650200.3656626).

### 1. Benchmarks
```
.
├── benchmarks
```
[Benchmarks](https://github.com/HPCRL/Ansor-AF-DS/tree/main/benchmarks) for re-collecting the data, including the following benchmarks:

- [bench_roller](https://github.com/HPCRL/bench_roller/tree/main) for evaluating the top50 performance of the rolle, tvm and Cuda toolkit(> 12.0) are required.
- [Ansor] source code of Ansor v0.9
- [Ansor-AF] source code of Ansor-AF
- [Ansor-DS] source code of Ansor-DS
- [Ansor-AF-DS] source code of Ansor-AF-DS
- [test](benchmarks/README.md) contains the scripts for re-collecting the data of Ansor, Ansor-AF, Ansor-DS, and Ansor-AF-DS. Please use the bash script to run the benchmarks.


#### Create Conda environment

To build TVM from source and install the necessary Python packages, follow these steps:

Create conda environment:
```
conda create -n ansor python=3.10
conda activate ansor
conda install -c conda-forge xgboost=1.5.0 numpy decorator attrs tornado psutil cloudpickle pandas scipy pytest
```

The conda environment setting was from the [official documentation of TVM](https://tvm.apache.org/docs/v0.9.0/install/from_source.html#developers-get-source-from-github)


#### Benchmark script setting and explanation

Use the following command to run tests:

bash run_tests_times_conv.sh conv2d cuda num_of_runs num_sm num_shared_mem network num_trials num_init_states threshold pz_num

For example:

```
bash run_tests_times_conv.sh conv2d cuda 1 128 48 yolo 5 64 0.6 0
```

This command runs the YOLO network on CUDA with the following parameters:
- 128 Streaming Multiprocessors (SMs)
- 48k shared memory
- 5 start points
- 64 initial configurations for building the model
- 0.6 threshold
- problem size 0 (leave it empty to test all the problem sizes)


#####  Benchmark the Ansor-AF-DS

Before proceeding, please make sure that both **CUDA** and **LLVM** are installed on your system. You can verify this by running the following commands in your terminal:

```
llvm-config --version
nvcc --version
```

Build Ansor-AF-DS first:
```
git clone git@github.com:HPCRL/Ansor-AF-DS.git --recursive
cd Ansor-AF-DS/benchmarks/Ansor_AF_DS

export TVM_HOME=$PWD && export PYTHONPATH=$TVM_HOME/python
mkdir -p build && cd ./build

cp "$TVM_HOME/cmake/config.cmake" ./

sed -i 's/set(USE_CUDA OFF)/set(USE_CUDA ON)/' config.cmake
sed -i 's/set(USE_LLVM OFF)/set(USE_LLVM ON)/' config.cmake

cmake ..
make -j8
```

Then go to the benchmarks folder and test: (The following setting is used for NVIDIA RTX 4090; please refer to the previous explanation and change it for your GPUs)
```
cd ../../
bash run_tests_times_conv.sh conv2d cuda 3 128 48 yolo 5 64 0.6
bash run_tests_times_conv.sh conv2d cuda 3 128 48 resnet 5 64 0.6
bash run_tests_times_mm.sh matmul cuda 3 128 48 5 64 0.6
```

##### Benchmark Ansor

Build TVM first:
```
cd Ansor-AF-DS/benchmarks/Ansor

export TVM_HOME=$PWD && export PYTHONPATH=$TVM_HOME/python
mkdir -p build && cd ./build

cp "$TVM_HOME/cmake/config.cmake" ./

sed -i 's/set(USE_CUDA OFF)/set(USE_CUDA ON)/' config.cmake
sed -i 's/set(USE_LLVM OFF)/set(USE_LLVM ON)/' config.cmake

cmake ..
make -j8
```

Then go to the benchmarks folder and test: (The following setting is used for NVIDIA RTX 4090; please refer to the previous explanation and change it for your GPUs)
```
cd ../../
bash run_tests_times_mm.sh matmul cuda 3 128 48 1000 64
bash run_tests_times_conv.sh conv2d cuda 3 128 48 yolo 1000 64 0.6
bash run_tests_times_conv.sh conv2d cuda 3 128 48 resnet 1000 64 0.6
```


### 2. Reproduce the variability data
```
.
├── cal_var
```
This folder contains the script and data to calculate the variability of Ansor-AF-DS(in 2 minutes and after 1000-trials) and Ansor(1000-trials)

#### Calculate the variability
```
python3 calc_var.py
```

### 3. Reproduce the figures
```
.
├── figures
```
This folder contains the scripts for reproducing the figures in the paper.

#### Draw all the figures
```
bash plot.sh
```

#### Scatter plot

```
python3 plot_scatter.py
```

#### Cudnn VS Ansor
```
python3 cudnn-ansor3090.py 
python3 cudnn-ansor4090.py 
```

#### Ablation 1

```
python3 plot_stack_ablation1.py
```

#### Ablation 2
```
python3 plot_stack_ablation2.py
```

#### Performance plot scripts
```
python3 plot_all_perf_stack3090.py
python3 plot_all_perf_stack4090.py
```

#### Variability plot scripts
```
python3 plot_var_perf_3090.py 
python3 plot_var_perf_4090.py
```

## Citation
If you found it useful, please consider citing our paper:
```
@inproceedings{10.1145/3650200.3656626,
author = {Li, Chendi and Xu, Yufan and Saravani, Sina Mahdipour and Sadayappan, Ponnuswamy},
title = {Accelerated Auto-Tuning of GPU Kernels for Tensor Computations},
year = {2024},
isbn = {9798400706103},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3650200.3656626},
doi = {10.1145/3650200.3656626},
abstract = {TVM is a state-of-the-art auto-tuning compiler for the synthesis of high-performance implementations of tensor computations. However, an extensive search in the vast design space via thousands of compile-execute trials is often needed to identify high-performance code versions, leading to high auto-tuning time. This paper develops new performance modeling and design space exploration strategies to accelerate the code optimization process within TVM. Experimental evaluation on a number of matrix-matrix multiplication and 2D convolution kernels demonstrates about an order-of-magnitude improvement in auto-tuning time to achieve the same level of code performance.},
booktitle = {Proceedings of the 38th ACM International Conference on Supercomputing},
pages = {549–561},
numpages = {13},
keywords = {Auto-tuning, Design space exploration, GPU kernel optimization, Neural networks, Performance modeling, Tile-size optimization},
location = {Kyoto, Japan},
series = {ICS '24}
}
```

## License


This project is licensed under the Apache License, Version 2.0 (the "License").  
You may not use this file except in compliance with the License.  

You may obtain a copy of the License at:

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
