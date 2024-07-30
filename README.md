# ICS24_Ansor_AF_DS

## Introduction

This repository contains the figures, tables data and source code in the paper [ICS'24: Accelerated Auto-Tuning of GPU Kernels for Tensor Computations](https://dl.acm.org/doi/10.1145/3650200.3656626).

```
.
├── benchmarks
├── cal_var
├── default_ansor_benchmarks
├── figures
```


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


#### Build TVM

To build TVM from source and install the necessary Python packages, follow these steps:

Create conda environment:
```
conda create -n ansor python=3.10
conda activate ansor
conda install -c conda-forge xgboost=1.5.0 numpy decorator attrs tornado psutil cloudpickle pandas scipy pytest
```

#### Usage

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

Build:
```
cd PATH_TO_ANSOR_AF_DS
mkdir build
export TVM_HOME=$PWD; export PYTHONPATH=$TVM_HOME/python; cd ./build || exit 1; rm * -rf && cp ~/config.cmake ./ && cmake .. && make -j8
```

Back to the benchmarks folder and test:
```
cd PATH_TO_BASH_SCRIPTS
bash run_tests_times_conv.sh conv2d cuda 3 128 48 yolo 5 64 0.6
bash run_tests_times_conv.sh conv2d cuda 3 128 48 resnet 5 64 0.6
bash run_tests_times_mm.sh matmul cuda 3 128 48 5 64 0.6
```

##### Benchmark Ansor

Build:
```
cd PATH_TO_ANSOR
mkdir build
export TVM_HOME=$PWD; export PYTHONPATH=$TVM_HOME/python; cd ./build || exit 1; rm * -rf && cp ~/config.cmake ./ && cmake .. && make -j8
```

Back to the benchmarks folder and test:
```
cd PATH_TO_BASH_SCRIPTS
bash run_tests_times_mm.sh matmul cuda 3 128 48 1000 64
bash run_tests_times_conv.sh conv2d cuda 3 128 48 yolo 1000 64 0.6
bash run_tests_times_conv.sh conv2d cuda 3 128 48 resnet 1000 64 0.6
```


### 2. Calculate the variability
```
.
├── cal_var
```
This folder contains the script and data to calculate the variability of Ansor-AF-DS(in 2 minutes and after 1000-trials) and Ansor(1000-trials)

#### Calculate the variability
```
python3 calc_var.py

```


### 3. Default Ansor benchmarks
```
.
├── default_ansor_benchmarks
```
This folder contains the example scripts for the default Ansor.


### 4. Figures
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

