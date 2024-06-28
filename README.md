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

Please build tvm from source and install necessary python packages.

create conda environment:
```
conda create -n ansor python=3.10
conda activate ansor
conda install -c conda-forge xgboost=1.5.0 numpy decorator attrs tornado psutil cloudpickle pandas scipy pytest
```

### 2. Calculate the variability
```
.
├── cal_var
```
This folder contains the script and data to calculate the variability of Ansor-AF-DS(in 2 minutes and after 1000-trials) and Ansor(1000-trials)


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

