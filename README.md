# ICS24_Ansor_AF_DS

## Introduction

This repository contains the figures, tables data and Ansor_AF_DS source in the paper ICS'24: "Accelerated Auto-Tuning of GPU Kernels for Tensor Computations".

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
Benchmarks for re-collecting the data, including the following benchmarks:

- [bench_roller](benchmarks/bench_roller/README.md) for evaluating the top50 performance of the rolle, tvm and Cuda toolkit(> 12.0) are required.
- [Ansor] source code of Ansor v0.9
- [Ansor-AF] source code of Ansor-AF
- [Ansor-DS] source code of Ansor-DS
- [Ansor-AF-DS] source code of Ansor-AF-DS
- [test](benchmarks/README.md) contains the scripts for re-collecting the data of Ansor, Ansor-AF, Ansor-DS, and Ansor-AF-DS. Please use the bash script to run the benchmarks.

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

