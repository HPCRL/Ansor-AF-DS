## Description

benchmark and log the end-to-end time


## Usage


Please build tvm from source and install necessary python packages.

Create conda environment:
```
conda create -n ansor python=3.10
conda activate ansor
conda install -c conda-forge xgboost=1.5.0 numpy decorator attrs tornado psutil cloudpickle pandas scipy pytest
```

bash run_tests_times_conv.sh conv2d cuda num_of_runs num_sm num_shared_mem network num_trials num_init_states threshold pz_num

for example:

```
bash run_tests_times_conv.sh conv2d cuda 1 128 48 yolo 5 64 0.6 0
```

means run the yolo network on cuda with 128 SMs, 48 shared memory banks, 5 trials, 64 initial states, 0.6 threshold, and problem size 0

Benchmark for different branch:

###  benchmark the Ansor-AF-DS
build
```
cd PATH_TO_ANSOR_AF_DS
mkdir build
export TVM_HOME=$PWD; export PYTHONPATH=$TVM_HOME/python; cd ./build || exit 1; rm * -rf && cp ~/config.cmake ./ && cmake .. && make -j8
```

back to the benchmarks folder and run
```
bash run_tests_times_conv.sh conv2d cuda 3 128 48 yolo 5 64 0.6
bash run_tests_times_conv.sh conv2d cuda 3 128 48 resnet 5 64 0.6
bash run_tests_times_mm.sh matmul cuda 3 128 48 5 64 0.6
```

### Benchmark Ansor

build
```
cd PATH_TO_ANSOR
mkdir build
export TVM_HOME=$PWD; export PYTHONPATH=$TVM_HOME/python; cd ./build || exit 1; rm * -rf && cp ~/config.cmake ./ && cmake .. && make -j8
```

back to the benchmarks folder and run
```
bash run_tests_times_mm.sh matmul cuda 3 128 48 1000 64
bash run_tests_times_conv.sh conv2d cuda 3 128 48 yolo 1000 64 0.6
bash run_tests_times_conv.sh conv2d cuda 3 128 48 resnet 1000 64 0.6
```
