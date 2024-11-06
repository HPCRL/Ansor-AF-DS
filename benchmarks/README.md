## Description

Benchmark and log the end-to-end time

## Usage
```
bash run_tests_times_conv.sh conv2d cuda num_of_runs num_sm num_shared_mem network num_trials num_init_states threshold pz_num
```

For example:

```
bash run_tests_times_conv.sh conv2d cuda 1 128 48 yolo 5 64 0.6 0
```

It means running the Yolo network on Cuda with 128 SMs, 48 shared memory banks, 5 trials, 64 initial states, 0.6 threshold, and problem size index is 0.

###  Benchmark the Ansor-AF-DS

Please use `echo $TVM_HOME` to make sure you are benchmarking the correct branch

```
cd PATH_TO_BASH_SCRIPTS
bash run_tests_times_conv.sh conv2d cuda 3 128 48 yolo 5 64 0.6
bash run_tests_times_conv.sh conv2d cuda 3 128 48 resnet 5 64 0.6
bash run_tests_times_mm.sh matmul cuda 3 128 48 5 64 0.6
```

### Benchmark Ansor
```
cd PATH_TO_BASH_SCRIPTS
bash run_tests_times_mm.sh matmul cuda 3 128 48 1000 64
bash run_tests_times_conv.sh conv2d cuda 3 128 48 yolo 1000 64 0.6
bash run_tests_times_conv.sh conv2d cuda 3 128 48 resnet 1000 64 0.6
```
