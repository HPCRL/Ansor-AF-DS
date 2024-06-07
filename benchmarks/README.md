## Description

benchmark and log the end-to-end time

## Usage

bash run_tests_times_conv.sh conv2d cuda num_of_runs num_sm num_shared_mem network num_trials num_init_states threshold pz_num

for example:

```
bash run_tests_times_conv.sh conv2d cuda 1 128 48 yolo 5 64 0.6 0
```

means run the yolo network on cuda with 128 SMs, 48 shared memory banks, 5 trials, 64 initial states, 0.6 threshold, and problem size 0
