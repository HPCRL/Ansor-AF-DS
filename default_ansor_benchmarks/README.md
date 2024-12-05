## Description

This folder contains a collection of benchmarks for the default Ansor. You could use it to generate the Ansor baseline.

## Getting Started

### Prerequisites

Please build TVM from the source and install the necessary Python packages. 

### Run matmul benchmarks
```
python batch_matmul_cuda.py
```

### Run yolo benchmarks (2D convolutions)

```
python batch_conv2d_cuda_yolo.py
```

### Run resnet benchmarks (2D convolutions)

```
python batch_conv2d_cuda_resnet.py
```


