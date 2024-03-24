
import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python


sizesYolo = [
    [1, 544, 544, 32, 3, 3, 3, 1, 1],    # Yolo0
    [1, 272, 272, 64, 32, 3, 3, 1, 1],   # Yolo2
    [1, 136, 136, 128, 64, 3, 3, 1, 1],  # yolo4
    [1, 136, 136, 64, 128, 1, 1, 1, 0],  # yolo5
    [1, 68, 68, 256, 128, 3, 3, 1, 1],   # yolo8
    [1, 68, 68, 128, 256, 1, 1, 1, 0],   # yolo9
    [1, 34, 34, 512, 256, 3, 3, 1, 1],   # yolo12
    [1, 34, 34, 256, 512, 1, 1, 1, 0],   # yolo13
    [1, 17, 17, 1024, 512, 3, 3, 1, 1],  # yolo18
    [1, 17, 17, 512, 1024, 1, 1, 1, 0],  # yolo19
]

class Conv2DParams:
    def __init__(self, N, H, W, CO, CI, KH, KW, strides, padding):
        self.N = N
        self.H = H
        self.W = W
        self.CO = CO
        self.CI = CI
        self.KH = KH
        self.KW = KW
        self.strides = strides
        self.padding = padding

conv_params = {}
for i, size in enumerate(sizesYolo):
    N, H, W, CO, CI, KH, KW, stride, pad = size
    key = "conv" + str(i+1)
    #N, H, W, CO, CI, KH, KW, strides, padding
    conv_params[key] = Conv2DParams(N, H, W, CO, CI, KH, KW, (stride, stride), (pad, pad))

@auto_scheduler.register_workload
def _conv2d(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

def test_conv2d():
    for ite, key in enumerate(conv_params.keys()):
        conv = conv_params[key]
        target = tvm.target.cuda()
        
        # Use the last layer in ResNet-50
        N, H, W, CO, CI, KH, KW, strides, padding = conv.N, conv.H, conv.W, conv.CO, conv.CI, conv.KH, conv.KW, conv.strides, conv.padding
        
        #N, H, W, CO, CI, KH, KW, strides, padding = 1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1)
        task = auto_scheduler.SearchTask(
            func=_conv2d, args=(N, H, W, CO, CI, KH, KW, strides, padding), target=target
        )

        # Inspect the computational graph
        print("Computational DAG:")
        print(task.compute_dag)

        log_file = "cuda_yolo_"+str(ite)+"_conv2d_N_"+str(N)+"_H_"+str(H)+"_W_"+str(W)+"_CO_"+str(CO)+"_CI_"+str(CI)+"_KH_"+str(KH)+"_KW_"+str(KW)+"_strides_"+str(strides)+"_padding_"+str(padding)+".json"

        measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=1000,  # change this to 1000 to achieve the best performance
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            verbose=2,
        )

        task.tune(tune_option)
        # Apply the best schedule
        sch, args = task.apply_best(log_file)

        # Kill the measurement process
        del measure_ctx
        
        
        print("Lowered TIR:")
        print(tvm.lower(sch, args, simple_mode=True))

        func = tvm.build(sch, args, target)

        # Check correctness
        data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
        weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
        conv_np = conv2d_nchw_python(data_np, weight_np, strides, padding)
        out_np = np.maximum(conv_np, 0.0)

        dev = tvm.gpu()
        data_tvm = tvm.nd.array(data_np, device=dev)
        weight_tvm = tvm.nd.array(weight_np, device=dev)
        out_tvm = tvm.nd.empty(out_np.shape, device=dev)
        func(data_tvm, weight_tvm, out_tvm)

        # Check results
        np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)

        # Evaluate execution time
        evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
        print(
            "Execution time of this operator: %.3f ms"
            % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000)
        )
        print("cuda yolo: "+str(ite)+" conv2d for N = %d, H = %d, W = %d, CO = %d, CI = %d, KH = %d, KW = %d, strides = %s, padding = %s, correctness check passed!\n" % (N, H, W, CO, CI, KH, KW, strides, padding))


if __name__ == '__main__':
    test_conv2d()