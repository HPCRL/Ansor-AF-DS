
import os
import csv
import time
import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python
import json
import threading

########### all pz 
sizesResnet = [
    [1, 224, 224, 64, 3, 7, 7, 2, 3],   # RESNET1
    [1, 56, 56, 64, 64, 1, 1, 1, 0],    # RESNET2
    [1, 56, 56, 64, 64, 3, 3, 1, 1],    # RESNET2
    [1, 56, 56, 256, 64, 1, 1, 1, 0],   # RESNET2
    [1, 56, 56, 128, 256, 1, 1, 2, 0],  # RESNET3
    [1, 28, 28, 128, 128, 3, 3, 1, 1],  # RESNET3
    [1, 28, 28, 512, 128, 1, 1, 1, 0],  # RESNET3
    [1, 28, 28, 256, 512, 1, 1, 2, 0],  # RESNET4
    [1, 14, 14, 256, 256, 3, 3, 1, 1],  # RESNET4
    [1, 14, 14, 1024, 256, 1, 1, 1, 0], # RESNET4
    [1, 14, 14, 512, 1024, 1, 1, 2, 0], # RESNET5
    [1, 7, 7, 512, 512, 3, 3, 1, 1],    # RESNET5
    [1, 7, 7, 2048, 512, 1, 1, 1, 0],   # RESNET5
]

sizesYolo = [
    [1, 544, 544, 32, 3, 3, 3, 1, 1],    # Yolo0
    [1, 272, 272, 64, 32, 3, 3, 1, 1],   # Yolo2
    [1, 136, 136, 128, 64, 3, 3, 1, 1],  # yolo4
    [1, 136, 136, 64, 128, 1, 1, 1, 0],  # yolo5
    [1, 68, 68, 256, 128, 3, 3, 1, 1],   # yolo8
    [1, 68, 68, 128, 256, 1, 1, 1, 0],   # yolo9
    [1, 68, 68, 512, 256, 3, 3, 1, 1],   # yolo14
    [1, 34, 34, 512, 256, 3, 3, 1, 1],   # yolo12
    [1, 34, 34, 256, 512, 1, 1, 1, 0],   # yolo13
    [1, 17, 17, 1024, 512, 3, 3, 1, 1],  # yolo18
    [1, 17, 17, 512, 1024, 1, 1, 1, 0],  # yolo19
]
########### all pz end

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
        
@auto_scheduler.register_workload
def conv2d(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]


def test_conv2d(network, sm_num, max_shared_memory_per_block, specify_pz, ntrials, tolerant_threashold=0.6, init_states=64):
    print(f"ntrials: {ntrials}", flush=True)
    
    if network == "yolo":
        sizes=sizesYolo
        print(f"\ntesting yolo with {len(sizes)} layers\n")
    elif network == "resnet":
        sizes=sizesResnet
        print(f"\ntesting resnet with {len(sizes)} layers\n")
    else:
        raise Exception("network not specified!")

    # if we have specify_pz, we only test that case
    if specify_pz != "-1":
        print("testing specified case: ", specify_pz, flush=True)
        if network == "yolo":
            sizes_tmp = [sizesYolo[int(specify_pz)]]
        elif network == "resnet":
            sizes_tmp = [sizesResnet[int(specify_pz)]]
        else:
            raise Exception("network not specified!")
    # otherwise, we test all cases
    else:
        print("network not specified, testing all cases!", flush=True)
        sizes_tmp = sizes

    conv_params = {}
    for i, size in enumerate(sizes):
        if size not in sizes_tmp:
            continue
        N, H, W, CO, CI, KH, KW, stride, pad = size
        key = "conv" + str(i+1)
        #N, H, W, CO, CI, KH, KW, strides, padding
        conv_params[key] = Conv2DParams(N, H, W, CO, CI, KH, KW, (stride, stride), (pad, pad))

    for ite, key in enumerate(conv_params.keys()):
        if specify_pz != "-1":
            ite = int(specify_pz)

        start_time = time.time()
        conv = conv_params[key]
        target = tvm.target.cuda()
        
        # Use the conv2d layer to test
        N, H, W, CO, CI, KH, KW, strides, padding = conv.N, conv.H, conv.W, conv.CO, conv.CI, conv.KH, conv.KW, conv.strides, conv.padding
                
        hardware_params = auto_scheduler.HardwareParams(target=target, num_cores=int(sm_num), max_shared_memory_per_block=int(max_shared_memory_per_block)*1024, max_threads_per_block=1024, \
                                                        max_vthread_extent=255, vector_unit_bytes=int(999), cache_line_bytes =int(999))
        task = auto_scheduler.SearchTask(
            func=conv2d, args=(N, H, W, CO, CI, KH, KW, strides, padding), target=target, hardware_params=hardware_params
        )

        # Inspect the computational graph
        print("pz:", ite, flush=True)
        print("Computational DAG:", flush=True)
        print(task.compute_dag, flush=True)

        log_file = "cuda_"+network+"_testCase_"+str(ite)+"_conv2d_N_"+str(N)+"_H_"+str(H)+"_W_"+str(W)+"_CO_"+str(CO)+"_CI_"+str(CI)+"_KH_"+str(KH)+"_KW_"+str(KW)+"_strides_"+str(strides)+"_padding_"+str(padding)+".json"
        
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=ntrials,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            verbose=2,
        )
        cost_model = auto_scheduler.XGBModel()
        search_policy = auto_scheduler.SketchPolicy(
            task, 
            program_cost_model=cost_model,
            params={
                "sample_init_min_population": init_states,
                "tolerant_threashold": tolerant_threashold,
            },
        )
        
        start_time = int(time.time())
        csv_file_path = log_file.replace('.json', '.csv')

        # write the start time to the csv file
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_file.write(f"start_time:{str(start_time)}\n")

        task.tune(tune_option, search_policy)
        
        end_time = time.time()
        search_time = end_time - start_time
        search_time /= 60
        
        print(f"Total search time: {search_time} minutes", flush=True)

        # Apply the best schedule
        try: 
            if not os.path.exists(log_file):
                log_file = "x" + log_file
            sch, args = task.apply_best(log_file)
            
            print("Lowered TIR:", flush=True)
            print(tvm.lower(sch, args, simple_mode=True), flush=True)

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
                % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000), flush=True)
            print("cuda testCase: "+str(ite)+" conv2d for N = %d, H = %d, W = %d, CO = %d, CI = %d, KH = %d, KW = %d, strides = %s, padding = %s, correctness check passed!\n" % (N, H, W, CO, CI, KH, KW, strides, padding), flush=True)

        except Exception as e:
            print(e, flush=True)
            continue

if __name__ == '__main__':
    
    # use arg to select network from bash script
    import sys
    if len(sys.argv) > 7:
        network = sys.argv[1]
        sm_num = sys.argv[2]
        max_shared_memory_per_block = sys.argv[3]
        specify_pz = sys.argv[4]
        ntrials = int(sys.argv[5])
        init_states = int(sys.argv[6])
        tolerant_threashold = float(sys.argv[7])
        print ("network:", network, flush=True)
        print ("sm_num: ", sm_num, flush=True)
        print ("max_shared_memory_per_block: ", max_shared_memory_per_block, flush=True)
    else:
        raise Exception("network , sm_num, max_shared_memory_per_block not specified!")

    test_conv2d(network, sm_num, max_shared_memory_per_block, specify_pz, ntrials, tolerant_threashold, init_states)