import os

import numpy as np
import tvm
from tvm import te, auto_scheduler
import time
import csv
import json
import threading

sizes=[
    #Bert large
[512,64,1024],      #BMATmul
[512,4096,1024],    #MLP1

    #Bert basic
[512,64,768],       #BMATmul
[512,3072,768],     #MLP1

    #Bert large
[512,1024,4096],    #MLP2
    #Bert basic
[512,768,3072],     #MLP2
]

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def _matmul(N, L, M, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    matmul = te.compute(
        (N, M),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="matmul",
    )
    return [A, B, matmul]

def parse_and_write(log_file_path, csv_file_path, start_time):
    if not os.path.exists(log_file_path):
        log_file_path = "x" + log_file_path
    # deal with FileNotFoundError exception
    try:
        with open(log_file_path, 'r') as log_file:
            log_content = log_file.read()
    except FileNotFoundError:
        # print(f'Json log might not be generated yet. Skip this round.', flush=True)
        return

    lines = log_content.splitlines()
    num_lines = len(lines)
    min_time_value = float('inf')

    for line in lines:
        try:
            data = json.loads(line)
            time_value = data['r'][0][0]

            if time_value < min_time_value:
                min_time_value = time_value
        except json.JSONDecodeError:
            continue

    if min_time_value != float('inf'):
        elapsed_time = int(time.time() - start_time)
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([elapsed_time, min_time_value, num_lines])

timer = None

def start_timer(interval, log_file_path, csv_file_path, start_time):
    global timer
    timer = threading.Timer(interval, start_timer, [interval, log_file_path, csv_file_path, start_time])
    timer.start()
    parse_and_write(log_file_path, csv_file_path, start_time)
    
def stop_timer():
    global timer
    if timer:
        timer.cancel()

file_name = 'search_info_matmul.csv'

def append_to_csv(row_data = []):
    with open(file_name, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(row_data)

def test_matmul(sm_num, max_shared_memory_per_block, specify_pz, ntrials, init_states=128):
    with open(file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["test_case", "M", "N", "K", "search_time(min)"])
    # if we have specify_pz, we only test that case
    if specify_pz != "-1":
        print("testing specified case: ", specify_pz, flush=True)
        sizes_tmp = [sizes[int(specify_pz)]]
    else:
        print("testing all cases", flush=True)
        sizes_tmp = sizes
        
    for i, size in enumerate(sizes):
        M=size[0]
        N=size[1]
        L=size[2]
        if size not in sizes_tmp:
            continue
        print("M=",M,"N=",N,"K=",L, flush=True)
        start_time = time.time()
        target = tvm.target.cuda()
        hardware_params = auto_scheduler.HardwareParams(target=target, num_cores=int(sm_num), max_shared_memory_per_block=int(max_shared_memory_per_block)*1024, max_threads_per_block=1024, \
                                                        max_vthread_extent=255, vector_unit_bytes=int(999), cache_line_bytes =int(999))

        task = tvm.auto_scheduler.SearchTask(func=_matmul, args=(M, L, N, "float32"), target=target, hardware_params=hardware_params)

        # Inspect the computational graph
        print("Computational DAG:", flush=True)
        print(task.compute_dag, flush=True)

        log_file = "cuda_testCase_" + str(i) +"_matmul_M"+str(M)+"_N"+str(N)+"_K"+str(L)+".json"
        
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=ntrials,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            verbose=2,
        )

        # Run auto-tuning (search)
        cost_model = auto_scheduler.XGBModel()
        search_policy = auto_scheduler.SketchPolicy(
            task, 
            program_cost_model=cost_model,
            params={ # just for first round model
                "sample_init_min_population": init_states,
                "tolerant_threashold": 0.6,
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
        
        # wait 3 seconds for the last log to be written
        time.sleep(3)
        # stop_timer()
        print(f"Total search time: {search_time} minutes", flush=True)
        
        # Write size and search time to CSV
        append_to_csv([i, M, N, L, search_time])
        
        # Apply the best schedule
        try: 
            if not os.path.exists(log_file):
                log_file = "x" + log_file
            sch, args = task.apply_best(log_file)

            func = tvm.build(sch, args, target)
            a_np = np.random.uniform(size=(M, L)).astype(np.float32)
            b_np = np.random.uniform(size=(L, N)).astype(np.float32)
            out_np = a_np.dot(b_np)

            dev = tvm.gpu()
            a_tvm = tvm.nd.array(a_np, device=dev)
            b_tvm = tvm.nd.array(b_np, device=dev)
            out_tvm = tvm.nd.empty(out_np.shape, device=dev)
            func(a_tvm, b_tvm, out_tvm)

            # Check results
            np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)

        except Exception as e:
            print(e, flush=True)
            continue
        
        # Evaluate execution time.
        evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
        print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(a_tvm, b_tvm, out_tvm).results) * 1000), flush=True)
        print("for M=",M,"N=",N,"K=",L,"matmul correctness check passed!", flush=True)

if __name__ == '__main__':
    # add timer to kill the process after 8 hours
    # import signal
    # def handler(signum, frame):
    #     raise Exception("end of time")
    # signal.signal(signal.SIGALRM, handler)
    # signal.alarm(8*60*60)
    
    # use arg to receive input from shell script
    import sys
    if len(sys.argv) > 1:
        sm_num = sys.argv[1]
        max_shared_memory_per_block = sys.argv[2]
        specify_pz = sys.argv[3]
        ntrials = int(sys.argv[4])
        init_states = int(sys.argv[5])
        print ("sm_num: ", sm_num, flush=True)
        print ("max_shared_memory_per_block: ", max_shared_memory_per_block, flush=True)
    else:
        raise Exception("network , sm_num, max_shared_memory_per_block not specified!")
    
    test_matmul(sm_num, max_shared_memory_per_block, specify_pz, ntrials, init_states)
    
    # get TEST_TVM_HOME environment variable
    test_tvm_home = os.environ.get("TEST_TVM_HOME")
    module_path = os.path.join(test_tvm_home, "modules/tg.py")    
    if os.path.exists(module_path):
        # load module.py from localtvm/zz_scripts
        # print("module.py found in", module_path)
        import sys
        sys.path.append(os.path.join(test_tvm_home, "modules"))
        from tg import send2TG
        send2TG("batch matmul test finished!")
    else:
        print("batch matmul test finished!", flush=True)