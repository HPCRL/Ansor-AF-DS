import os

import numpy as np
import tvm
from tvm import te, auto_scheduler

sizes=[
    #Bert large
[512,64,1024],      #BMATmul
[512,4096,1024],    #MLP1
[512,1024,4096],    #MLP2

    #Bert basic
[512,64,768],       #BMATmul
[512,3072,768],     #MLP1
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

def test_matmul():
        
    for i, size in enumerate(sizes):
        M=size[0]
        N=size[1]
        L=size[2]
        print("M=",M,"N=",N,"K=",L)
        target = tvm.target.cuda()
        task = tvm.auto_scheduler.SearchTask(func=_matmul, args=(M, L, N, "float32"), target=target)

        # Inspect the computational graph
        print("Computational DAG:")
        print(task.compute_dag)

        log_file = "cuda_testCase_" + str(i) +"_matmul_M"+str(M)+"_N"+str(N)+"_K"+str(L)+".json"
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=1000,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            verbose=2,
        )

        # Run auto-tuning (search)
        task.tune(tune_option)
        # Apply the best schedule
        sch, args = task.apply_best(log_file)
        
        func = tvm.build(sch, args, target)
        a_np = np.random.uniform(size=(N, L)).astype(np.float32)
        b_np = np.random.uniform(size=(L, M)).astype(np.float32)
        out_np = a_np.dot(b_np)

        dev = tvm.gpu()
        a_tvm = tvm.nd.array(a_np, device=dev)
        b_tvm = tvm.nd.array(b_np, device=dev)
        out_tvm = tvm.nd.empty(out_np.shape, device=dev)
        func(a_tvm, b_tvm, out_tvm)

        # Check results
        np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)

        # Evaluate execution time.
        evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
        print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(a_tvm, b_tvm, out_tvm).results) * 1000)
        )
        print("for M=",M,"N=",N,"K=",L,"matmul correctness check passed!")

if __name__ == '__main__':
    test_matmul()