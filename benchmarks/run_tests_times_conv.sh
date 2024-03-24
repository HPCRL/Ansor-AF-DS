#!/bin/bash


test_type=$1
platform=$2
run_time=$3
sm_num=$4
max_shared_memory=$5
network=$6
ntrials=$7
init_states=$8
threshold=$9
specify_pz=${10}

# set default value to 0.6
if [[ $threshold == "" ]]; then
    threshold=0.6
fi

echo "debug------------------"
echo "test_type: $test_type"
echo "platform: $platform"
echo "run_time: $run_time"
echo "sm_num: $sm_num"
echo "max_shared_memory: $max_shared_memory"
echo "network: $network"
echo "ntrials: $ntrials"
echo "init_states: $init_states"
echo "specify_pz: $specify_pz"
echo "threshold: $threshold"
echo "debug end------------------"

# if we have specify_pz is not such parameter
if [[ $specify_pz == "" ]]; then
    echo "Specify PZ set to -1"
    specify_pz=-1
else 
    # if we have specify_pz is a number
    if [[ $specify_pz =~ ^[0-9]+$ ]]; then
        echo "Specify PZ: "$specify_pz
    else
        echo "Specify PZ set to -1"
        specify_pz=-1
    fi
fi

for ((i=1; i<=run_time; i++)); do
    echo "Run $i of $run_time"
    
    new_dir="run_time${i}_$network"
    mkdir -p "${new_dir}"
    
    if [[ $test_type == "conv2d" ]]; then
        if [[ $platform == "x86" ]]; then
            python3 test/batch_conv2d_x86.py 2>&1 | tee conv2d_x86.log
        elif [[ $platform == "cuda" ]]; then
            if [[ $specify_pz != -1 ]]; then
                log_file="conv2d_cuda_pz${specify_pz}.log"
            else
                log_file="conv2d_cuda.log"
            fi
            python3 test/batch_conv2d_cuda.py $network $sm_num $max_shared_memory $specify_pz $ntrials $init_states $threshold 2>&1 | tee $log_file
        else
            echo "Invalid platform entered."
            exit 1
        fi
    else
        echo "Invalid test type entered."
        exit 1
    fi
    
    mv *.json "${new_dir}/"
    mv *.log "${new_dir}/"
    mv *.csv "${new_dir}/"
    
    echo "Completed run $i, files moved to ${new_dir}"
done

echo "All tests completed."