#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

declare -A algos
algos["IQL"]="iql" #mapping dict values to params, change algos here
#algos["CQL"]="cql"
#algos["BEAR"]="bear"
#algos["TD3PLUSBC"]="td3plusbc"
#algos["PLASP"]="plasp"
#algos["BCQ"]="bcq"


declare -A datasets #change datasets here
#datasets["kitchen-complete-v0"]="kitchen"
datasets["kitchen-mixed-v0"]="kitchen"
#datasets["kitchen-partial-v0"]="kitchen"

GPUs=(0 1 2) #Set GPUs here

unlearning_rates=(0.20) #set here

tasks_per_gpu=1 #task per gpu

gpu_counter=0 # set default

for dataset_key in "${!datasets[@]}"; do
    dataset=${datasets[$dataset_key]}
    for unlearning_rate in "${unlearning_rates[@]}"; do
        for algo_key in "${!algos[@]}"; do
            algo=${algos[$algo_key]}
            model_path="./params/${algo}_${dataset}_2.json" #change params here for each datasets
            for seed in 0 1 2 3 4; do
                gpu_id=${GPUs[$((seed % ${#GPUs[@]}))]}
                command="CUDA_VISIBLE_DEVICES=$gpu_id python ./mujoco_exact_unlearning.py --seed=$seed --dataset='$dataset_key' --model='$model_path' --algo='$algo_key' --unlearning_rate='$unlearning_rate'"
                echo $command
                eval $command
                sleep 2

                ((gpu_counter++))

                if [ $(( (seed + 1) % tasks_per_gpu )) -eq 0 ]; then
                    wait
                fi
            done
        done
    done
done

wait

