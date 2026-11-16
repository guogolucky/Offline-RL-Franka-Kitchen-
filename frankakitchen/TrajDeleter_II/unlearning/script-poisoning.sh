#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

declare -A algos
algos["IQL"]="iql"

datasets=(
    "kitchen-complete-v0" #add datasets here
)

GPUs=(0)
tasks_per_gpu=1

for dataset_key in "${datasets[@]}"; do

    for algo_key in "${!algos[@]}"; do

        model_path="./params/iql_kitchen_2.json" #change params here

        for seed in 0 1 2 3 4; do

            gpu_id=${GPUs[$((seed % ${#GPUs[@]}))]}

            command="CUDA_VISIBLE_DEVICES=$gpu_id python ./poisoning_training.py \
                --seed=$seed \
                --dataset='$dataset_key' \
                --model='$model_path' \
                --algo='$algo_key' \
                --poison_rate=0.01"

            echo "$command"
            eval "$command"

            sleep 1

            if [ $(( (seed + 1) % tasks_per_gpu )) -eq 0 ]; then
                wait
            fi

        done
    done
done

wait