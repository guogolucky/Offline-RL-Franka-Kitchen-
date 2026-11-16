import os
import glob
import concurrent.futures
import csv
import time
import subprocess

# envs = ["hopper-medium-v0", "halfcheetah-medium-v0", "walker2d-medium-v0"]
# envs = ["halfcheetah-medium-expert-v0", "walker2d-medium-expert-v0", "hopper-medium-expert-v0"]

envs = [
    #"kitchen-complete-v0",
    #"kitchen-mixed-v0",
    "kitchen-mixed-v0"
]

# 0.11 =  1% unlearning 
# 0.15 =  5% unlearning...
unlearning_rates = [0.15]

unlearning_steps = [10000]

seeds = [0, 1, 2, 3, 4] #run 5 seeds

# gpus = ['0', '1', '2']
# algos = ['BCQ', 'CQL', 'BEAR', 'IQL', 'PLASP', 'TD3PlusBC']

gpus = ['0'] #num of gpus
algos = ['CQL'] #change algos here

# os.environ['LD_LIBRARY_PATH'] = '/bigtemp/fzv6en/anaconda3/envs/d4rl/lib'
# os.environ['LD_LIBRARY_PATH'] += ':/u/fzv6en/.mujoco/mujoco210/bin'
# os.environ['LD_LIBRARY_PATH'] += ':/usr/lib/nvidia'

# 1 = run 1 by 1
# 24 = default (need 3 gpus)
max_workers = 1

def get_directories(path):
    directories = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return directories

def run_command_on_gpu(command, gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    subprocess.run(command)
    time.sleep(1)
    # with open(output_file, "a", newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     csv_writer.writerow([env, unlearning_rate, unlearning_step, algo, start_time, result.stdout.strip()])


with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    gpu_index = 0

    for env in envs:
        for seed in seeds:
            for unlearning_rate in unlearning_rates:
                for unlearning_step in unlearning_steps:
                    for algo in algos:
                        fully_trained_model = "./Poisoned_agents/" + str(env) + '/' + str(algo)+ '_0.05/params.json' #change here to run diff agents
                        fully_trained_params = "./Poisoned_agents/" + str(env) + '/' + str(algo) + '_0.05/model.pt' 
                        arguments = [
                            '--dataset', env,
                            '--model', fully_trained_model,
                            '--model_params', fully_trained_params,
                            '--number_of_unlearning', unlearning_step,
                            '--seed', seed,
                            '--unlearning_rate', unlearning_rate,
                            '--algo', algo,
                            '--gpu', gpus[gpu_index] 
                        ]
                        script_path = 'mujoco_trajdeleter.py'
                        command = ['python', script_path] + [str(arg) for arg in arguments]

                        futures.append(executor.submit(run_command_on_gpu, command, gpus[gpu_index]))

                        gpu_index = (gpu_index + 1) % len(gpus)

    concurrent.futures.wait(futures)
