import os
import glob
import concurrent.futures
import csv
import time
import subprocess

# envs = ["hopper-medium-v0", "halfcheetah-medium-v0", "walker2d-medium-v0"]
envs = ["kitchen-partial-v0"] #change datasets here
unlearning_rates = [0.15] #change % here
seeds = [0, 1, 2, 3, 4] #5 seeds
unlearning_steps = [10000]


gpus = ['0'] #set GPUs (if changer, take note of max_workers)

# os.environ['LD_LIBRARY_PATH'] = '/anaconda3/envs/d4rl/lib'
# os.environ['LD_LIBRARY_PATH'] += ':/.mujoco/mujoco210/bin'
# os.environ['LD_LIBRARY_PATH'] += ':/usr/lib/nvidia'


max_workers = 1 #how many tasks assigned to each GPUs


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
                file_folder = f"./Poisoned_agents/{env}/CQL_0.05" #这里换 folders
                folders = [file_folder]
                #folders = get_directories(file_folder)
                for folder in folders:
                    model_param = os.path.join(folder, 'params.json') #fixed (naming)
                    search_pattern = os.path.join(folder, 'model.pt')
                    model_files = glob.glob(search_pattern)
                    if not model_files:
                        continue
                    model = os.path.join(folder, os.path.basename(model_files[0]))


                    part = os.path.basename(folder).split('_')[0]
                    start_time = os.path.basename(folder).split('_')[1]
                    algo = "PLASP" if part == "PLASWithPerturbation" else part #可选
                    if algo == 'BEAR' or 'TD3PLUSBC':
                        arguments = [
                            '--dataset', env,
                            '--model', model_param,
                            '--model_params', model,
                            '--number_of_finetuning', unlearning_step,
                            '--seed', seed,
                            '--unlearning_rate', unlearning_rate,
                            '--algo', algo,
                            '--gpu', gpus[gpu_index]  # 分配GPU
                        ]
                        script_path = 'mujoco_finetuning.py'
                        command = ['python', script_path] + [str(arg) for arg in arguments]

                        futures.append(executor.submit(run_command_on_gpu, command, gpus[gpu_index]))

                        gpu_index = (gpu_index + 1) % len(gpus)

    concurrent.futures.wait(futures)
