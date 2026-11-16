import subprocess
import os
import concurrent.futures
import csv

envs = ["kitchen-mixed-v0"]  # change here
algos = ["IQL"]  # change here

#0.01 → 1%
#0.05 → 5%
unlearning_rates = [0.01]# change here
gpus = ["0"]

# change multiple folders here (hardcoded)
folders = [
    "Kitchen_poison_fine_tuning/kitchen-mixed-v0-10000-0.009999999999999995/IQL_20261022014700",
    "Kitchen_poison_fine_tuning/kitchen-mixed-v0-10000-0.009999999999999995/IQL_20261022014922",
    "Kitchen_poison_fine_tuning/kitchen-mixed-v0-10000-0.009999999999999995/IQL_20261022015142",
    "Kitchen_poison_fine_tuning/kitchen-mixed-v0-10000-0.009999999999999995/IQL_20261022015405",
    "Kitchen_poison_fine_tuning/kitchen-mixed-v0-10000-0.009999999999999995/IQL_20261022015639"
    ]

output_csv = "result_trajaditor_8000_1.0.csv"  # ignore this (output csv)

max_workers = 1 


def run_command_on_gpu(command, env, unlearning_rate, algo, gpu_id, output_file):

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["PYTHONUNBUFFERED"] = "1"

    result = subprocess.run(
        command,
        text=True,
        capture_output=True
    )

    print(result.stdout, flush=True)

    os.makedirs("trajauditor logs", exist_ok=True) #write to a log

    log = ("trajauditor logs/" "PT-FT" + "-" + env + "_" + algo + "_" + str(unlearning_rate) + ".txt") #change here to write different logs

    with open(log, "a") as f:
        f.write("Command:\n")
        f.write(" ".join(command))
        f.write("\n\n")
        f.write(result.stdout)
        f.write("\n")
        f.write(result.stderr)


    with open(output_file, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            env,
            unlearning_rate,
            algo,
            gpu_id,
            result.returncode
        ])


# initial csv
with open(output_csv, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow([
        "Environment",
        "Unlearning_rate",
        "Algorithm",
        "GPU",
        "Output"
    ])


with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    gpu_index = 0

    #stage1_step = 8000
    #stage2_step = 10000 - stage1_step
    #lamda = 1.0
    checkpoint_step = 10000 #for rr and ft
    #checkpoint_step = "model" #for eu

    for env in envs:
        for unlearning_rate in unlearning_rates:
            for algo in algos:
                for folder in folders:

                    model_param = os.path.join(folder, "params.json")
                    #model = os.path.join(folder, "model.pt")
                    model = os.path.join(folder, f"model_{checkpoint_step}.pt")
                    #model = os.path.join(folder, f"model_{stage2_step}.pt")

                    print("folder:", folder, flush=True)
                    print("params:", model_param, flush=True)
                    print("model:", model, flush=True)

                    if not os.path.exists(folder):
                        print("Missing folder:", folder, flush=True)
                        continue

                    if not os.path.exists(model_param):
                        print("Missing params.json:", model_param, flush=True)
                        continue

                    if not os.path.exists(model):
                        print("Missing model checkpoint:", model, flush=True)
                        continue

                    fully_trained_model = ("./Fully_trained_agents/" + str(env) + "/" + str(algo) + "/params.json")

                    fully_trained_params = ("./Fully_trained_agents/" + str(env) + "/" + str(algo) + "/model.pt")

                    # stay same
                    exact_unlearning_model = model_param
                    exact_unlearning_params = model

                    script_path = "mujoco_trajauditor.py"

                    arguments = [
                        "--dataset", env,
                        "--fully_trained_model", fully_trained_model,
                        "--fully_trained_params", fully_trained_params,
                        "--exact_unlearning_model", exact_unlearning_model,
                        "--exact_unlearning_params", exact_unlearning_params,
                        "--algo", algo,
                        "--unlearning_rate", unlearning_rate + 0.1,  # here add 10% for evaluation
                        "--gpu", gpus[gpu_index],
                    ]

                    command = ["python", "-u", script_path] + [str(arg) for arg in arguments]

                    futures.append(
                        executor.submit(
                            run_command_on_gpu,
                            command,
                            env,
                            unlearning_rate,
                            algo,
                            gpus[gpu_index],
                            output_csv
                        )
                    )

                    gpu_index = (gpu_index + 1) % len(gpus)

    concurrent.futures.wait(futures)