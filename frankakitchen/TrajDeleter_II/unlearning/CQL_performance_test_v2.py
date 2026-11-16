import gym
import d4rl
import numpy as np
import d3rlpy
import os
import re

from d3rlpy.metrics.scorer import evaluate_on_environment


def log_txt(path, txt):
    with open(path, "a") as f:
        f.write(txt + "\n")


def evaluate(seed, folder, log):

    d3rlpy.seed(seed)

    env = gym.make("kitchen-mixed-v0") #换数据集


    algo = d3rlpy.algos.CQL.from_json(
        "./params/cql_kitchen.json", # change params
        use_gpu=True
    )


    ckpts = sorted(
        [f for f in os.listdir(folder) if f.endswith(".pt")],
        key=lambda x: int(re.findall(r"\d+", x)[-1])
    )


    best = (-np.inf, "")


    log_txt(log, f"\n===== Seed {seed} =====")


    for ckpt in ckpts:


        # Load checkpoint
        algo.load_model(
            os.path.join(folder, ckpt)
        )


        # Evaluate over 100 test episodes
        scorer = evaluate_on_environment(
            env,
            n_trials=100
        )


        # Average cumulative return
        mean_return = scorer(algo)


        log_txt(
            log,
            f"{ckpt}: Return = {mean_return:.2f}"
        )


        # Select best checkpoint
        if mean_return > best[0]:

            best = (
                mean_return,
                ckpt
            )


    log_txt(
        log,
        f"BEST CHECKPOINT: {best[1]} | Return = {best[0]:.2f}"
    )


    return best[0]



def run_experiment(name, folders, log):

    open(log, "w").write(
        f"{name} CQL Evaluation (100 trajectories, 5 seeds)\n"
    )


    results = []


    for seed, folder in enumerate(folders):

        result = evaluate(
            seed,
            folder,
            log
        )

        results.append(result)



    # Mean and std across seeds
    final_mean = np.mean(results)
    final_std = np.std(results)


    log_txt(
        log,
        "\n===== Final Result ====="
    )


    log_txt(
        log,
        f"Mean Return: {final_mean:.2f} ± {final_std:.2f}"
    )


    print(
        f"{name} Performance: {final_mean:.2f} ± {final_std:.2f}"
    )



if __name__ == "__main__":


    os.makedirs("performance logs", exist_ok=True)



    # =====================================
    # 1st batch, change folder here
    # =====================================

    folder_one= [

        "./Mujoco_our_method_8000_1.0/stage2/kitchen-mixed-v0-10000-0.009999999999999995/CQL_20260910210559",
        "./Mujoco_our_method_8000_1.0/stage2/kitchen-mixed-v0-10000-0.009999999999999995/CQL_20260910211546",
        "./Mujoco_our_method_8000_1.0/stage2/kitchen-mixed-v0-10000-0.009999999999999995/CQL_20260910212528",
        "./Mujoco_our_method_8000_1.0/stage2/kitchen-mixed-v0-10000-0.009999999999999995/CQL_20260910213507",
        "./Mujoco_our_method_8000_1.0/stage2/kitchen-mixed-v0-10000-0.009999999999999995/CQL_20260910214445"

    ]


    log_one = (
        "performance logs/TD-Stage2-KM-CQL-0.01.txt" #log 名字
    )



    # =====================================
    # 2nd batch , change folder here
    # =====================================

    folder_five = [

        "./Mujoco_our_method_8000_1.0/stage2/kitchen-mixed-v0-10000-0.04999999999999999/CQL_20260910200813",
        "./Mujoco_our_method_8000_1.0/stage2/kitchen-mixed-v0-10000-0.04999999999999999/CQL_20260910201657",
        "./Mujoco_our_method_8000_1.0/stage2/kitchen-mixed-v0-10000-0.04999999999999999/CQL_20260910202548",
        "./Mujoco_our_method_8000_1.0/stage2/kitchen-mixed-v0-10000-0.04999999999999999/CQL_20260910203447",
        "./Mujoco_our_method_8000_1.0/stage2/kitchen-mixed-v0-10000-0.04999999999999999/CQL_20260910204336"
    ]


    log_five = (
        "performance logs/TD-Stage2-KM-CQL-0.05.txt" #log 名字
    
    )



    # =====================================
    # 1st 
    # =====================================

    run_experiment(
       "TD-Stage2-KM-CQL-0.01", #first line within the file
        folder_one, # run
        log_one
    )



    # =====================================
    # 2nd
    # =====================================

    run_experiment(
        "TD-Stage2-KM-CQL-0.05", #first line within the file
        folder_five,
        log_five
    )
