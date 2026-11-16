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


    algo = d3rlpy.algos.IQL.from_json(
        "./params/iql_kitchen_2.json", #change if needed
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
        f"{name} IQL Evaluation (100 trajectories, 5 seeds)\n"
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
    # 1st (1%) change here folder
    # =====================================

    folder_one= [

        "./Mujoco_our_method_8000_1.0/stage2/kitchen-mixed-v0-10000-0.009999999999999995/IQL_20261001115525",
        "./Mujoco_our_method_8000_1.0/stage2/kitchen-mixed-v0-10000-0.009999999999999995/IQL_20261001120004",
        "./Mujoco_our_method_8000_1.0/stage2/kitchen-mixed-v0-10000-0.009999999999999995/IQL_20261001120447",
        "./Mujoco_our_method_8000_1.0/stage2/kitchen-mixed-v0-10000-0.009999999999999995/IQL_20261001120937",
        "./Mujoco_our_method_8000_1.0/stage2/kitchen-mixed-v0-10000-0.009999999999999995/IQL_20261001121417"
    ]


    log_one = (
        "performance logs/TD-Stage2-KM-IQL-0.01.txt" #set log name
    )



    # =====================================
    # 2nd (5%) #change here folder
    # =====================================

    folder_five = [

        "./Mujoco_our_method_8000_1.0/stage2/kitchen-mixed-v0-10000-0.04999999999999999/IQL_20260903194231",
        "./Mujoco_our_method_8000_1.0/stage2/kitchen-mixed-v0-10000-0.04999999999999999/IQL_20260903194935",
        "./Mujoco_our_method_8000_1.0/stage2/kitchen-mixed-v0-10000-0.04999999999999999/IQL_20260903195639",
        "./Mujoco_our_method_8000_1.0/stage2/kitchen-mixed-v0-10000-0.04999999999999999/IQL_20260903200346",
        "./Mujoco_our_method_8000_1.0/stage2/kitchen-mixed-v0-10000-0.04999999999999999/IQL_20260903201053"
    ]


    log_five = (
        "performance logs/TD-Stage2-KM-IQL-0.05.txt"
    )



    # =====================================
    # 1st
    # =====================================

    run_experiment(
        "TD-Stage2-KM-IQL-0.01", #first line within the file
        folder_one, # run
        log_one
    )



    # =====================================
    # 2nd
    # =====================================

    run_experiment(
        "TD-Stage2-KM-IQL-0.05", #first line within the file
        folder_five,
        log_five
    )
