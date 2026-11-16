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

    env = gym.make("kitchen-mixed-v0") #change here


    algo = d3rlpy.algos.IQL.from_json(
        "./params/iql_kitchen_2.json", #换
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


        # Evaluate over 100 test trajectories
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



if __name__ == "__main__":

    #change here
    folders = [
        "./Weak_trained/kitchen-mixed-v0/IQL/seed_0/IQL_20261026023434",
        "./Weak_trained/kitchen-mixed-v0/IQL/seed_1/IQL_20261026023434",
        "./Weak_trained/kitchen-mixed-v0/IQL/seed_2/IQL_20261026023434",
        "./Weak_trained/kitchen-mixed-v0/IQL/seed_3/IQL_20261026063609",
        "./Weak_trained/kitchen-mixed-v0/IQL/seed_4/IQL_20261026063609"
    ]


    os.makedirs("performance logs", exist_ok=True)


    log = "performance logs/km-weak-trained.txt" #here


    open(log, "w").write(
        "IQL Evaluation (100 episodes, 5 seeds)\n"
    )


    results = []

    for seed, folder in enumerate(folders):

        result = evaluate(
            seed,
            folder,
            log
        )

        results.append(result)



    # Mean and std across 5 training seeds
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
        f"Final Performance: {final_mean:.2f} ± {final_std:.2f}"
    )
