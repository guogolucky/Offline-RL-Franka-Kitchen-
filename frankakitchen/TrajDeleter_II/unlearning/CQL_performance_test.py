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


    algo = d3rlpy.algos.CQL.from_json(
        "./params/cql_kitchen.json", #换
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


        # Evaluate using 100 eps
        scorer = evaluate_on_environment(
            env,
            n_trials=100
        )


        # Average cumulative return over 100 episodes
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
           "./Poison_retrained/kitchen-mixed-v0/CQL_20260906153303",
           "./Poison_retrained/kitchen-mixed-v0/CQL_20260906205521",
           "./Poison_retrained/kitchen-mixed-v0/CQL_20260907022602",
           "./Poison_retrained/kitchen-mixed-v0/CQL_20260907073249",
           "./Poison_retrained/kitchen-mixed-v0/CQL_20260907122730"
    ]


    os.makedirs("performance logs", exist_ok=True)


    log = "performance logs/kitchen_mixed_CQL_3.txt" #here


    open(log, "w").write(
        "CQL Evaluation (100 episodes, 5 seeds)\n"
    )


    # Evaluate each seed
    results = []

    for seed, folder in enumerate(folders):

        result = evaluate(
            seed,
            folder,
            log
        )

        results.append(result)



    # Average best checkpoint performance across 5 seeds
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
