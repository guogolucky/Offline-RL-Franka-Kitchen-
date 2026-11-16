# TrajDeleter: Enabling Trajectory Forgetting in Offline Reinforcement Learning Agents

## Intro

Replication Package for "[TrajDeleter: Enabling Trajectory Forgetting in Offline Reinforcement Learning Agents]([https://arxiv.org/abs/2210.04688](https://arxiv.org/abs/2404.12530))", NDSS 2025.

The descriptions of folds are as follows:

| fold_name | descriptions |
| ------ | ---------- |
| Shadow Models     |  Shadow agents for auditing |
| Exact unlearning agents      |  Retraining agents from scratch           |
| Fully trained agents      |  Agents Trained using the entire dataset          |

## Selected offline RL algorithms
| algorithm | discrete control | continuous control | 
|:-|:-:|:-:|
| [Conservative Q-Learning (CQL)](https://arxiv.org/abs/2006.04779) | ✓ | ✓ |
| [Implicit Q-learning (IQL)](https://arxiv.org/abs/2110.06169) | ✓ | ✓ |

## Project structure

The structure of this project in `unlearning' folder is as follows：
```
Unlearning
    -- CQL_performance_test.py ------------------ evaluate the average cumulative rewards obtained by agents (each checkpoints).
    -- CQL_performance_test_v2.py ------------------ evaluate the average cumulative rewards obtained by agents for 1% and 5% unlearning/poisoning rates (each checkpoint).
    -- IQL_performance_test.py ------------------ evaluate the average cumulative rewards obtained by agents (each checkpoints).
    -- IQL_performance_test_v2.py ------------------ evaluate the average cumulative rewards obtained by agents for 1% and 5% unlearning/poisoning rates (each checkpoint).
    -- env_test.py ------------------ download and test the dataset into environment
    -- efficacy_evaluation.py ------------------ to evaluate the robustness of TrajAuditor using metrices (Precision, Recall and F1-scores).
    -- kitchen_exact_unlearning.py ------------------ unlearning with retraining the agents from scratch.
    -- kitchen_finetuning.py ------------------ unlearning with fine-tuning method.
    -- kitchen_fully_training.py ------------------ train the original agents using the whole dataset.
    -- kitchen_random_reward.py ------------------ unlearning with random_reward method.
    -- kitchen_auditor.py ------------------ code for trajauditor evaluation method
    -- kitchen_trajdeleter.py ------------------ unlearning using our proposed method.
    -- performance_test.py ------------------ evaluate the average cumulative rewards obtained by agents. (not used)
    -- poisoning_retrain.py ------------------ unlearning the poisoned trajectories using retraining method.
    -- poisoning_training.py ------------------ training agents using the poisoned dataset.
    -- script-aditor.py ------------------ script files of trajauditor and achieve the results in RQ1.
    -- script-exact-unlearning.py ------------------ script files of retraining.
    -- script-fine-tune.py ------------------ script files of fine tuning.
    -- script-random-reward.py ------------------ script files of random reward method.
    -- script-mujoco-test.py ------------------ script files of agents' performance testing. (not used)
    -- script-trajdeleter.py ------------------ script files of our proposed unlearning method.
    -- script-performance-test.py ------------------ script files of testing the cumulative returns of agents. (not used)
    -- script-poisoning-fine-tuning.py ------------------ script files of fine-tuning the poisoned dataset.
    -- script-poisoning-retrain.py ------------------ script files of retraining the poisoned dataset.
    -- script-poisoning.py ------------------ script files of poisoning agents.
    -- script-trajauditor.py ------------------ script files of trajauditor.
    packages.txt ------------------ the list of env.
    params ------------------ the files of hype-parameters settings of offline reinforcement learning algorithms.
    
```


## Installation


Our CUDA version is -. We ran majority of the experiments on Swinburne HPC which have different GPU nodes.

### Install with Apptainer (ONLY FOR SWINBURNE HPC)

To install the Apptainer with all the dependencies needed for the environment,

1) Navigate to containers/image.def directory,
2) Enter "module load apptainer",
3) Enter "apptainer build containers/image.sif containers/image.def"
4) Whenever need to run a script, enter "module load apptainer" to intialize the environment to ensure no error with running the scripts

## How to run

⚠️ The codes and scripts for replicating our experiments can be found in `README.md` under the `unlearning` folder. 

To submit batch jobs to HPC (for long run trainings),

1) cql_performance.sh/cql_performance_2.sh - to run scripts for evaluating model cumulative reward return (CQL)
2) iql_performance.sh/iql_performance_2.sh - to run scripts for evaluating model cumulative reward return (IQL)
3) kitchen-download-dataset.sh - to download the kitchen datasets
4) kitchen-exact-unlearning.sh - to run exact unlearning scripts
5) kitchen-fine-tune.sh - to run fine-tuning unlearning scripts
6) kitchen-poison-fine-tune.sh - to run poison fine-tuning unleanrning scripts
7) kitchen-poisoning-retrain.sh - to run retrain clean dataset scripts for comparison against poisoning training
8) kitchen-poisoning-training.sh - to run training dataset with poison trajectories
9) kitchen-random-reward.sh - to run random-reward unlearning
10) kitchen-trajauditor.sh - to evaluate trajdeleter unlearning (%PPR and retention rate)
11) kitchen-trajdeleter.sh - to run trajdeleter scripts

## Acknowledgement

- The codes for achieving the offline RL algorithms are based on the [D3RLPY](https://github.com/takuseno/d3rlpy).
- The offline datasets for our evaluations are from [UC Berkerly dataset library](http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/).
