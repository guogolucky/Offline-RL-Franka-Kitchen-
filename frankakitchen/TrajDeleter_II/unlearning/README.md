### Franka Kitchen Tasks

#### Download and test the offline dataset:

Please run:
```
python download_kitchen_datasets.py
```

#### (1) Training original agents:

The hyper-parameters settings of offline RL algorithms are recorded in fold './params'.

- Please run 
```
module load apptainer
python mujoco_fully_training.py --dataset <dataset_name> --seed <seed> --gpu <gpu_id> --model <params> --algo <algo> / sbatch script-poisoning-retrain.sh (NOTE - is the same code, so is applicable, only naming is different)
```

**Potential errors:**

```
There should be no error replicating the environment if the Apptainer (singularity) is installed properly and is running on Swinburne HPC. (You face some warning issues such as "old gym version", "carla not found", these are minor issues / not related which will affect the trainings
```

In the above scripts, `<dataset_name>` specifies the dataset name. The options are as follows:

| tasks info | dataset name |
| ------ | ----------- |
| complete sequence together     |  kitchen-complete-v0           |
| mixed of complete and incomplete sequence      | kitchen-mixed-v0           |
| sequence that are not completed together      | kitchen-partial-v0           |
 
After training, the trained models are saved into the folder `./Fully_trained/<dataset_name>`.


#### (2) For measuring Trajauditor efficacy:

The hyper-parameters settings of offline RL algorithms are recorded in fold './params'.

Please run,
```
python script-auditor.py (this doesn't take long to evaluate, so slurm jobs bash scripts are not needed)
```

After auditing, the results are saved in the `./<output_csv>` folder. If you want to change the file where the results are stored, please update the `output_csv` name in our code.


#### (3) For unlearning specific % trajectories:

1. The agents used for the unlearning experiments in the `./Fully_trained/<dataset_name>` folder. The weights of the agents are named as <xx>.pt, and the hyper-parameters settings of the offline RL algorithm are named as <xx>.json.

It is noticed that in d3rlpy, 10\% trajectories in dataset are used as testing trajecotroies. Therefore, if you want to unlearn 1\% trajectories, you should set the unlearning rate as 0.11.

```
python kitchen_trajdeleter.py --dataset <dataset_name> --seed <seed> --gpu <gpu_id> --unlearning_rate <poison_rate> --model <path-of-the-hyperparameters-of-orignial-agent> \
                             --model_params <path-of-the-weights-of-orignial-agent>  --number_of_unlearning <the-number-of-unlearning-rate> \
```

After unlearning, the unlearned agents are saved in the `./Kitchen_Trajdeleter_<stage1_step>_<balancing factor>/<dataset_name>` folder. Additionally, you can edit line 58 and line 71 to change the `stage1_step` and `lamda` variable, which controls the number of steps for unlearning and the balancing factor.

Besides, you could run ` python script-mujoco-trajdeleter.py / sbatch kitchen-trajdeleter.sh` for all algorithms and tasks to unlearn concurrently.

2. For Retraining from scratch,

```
bash script-exact-unlearning.sh / sbatch kitchen-exact-unlearning.sh
```

After unlearning, the unlearned agents are saved into the folder `./Exact_unlearning/<dataset_name>``.


3. For Fine-tuning,

```
python script-fine-tuning.py
```

After unlearning, the unlearned agents are saved into the folder `./Kitchen_fine_tuning/<dataset_name>``.

4. For Random reward,

```
python script-random-reward.py
```

After unlearning, the unlearned agents are saved into the folder `./Kitchen_noise_reward/<dataset_name>``.

5. Evaluation:

To test the obtained cumulative rewards of the agents:
```
python IQL/CQL_performance_test.py / IQL/CQL_performance_test_v2.py  --task <dataset_name> --seed <seed> --gpu <gpu_id> -model <path-of-the-hyperparameters-of-models> \
                               -model <path-of-the-weights-of-models>
```

Besides, you could run `sbatch IQL/CQL_performance.sh / IQL_CQL_performance_2.sh` to let it run via slurm job

To test the successful unlearning rates of the unlearned agents (%PPR and retention rate):
```
python script-trajauditor.py / sbatch kitchen-trajauditor.sh for long runs and to save the output onto a folder (refer to code for on where the folder path is saved to)
```
Please change variable line 41-43 to evaluate the efficacy of unlearning methods.

#### (4) For hyper-parameters analysis:

You can edit line 58 and line 71 to change the `stage1_step` and `lamda` variable, which controls the number of steps for unlearning and the balancing factor.

#### (5) For defending against trajectory poisoning,

To poison the agents:

```
bash script-poisoning.sh
```

Besides, you can run ` python poisoning_training.py` for specific algorithms and tasks.

To retrain the agents on the clean dataset:

```
bash script-poisoning-retain.sh / sbatch kitchen-poisoning-retrain.sh
```

You can also run `python poisoning_retrain.py` for specific algorithms and tasks.

#### (6) For defending against trajectory poisoning,

To unlearn the poisoned agents using TrajDeleter:

```
python kitchen-trajdeleter.py (please remember to change the path to the poisoned agent directory before unlearning) / sbatch kitchen-trajdeleter.sh for long training runs
```

To fine-tune unlearn the poisoned agents:

```
bash script-poisoning-fine-tune.py / sbatch kitchen-poison-fine-tune.sh for long training runs
```


Any questions or confusion can be emailed to 105318371@student.swin.edu.au" or email me via GitHub
