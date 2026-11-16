#!/bin/bash
#SBATCH --job-name=kitchen-poisoning-retrain-cql-training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:3
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err

mkdir -p logs/

module load apptainer


apptainer exec --nv --writable-tmpfs --fakeroot \
  containers/image.sif \
  bash -lc '
  cd unlearning
    bash script-poisoning-retrain.sh
  '
