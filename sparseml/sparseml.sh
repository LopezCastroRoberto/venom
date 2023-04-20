#!/bin/bash

#SBATCH --job-name=sparsifier
#SBATCH --account=g34
#SBATCH --mem=128GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=amdrtx
module load cuda/11.7.1
#conda init zsh
source activate sparseml_me

echo "SLURM_JOB_NUM_NODES $SLURM_JOB_NUM_NODES"
echo "HOSTNAME $HOSTNAME"

echo "Args train.sh: $@"

srun ./integrations/huggingface-transformers/scripts/oBERTnmv_squad_gradual.sh

