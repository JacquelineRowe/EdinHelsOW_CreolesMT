#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=40g
#SBATCH --account=plgmodularnlp-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err

ml ML-bundle/24.06a

source /net/storage/pr3/plgrid/plggmultilingualnlp/mklimasz/creolesMT/venv/bin/activate

# Usage example: sbatch scripts/finetune-mbart/slurm_final_generate_test.sh MB1 m2m
# $1 is the model id, such as MB1, MB2...
# $2 is the language directions, either m2o, o2m, or m2m.
# It will generate the test sets and save them into the directory "submission"

bash scripts/finetune-mbart/final_generate_test.sh $1 $2
