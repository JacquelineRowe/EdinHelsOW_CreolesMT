#!/bin/bash
#SBATCH --job-name=convert
#SBATCH --output=logs/convert_%j.out
#SBATCH --error=logs/convert_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=16G
#SBATCH --partition=small
#SBATCH --account=project_2005815

# Load environment
source /scratch/project_2005815/members/degibert/fairseq-dummy/venv/bin/activate

python transformers/src/transformers/models/mbart/convert_mbart_original_checkpoint_to_pytorch.py \
  models/mbart50-many-to-many-ft-XX-to-EN/checkpoint_best.pt \
  hf/mbart50-many-to-many-ft-XX-to-EN/ \
  --hf_config facebook/mbart-large-50 \
  --mbart_50 \
  --finetuned

python transformers/src/transformers/models/mbart/convert_mbart_original_checkpoint_to_pytorch.py \
  models/mbart50-many-to-many-ft-XX-to-EN_mod_set-1_por/checkpoint_best.pt \
  hf/mbart50-many-to-many-ft-XX-to-EN_mod_set-1_por/ \
  --hf_config facebook/mbart-large-50 \
  --mbart_50 \
  --finetuned