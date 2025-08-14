#!/bin/bash
#SBATCH --job-name=mbart
#SBATCH --output=logs/mbart_set_%j.out
#SBATCH --error=logs/mbart_set_%j.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpumedium
#SBATCH --account=project_2005815
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:a100:4

set=$1
echo "Running for Set ${set}..."

# Load environment
source /scratch/project_2005815/members/degibert/fairseq-dummy/venv/bin/activate

export MASTER_ADDR=localhost
export MASTER_PORT=12345
export WORLD_SIZE=4
export RANK=0
export PYTORCH_WEIGHTS_ONLY=0

lang_pairs="en_XX-tr_TR,en_XX-my_MM,en_XX-ml_IN,en_XX-gl_ES,en_XX-tl_XX,en_XX-te_IN,en_XX-mk_MK,tr_TR-en_XX,my_MM-en_XX,ml_IN-en_XX,gl_ES-en_XX,tl_XX-en_XX,te_IN-en_XX,mk_MK-en_XX"
path_2_data="data/mbart/set_${set}/data-bin"
lang_list="models/mbart50-many-to-many/ML50_langs.txt"
pretrained_model="models/mbart50-many-to-many/model_modified.pt"
CHECKPOINTS_DIR="models/mbart50-many-to-many-ft-ALL-to-ALL-mod_set-${set}"

mkdir -p $CHECKPOINTS_DIR

CUDA_VISIBLE_DEVICES=0,1,2,3  fairseq-train $path_2_data \
   --save-dir $CHECKPOINTS_DIR --wandb-project CreolesMT \
  --finetune-from-model $pretrained_model --log-file $CHECKPOINTS_DIR/train.log \
  --encoder-normalize-before --decoder-normalize-before \
  --arch mbart_large --layernorm-embedding \
  --task translation_multi_simple_epoch \
  --sampling-method "temperature" \
  --sampling-temperature 2 \
  --encoder-langtok "src" \
  --decoder-langtok \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --lr 3e-05 --warmup-updates 2500 --max-update 40000 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 1024 --update-freq 2 \
  --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
  --seed 222 --log-format simple --log-interval 2 \
  --validate-interval-updates 5000 --patience 10 --distributed-world-size 4 --distributed-init-method env:// --fp16
