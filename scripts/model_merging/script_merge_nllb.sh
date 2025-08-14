#!/usr/bin/env bash
export HF_HOME=/net/storage/pr3/plgrid/plggmultilingualnlp/huggingface
source /net/storage/pr3/plgrid/plggmultilingualnlp/mklimasz/creolesMT/venv/bin/activate
cd "/net/storage/pr3/plgrid/plggmultilingualnlp/mklimasz/creolesMT/scripts/merging"

MODEL_ID=NLLB15
MERGE_ID=H4

MODELS=(
# NLLB 9
#"/net/storage/pr3/plgrid/plggmultilingualnlp/creole-nllb/NLLB9/checkpoints/nllb-best-global_step=49999.0-eval_chrf=0.00.ckpt"
#"/net/storage/pr3/plgrid/plggmultilingualnlp/creole-nllb/NLLB9/checkpoints/nllb-best-global_step=59999.0-eval_chrf=0.00.ckpt"
#"/net/storage/pr3/plgrid/plggmultilingualnlp/creole-nllb/NLLB9/checkpoints/nllb-best-global_step=129999.0-eval_chrf=0.00.ckpt"
#"/net/storage/pr3/plgrid/plggmultilingualnlp/creole-nllb/NLLB9/checkpoints/nllb-best-global_step=139999.0-eval_chrf=0.00.ckpt"
# NLLB 15
"/net/storage/pr3/plgrid/plggmultilingualnlp/creole-nllb/$MODEL_ID/checkpoints/nllb-best-global_step=179999.0-eval_chrf=0.00.ckpt"
"/net/storage/pr3/plgrid/plggmultilingualnlp/creole-nllb/$MODEL_ID/checkpoints/nllb-best-global_step=184999.0-eval_chrf=0.00.ckpt"
"/net/storage/pr3/plgrid/plggmultilingualnlp/creole-nllb/$MODEL_ID/checkpoints/nllb-best-global_step=189999.0-eval_chrf=0.00.ckpt"
)

for MODEL in "${MODELS[@]}"
do
    BASE_NAME=$(basename $MODEL)
    OUTPUT="models"/"${MODEL_ID}__"${BASE_NAME/".0-eval_chrf=0.00.ckpt"/_weights.pt}
    python prepare_nllb_checkpoints.py --model $MODEL --output_path $OUTPUT
done



source /net/storage/pr3/plgrid/plggmultilingualnlp/mklimasz/creolesMT/venv-mergekit/bin/activate


mergekit-pytorch configs/$MERGE_ID.yaml "./models/$MERGE_ID" --cuda

source /net/storage/pr3/plgrid/plggmultilingualnlp/mklimasz/creolesMT/venv/bin/activate

mkdir ./models/$MERGE_ID/checkpoints/
python post_process_nllb_checkpoints.py \
    $MODEL \
    "./models/$MERGE_ID" \
    "./models/$MERGE_ID/checkpoints/nllb-best-global_step=0.0-eval_chrf=0.00.ckpt"

cp -r /net/storage/pr3/plgrid/plggmultilingualnlp/creole-nllb/$MODEL_ID/translation ./models/$MERGE_ID/

cd /net/storage/pr3/plgrid/plggmultilingualnlp/mklimasz/creolesMT/models

ln -s /net/storage/pr3/plgrid/plggmultilingualnlp/mklimasz/creolesMT/scripts/merging/models/$MERGE_ID $MERGE_ID
