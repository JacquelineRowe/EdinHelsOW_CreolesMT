#!/usr/bin/env bash
export HF_HOME=/net/storage/pr3/plgrid/plggmultilingualnlp/huggingface
source /net/storage/pr3/plgrid/plggmultilingualnlp/mklimasz/creolesMT/venv/bin/activate
cd "/net/storage/pr3/plgrid/plggmultilingualnlp/mklimasz/creolesMT/scripts/merging"

MODEL_ID=KMT10-NEW
MERGE_ID=H6

MODELS=(
# KMT3-NEW1
#"/net/storage/pr3/plgrid/plggmultilingualnlp/creole-nllb/$MODEL_ID/checkpoints/kreyolmt-best-global_step=14999.0-eval_chrf=0.00.ckpt"
#"/net/storage/pr3/plgrid/plggmultilingualnlp/creole-nllb/$MODEL_ID/checkpoints/kreyolmt-best-global_step=19999.0-eval_chrf=0.00.ckpt"
#"/net/storage/pr3/plgrid/plggmultilingualnlp/creole-nllb/$MODEL_ID/checkpoints/kreyolmt-best-global_step=24999.0-eval_chrf=0.00.ckpt"
# KMT10-NEW
"/net/storage/pr3/plgrid/plggmultilingualnlp/creole-nllb/$MODEL_ID/checkpoints/kreyolmt-best-global_step=29999.0-eval_chrf=0.00.ckpt"
"/net/storage/pr3/plgrid/plggmultilingualnlp/creole-nllb/$MODEL_ID/checkpoints/kreyolmt-best-global_step=34999.0-eval_chrf=0.00.ckpt"
"/net/storage/pr3/plgrid/plggmultilingualnlp/creole-nllb/$MODEL_ID/checkpoints/kreyolmt-best-global_step=39999.0-eval_chrf=0.00.ckpt"
)

for MODEL in "${MODELS[@]}"
do
    BASE_NAME=$(basename $MODEL)
    OUTPUT="models"/"${MODEL_ID}__"${BASE_NAME/".0-eval_chrf=0.00.ckpt"/_weights.pt}
    python prepare_nllb_checkpoints.py --model $MODEL --output_path $OUTPUT --prefix "kreyolmt."
done



source /net/storage/pr3/plgrid/plggmultilingualnlp/mklimasz/creolesMT/venv-mergekit/bin/activate


mergekit-pytorch configs/$MERGE_ID.yaml "./models/$MERGE_ID" --cuda

source /net/storage/pr3/plgrid/plggmultilingualnlp/mklimasz/creolesMT/venv/bin/activate

mkdir ./models/$MERGE_ID/checkpoints/
python post_process_nllb_checkpoints.py \
    $MODEL \
    "./models/$MERGE_ID" \
    "./models/$MERGE_ID/checkpoints/kreyolmt-best-global_step=0.0-eval_chrf=0.00.ckpt" \
    --prefix "kreyolmt."

cp -r /net/storage/pr3/plgrid/plggmultilingualnlp/creole-nllb/$MODEL_ID/translation ./models/$MERGE_ID/

cd /net/storage/pr3/plgrid/plggmultilingualnlp/mklimasz/creolesMT/models

ln -s /net/storage/pr3/plgrid/plggmultilingualnlp/mklimasz/creolesMT/scripts/merging/models/$MERGE_ID $MERGE_ID
