#!/usr/bin/env bash

export HF_HOME=/net/storage/pr3/plgrid/plggmultilingualnlp/huggingface

source /net/storage/pr3/plgrid/plggmultilingualnlp/mklimasz/creolesMT/venv/bin/activate

cd "/net/storage/pr3/plgrid/plggmultilingualnlp/mklimasz/creolesMT/scripts/merging"


MODEL_ID=MB7

cat "/net/storage/pr3/plgrid/plggmultilingualnlp/degibert/creolesMT/final/models/$MODEL_ID/dev_eval/best_checkpoints.txt"


MODELS=(
## MB7
    "/net/storage/pr3/plgrid/plggmultilingualnlp/degibert/creolesMT/final/models/$MODEL_ID/checkpoint.best_loss_5.7401.pt"
    "/net/storage/pr3/plgrid/plggmultilingualnlp/degibert/creolesMT/final/models/$MODEL_ID/checkpoint.best_loss_5.7442.pt"
## MB 8
#    "/net/storage/pr3/plgrid/plggmultilingualnlp/degibert/creolesMT/final/models/$MODEL_ID/checkpoint.best_loss_5.5082.pt"
#    "/net/storage/pr3/plgrid/plggmultilingualnlp/degibert/creolesMT/final/models/$MODEL_ID/checkpoint_8_10000.pt"
#    "/net/storage/pr3/plgrid/plggmultilingualnlp/degibert/creolesMT/final/models/$MODEL_ID/checkpoint_23_30000.pt"
#    "/net/storage/pr3/plgrid/plggmultilingualnlp/degibert/creolesMT/final/models/$MODEL_ID/checkpoint_4_5000.pt"
#    "/net/storage/pr3/plgrid/plggmultilingualnlp/degibert/creolesMT/final/models/$MODEL_ID/checkpoint_last.pt"
)

for MODEL in "${MODELS[@]}"
do
    BASE_NAME=$(basename $MODEL)
    OUTPUT="models"/"${MODEL_ID}__"${BASE_NAME/.pt/_weights.pt}
    python prepare_pt_checkpoints.py --model $MODEL --output_path $OUTPUT

    OUTPUT="models/${MODEL_ID}__${BASE_NAME/.pt/_hf/}/"
    python convert_mbart_original_checkpoint_to_pytorch.py $MODEL $OUTPUT \
    --hf_config facebook/mbart-large-50 \
    --spm_model "/net/storage/pr3/plgrid/plggmultilingualnlp/mklimasz/creolesMT/models/mbart50-many-to-many/sentence.bpe.model" \
    --lang_list "/net/storage/pr3/plgrid/plggmultilingualnlp/mklimasz/creolesMT/models/mbart50-many-to-many/ML50_langs.txt" \
    --mbart_50 \
    --finetuned
done



source /net/storage/pr3/plgrid/plggmultilingualnlp/mklimasz/creolesMT/venv-mergekit/bin/activate


mergekit-yaml configs/H1.yaml "./models/MB7+MB8__Linear" --cuda

mergekit-multi configs/H2.yaml --intermediate-dir ./intermidates --out-path ./models/MB7+MB8+kreyol-mt__Slerp --cuda


mergekit-yaml configs/H3.yaml "./models/KMT3+kreyol-mt__Linear" --cuda
