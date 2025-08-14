#!/bin/bash
#SBATCH --job-name=mbart
#SBATCH --output=logs/mbart_final_%j.out
#SBATCH --error=logs/mbart_final_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpusmall
#SBATCH --account=project_2005815
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1

# Usage check
if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <ID> <BASE_MODEL> <LANGS> <POR_EMB> <POR_DATA> <DIST_DATA>"
  echo "Example: $0 MB1 many-to-many m2o True False True"
  exit 1
fi

ID=$1
BASE_MODEL=$2
LANGS=$3
POR_EMB=$4
POR_DATA=$5
DIST_DATA=$6

if [[ "$LANGS" == "o2m" ]]; then
    LANG_PAIRS="en_XX-tr_TR,en_XX-my_MM,en_XX-ml_IN,en_XX-tl_XX,en_XX-gl_ES,en_XX-te_IN,en_XX-mk_MK"
elif [[ "$LANGS" == "m2m" ]]; then
    LANG_PAIRS="tr_TR-en_XX,my_MM-en_XX,ml_IN-en_XX,gl_ES-en_XX,tl_XX-en_XX,te_IN-en_XX,mk_MK-en_XX,en_XX-tr_TR,en_XX-my_MM,en_XX-ml_IN,en_XX-gl_ES,en_XX-tl_XX,en_XX-te_IN,en_XX-mk_MK"
else
    LANG_PAIRS="tr_TR-en_XX,my_MM-en_XX,ml_IN-en_XX,tl_XX-en_XX,gl_ES-en_XX,te_IN-en_XX,mk_MK-en_XX"
fi

echo "================================================="
echo "Training model ID:       $ID"
echo "Using base model:        $BASE_MODEL"
echo "Language direction:      $LANGS"
echo "Use Portuguese emb:      $POR_EMB"
echo "Use Portuguese data:     $POR_DATA"
echo "Use Distilled data:      $DIST_DATA"
echo "================================================="

# Load environment
source /scratch/project_2005815/members/degibert/fairseq-dummy/venv/bin/activate

# Training config
export MASTER_ADDR=localhost
export MASTER_PORT=12345
export WORLD_SIZE=4
export RANK=0
export PYTORCH_WEIGHTS_ONLY=0

DATA="data/mbart/set_5/data-bin"
PRETRAINED_MODEL="models/mbart50-$BASE_MODEL/model_fixed.pt"

if [[ "$POR_EMB" == "True" ]]; then
    PRETRAINED_MODEL="models/mbart50-$BASE_MODEL/model_modified.pt"
fi

if [[ "$POR_DATA" == "True" && "$LANGS" == "o2m" ]]; then
    LANG_PAIRS="${LANG_PAIRS},en_XX-pt_XX"
elif [[ "$POR_DATA" == "True" && "$LANGS" == "m2m" ]]; then
    LANG_PAIRS="${LANG_PAIRS},pt_XX-en_XX,en_XX-pt_XX"
elif [[ "$POR_DATA" == "True" && "$LANGS" == "m2o" ]]; then
    LANG_PAIRS="${LANG_PAIRS},pt_XX-en_XX"
fi

if [[ "$DIST_DATA" == "True" && "$LANGS" == "o2m" ]]; then
    #LANG_PAIRS="en_XX-tr_TR,en_XX-my_MM,en_XX-ml_IN,en_XX-tl_XX"
    DATA="data/mbart/set_5_distilled/${LANGS}/data-bin"
elif [[ "$DIST_DATA" == "True" && "$LANGS" == "m2o" ]]; then
    #LANG_PAIRS="tr_TR-en_XX,my_MM-en_XX,ml_IN-en_XX,tl_XX-en_XX"
    DATA="data/mbart/set_5_distilled/${LANGS}/data-bin"
fi

MODEL_DIR="final/models/${ID}"
SPM_MODEL="models/mbart50-$BASE_MODEL/sentence.bpe.model"
DICTIONARY="models/mbart50-$BASE_MODEL/dict.en_XX.txt"
LANG_LIST="models/mbart50-$BASE_MODEL/ML50_langs.txt"

mkdir -p $MODEL_DIR

echo "Language Pairs:          $LANG_PAIRS"
echo "Pretrained Model:        $PRETRAINED_MODEL"
echo "Data path:               $DATA"
echo "Model output path:       $MODEL_DIR"
echo "Starting training..."
echo "================================================="

# Train

if [[ ! -f "$MODEL_DIR/checkpoint_best.pt" ]]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train $DATA \
    --save-dir $MODEL_DIR --wandb-project CreolesMT \
    --finetune-from-model $PRETRAINED_MODEL --log-file $MODEL_DIR/train.log \
    --encoder-normalize-before --decoder-normalize-before \
    --arch mbart_large --layernorm-embedding \
    --task translation_multi_simple_epoch \
    --sampling-method "temperature" --sampling-temperature 2 \
    --encoder-langtok "src" --decoder-langtok \
    --lang-dict "$LANG_LIST" \
    --lang-pairs "$LANG_PAIRS" \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --lr 3e-05 --warmup-updates 2500 --max-update 40000 \
    --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
    --max-tokens 1024 --update-freq 2 --keep-best-checkpoints 3 \
    --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
    --seed 222 --log-format simple --log-interval 2 \
    --validate-interval-updates 5000 --patience 10 \
    --distributed-world-size 4 --distributed-init-method env:// --fp16
else
   echo "Model $ID already trained, skipping."
fi

# Evaluation
for type in "all" "kmt" "ours"; do
    echo "Evaluating on $type set"

    mkdir -p "final/results/${ID}"
    mkdir -p "final/results/${ID}/${type}"
    MODEL_SCORE_LOG="final/results/${ID}/${type}/scores.txt"
    echo -e "LangPair\tBLEU\tchrF" > "$MODEL_SCORE_LOG"

    declare -A TAG_LANGS=(  [en_XX]=eng  [tr_TR]=pap  [my_MM]=kea  [ml_IN]=pov  [gl_ES]=aoa  [tl_XX]=cri  [te_IN]=fab  [mk_MK]=pre )

    for PAIR in ${LANG_PAIRS//,/ }; do
        SRC=$(echo $PAIR | cut -d'-' -f1)
        TGT=$(echo $PAIR | cut -d'-' -f2)
        SRC_LANG=${TAG_LANGS[$SRC]}
        TGT_LANG=${TAG_LANGS[$TGT]}
        NEWPAIR="${SRC_LANG}-${TGT_LANG}"

        if [[ "$type" == "ours" ]]; then
            if [[ "$SRC_LANG" == "fab" || "$SRC_LANG" == "aoa" || "$SRC_LANG" == "pre" || \
                "$TGT_LANG" == "fab" || "$TGT_LANG" == "aoa" || "$TGT_LANG" == "pre" ]]; then
                echo "Skipping $PAIR for 'ours' set (excluded language)."
                continue
            fi
        fi

        DEST_DIR="data/mbart/test/${type}/data-bin/"
        RESULT_SUBDIR="final/results/${ID}/${type}/${NEWPAIR}"
        mkdir -p "$RESULT_SUBDIR"

        GEN_FILE="$RESULT_SUBDIR/generate-test.txt"
        GEN_SORTED="$RESULT_SUBDIR/generate-test_sorted.txt"
        HYP_FILE="$RESULT_SUBDIR/hyp.txt"
        REF_FILE="$RESULT_SUBDIR/ref.txt"
        SCORE_FILE="$RESULT_SUBDIR/scores.txt"
        SRC_FILE="$RESULT_SUBDIR/src.txt"

        if [[ ! -f "$GEN_FILE" ]]; then
            echo "Evaluating ${ID}/${type}/${NEWPAIR}..."

            MODEL_PATH="$MODEL_DIR/checkpoint_best.pt"

            CUDA_VISIBLE_DEVICES=0 fairseq-generate $DEST_DIR \
                --lang-dict "$LANG_LIST" \
                --lang-pairs "$LANG_PAIRS" \
                --gen-subset test \
                --task translation_multi_simple_epoch \
                --path $MODEL_PATH \
                --remove-bpe sentencepiece \
                --batch-size 32 \
                --results-path "$RESULT_SUBDIR" \
                --source-lang $SRC --target-lang $TGT \
                --fp16 --required-batch-size-multiple 1

            awk '
            /^(S|T|H|D|P)-[0-9]+/ {
            # Extract type and ID
            split($1, parts, "-")
            type = parts[1]
            id = parts[2]

            key = sprintf("%05d", id)  # zero-pad for numeric sorting
            block[key][type] = $0
            ids[key] = 1
            }
            END {
            n = asorti(ids, sorted_ids)
            for (i = 1; i <= n; i++) {
                id = sorted_ids[i]
                for (t in block[id]) {
                print block[id][t]
                }
            }
            }
            ' "$GEN_FILE" > "$GEN_SORTED"

            grep -P "^H" $GEN_SORTED | sort -V | cut -f 3- > $HYP_FILE
            grep -P "^T" $GEN_SORTED | sort -V | cut -f 2- > $REF_FILE
            sed -i 's/__[a-z][a-z]_[A-Z][A-Z]__ //' "$HYP_FILE"
            grep -P '^S-' "$GEN_SORTED" | cut -f2-  > "$SRC_FILE"

            # Optional character fixes
            sed -i "s/@\"/\"/g; s/調\"/\"/g; s/付\"/\"/g; s/혼'/'/g; s/ච'/'/g; s/완-/-/g; s/罪-/-/g; s/«/<</g; s/»/>>/g; s/‚/,/g; s/ ' /'/g" "$HYP_FILE"

            BLEU=$(sacrebleu $REF_FILE < $HYP_FILE | jq -r '.score')
            CHRF=$(sacrebleu $REF_FILE --metrics chrf < $HYP_FILE | jq -r '.score')

            echo -e "$NEWPAIR\t$BLEU\t$CHRF" >> "$MODEL_SCORE_LOG"
            echo -e "BLEU: $BLEU\nchrF: $CHRF" > "$SCORE_FILE"
        else
            echo "$ID/$type/$NEWPAIR: already evaluated, skipping."
        fi

        echo "$ID/$type/$NEWPAIR done."
    done
done

if [[ ! -f "final/hf/${ID}/config.json" ]]; then
    # Convert to HuggingFace
    python transformers/src/transformers/models/mbart/convert_mbart_original_checkpoint_to_pytorch.py \
    ${MODEL_DIR}/checkpoint_best.pt \
    final/hf/${ID} \
    --hf_config facebook/mbart-large-50 \
    --mbart_50 \
    --finetuned
fi