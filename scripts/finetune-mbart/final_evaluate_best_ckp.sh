#!/bin/bash
#SBATCH --job-name=eval_best
#SBATCH --output=logs/eval_best_%j.out
#SBATCH --error=logs/eval_best_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpusmall
#SBATCH --account=project_2005815
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1

ID=$1
LANGS=$2
MODEL_DIR="final/models/${ID}"
BEST_CKPT_FILE="$MODEL_DIR/dev_eval/best_checkpoints.txt"
LANG_LIST="models/mbart50-many-to-many/ML50_langs.txt"

if [[ "$LANGS" == "o2m" ]]; then
    LANG_PAIRS="en_XX-tr_TR,en_XX-my_MM,en_XX-ml_IN,en_XX-gl_ES,en_XX-tl_XX,en_XX-te_IN,en_XX-mk_MK"
elif [[ "$LANGS" == "m2m" ]]; then
    LANG_PAIRS="tr_TR-en_XX,my_MM-en_XX,ml_IN-en_XX,gl_ES-en_XX,tl_XX-en_XX,te_IN-en_XX,mk_MK-en_XX,en_XX-tr_TR,en_XX-my_MM,en_XX-ml_IN,en_XX-gl_ES,en_XX-tl_XX,en_XX-te_IN,en_XX-mk_MK"
else
    LANG_PAIRS="tr_TR-en_XX,my_MM-en_XX,ml_IN-en_XX,gl_ES-en_XX,tl_XX-en_XX,te_IN-en_XX,mk_MK-en_XX"
fi

# Load environment
source /scratch/project_2005815/members/degibert/fairseq-dummy/venv/bin/activate

# Evaluation
for type in "all" "kmt" "ours"; do
    echo "Evaluating on $type set"

    mkdir -p "final/results/${ID}_best"
    mkdir -p "final/results/${ID}_best/${type}"
    MODEL_SCORE_LOG="final/results/${ID}_best/${type}/scores.txt"
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
        RESULT_SUBDIR="final/results/${ID}_best/${type}/${NEWPAIR}"
        mkdir -p "$RESULT_SUBDIR"

        GEN_FILE="$RESULT_SUBDIR/generate-test.txt"
        GEN_SORTED="$RESULT_SUBDIR/generate-test_sorted.txt"
        HYP_FILE="$RESULT_SUBDIR/hyp.txt"
        REF_FILE="$RESULT_SUBDIR/ref.txt"
        SCORE_FILE="$RESULT_SUBDIR/scores.txt"
        SRC_FILE="$RESULT_SUBDIR/src.txt"

        if [[ ! -f "$GEN_FILE" ]]; then
            echo "Evaluating ${ID}/${type}/${NEWPAIR}, best checkpoint per language pair..."

             # Obtain best checkpoint per language pair
            if [[ -f "$BEST_CKPT_FILE" ]]; then
                CKPT_NAME=$(awk -v pair="$NEWPAIR" '$1 == pair { print $2 }' "$BEST_CKPT_FILE")
                if [[ -z "$CKPT_NAME" ]]; then
                    echo "⚠️ No entry for $NEWPAIR in best_checkpoints.txt, using checkpoint_best.pt"
                    MODEL_PATH="$MODEL_DIR/checkpoint_best.pt"
                else
                    MODEL_PATH="$MODEL_DIR/$CKPT_NAME.pt"
                    echo "Evaluating best checkpoint for this langauge pair: $CKPT_NAME.pt"
                fi
            else
                echo "⚠️ No best_checkpoints.txt found, using checkpoint_best.pt for all"
                MODEL_PATH="$MODEL_DIR/checkpoint_best.pt"
            fi

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
                --encoder-langtok "src" --decoder-langtok --fp16 --required-batch-size-multiple 1

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