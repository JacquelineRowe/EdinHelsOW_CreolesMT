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

###############################################################################
# Script for evaluating multilingual models using Fairseq's generation script
# It expects the test set to be in data/test
# Usage:
#   ./final_generate_best.sh <MODEL_ID> <LANG_SETTING>
# Where:
#   MODEL_ID is the name of your trained model (e.g., MB2)
#   LANG_SETTING is one of: o2m, m2m, or default (m2o)
###############################################################################

set -e  # Exit on error
set -u  # Treat unset variables as errors
set -o pipefail

# === Parse arguments ===
if [[ $# -ne 2 ]]; then
    echo "❌ Usage: $0 <MODEL_ID> <LANG_SETTING>"
    echo "Example: $0 MB2 o2m"
    exit 1
fi

ID=$1
LANGS=$2

# === Paths ===
MODEL_DIR="final/models/${ID}"
BEST_CKPT_FILE="$MODEL_DIR/dev_eval/best_checkpoints.txt"
LANG_LIST="final/models/ML50_langs.txt"
DATA_PATH="data/test"
DATA_PATH_TOK="${DATA_PATH}/tokenised"
DATA_PATH_BIN="${DATA_PATH}/data-bin"

# === Static parameters ===
SPM_PATH="/net/storage/pr3/plgrid/plggmultilingualnlp/mklimasz/sentencepiece/build/src"
SPM_MODEL="final/models/sentence.bpe.model"
BPE_TYPE="sentencepiece"
DICTIONARY="final/models/dict.en_XX.txt"           # Make sure this exists!

PAIRS=(eng-pap eng-pov eng-cri eng-kea eng-aoa eng-fab eng-pre)
declare -A LANG_TAGS=(
  [eng]=en_XX [pap]=tr_TR [kea]=my_MM [pov]=ml_IN [aoa]=gl_ES [cri]=tl_XX [fab]=te_IN [pre]=mk_MK
)

echo "🔄 Starting preprocessing..."

# === Tokenize + Binarize ===
if [[ -d "$DATA_PATH_BIN" && "$(ls -A "$DATA_PATH_BIN")" ]]; then
    echo "📦 Binarized data already exists in $DATA_PATH_BIN — skipping tokenization and binarization. Delete if you want to overwrite."
else
    echo "🔄 Tokenizing and binarizing data into $DATA_PATH_BIN..."
    mkdir -p "$DATA_PATH_TOK" "$DATA_PATH_BIN"

    for PAIR in "${PAIRS[@]}"; do
        SRC=$(echo "$PAIR" | cut -d'-' -f1)
        TGT=$(echo "$PAIR" | cut -d'-' -f2)
        SRC_LANG=${LANG_TAGS[$SRC]}
        TGT_LANG=${LANG_TAGS[$TGT]}

        echo "🧽 Tokenizing test.${PAIR}.${SRC} and .${TGT}"

        $SPM_PATH/spm_encode --model="$SPM_MODEL" < "$DATA_PATH/test.${PAIR}.${SRC}" > "$DATA_PATH_TOK/test.$PAIR.$SRC_LANG"
        $SPM_PATH/spm_encode --model="$SPM_MODEL" < "$DATA_PATH/test.${PAIR}.${TGT}" > "$DATA_PATH_TOK/test.$PAIR.$TGT_LANG"

        rm -f "$DATA_PATH_BIN/dict.en_XX.txt"

        echo "📦 Binarizing $PAIR..."
        fairseq-preprocess \
            --source-lang "$SRC_LANG" --target-lang "$TGT_LANG" \
            --testpref "$DATA_PATH_TOK/test.$PAIR" \
            --bpe "$BPE_TYPE" \
            --destdir "$DATA_PATH_BIN" \
            --joined-dictionary \
            --srcdict "$DICTIONARY" \
            --workers 10
    done
    echo "✅ Preprocessing complete."
fi

# === Language pairs for generation ===
if [[ "$LANGS" == "o2m" ]]; then
    LANG_PAIRS="en_XX-tr_TR,en_XX-my_MM,en_XX-ml_IN,en_XX-tl_XX,en_XX-gl_ES,en_XX-te_IN,en_XX-mk_MK"
elif [[ "$LANGS" == "m2m" ]]; then
    LANG_PAIRS="tr_TR-en_XX,my_MM-en_XX,ml_IN-en_XX,gl_ES-en_XX,tl_XX-en_XX,te_IN-en_XX,mk_MK-en_XX,en_XX-tr_TR,en_XX-my_MM,en_XX-ml_IN,en_XX-gl_ES,en_XX-tl_XX,en_XX-te_IN,en_XX-mk_MK"
else
    LANG_PAIRS="tr_TR-en_XX,my_MM-en_XX,ml_IN-en_XX,tl_XX-en_XX,gl_ES-en_XX,te_IN-en_XX,mk_MK-en_XX"
fi

declare -A TAG_LANGS=(  [en_XX]=eng  [tr_TR]=pap  [my_MM]=kea  [ml_IN]=pov  [gl_ES]=aoa  [tl_XX]=cri  [te_IN]=fab  [mk_MK]=pre )

# === Generation ===
OUTPUT_DIR="submission/${ID}"
mkdir -p "$OUTPUT_DIR"

echo "🚀 Starting generation for LANGS=$LANGS..."

for PAIR in ${LANG_PAIRS//,/ }; do
    SRC=$(echo "$PAIR" | cut -d'-' -f1)
    TGT=$(echo "$PAIR" | cut -d'-' -f2)
    SRC_LANG=${TAG_LANGS[$SRC]}
    TGT_LANG=${TAG_LANGS[$TGT]}
    NEWPAIR="${SRC_LANG}-${TGT_LANG}"
    RESULT_SUBDIR="$OUTPUT_DIR/$NEWPAIR"
    mkdir -p "$RESULT_SUBDIR"

    GEN_FILE="$RESULT_SUBDIR/generate-test.txt"
    GEN_SORTED="$RESULT_SUBDIR/generate-test_sorted.txt"
    HYP_FILE="$RESULT_SUBDIR/hyp.txt"
    REF_FILE="$RESULT_SUBDIR/ref.txt"
    SCORE_FILE="$RESULT_SUBDIR/scores.txt"
    SRC_FILE="$RESULT_SUBDIR/src.txt"

    if [[ -f "$GEN_FILE" ]]; then
        echo "⚠️ $ID/$NEWPAIR already generated. Skipping. Delete the folder to regenerate."
        continue
    fi

    echo "🔍 Evaluating $NEWPAIR..."

    # Get checkpoint
    if [[ -f "$BEST_CKPT_FILE" ]]; then
        CKPT_NAME=$(awk -v pair="$NEWPAIR" '$1 == pair { print $2 }' "$BEST_CKPT_FILE")
        if [[ -z "$CKPT_NAME" ]]; then
            echo "⚠️ No checkpoint entry for $NEWPAIR. Using default checkpoint_best.pt"
            MODEL_PATH="$MODEL_DIR/checkpoint_best.pt"
        else
            MODEL_PATH="$MODEL_DIR/$CKPT_NAME.pt"
            echo "✅ Using best checkpoint for $NEWPAIR: $CKPT_NAME.pt"
        fi
    else
        echo "⚠️ No best_checkpoints.txt found. Using checkpoint_best.pt"
        MODEL_PATH="$MODEL_DIR/checkpoint_best.pt"
    fi

    # Generate
    CUDA_VISIBLE_DEVICES=0 fairseq-generate "$DATA_PATH_BIN" \
        --lang-dict "$LANG_LIST" \
        --lang-pairs "$LANG_PAIRS" \
        --gen-subset test \
        --task translation_multi_simple_epoch \
        --path "$MODEL_PATH" \
        --remove-bpe sentencepiece \
        --batch-size 32 \
        --results-path "$RESULT_SUBDIR" \
        --source-lang "$SRC" --target-lang "$TGT" \
        --encoder-langtok "src" \
        --decoder-langtok \
        --fp16  --required-batch-size-multiple 1

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
    
    if grep -q -P "^T" "$GEN_SORTED"; then
    	grep -P "^T" "$GEN_SORTED" | sort -V | cut -f 2- > "$REF_FILE"
    else
	echo "⚠️ No reference lines (^T) found in $GEN_SORTED — skipping ref.txt creation."
    fi

    sed -i 's/__[a-z][a-z]_[A-Z][A-Z]__ //' "$HYP_FILE"
    grep -P '^S-' "$GEN_SORTED" | cut -f2-  > "$SRC_FILE"

    # Character fixes
    sed -i "s/@\"/\"/g; s/調\"/\"/g; s/付\"/\"/g; s/혼'/'/g; s/ච'/'/g; s/완-/-/g; s/罪-/-/g; s/«/<</g; s/»/>>/g; s/‚/,/g; s/ ' /'/g" "$HYP_FILE"

    echo "✅ Done with $ID/$NEWPAIR"
done


echo "🎉 All done! Results in: submission/${ID}"
