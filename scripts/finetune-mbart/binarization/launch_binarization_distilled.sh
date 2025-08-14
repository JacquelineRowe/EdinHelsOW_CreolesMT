#!/bin/bash
#SBATCH --job-name=binarize
#SBATCH --output=logs/binarize_%j.out
#SBATCH --error=logs/binarize_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=16G
#SBATCH --partition=small
#SBATCH --account=project_2005815

echo "Processing multiple language pairs..."

# Load environment
source /scratch/project_2005815/members/degibert/fairseq-dummy/venv/bin/activate

export WORKDIR_ROOT=/scratch/project_2005815/members/degibert/creolesMT
export SPM_PATH=$WORKDIR_ROOT/sentencepiece/build/src/

# Define language pairs (ISO codes)
PAIRS=(eng-pap eng-pov eng-cri eng-kea)

declare -A LANG_TAGS=(
  [eng]=en_XX [pap]=tr_TR [kea]=my_MM [pov]=ml_IN [aoa]=gl_ES [cri]=tl_XX [fab]=te_IN [pre]=mk_MK
)

# Tokenization model and dictionary
SPM_MODEL="models/mbart50-many-to-many/sentence.bpe.model"
DICTIONARY="models/mbart50-many-to-many/dict.en_XX.txt"
BPE_TYPE="sentencepiece"

# Paths
SRC_DIR="data/preprocessed_lusophone/set_5/"
TRG_DIR="data/distilled/set_5"
OUT_DIR="data/mbart/set_5_distilled/o2m/tokenised"
DEST_DIR="data/mbart/set_5_distilled/o2m/data-bin/"

mkdir -p "$OUT_DIR"
mkdir -p "$DEST_DIR"

for PAIR in "${PAIRS[@]}"; do
    echo "Processing $PAIR..."

    TGT=$(echo $PAIR | cut -d'-' -f2)
    SRC=$(echo $PAIR | cut -d'-' -f1)
    NEWPAIR="${SRC}-${TGT}"

    SRC_LANG=${LANG_TAGS[$SRC]}
    TGT_LANG=${LANG_TAGS[$TGT]}

    TRAIN_PREF="train.${PAIR}"
    VALID_PREF="validation.${PAIR}"

    # Tokenize
    $SPM_PATH/spm_encode --model=$SPM_MODEL < "$SRC_DIR/$TRAIN_PREF.${SRC}_filtered" > "$OUT_DIR/train.$NEWPAIR.$SRC_LANG"
    $SPM_PATH/spm_encode --model=$SPM_MODEL < "$TRG_DIR/$TRAIN_PREF.${SRC}_filtered.${TGT}" > "$OUT_DIR/train.$NEWPAIR.$TGT_LANG"
    $SPM_PATH/spm_encode --model=$SPM_MODEL < "$SRC_DIR/$VALID_PREF.${SRC}" > "$OUT_DIR/dev.$NEWPAIR.$SRC_LANG"
    $SPM_PATH/spm_encode --model=$SPM_MODEL < "$TRG_DIR/$VALID_PREF.${SRC}.${TGT}" > "$OUT_DIR/dev.$NEWPAIR.$TGT_LANG"

    rm -f "$DEST_DIR/dict.en_XX.txt"

    # Binarize
    fairseq-preprocess \
        --source-lang $SRC_LANG --target-lang $TGT_LANG \
        --trainpref "$OUT_DIR/train.$NEWPAIR" \
        --validpref "$OUT_DIR/dev.$NEWPAIR" \
        --bpe $BPE_TYPE \
        --destdir $DEST_DIR \
        --joined-dictionary \
        --srcdict $DICTIONARY \
        --workers 40

done
