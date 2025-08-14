#!/bin/bash
#SBATCH --job-name=binarize
#SBATCH --output=logs/binarize_%A_%a.out
#SBATCH --error=logs/binarize_%A_%a.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=16G
#SBATCH --partition=test
#SBATCH --account=project_2005815

# Load environment
source /scratch/project_2005815/members/degibert/fairseq-dummy/venv/bin/activate

export WORKDIR_ROOT=/scratch/project_2005815/members/degibert/creolesMT
export SPM_PATH=$WORKDIR_ROOT/sentencepiece/build/src/

SRC="por"
TGT="eng"
PAIR=$SRC-$TGT

declare -A LANG_TAGS=(
  [eng]=en_XX [por]=pt_XX
)

SRC_LANG=${LANG_TAGS[$SRC]}
TGT_LANG=${LANG_TAGS[$TGT]}

RAW_DIR="new_data/por"
TRAIN_PREF="${RAW_DIR}/train.${PAIR}"
OUT_DIR="data/mbart/tokenised"

# Tokenize
SPM_MODEL="models/mbart50-many-to-many/sentence.bpe.model"

#cat "$TRAIN_PREF.${SRC}" | $SPM_PATH/spm_encode --model=$SPM_MODEL > $OUT_DIR/train.$PAIR.$SRC_LANG
#cat "$TRAIN_PREF.${TGT}" | $SPM_PATH/spm_encode --model=$SPM_MODEL > $OUT_DIR/train.$PAIR.$TGT_LANG

# Binarize
DEST_DIR="data/mbart/data-bin/"

BPE_TYPE="sentencepiece"
DICTIONARY="models/mbart50-many-to-many/dict.en_XX.txt"

fairseq-preprocess \
    --source-lang $SRC_LANG --target-lang $TGT_LANG \
    --trainpref "$OUT_DIR/train.$PAIR"  \
    --validpref "$RAW_DIR/valid.$PAIR" \
    --bpe $BPE_TYPE \
    --destdir $DEST_DIR \
    --joined-dictionary \
    --srcdict $DICTIONARY \
    --workers 40