#!/bin/bash
#SBATCH --job-name=binarize
#SBATCH --output=logs/binarize_%A_%a.out
#SBATCH --error=logs/binarize_%A_%a.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=16G
#SBATCH --partition=small
#SBATCH --array=0
#SBATCH --account=project_2005815

set=$1
echo "Processing Test Set..."

# Load environment
source /scratch/project_2005815/members/degibert/fairseq-dummy/venv/bin/activate

export WORKDIR_ROOT=/scratch/project_2005815/members/degibert/creolesMT
export SPM_PATH=$WORKDIR_ROOT/sentencepiece/build/src/

# Define language pairs (ISO codes)
PAIRS=( eng-kea ) # eng-pap eng-kea eng-pov eng-aoa eng-cri eng-fab eng-pre
PAIR=${PAIRS[$SLURM_ARRAY_TASK_ID]}
SRC=$(echo $PAIR | cut -d'-' -f2)
TGT=$(echo $PAIR | cut -d'-' -f1)
NEWPAIR=$SRC-$TGT

declare -A LANG_TAGS=(
  [eng]=en_XX [pap]=tr_TR [kea]=my_MM [pov]=ml_IN [aoa]=gl_ES [cri]=tl_XX [fab]=te_IN [pre]=mk_MK
)

declare -A DIR=(
  [""]=all ["_kmt"]=kmt ["_ours"]=ours
)


SRC_LANG=${LANG_TAGS[$SRC]}
TGT_LANG=${LANG_TAGS[$TGT]}

RAW_DIR="data/preprocessed_lusophone/set_5/"

for type in "_kmt" "" "_ours"; do
  TEST_PREF="${RAW_DIR}/test${type}.${PAIR}"
  OUT_DIR="data/mbart/test/${DIR[$type]}/tokenised"
  mkdir -p "data/mbart/test"
  mkdir -p "data/mbart/test/${DIR[$type]}"
  mkdir -p "$OUT_DIR"


  # Tokenize
  SPM_MODEL="models/mbart50-many-to-many/sentence.bpe.model"

  cat "$TEST_PREF.${SRC}" | $SPM_PATH/spm_encode --model=$SPM_MODEL > $OUT_DIR/test.$NEWPAIR.$SRC_LANG
  cat "$TEST_PREF.${TGT}" | $SPM_PATH/spm_encode --model=$SPM_MODEL > $OUT_DIR/test.$NEWPAIR.$TGT_LANG

  # Binarize
  DEST_DIR="data/mbart/test/${DIR[$type]}/data-bin/"
  mkdir -p $DEST_DIR

  BPE_TYPE="sentencepiece"
  DICTIONARY="models/mbart50-many-to-many/dict.en_XX.txt"

  rm -f "$DEST_DIR/dict.en_XX.txt"

  fairseq-preprocess \
      --source-lang $SRC_LANG --target-lang $TGT_LANG \
      --testpref "$OUT_DIR/test.$NEWPAIR"  \
      --bpe $BPE_TYPE \
      --destdir $DEST_DIR \
      --joined-dictionary \
      --srcdict $DICTIONARY \
      --workers 40
done