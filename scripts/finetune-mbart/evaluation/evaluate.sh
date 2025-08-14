#!/bin/bash
#SBATCH --job-name=mbart_eval
#SBATCH --output=logs/mbart_eval_%A_%a.out
#SBATCH --error=logs/mbart_eval_%A_%a.err
#SBATCH --time=00:06:00
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpusmall
#SBATCH --account=project_2005815
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-6

# Activate environment
source /scratch/project_2005815/members/degibert/fairseq-dummy/venv/bin/activate

export WORKDIR_ROOT=/scratch/project_2005815/members/degibert/creolesMT
export SPM_PATH=$WORKDIR_ROOT/sentencepiece/build/src/

# Language mapping
declare -A LANG_TAGS=(
  [eng]=en_XX [pap]=tr_TR [kea]=my_MM [pov]=ml_IN [aoa]=gl_ES [cri]=tl_XX [fab]=te_IN [pre]=mk_MK
)

PAIRS=(eng-pap eng-kea eng-pov eng-aoa eng-cri eng-fab eng-pre)
PAIR=${PAIRS[$SLURM_ARRAY_TASK_ID]}
SRC=$(echo $PAIR | cut -d'-' -f2)
TGT=$(echo $PAIR | cut -d'-' -f1)
NEWPAIR=$SRC-$TGT

SRC_LANG=${LANG_TAGS[$SRC]}
TGT_LANG=${LANG_TAGS[$TGT]}

echo "Evaluating ${PAIR}..."

# File paths
RAW_DIR="data/preprocessed/data_all/raw/"
OUT_DIR="data/mbart/tokenised"
TEST_PREF="${RAW_DIR}/test.${PAIR}"
DEST_DIR="data/mbart/data-bin-test/"
RESULTS_PATH="models/mbart50-many-to-many-ft/results/$NEWPAIR"
MODEL_PATH="models/mbart50-many-to-many-ft/checkpoint_best.pt"
LANG_LIST="models/mbart50-many-to-many/ML50_langs.txt"
LANG_PAIRS="tr_TR-en_XX,my_MM-en_XX,ml_IN-en_XX,gl_ES-en_XX,tl_XX-en_XX,te_IN-en_XX,mk_MK-en_XX"
SPM_MODEL="models/mbart50-many-to-many/sentence.bpe.model"
DICTIONARY="models/mbart50-many-to-many/dict.en_XX.txt"

mkdir -p "$OUT_DIR" "$DEST_DIR" "$RESULTS_PATH"

# SentencePiece encode
$SPM_PATH/spm_encode --model=$SPM_MODEL < "$TEST_PREF.${SRC}" > "$OUT_DIR/test.$NEWPAIR.$SRC_LANG"
$SPM_PATH/spm_encode --model=$SPM_MODEL < "$TEST_PREF.${TGT}" > "$OUT_DIR/test.$NEWPAIR.$TGT_LANG"

# Binarize test set
fairseq-preprocess \
    --source-lang $SRC_LANG --target-lang $TGT_LANG \
    --testpref "$OUT_DIR/test.$NEWPAIR" \
    --destdir $DEST_DIR \
    --bpe sentencepiece \
    --joined-dictionary \
    --srcdict $DICTIONARY \
    --workers 10

# Generate translations
CUDA_VISIBLE_DEVICES=0 fairseq-generate $DEST_DIR \
  --lang-dict "$LANG_LIST" \
  --gen-subset test \
  --task translation_multi_simple_epoch \
  --path $MODEL_PATH \
  --remove-bpe sentencepiece \
  --batch-size 32 \
  --results-path "$RESULTS_PATH" \
  --source-lang $SRC_LANG --target-lang $TGT_LANG

# Postprocess and evaluate
GEN_FILE="$RESULTS_PATH/generate-test.txt"
HYP_FILE="$RESULTS_PATH/hyp.txt"
REF_FILE="$RESULTS_PATH/ref.txt"

grep -P "^H" $GEN_FILE | sort -V | cut -f 3- > $HYP_FILE
grep -P "^T" $GEN_FILE | sort -V | cut -f 2- > $REF_FILE

sed -i 's/__en_XX__ //' $HYP_FILE

# Evaluate with sacreBLEU and chrF
echo -e "\nBLEU:"
sacrebleu $REF_FILE < $HYP_FILE

echo -e "\nchrF:"
sacrebleu $REF_FILE --metrics chrf < $HYP_FILE