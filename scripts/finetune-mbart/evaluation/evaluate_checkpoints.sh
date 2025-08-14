#!/bin/bash
#SBATCH --job-name=eval_ckpts
#SBATCH --output=logs/eval_ckpts_%j.out
#SBATCH --error=logs/eval_ckpts_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpusmall
#SBATCH --account=project_2005815
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1

set -e
source /scratch/project_2005815/members/degibert/fairseq-dummy/venv/bin/activate

MODEL_NAME="$1"
DIRECTION=$2     # "m2m", "o2m", or "m2o"

case "$DIRECTION" in
  "o2m")
    LANG_PAIRS="en_XX-tr_TR,en_XX-my_MM,en_XX-ml_IN,en_XX-tl_XX,en_XX-gl_ES,en_XX-te_IN,en_XX-mk_MK"
    ;;
  "m2o")
    LANG_PAIRS="tr_TR-en_XX,my_MM-en_XX,ml_IN-en_XX,tl_XX-en_XX,gl_ES-en_XX,te_IN-en_XX,mk_MK-en_XX"
    ;;
  "m2m")
    LANG_PAIRS="tr_TR-en_XX,my_MM-en_XX,ml_IN-en_XX,gl_ES-en_XX,tl_XX-en_XX,te_IN-en_XX,mk_MK-en_XX,en_XX-tr_TR,en_XX-my_MM,en_XX-ml_IN,en_XX-gl_ES,en_XX-tl_XX,en_XX-te_IN,en_XX-mk_MK"
    ;;
  *)
    echo "Unknown direction: $DIRECTION. Use one of: m2m, o2m, m2o"
    exit 1
    ;;
esac

export WORKDIR_ROOT=/scratch/project_2005815/members/degibert/creolesMT
export SPM_PATH=$WORKDIR_ROOT/sentencepiece/build/src/
MODEL_DIR="$WORKDIR_ROOT/final/models/$MODEL_NAME"
BASE_MODEL="$WORKDIR_ROOT/models/mbart50-many-to-many"
LANG_LIST="$BASE_MODEL/ML50_langs.txt"

declare -A TAG_LANGS=( [en_XX]=eng [tr_TR]=pap [my_MM]=kea [ml_IN]=pov [gl_ES]=aoa [tl_XX]=cri [te_IN]=fab [mk_MK]=pre )

DEST_DIR="$WORKDIR_ROOT/data/mbart/set_5/data-bin"

mkdir -p "$MODEL_DIR/dev_eval"

# ============ MAIN LOOP ============ #
for CKPT in "$MODEL_DIR"/checkpoint*.pt; do
  CKPT_NAME=$(basename "$CKPT" .pt)
  echo "🔍 Evaluating checkpoint: $CKPT_NAME"

  for PAIR in ${LANG_PAIRS//,/ }; do
    SRC=$(echo $PAIR | cut -d'-' -f1)
    TGT=$(echo $PAIR | cut -d'-' -f2)
    SRC_LANG=${TAG_LANGS[$SRC]}
    TGT_LANG=${TAG_LANGS[$TGT]}
    NEWPAIR="${SRC_LANG}-${TGT_LANG}"

    RESULT_SUBDIR="$MODEL_DIR/dev_eval/$CKPT_NAME/$NEWPAIR"
    mkdir -p "$RESULT_SUBDIR"

    GEN_FILE="$RESULT_SUBDIR/generate-valid.txt"
    GEN_SORTED="$RESULT_SUBDIR/generate-valid_sorted.txt"
    HYP_FILE="$RESULT_SUBDIR/hyp.txt"
    REF_FILE="$RESULT_SUBDIR/ref.txt"
    SCORE_FILE="$RESULT_SUBDIR/scores.txt"
    SRC_FILE="$RESULT_SUBDIR/src.txt"

    if [[ ! -f "$GEN_FILE" ]]; then
      echo "⏳ $CKPT_NAME/$NEWPAIR..."
      CUDA_VISIBLE_DEVICES=0 fairseq-generate $DEST_DIR \
        --lang-dict "$LANG_LIST" \
        --lang-pairs "$LANG_PAIRS" \
        --gen-subset valid \
        --task translation_multi_simple_epoch \
        --path "$CKPT" \
        --remove-bpe sentencepiece \
        --batch-size 32 \
        --results-path "$RESULT_SUBDIR" \
        --source-lang "$SRC" \
        --target-lang "$TGT" --fp16 --required-batch-size-multiple 1

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

      
      # Remove special characters
      sed -i "s/@\"/\"/g" "$HYP_FILE"
      sed -i "s/調\"/\"/g" "$HYP_FILE"
      sed -i "s/付\"/\"/g" "$HYP_FILE"
      sed -i "s/혼'/'/g" "$HYP_FILE"
      sed -i "s/ච'/'/g" "$HYP_FILE"
      sed -i "s/완-/-/g" "$HYP_FILE"
      sed -i "s/罪-/-/g" "$HYP_FILE"
      sed -i "s/«/<</g" "$HYP_FILE"
      sed -i "s/»/>>/g" "$HYP_FILE"
      sed -i "s/‚/,/g" "$HYP_FILE"
      sed -i "s/ ' /'/g" "$HYP_FILE"

      BLEU=$(sacrebleu "$REF_FILE" < "$HYP_FILE" | jq -r '.score')
      CHRF=$(sacrebleu "$REF_FILE" --metrics chrf < "$HYP_FILE" | jq -r '.score')

      echo -e "BLEU: $BLEU\nchrF: $CHRF" > "$SCORE_FILE"
    else
      echo "✅ $CKPT_NAME/$NEWPAIR already evaluated."
    fi
  done
done

# ============ BEST CHECKPOINT PER PAIR ============ #
BEST_CKPT_FILE="$MODEL_DIR/dev_eval/best_checkpoints.txt"
echo -e "# LangPair\tCheckpoint" > "$BEST_CKPT_FILE"

echo -e "\n📈 Best Checkpoints per Language Pair (based on chrF):"
for PAIR in ${LANG_PAIRS//,/ }; do
  SRC=$(echo $PAIR | cut -d'-' -f1)
  TGT=$(echo $PAIR | cut -d'-' -f2)
  SRC_LANG=${TAG_LANGS[$SRC]}
  TGT_LANG=${TAG_LANGS[$TGT]}
  NEWPAIR="${SRC_LANG}-${TGT_LANG}"

  best_ckpt=""
  best_chrf=0

  for CKPT_DIR in "$MODEL_DIR"/dev_eval/*/; do
    CKPT_NAME=$(basename "$CKPT_DIR")
    SCORE_FILE="$CKPT_DIR/$NEWPAIR/scores.txt"
    if [[ -f "$SCORE_FILE" ]]; then
      CHRF=$(grep "chrF:" "$SCORE_FILE" | cut -d' ' -f2)
      if [[ $(echo "$CHRF > $best_chrf" | bc -l) -eq 1 ]]; then
        best_chrf=$CHRF
        best_ckpt=$CKPT_NAME
      fi
    fi
  done

  echo "$NEWPAIR: $best_ckpt (chrF = $best_chrf)"
  echo -e "$NEWPAIR\t$best_ckpt\t$best_chrf" >> "$BEST_CKPT_FILE"
done