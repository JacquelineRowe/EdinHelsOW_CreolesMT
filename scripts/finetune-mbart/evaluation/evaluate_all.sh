#!/bin/bash
#SBATCH --job-name=mbart_eval_all
#SBATCH --output=logs/mbart_eval_all_%j.out
#SBATCH --error=logs/mbart_eval_all_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=10
#SBATCH --partition=gputest
#SBATCH --account=project_2005815
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --dependency=afterok:4757700

# Activate environment
source /scratch/project_2005815/members/degibert/fairseq-dummy/venv/bin/activate

export WORKDIR_ROOT=/scratch/project_2005815/members/degibert/creolesMT
export SPM_PATH=$WORKDIR_ROOT/sentencepiece/build/src/

# Language mapping
declare -A TAG_LANGS=(  [en_XX]=eng  [tr_TR]=pap  [my_MM]=kea  [ml_IN]=pov  [gl_ES]=aoa  [tl_XX]=cri  [te_IN]=fab  [mk_MK]=pre )

PAIRS=(eng-pap eng-kea eng-pov eng-aoa eng-cri eng-fab eng-pre)
MODELS=(distill-many-to-many-ft-EN-to-XX) #mbart50-one-to-many-ft-EN-to-XX_mod_set-1 ) # mbart50-many-to-one-ft-XX-to-EN  mbart50-many-to-many-ft-ALL-to-ALL-mod_set-1 mbart50-many-to-many-ft-ALL-to-ALL-mod_set-2 mbart50-many-to-many-ft-ALL-to-ALL-mod_set-3)
#mbart50-many-to-one-ft-XX_to_EN_mod_set-1_por mbart50-many-to-many-ft-XX-to-EN mbart50-many-to-many-ft-ALL-to-ALL  mbart50-many-to-many-ft-EN-to-XX   mbart50-one-to-many-ft-EN-to-XX mbart50-many-to-many-ft-ALL-to-ALL-modified mbart50-many-to-many-ft-ALL-to-ALL-por
for MODEL in "${MODELS[@]}"; do
  MODEL_DIR="models/$MODEL"
  BEST_CKPT_FILE="$MODEL_DIR/dev_eval/best_checkpoints.txt"
  BASE_MODEL="models/mbart50-many-to-many"
  LANG_LIST="$BASE_MODEL/ML50_langs.txt"
  if [[ "$MODEL" == *"EN-to-XX"* ]]; then
    LANG_PAIRS="en_XX-tr_TR,en_XX-my_MM,en_XX-ml_IN,en_XX-tl_XX" #en_XX-gl_ES,en_XX-te_IN,en_XX-mk_MK"
  elif [[ "$MODEL" == *"ALL-to-ALL"* ]]; then
    LANG_PAIRS="tr_TR-en_XX,my_MM-en_XX,ml_IN-en_XX,gl_ES-en_XX,tl_XX-en_XX,te_IN-en_XX,mk_MK-en_XX,en_XX-tr_TR,en_XX-my_MM,en_XX-ml_IN,en_XX-gl_ES,en_XX-tl_XX,en_XX-te_IN,en_XX-mk_MK"
  else
    LANG_PAIRS="tr_TR-en_XX,my_MM-en_XX,ml_IN-en_XX,tl_XX-en_XX" #gl_ES-en_XX,,te_IN-en_XX,mk_MK-en_XX"
  fi
  SPM_MODEL="$BASE_MODEL/sentence.bpe.model"
  DICTIONARY="$BASE_MODEL/dict.en_XX.txt"

  mkdir -p "results"
  mkdir -p "results/$MODEL"

  MODEL_SCORE_LOG="results/$MODEL/scores.txt"
  echo -e "LangPair\tBLEU\tchrF" >> "$MODEL_SCORE_LOG"

  for PAIR in ${LANG_PAIRS//,/ }; do
      SRC=$(echo $PAIR | cut -d'-' -f1)
      TGT=$(echo $PAIR | cut -d'-' -f2)

      SRC_LANG=${TAG_LANGS[$SRC]}
      TGT_LANG=${TAG_LANGS[$TGT]}

      NEWPAIR="${SRC_LANG}-${TGT_LANG}"

      DEST_DIR="data/mbart/data-bin-test/"
      RESULT_SUBDIR="results/$MODEL/$NEWPAIR"
      mkdir -p "$RESULT_SUBDIR"

      GEN_FILE="$RESULT_SUBDIR/generate-test.txt"
      HYP_FILE="$RESULT_SUBDIR/hyp.txt"
      REF_FILE="$RESULT_SUBDIR/ref.txt"
      SCORE_FILE="$RESULT_SUBDIR/scores.txt"

      if [[ ! -f "$GEN_FILE" ]]; then
        echo "Evaluating $MODEL/$NEWPAIR..."

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
        
        # Generate translations
        CUDA_VISIBLE_DEVICES=0 fairseq-generate $DEST_DIR \
          --lang-dict "$LANG_LIST" \
          --lang-pairs "$LANG_PAIRS" \
          --gen-subset test \
          --task translation_multi_simple_epoch \
          --path $MODEL_PATH \
          --remove-bpe sentencepiece \
          --batch-size 32 \
          --results-path "$RESULT_SUBDIR" \
          --source-lang $SRC --target-lang $TGT --beam 4 --lenpen 0.4 

        # Extract outputs
        grep -P "^H" $GEN_FILE | sort -V | cut -f 3- > $HYP_FILE
        grep -P "^T" $GEN_FILE | sort -V | cut -f 2- > $REF_FILE
        sed -i 's/__[a-z][a-z]_[A-Z][A-Z]__ //' "$HYP_FILE"
        
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

        # Compute scores
        BLEU=$(sacrebleu $REF_FILE < $HYP_FILE | jq  -r '.score')
        CHRF=$(sacrebleu $REF_FILE --metrics chrf < $HYP_FILE | jq  -r '.score')

        echo -e "$NEWPAIR\t$BLEU\t$CHRF" >> "$MODEL_SCORE_LOG"
        echo -e "BLEU: $BLEU\nchrF: $CHRF" > "$SCORE_FILE"
      else
        echo "$MODEL/$NEWPAIR: already done, skipping. Remove if you want to overwrite."
      fi
          
      echo "$MODEL/$NEWPAIR done."
    done
done
