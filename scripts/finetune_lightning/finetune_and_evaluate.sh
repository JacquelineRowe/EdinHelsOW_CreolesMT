#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=120g
#SBATCH --account=plgmodularnlp-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=eval-%j.out
#SBATCH --error=eval-%j.err

ml ML-bundle/24.06a
export HF_HOME=/net/storage/pr3/plgrid/plggmultilingualnlp/huggingface
mkdir -p $HF_HOME

cd "$PLG_GROUPS_STORAGE/plggmultilingualnlp/creole-nllb"
source venv/bin/activate

cd "$PLG_GROUPS_STORAGE/plggmultilingualnlp/creole-nllb/creolesMT/scripts/finetune_lightning"

# Usage check
if [[ $# -lt 12 ]]; then
  echo "Usage: $0 <ID> <BASE_MODEL> <DIRECTION>  <DATASET> <POR_EMB> <POR_DATA> <DIST_DATA> <POV_NORM_TRAIN> <POV_NORM_EVAL> <KMT_TAGGING> <PER_LANG_EVAL> <SYN_CRI>"
  echo "Example: $0 "KMT2" "kmt" "eng-XX" "set_5" True False True False True True True 0"
  exit 1
fi

ID=$1 # whatever ID you want to identify the model with e.g. NLLB3
BASE_MODEL=$2 #kmt, nllb-600M, nllb-1.3B, mbart
DIRECTION=$3 # either eng-XX, XX-eng or XX-XX
DATASET=$4  # e.g. set_5 or set_6 
POR_EMB=$5 # Boolean - initialises language tokens with embeddings of portuguese token
POR_DATA=$6 # Boolean - include additional Portuguese data or not
DIST_DATA=$7 # Boolean - use distilled data (forward translated by KMT) instead of normal data for pap, kea, cri and pov 
POV_NORM_TRAIN=$8 # Boolean - use normalised pov data for training pov->eng 
POV_NORM_EVAL=$9 # Boolean - use normalised pov data for translating pov-eng at evaluation
KMT_TAGGING=${10} # Boolean - using {text} </s> {tag} instead of the regular {tag} {text} </s> in training, validation and evaluation
PER_LANG_EVAL=${11} # Boolean - whether to evaluate on best checkpoint per language or not 
SYN_CRI=${12} # integer - how many k synthetic cri sentences to include (0, 5, 25 or 100)

echo "================================================="
echo "Training model ID:       $ID"
echo "Using base model:        $BASE_MODEL"
echo "Language direction:      $DIRECTION"
echo "Dataset:                 $DATASET"
echo "Use Portuguese emb:      $POR_EMB"
echo "Use Portuguese data:     $POR_DATA"
echo "Use Distilled data:      $DIST_DATA"
echo "Use pov-norm (train):    $POV_NORM_TRAIN"
echo "Use pov-norm (eval):     $POV_NORM_EVAL"
echo "KreyolMT Tagging :       $KMT_TAGGING"
echo "Per language eval:       $PER_LANG_EVAL"
echo "Synthetic cri data:      $SYN_CRI"
echo "================================================="

# Train
export WANDB_TAGS=$ID

# use direction and por_data parameters to define what language pairs are passed to data and model modules 
if [[ $DIRECTION == "eng-XX" ]]; then
    LANG_PAIRS=("eng-pap" "eng-kea" "eng-pov" "eng-aoa" "eng-cri" "eng-fab" "eng-pre")
    if [[ $POR_DATA == True ]]; then
        LANG_PAIRS+=("eng-por")
    fi

elif [[ $DIRECTION == "XX-eng" ]]; then
    LANG_PAIRS=("pap-eng" "kea-eng" "pov-eng" "aoa-eng" "cri-eng" "fab-eng" "pre-eng")
    if [[ $POR_DATA == True ]]; then
        LANG_PAIRS+=("por-eng")
    fi

elif [[ $DIRECTION == "XX-XX" ]]; then
    LANG_PAIRS=("eng-pap" "eng-kea" "eng-pov" "eng-aoa" "eng-cri" "eng-fab" "eng-pre" "pap-eng" "kea-eng" "pov-eng" "aoa-eng" "cri-eng" "fab-eng" "pre-eng")
    if [[ $POR_DATA == True ]]; then
        LANG_PAIRS+=("eng-por" "por-eng")
    fi

else
    echo "error: direction other than eng-XX, XX-eng and XX-XX specified"
    exit 1
fi

echo "Language pairs: "
for pair in ${LANG_PAIRS[@]}; do
    echo "$pair"
done 

LANG_PAIR_STRING=$(printf '"%s",' "${LANG_PAIRS[@]}")
LANG_PAIR_STRING="[${LANG_PAIR_STRING%,}]"

if [[ $BASE_MODEL == "kmt" ]]; then
    BASE_MODEL_NAME="jhu-clsp/kreyol-mt"
elif [[ $BASE_MODEL == "nllb-600M" ]]; then
    BASE_MODEL_NAME="facebook/nllb-200-distilled-600M"
elif [[ $BASE_MODEL == "nllb-1.3B" ]]; then
    BASE_MODEL_NAME="facebook/nllb-200-distilled-1.3B"
fi

python3 finetune_script.py \
  --ID $ID \
  --base_model $BASE_MODEL_NAME \
  --dataset $DATASET \
  --lang_pairs "$LANG_PAIR_STRING" \
  --por_emb "$POR_EMB" \
  --dist_data "$DIST_DATA" \
  --pov_norm_train "$POV_NORM_TRAIN" \
  --kmt_tagging "$KMT_TAGGING" \
  --syn_cri "$SYN_CRI"


cd "$PLG_GROUPS_STORAGE/plggmultilingualnlp/creole-nllb"

echo "**************************************************************************"
echo "TRAINING COMPLETE"
echo "**************************************************************************"

if [[ $PER_LANG_EVAL == True ]]; then
    python3 creolesMT/scripts/finetune_lightning/evaluate_per_checkpoint.py \
        --ID "$ID" \
        --base_model "$BASE_MODEL_NAME" \
        --dataset $DATASET \
        --lang_pairs "$LANG_PAIR_STRING" \
        --pov_norm_eval "$POV_NORM_EVAL" \
        --kmt_tagging "$KMT_TAGGING"
fi
    
python3 creolesMT/scripts/finetune_lightning/evaluate_all.py \
    --ID "$ID" \
    --base_model "$BASE_MODEL_NAME" \
    --dataset $DATASET \
    --lang_pairs "$LANG_PAIR_STRING" \
    --pov_norm_eval "$POV_NORM_EVAL" \
    --kmt_tagging "$KMT_TAGGING" \
    --per_lang_ckpts "$PER_LANG_EVAL"





