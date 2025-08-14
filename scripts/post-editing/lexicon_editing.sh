#!/bin/bash -l

ml ML-bundle/24.06a
export HF_HOME=/net/storage/pr3/plgrid/plggmultilingualnlp/huggingface
export GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
export MISTRAL_API_KEY=os.getenv("MISTRAL_API_KEY")
export OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
mkdir -p $HF_HOME

cd "$PLG_GROUPS_STORAGE/plggmultilingualnlp/creole-nllb"
source venv/bin/activate

LLM=$1 # e.g. hf: "mistral_hf" "mistral_hf_large" API: "mistral" "gemini" or "openai"
DIRECTION=$2 # XX-eng, eng-XX, XX-XX
CONTINUE=$3 # True or false i.e. whether to restrat midrun or re-write over existing files and start afresh 

if [[ $DIRECTION == "eng-XX" ]]; then
    LANG_PAIRS=("eng-pap" "eng-kea" "eng-pov" "eng-aoa" "eng-cri" "eng-fab" "eng-pre")

elif [[ $DIRECTION == "XX-eng" ]]; then
    LANG_PAIRS=("pap-eng" "kea-eng" "pov-eng" "aoa-eng" "cri-eng" "fab-eng" "pre-eng")

elif [[ $DIRECTION == "XX-XX" ]]; then
    LANG_PAIRS=("eng-pap" "eng-kea" "eng-pov" "eng-aoa" "eng-cri" "eng-fab" "eng-pre" "pap-eng" "kea-eng" "pov-eng" "aoa-eng" "cri-eng" "fab-eng" "pre-eng")
fi

echo "Language pairs: "
for pair in ${LANG_PAIRS[@]}; do
    echo "$pair"
done 

LANG_PAIR_STRING=$(printf '"%s",' "${LANG_PAIRS[@]}")
LANG_PAIR_STRING="[${LANG_PAIR_STRING%,}]"

if [[ $CONTINUE == True ]]; then 
    python3 /home/jrowe/creolesMT/scripts/lexicon/prompt_lexicons.py \
        --test_set="kmt" \
        --lang_pairs=$LANG_PAIR_STRING \
        --llm="$LLM" \
        --continue_fixing
else
    python3 /home/jrowe/creolesMT/scripts/lexicon/prompt_lexicons.py \
    --test_set="kmt" \
    --lang_pairs=$LANG_PAIR_STRING \
    --llm="$LLM" 
fi