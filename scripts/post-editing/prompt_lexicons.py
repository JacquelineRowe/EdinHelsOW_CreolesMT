
import os
import glob
import json
import time
import tiktoken
import gc
# Set Hugging Face cache paths
# os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.getcwd(), ".hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), ".hf_cache")
os.environ["HF_METRICS_CACHE"] = os.path.join(os.getcwd(), ".hf_cache")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from pathlib import Path
import transformers
import torch
import evaluate
import pandas as pd
import requests
import openai
from mistralai import Mistral

# git_repo_path = "/net/storage/pr3/plgrid/plggmultilingualnlp/creole-nllb/creolesMT" 
git_repo_path = "/home/jrowe/creolesMT"

lang_maps = {
    "pap": "Papiamento (papi1253)",
    "kea": "Kabuverdianu (kabu1256)",
    "pov": "Guinea-Bissau Creole (uppe1455)",
    "cri": "Saotomense (saot1239)",
    "aoa": "Angolar (ango1258)",
    "pre": "Principense (prin1242)",
    "fab": "Annobonese (fada1250)",
    "eng": "English"
}

model_map = {
    "pap-eng": "H2",
    "kea-eng": "H7",
    "pov-eng": "H7",
    "aoa-eng": "H1",
    "cri-eng": "H7",
    "fab-eng": "H1",
    "pre-eng": "H1",
    "eng-pap": "H4",
    "eng-kea": "H8",
    "eng-pov": "H6",
    # "eng-aoa": "H1",
    "eng-cri": "H8",
    # "eng-fab": "H1",
    # "eng-pre": "H1"
        }

chrf = evaluate.load("chrf")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Available CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

def load_file_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# Prompt strategies - edited to mimic prompts
def build_prompt(source, model_translation, lexicon_str, src_lang, tgt_lang, mode: str):
    src_code = lang_maps[src_lang]
    tgt_code = lang_maps[tgt_lang]

    if mode == "with_translation_ours":
        return (
            f"You are given a source sentence and a translation.\n"
            f"Improve the translation from {src_code} into {tgt_code}.\n"
            f"You must return ONLY the corrected translation sentence, without explanation or extra text.\n"
            f"Source: {source}\n"
            f"Translation: {model_translation}\n"
        )
    elif mode == "with_translation_gatitos":
        return (
            f"You are asked to edit the following translation from {src_code} into {tgt_code}. The proposed \
            translation is high-quality, but may have some incorrect words. \n\
            Please output only the translation of the text wihtout any other explanation. \n\
            {src_code}: {source}\n\
            {tgt_code}: {model_translation}"
        )
    elif mode == "with_both_ours":
        return (
            f"You are given a source sentence, a translation and a lexicon.\n"
            f"Improve the translation from {src_code} into {tgt_code}.\n"
            f"You must return ONLY the corrected translation sentence, without explanation or extra text.\n"
            f"Source: {source}\n"
            f"Translation: {model_translation}\n"
            f"Lexicon:\n{lexicon_str}\n"
        )
    elif mode == "with_both_gatitos":
        return (
            f"You are asked to edit the following translation from {src_code} into {tgt_code}. The proposed \
            translation is high-quality, but may have some incorrect words. \n\
            Note the following translations:\
            Lexicon:\n{lexicon_str}\n\
            Please output only the translation of the text wihtout any other explanation. \n\
            {src_code}: {source}\n\
            {tgt_code}: {model_translation}"
        )
    else:
        raise ValueError("Unknown mode")

def generate_fixed_batch_hf(prompts, pipeline, max_new_tokens=50):
    # this generates llm-edited translations with models accessed via Hugging Face
    messages_batch = [
        [
            {"role": "system", "content": "You are an expert translator in Portuguese-based creole translation."},
            {"role": "user", "content": prompt},
        ]
        for prompt in prompts
    ]

    outputs = pipeline(messages_batch, max_new_tokens=max_new_tokens, batch_size=4, do_sample=False, return_full_text=False) #do sample means we use greedy decoding (as in gatitos paper) 
    fixed_translations = []
    for out in outputs:
        text = out[0]["generated_text"][-1]["content"].strip()
        fixed_translations.append(text.split("\n")[0])
    return fixed_translations

def fix_trans_api_url(api_key, api_url, headers, payload, retries, wait):
    for index, attempt in enumerate(range(retries)):
        try:
            response = requests.post(
            api_url,
            headers=headers,
            json=payload
            )

            if response.status_code == 429:
                print("Rate limited. Retrying in", wait*(index+1), "seconds...")
                time.sleep(wait*(index+1))
                continue

            response.raise_for_status()
            data = response.json()  

            if response.status_code == 200:
                if "gemini" in api_url:
                    result = data["candidates"][0]["content"]["parts"][0]["text"].strip().replace("\n", " ").replace("\r", " ") # make sure no newline charactcers added in as this messes up translation alignment in txt files
                else: 
                    result = data.get("choices")[0]["message"]["content"].strip().replace("\n", " ").replace("\r", " ")
                return result
            time.sleep(wait*(index+1))

        except Exception:
            continue

    return "ERROR: Incomplete response"


def generate_fixed_batch_api(prompts, model_name):

    fixed_translations = []

    if model_name.startswith("openai") or model_name.startswith("gemini"):
        
        if model_name.startswith("openai"):
            API_KEY = os.getenv("OPENAI_API_KEY")
            API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
                }

        elif model_name.startswith("gemini"):
            API_KEY = os.getenv("GEMINI_API_KEY")
            # note to check whether other urls cause much difference 
            API_URL = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent")
            headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": API_KEY,
            }

        for i, prompt in enumerate(prompts):
            print(f"Calling {model_name} API for prompt {i+1}/{len(prompts)}")
            if model_name.startswith("openai"):
                payload = {"model": "gpt-3.5-turbo",
                 "messages": [{"role": "user", "content": prompt}],
                 "temperature": 0.0,   # set to 0.0 for deterministic output
                 "top_p": 1.0,  # set to 1.0 for deterministic output,
                 "n": 1,
                }
            elif model_name.startswith("gemini"):
                payload = {"contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                    "temperature": 0.0, # set to 0.0 for deterministic output
                    "topP": 1.0, # set to 1.0 for deterministic output
                    "candidateCount": 1 
                    }
                }
            
            result = fix_trans_api_url(
                api_key=API_KEY, 
                api_url=API_URL, 
                headers=headers, 
                payload=payload,
                retries=5, # how many times to retry if rate limited
                wait=10 # how many seconds to wait before retrying
                ) 
            
            time.sleep(1) # to avoid maxing out rates of models

            fixed_translations.append(result)

        return fixed_translations

    elif model_name.startswith("mistral"):

        model_name="mistral-large-latest"
        API_KEY = os.getenv("MISTRAL_API_KEY")
        client = Mistral(api_key=API_KEY)
        for i, prompt in enumerate(prompts):
            print(f"Calling {model_name} API for prompt {i+1}/{len(prompts)}")
            payload = [{"role": "user", "content": prompt}] 
            for attempt in range(5):
                try:
                    chat_response = client.chat.complete(
                        model=model_name,
                        messages=payload,
                        temperature=0.0,
                        top_p=1.0,
                    )
                    response = chat_response.choices[0].message.content
                    fixed_translations.append(response.strip()) 
                    break  
                except Exception as e:
                    if "Status 429" in str(e):
                        wait = 10 * (attempt + 1)
                        print(f"Rate limit. Waiting {wait} seconds...")
                        time.sleep(wait)
                    else:
                        print(f"Unhandled API error: {e}")
                        fixed_translations.append("ERROR: API failure")
                        break

            time.sleep(1) # to avoid maxing out rates of models

        return fixed_translations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set", required=True, help="Test set (all, ours, kmt)")
    parser.add_argument("--lang_pairs", required=True, help="Language pairs (e.g., pap-eng)")
    parser.add_argument("--llm", required=True, help="Model id from Huggingface to use for generation")
    parser.add_argument("--continue_fixing", action='store_true', help="Continue from existing files or start afresh")
    args = parser.parse_args()
    lang_pairs = json.loads(args.lang_pairs)
    print(lang_pairs)

    results = []
    for lang_pair in lang_pairs:
        print("lang_pair", lang_pair)
        src_lang, tgt_lang = lang_pair.split("-")
        if src_lang == "eng":
            creole = tgt_lang
        else:
            creole = src_lang
        if src_lang in ["pap", "kea", "fab", "pov"] or tgt_lang in ["pap", "kea", "fab", "pov"]:
            lexicon = f"{git_repo_path}/data/lexicons/eng-{creole}.tsv" 
        elif src_lang in ["aoa", "pre", "cri"] or tgt_lang in ["aoa", "pre", "cri"]:
            lexicon = f"{git_repo_path}/data/lexicons/por-{creole}.tsv" 

        max_context_length = 15000 if args.llm == "openai" else 30000
        # Paths
        # results_dir = "/net/storage/pr3/plgrid/plggmultilingualnlp/creole-nllb/llm_editing"
        results_dir = "/home/jrowe/llm_editing"
        base = Path(f"{results_dir}/{lang_pair}/{args.test_set}")
        source_path = base / "src.txt"
        hyp_path = base / "hyp.txt"
        ref_path = base / "ref.txt"

        # Load data
        sources = load_file_lines(source_path)
        translations = load_file_lines(hyp_path)
        references = load_file_lines(ref_path)

        # calculate baseline chrf 
        chrf_scores = {}
        baseline = chrf.compute(predictions=translations, references=references)["score"]

        # load and format lexicon 
        lexicon = dict(line.strip().split('\t') for line in load_file_lines(lexicon))
        if src_lang != "eng":  # Reverse if translating from creole to English
            lexicon = {v: k for k, v in lexicon.items()}
        lexicon_str = "; ".join([f"{src} means {tgt}" for src, tgt in list(lexicon.items())])

        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        # if its a long lexicon, shorten it by pickking just shorted entries 
        def truncate_lexicon(lexicon_dict, max_tokens=15000, model="gpt-3.5-turbo"):
            encoding = tiktoken.encoding_for_model(model)
            entries = list(lexicon_dict.items())

            # Sort by shortest source word length
            entries = sorted(entries, key=lambda x: len(x[0]))

            lexicon_str = ""
            for i in range(len(entries)):
                candidate = "; ".join([f"{src} means {tgt}" for src, tgt in entries[:i+1]])
                if len(encoding.encode(candidate)) >= max_tokens:
                    return "; ".join([f"{src} means {tgt}" for src, tgt in entries[:i]])
            return "; ".join([f"{src} means {tgt}" for src, tgt in entries])  # all fit

        if len(encoding.encode(lexicon_str)) >= max_context_length:
            print("TRUNCATING LEXICON")
            lexicon_str = truncate_lexicon(lexicon)

        # generate prompts 
        all_prompts = {}
        for condition in ["with_translation_ours", "with_translation_gatitos", "with_both_ours", "with_both_gatitos"]:
            condition_prompts = [
                build_prompt(source, translation, lexicon_str, src_lang, tgt_lang, condition)
                for source, translation in zip(sources, translations)
            ]
            all_prompts[condition] = condition_prompts
        
        out_dir = base / f"llm_fix_{args.llm}"
        out_dir.mkdir(exist_ok=True)

        # if model is on hf, use transformers pipeline
        if args.llm == "mistral_hf" or args.llm == "mistral_hf_large":
            if args.llm == "mistral_hf": 
                model_name = "mistralai/Mistral-7B-Instruct-v0.3"
            elif args.llm == "mistral_hf_large":
                model_name = "mistralai/Mistral-Small-24B-Instruct-2501"
            pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=0,
            #device_map="auto",
            )

            if pipeline.tokenizer.pad_token is None:
                pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token

            for condition, prompts in all_prompts.items():
                condition_path = Path(out_dir) / condition
                if args.continue_fixing and condition_path.exists():
                    with open(condition_path, "r", encoding="utf-8") as f:
                        llm_fixes = [line.strip() for line in f]
                else:
                    llm_fixes = generate_fixed_batch_hf(prompts, pipeline)

                    with open(condition_path, "w", encoding="utf-8") as f:
                        for line in llm_fixes:
                            print(line)
                            f.write(line + "\n")

                # Evaluation
                chrf_score = chrf.compute(predictions=llm_fixes, references=references)["score"]
                chrf_scores[condition] = chrf_score

                # clear memory 
                del llm_fixes
                torch.cuda.empty_cache()
                gc.collect()

        # otherwise use API calls 
        else:
            for condition, prompts in all_prompts.items():
                condition_path = Path(out_dir) / condition
                if args.continue_fixing and condition_path.exists():
                    with open(condition_path, "r", encoding="utf-8") as f:
                        llm_fixes = [line.strip() for line in f]
                else:
                    llm_fixes = generate_fixed_batch_api(prompts, args.llm)
                    print(llm_fixes[0])

                    with open(condition_path, "w", encoding="utf-8") as f:
                        for line in llm_fixes:
                            print(line)
                            f.write(line + "\n")

                # Evaluation
                chrf_score = chrf.compute(predictions=llm_fixes, references=references)["score"]
                chrf_scores[condition] = chrf_score
        
        results.append({
            "language": lang_pair,
            "direction": f"{src_lang} -> {tgt_lang}",
            "model": model_map[lang_pair],
            "baseline": baseline,
            **chrf_scores
        })

        df = pd.DataFrame(results)
        df.to_csv(f"{results_dir}/lexicon_eval_{args.llm}.csv", index=False, encoding="utf-8")

    print(df)
    print(f"Fixed outputs saved to {out_dir}")

if __name__ == "__main__":
    main()