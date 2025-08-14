import os

os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.getcwd(), ".hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), ".hf_cache")
os.environ["HF_METRICS_CACHE"] = os.path.join(os.getcwd(), ".hf_cache")

import torch
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import pandas as pd
import evaluate
import json

def safe_filename(text):
    return text.replace("/", "__").replace(":", "_")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Available CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

out_dir = "data/nllb"

models = [
    "facebook/nllb-200-distilled-600M",
    "facebook/nllb-200-1.3B",
    "facebook/nllb-200-distilled-1.3B",
    "facebook/nllb-200-3.3B"
]


chrf = evaluate.load("chrf")
results = []
batch_size = 64

lang_codes = {"eng":"eng_Latn",
    "hat":"hat_Latn",
    "pap": "pap_Latn",
    "por":"por_Latn",
    "fra":"fra_Latn",
    "tpi":"tpi_Latn",
    "sag":"sag_Latn",
    "kea":"kea_Latn"
    }

languages = ["hat-eng", "hat-fra", "pap-por", "pap-eng", "tpi-eng", "sag-eng", "kea-eng"]

for model_name in models:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name,torch_dtype=torch.float16).to(device)
    model.eval()

    for langpair in tqdm(languages, desc="Languages"):
        print(f"Evaluating language: {langpair}")
        try:
            dataset = load_dataset("jhu-clsp/kreyol-mt", langpair, split="test")
            fields = langpair.split("-")
            directions = [
                {"src": fields[0], "tgt": fields[1]},
                {"src": fields[1], "tgt": fields[0]}
            ]

            for direction in directions:
                src_key = direction["src"]
                tgt_key = direction["tgt"]

                references = []
                source_texts = []
                for item in dataset["translation"]:
                    
                    if item["src_lang"] == src_key:
                        source_texts.append(item["src_text"])
                        references.append(item["tgt_text"])
                    else:
                        source_texts.append(item["tgt_text"])
                        references.append(item["src_text"])

                file_name = f"{src_key}_to_{tgt_key}_{safe_filename(model_name)}.json"
                file_path = os.path.join(out_dir, file_name)

                print(f"  Model: {model_name} | Direction: {src_key} → {tgt_key}")
                
                if os.path.exists(file_path):
                    print(f"    Skipping {file_name}, already exists.")
                    with open(file_path, "r") as f:
                        predictions = json.load(f)
                else:
                    tokenizer.src_lang = lang_codes[src_key]
                    forced_bos_token_id = tokenizer.convert_tokens_to_ids(lang_codes[tgt_key])

                    predictions = []
                    inputs_batch = []
                    batch_refs = []
                    
                    for i in tqdm(range(0, len(source_texts), batch_size), desc=f"  Translating {src_key}->{tgt_key} with {safe_filename(model_name)}", leave=False):
                        batch_sources = source_texts[i:i + batch_size]
                        batch_refs = references[i:i + batch_size]
                        
                        # Prefix source text with the language token for correct behavior
                        inputs_batch = [f"{lang_codes[src_key]} {text}" for text in batch_sources]

                        encoded = tokenizer(
                            inputs_batch,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                        )

                        encoded = {k: v.to(device) for k, v in encoded.items() if k != "token_type_ids"}

                        with torch.no_grad():
                            batch_output_ids = model.generate(
                                **encoded,
                                use_cache=True,
                                forced_bos_token_id=forced_bos_token_id,
                                max_new_tokens=128
                            )

                        batch_preds = []
                        for output_ids in batch_output_ids:
                            batch_preds.append(tokenizer.decode(output_ids, skip_special_tokens=True))

                        predictions.extend([
                            {"source": src, "prediction": pred}
                            for src, pred in zip(batch_sources, batch_preds)
                        ])

                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(predictions, f, ensure_ascii=False, indent=2)
                
                pred_only = [p["prediction"] for p in predictions]

                score = chrf.compute(predictions=pred_only, references=references)["score"]
                results.append({
                    "language": langpair,
                    "direction": f"{src_key}->{tgt_key}",
                    "model": model_name,
                    "chrf": score
                })

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                "language": langpair,
                "direction": f"{src_key}->{tgt_key}",
                "model": model_name,
                "chrf": None,
                "error": str(e)
            })

df = pd.DataFrame(results)
df.to_csv("nllb_evaluation_batched.csv", index=False, encoding="utf-8")
print(df)
