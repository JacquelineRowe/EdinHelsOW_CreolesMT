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

out_dir = "data/flores"

nllb_models = [
    "facebook/nllb-200-distilled-600M",
    "facebook/nllb-200-1.3B",
    "facebook/nllb-200-distilled-1.3B",
    "facebook/nllb-200-3.3B"
]

kreyolmt_models = [
    "jhu-clsp/kreyol-mt-pubtrain",
    "jhu-clsp/kreyol-mt-scratch",
    "jhu-clsp/kreyol-mt-scratch-pubtrain",
    "jhu-clsp/kreyol-mt"
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
dictmap = {'acf': 'ar_AR', 'ara': 'cs_CZ', 'aze': 'it_IT', 'bzj': 'hi_IN', 'cab': 'az_AZ', 'ceb': 'et_EE',
           'crs': 'fi_FI', 'deu': 'de_DE', 'djk': 'gu_IN', 'eng': 'en_XX', 'fra': 'fr_XX', 'gcf': 'ja_XX',
           'gul': 'kk_KZ', 'hat': 'ko_KR', 'icr': 'lt_LT', 'jam': 'lv_LV', 'kea': 'my_MM', 'kri': 'ne_NP',
           'ktu': 'nl_XX', 'mart1259': 'ro_RO', 'mfe': 'ru_RU', 'nep': 'si_LK', 'pap': 'tr_TR', 'pcm': 'vi_VN',
           'por': 'pt_XX', 'sag': 'af_ZA', 'spa': 'es_XX', 'srm': 'bn_IN', 'srn': 'fa_IR', 'tpi': 'he_IL',
           'zho': 'hr_HR', 'wes': 'zh_CN', 'trf': 'id_ID', 'svc': 'ka_GE', 'rcf': 'km_KH', 'pre': 'mk_MK',
           'pov': 'ml_IN', 'mue': 'mn_MN', 'lou': 'mr_IN', 'gyn': 'pl_PL', 'gpe': 'ps_AF', 'gcr': 'sv_SE',
           'fpe': 'sw_KE', 'fng': 'ta_IN', 'fab': 'te_IN', 'dcr': 'th_TH', 'cri': 'tl_XX', 'bzk': 'uk_UA',
           'brc': 'ur_PK', 'bah': 'xh_ZA', 'aoa': 'gl_ES'}

models = nllb_models + kreyolmt_models

# Pre-load all needed flores devtest sets just once
flores_data = {}
for lang in set(l for pair in languages for l in pair.split("-")):
    print(f"Loading FLORES data for {lang}")
    flores_data[lang] = [
        item['sentence']
        for item in load_dataset("facebook/flores", lang_codes[lang], split="devtest")
    ]

for model_name in models:
    print(f"Evaluating model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              do_lower_case=False,
                                              use_fast=False,
                                              keep_accents=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name,torch_dtype=torch.float16).to(device)
    model.eval()

    for langpair in tqdm(languages, desc="Languages"):
        print(f"Evaluating language: {langpair}")
        try:
            creole, lexifier = langpair.split("-")
            creole_dataset = flores_data[creole]
            lexifier_dataset = flores_data[lexifier]
            
            datasets = {creole: creole_dataset, lexifier: lexifier_dataset}

            directions = [
                {"src": creole, "tgt": lexifier},
                {"src": lexifier, "tgt": creole}
            ]

            for direction in directions:
                src_key = direction["src"]
                tgt_key = direction["tgt"]

                references = []
                source_texts = []

                source_texts = datasets[src_key]
                references = datasets[tgt_key]

                file_name = f"{src_key}_to_{tgt_key}_{safe_filename(model_name)}.json"
                file_path = os.path.join(out_dir, file_name)

                print(f"  Model: {model_name} | Direction: {src_key} → {tgt_key}")
                
                if os.path.exists(file_path):
                    print(f"    Skipping {file_name}, already exists.")
                    with open(file_path, "r") as f:
                        predictions = json.load(f)
                else:
                    if "kreyol-mt" in model_name:
                        bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
                        eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
                        pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")

                        if "scratch" in model_name:
                            src_tag = f"<2{src_key}>"
                            tgt_tag = f"<2{tgt_key}>"

                        else:
                            src_tag = dictmap[src_key]
                            tgt_tag = dictmap[tgt_key]
                    else:
                        tokenizer.src_lang = lang_codes[src_key]
                        forced_bos_token_id = tokenizer.convert_tokens_to_ids(lang_codes[tgt_key])

                    predictions = []
                    inputs_batch = []
                    batch_refs = []
                    
                    for i in tqdm(range(0, len(source_texts), batch_size), desc=f"  Translating {src_key}->{tgt_key} with {safe_filename(model_name)}", leave=False):
                        batch_sources = source_texts[i:i + batch_size]
                        batch_refs = references[i:i + batch_size]
                        
                        if "nllb" in model_name:
                            # Prefix source text with the language token for correct behavior
                            inputs_batch = [f"{lang_codes[src_key]} {text}" for text in batch_sources]
                        else:
                            inputs_batch = [f"{text} </s> {src_tag}" for text in batch_sources]

                        encoded = tokenizer(
                            inputs_batch,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                        )

                        encoded = {k: v.to(device) for k, v in encoded.items() if k != "token_type_ids"}

                        if "kreyol-mt" in model_name:
                            decoder_start_token_id = tokenizer._convert_token_to_id_with_added_voc(tgt_tag)

                            with torch.no_grad():
                                batch_output_ids = model.generate(
                                    **encoded,
                                    use_cache=True,
                                    num_beams=4,
                                    max_length=100,
                                    min_length=1,
                                    early_stopping=True,
                                    pad_token_id=pad_id,
                                    bos_token_id=bos_id,
                                    eos_token_id=eos_id,
                                    decoder_start_token_id=decoder_start_token_id
                                )

                        else:
                            with torch.no_grad():
                                batch_output_ids = model.generate(
                                    **encoded,
                                    use_cache=True,
                                    forced_bos_token_id=forced_bos_token_id,
                                    max_new_tokens=128
                                )

                        batch_preds = []
                        for output_ids in batch_output_ids:
                            batch_preds.append(tokenizer.decode(output_ids,
                                                            skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=False))

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
df.to_csv("flores_evaluation_batched.csv", index=False, encoding="utf-8")
print(df)
