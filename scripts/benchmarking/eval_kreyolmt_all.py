import os
import traceback

# os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".hf_cache")
# os.environ["HF_DATASETS_CACHE"] = os.path.join(os.getcwd(), ".hf_cache")
# os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), ".hf_cache")
# os.environ["HF_METRICS_CACHE"] = os.path.join(os.getcwd(), ".hf_cache")

import torch
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm import tqdm
import json
from sacrebleu.metrics import CHRF


def safe_filename(text):
    return text.replace("/", "__").replace(":", "_")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Available CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

out_dir = "data/kreyol_mt"

models = [
    "jhu-clsp/kreyol-mt-pubtrain",
    "jhu-clsp/kreyol-mt-scratch",
    "jhu-clsp/kreyol-mt-scratch-pubtrain",
    "jhu-clsp/kreyol-mt"
]
languages = get_dataset_config_names("jhu-clsp/kreyol-mt")
dictmap = {'acf': 'ar_AR', 'ara': 'cs_CZ', 'aze': 'it_IT', 'bzj': 'hi_IN', 'cab': 'az_AZ', 'ceb': 'et_EE',
           'crs': 'fi_FI', 'deu': 'de_DE', 'djk': 'gu_IN', 'eng': 'en_XX', 'fra': 'fr_XX', 'gcf': 'ja_XX',
           'gul': 'kk_KZ', 'hat': 'ko_KR', 'icr': 'lt_LT', 'jam': 'lv_LV', 'kea': 'my_MM', 'kri': 'ne_NP',
           'ktu': 'nl_XX', 'mart1259': 'ro_RO', 'mfe': 'ru_RU', 'nep': 'si_LK', 'pap': 'tr_TR', 'pcm': 'vi_VN',
           'por': 'pt_XX', 'sag': 'af_ZA', 'spa': 'es_XX', 'srm': 'bn_IN', 'srn': 'fa_IR', 'tpi': 'he_IL',
           'zho': 'hr_HR', 'wes': 'zh_CN', 'trf': 'id_ID', 'svc': 'ka_GE', 'rcf': 'km_KH', 'pre': 'mk_MK',
           'pov': 'ml_IN', 'mue': 'mn_MN', 'lou': 'mr_IN', 'gyn': 'pl_PL', 'gpe': 'ps_AF', 'gcr': 'sv_SE',
           'fpe': 'sw_KE', 'fng': 'ta_IN', 'fab': 'te_IN', 'dcr': 'th_TH', 'cri': 'tl_XX', 'bzk': 'uk_UA',
           'brc': 'ur_PK', 'bah': 'xh_ZA', 'aoa': 'gl_ES'}

chrf = CHRF()
results = []
batch_size = 256
src_key = ""
tgt_key = ""

for model_name in models:
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              do_lower_case=False,
                                              use_fast=False,
                                              keep_accents=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    model.eval()

    for langpair in tqdm(languages, desc=f"Languages for model {model_name}"):
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
                    print(f"Skipping {file_name}, already exists.")
                    predictions = []
                    with open(file_path, "r") as f:
                        predictions.extend([json.loads(l) for l in f])
                else:
                    bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
                    eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
                    pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")

                    if "scratch" in model_name:
                        src_tag = f"<2{src_key}>"
                        tgt_tag = f"<2{tgt_key}>"

                    else:
                        src_tag = dictmap[src_key]
                        tgt_tag = dictmap[tgt_key]

                    predictions = []
                    inputs_batch = []
                    batch_refs = []

                    for i in tqdm(range(0, len(source_texts), batch_size),
                                  desc=f"  Translating {src_key}->{tgt_key} with {safe_filename(model_name)}",
                                  leave=False):
                        batch_sources = source_texts[i:i + batch_size]
                        batch_refs = references[i:i + batch_size]

                        inputs_batch = [f"{text} </s> {src_tag}" for text in batch_sources]

                        encoded = tokenizer(
                            inputs_batch,
                            return_tensors="pt",
                            padding=True,
                            add_special_tokens=False
                        )

                        encoded = {k: v.to(device) for k, v in encoded.items() if k != "token_type_ids"}

                        decoder_start_token_id = tokenizer._convert_token_to_id_with_added_voc(tgt_tag)

                        with torch.no_grad():
                            batch_output_ids = model.generate(
                                **encoded,
                                use_cache=True,
                                num_beams=4,
                                max_length=60,
                                min_length=1,
                                early_stopping=True,
                                pad_token_id=pad_id,
                                bos_token_id=bos_id,
                                eos_token_id=eos_id,
                                decoder_start_token_id=decoder_start_token_id
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

                    with open(file_path, "w") as f:
                        for pred in predictions:
                            json.dump(pred, f)
                            f.write("\n")

                pred_only = [p["prediction"] for p in predictions]

                score = chrf.corpus_score(pred_only, [references]).score
                results.append({
                    "language": langpair,
                    "direction": f"{src_key}->{tgt_key}",
                    "model": model_name,
                    "chrf": score
                })

        except Exception as e:
            print(f"    ERROR: {e}")
            print(traceback.format_exc())
            results.append({
                "language": langpair,
                "direction": f"{src_key}->{tgt_key}",
                "model": model_name,
                "chrf": None,
                "error": str(e)
            })

df = pd.DataFrame(results)
df.to_csv("kreyol_mt_model_evaluation_batched.csv", index=False)
print(df)
