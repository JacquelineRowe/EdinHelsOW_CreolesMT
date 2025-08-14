import os

os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.getcwd(), ".hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), ".hf_cache")
os.environ["HF_METRICS_CACHE"] = os.path.join(os.getcwd(), ".hf_cache")

import sys
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import traceback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Available CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

model_name = "jhu-clsp/kreyol-mt"
batch_size = 32

# Best model per language pair
models = { "cri": "kreyol-mt",
          "kea": "kreyol-mt",
          "pap": "kreyol-mt-pubtrain",
          "pov": "kreyol-mt-scratch",
          "pre": "kreyol-mt-pubtrain"
}

dictmap = {
    'acf': 'ar_AR', 'ara': 'cs_CZ', 'aze': 'it_IT', 'bzj': 'hi_IN', 'cab': 'az_AZ', 'ceb': 'et_EE',
    'crs': 'fi_FI', 'deu': 'de_DE', 'djk': 'gu_IN', 'eng': 'en_XX', 'fra': 'fr_XX', 'gcf': 'ja_XX',
    'gul': 'kk_KZ', 'hat': 'ko_KR', 'icr': 'lt_LT', 'jam': 'lv_LV', 'kea': 'my_MM', 'kri': 'ne_NP',
    'ktu': 'nl_XX', 'mart1259': 'ro_RO', 'mfe': 'ru_RU', 'nep': 'si_LK', 'pap': 'tr_TR', 'pcm': 'vi_VN',
    'por': 'pt_XX', 'sag': 'af_ZA', 'spa': 'es_XX', 'srm': 'bn_IN', 'srn': 'fa_IR', 'tpi': 'he_IL',
    'zho': 'hr_HR', 'wes': 'zh_CN', 'trf': 'id_ID', 'svc': 'ka_GE', 'rcf': 'km_KH', 'pre': 'mk_MK',
    'pov': 'ml_IN', 'mue': 'mn_MN', 'lou': 'mr_IN', 'gyn': 'pl_PL', 'gpe': 'ps_AF', 'gcr': 'sv_SE',
    'fpe': 'sw_KE', 'fng': 'ta_IN', 'fab': 'te_IN', 'dcr': 'th_TH', 'cri': 'tl_XX', 'bzk': 'uk_UA',
    'brc': 'ur_PK', 'bah': 'xh_ZA', 'aoa': 'gl_ES'
}

def translate_file(file_path, src_tag, tgt_tag, tokenizer, model, model_name, bos_id, eos_id, pad_id):
    try:
        print(f"🔁 Translating file: {file_path}")
        decoder_start_token_id = tokenizer._convert_token_to_id_with_added_voc(tgt_tag)

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            print(f"⚠️ Skipping empty file: {file_path}")
            return

        preds = []
        for i in tqdm(range(0, len(lines), batch_size), desc="  Batches", leave=False):
            batch_sources = lines[i:i + batch_size]
            inputs_batch = [f"{text} </s> {src_tag}" for text in batch_sources]

            encoded = tokenizer(
                inputs_batch,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False
            )

            encoded = {k: v.to(device) for k, v in encoded.items() if k != "token_type_ids"}

            with torch.no_grad():
                batch_output_ids = model.generate(
                    **encoded,
                    use_cache=True,
                    num_beams=4,
                    max_length=200,
                    min_length=1,
                    early_stopping=True,
                    pad_token_id=pad_id,
                    bos_token_id=bos_id,
                    eos_token_id=eos_id,
                    decoder_start_token_id=decoder_start_token_id
                )


            for output_ids in batch_output_ids:
                pred = tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                preds.append(pred)

        out_path = file_path.replace(".txt", ".syn_eng.txt")
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write("\n".join(preds))
        print(f"✅ Finished: {file_path} -> {out_path}")
    except Exception:
        print(f"❌ Error processing {file_path}:\n{traceback.format_exc()}")


def process_language(lang_key):
    model_name = f"jhu-clsp/{models[lang_key]}"
    print(f"🔧 Using model: {model_name} for language: {lang_key}")

    if "scratch" in model_name:
        src_tag = f"<2{lang_key}>"
        tgt_tag = "<2eng>"
    else:
        src_tag = dictmap[lang_key]
        tgt_tag = dictmap["eng"]

    lang_path = os.path.join("new_data", lang_key, "monolingual")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, do_lower_case=False, use_fast=False, keep_accents=True
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()   

    bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
    eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
    pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>") 

    all_files = []
    for root, dirs, files in os.walk(lang_path):
        for file in files:
            if file.endswith(".txt"):
                all_files.append(os.path.join(root, file))

    for path in all_files:
        translate_file(path, src_tag, tgt_tag, tokenizer, model, model_name, bos_id, eos_id, pad_id)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python translate_lang.py <lang_key>")
        sys.exit(1)
    lang = sys.argv[1]
    process_language(lang)
