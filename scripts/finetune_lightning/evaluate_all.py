import os

os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.getcwd(), ".hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), ".hf_cache")
os.environ["HF_METRICS_CACHE"] = os.path.join(os.getcwd(), ".hf_cache")

import sys
import glob
import numpy as np
import torch
import fire
from typing import List
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import pandas as pd
import re
import evaluate
import json
import pytorch_lightning as pl

sys.path.insert(0, '/net/storage/pr3/plgrid/plggmultilingualnlp/creole-nllb/creolesMT/scripts/finetune-lightning')
from model_module import CreoleMT_model

def main(
    ID: str,
    base_model: str,
    dataset: str,
    lang_pairs: List[str],
    pov_norm_eval: bool = False,
    kmt_tagging: bool = False,
    per_lang_ckpts: bool = False,
    main_dir: str = "/net/storage/pr3/plgrid/plggmultilingualnlp",
    batch_size: int = 64,
    use_bfloat16: bool = True,
    **kwargs,
):

    local_data_dir = f"{main_dir}/creole-nllb/creolesMT/data/preprocessed_lusophone/{dataset}"
    out_dir = f"{main_dir}/creole-nllb/evaluations_final/{ID}"
    os.makedirs(out_dir, exist_ok=True)

    model_dir = f"{main_dir}/creole-nllb/{ID}"
    print("model directory: ", model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Available CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
    chrf = evaluate.load("chrf")
    results = []

    if base_model == "jhu-clsp/kreyol-mt":
        LANG_CODES_MAPPING = {
            "pov": "ml_IN",
            "aoa": "gl_ES",
            "cri": "tl_XX",
            "fab": "te_IN",
            "pre": "mk_MK",
            "acf": "ar_AR",
            "pap": "tr_TR",
            "kea": "my_MM",
            "eng": "en_XX",
            "por": "pt_XX"
        }

    elif base_model == "facebook/nllb-200-distilled-600M" or base_model ==  "facebook/nllb-200-distilled-1.3B":
        LANG_CODES_MAPPING = {
            "pov": "pov_Latn",
            "aoa": "aoa_Latn",
            "cri": "cri_Latn",
            "fab": "fab_Latn",
            "pre": "pre_Latn",
            "acf": "acf_Latn",
            "pap": "pap_Latn",
            "kea": "kea_Latn",
            "eng": "eng_Latn",
            "por": "por_Latn"
        }

    # look up and store step numbers and paths of all saved checkpoints in that model folder 
    saved_checkpoints = {}

    checkpoint_path_pattern = os.path.join(model_dir, "checkpoints", "best-global_step=*-eval_chrf=0.00.ckpt")
    checkpoint_files = glob.glob(checkpoint_path_pattern)
    if checkpoint_files:
        for checkpoint_file in checkpoint_files: 
            checkpoint_numbers = re.search(r'global_step=(\d+)', checkpoint_file)
            if checkpoint_numbers: 
                saved_checkpoints[int(checkpoint_numbers.group(1))] = checkpoint_file
    else:
        raise FileNotFoundError(f"No model path found: {checkpoint_path_pattern}")
    
    tokenizer_path = os.path.join(model_dir,"translation", "tokenizer", base_model.split("/")[-1])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # if option is set, load df of best checkpoints for each language direction
    if per_lang_ckpts == True:
        if pov_norm_eval == False:
            extension = ''
        else:
            extension = '_pov_norm'
        try:
            all_ckpts_scores = pd.read_csv(f"{model_dir}/dev_eval/results_dev_{ID}_all_checkpoints{extension}.csv")
        except Exception as e:
            print(f"    ERROR: {e}")
            print(f"Please calculate checkpoint scores on dev set first: {model_dir}/dev_eval/results_{ID}_all_checkpoints{extension}.csv")
            sys.exit(1)

        all_ckpts_on_full_dev = all_ckpts_scores[all_ckpts_scores["split"] == "full"]
        best_ckpts = all_ckpts_on_full_dev.loc[all_ckpts_on_full_dev.groupby("direction")["chrf"].idxmax()]
        print(best_ckpts["direction"].unique())
         # Print out best checkpoints for future reference
        with open(f"{out_dir}/best_checkpoints{extension}.txt", "w+") as f:
            for langpair in lang_pairs:
                src_key, tgt_key = langpair.split("-")
                if src_key == "por" or tgt_key == "por":
                    continue
                else:
                    best_checkpoint_num = best_ckpts.loc[best_ckpts["direction"] == f"{src_key}->{tgt_key}", "checkpoint"]
                    if not best_checkpoint_num.empty:
                        f.write(langpair + str(best_checkpoint_num.iloc[0]) + "\n")
                    else:
                        print(f"Warning: No checkpoint found for direction {src_key}->{tgt_key}")


    # evaluate each language pair in turn   
    for langpair in tqdm(lang_pairs, total=len(lang_pairs), desc="Languages"):
        src_key, tgt_key = langpair.split("-")
        if src_key == "por" or tgt_key == "por":
            continue
        else:
            print(f"Evaluating language: {src_key}-{tgt_key}")

            # LOAD MODEL
            if per_lang_ckpts == True:
                best_checkpoint = best_ckpts.loc[best_ckpts["direction"] == f"{src_key}->{tgt_key}", "checkpoint"]
                if best_checkpoint.empty:
                    print(f"    ERROR: No best checkpoint found for direction {src_key}->{tgt_key}. Skipping...")
                    results.append({
                        "language": f"{src_key}-{tgt_key}",
                        "direction": f"{src_key}->{tgt_key}",
                        "model": ID,
                        "checkpoint": None,
                        "chrf": None,
                        "error": "No best checkpoint found for this direction."
                    })
                    continue  # Skip this language pair
                checkpoint_num = best_checkpoint.iloc[0]
                print("Selecting best checkpoint: ", checkpoint_num)
                # # look up best checkpoint for this lang pair in df 
                
                # checkpoint_num = best_ckpts.loc[best_ckpts["direction"] == f"{src_key}->{tgt_key}", "checkpoint"].iloc[0]
                # print("Selecting best checkpoint: ", checkpoint_num)
            else:
                # look up most recent checkpoint 
                sorted_checkpoints = dict(sorted(saved_checkpoints.items()))
                checkpoint_num = max(sorted_checkpoints.keys())
                print("Selecting most recent checkpoint: ", checkpoint_num)

            model_path = os.path.join(model_dir, "checkpoints", f"best-global_step={checkpoint_num}.0-eval_chrf=0.00.ckpt")
            print(f" Model: {model_path} |  Checkpoint Num: {checkpoint_num}  | Direction: {src_key} → {tgt_key}")

            # load relevant checkpoint 
            lightning_model = CreoleMT_model.load_from_checkpoint(
                model_path,  
                torch_dtype=torch.bfloat16)
            lightning_model.eval()
            lightning_model.freeze()
            model = lightning_model.CreoleMT.to(device)
            model.eval()            

            # LOAD DATA 
            # select eng-XX direction as files are named eng_xx
            if src_key == "eng":
                pass
            else:
                langpair = f"{tgt_key}-{src_key}"
            if pov_norm_eval == True and src_key == "pov":
                src_extension = "_norm"
                tgt_extension = ""
            else:
                src_extension = ""
                tgt_extension = ""
            
            src_tag = LANG_CODES_MAPPING[src_key]
            tgt_tag = LANG_CODES_MAPPING[tgt_key]

            if src_key != "por" and tgt_key != "por":
                try:
                    if "split" in dataset:
                        # collect sets of ours and kmt src texts for generating per-split evaluation scores later
                        ours_src_path = os.path.join(local_data_dir, f"test_ours.{langpair}.{src_key}{src_extension}")
                        kmt_src_path = os.path.join(local_data_dir, f"test_kmt.{langpair}.{src_key}{src_extension}")

                        if os.path.exists(ours_src_path):
                            with open(ours_src_path, 'r', encoding='utf-8') as f:
                                ours_sources = set(line.strip() for line in f if line.strip())
                        else:
                            ours_sources = set()

                        if os.path.exists(kmt_src_path):
                            with open(kmt_src_path, 'r', encoding='utf-8') as f:
                                kmt_sources = set(line.strip() for line in f if line.strip())
                        else:
                            kmt_sources = set()

                    src_path = os.path.join(local_data_dir, f"test.{langpair}.{src_key}{src_extension}")
                    tgt_path = os.path.join(local_data_dir, f"test.{langpair}.{tgt_key}{tgt_extension}")

                    with open(src_path, 'r', encoding='utf-8') as f_src:
                        source_texts = [line.strip() for line in f_src if line.strip()]
                    with open(tgt_path, 'r', encoding='utf-8') as f_tgt:
                        references = [line.strip() for line in f_tgt if line.strip()]
                        
                    assert len(source_texts) == len(references), "Mismatch between source and target line counts!"
                    
                    file_name = f"{src_key}_to_{tgt_key}_{ID}_{checkpoint_num}{src_extension}.json"
                    file_path = os.path.join(out_dir, file_name)

                    print(f"Loaded {len(source_texts)} lines from {dataset} for {src_key} - {tgt_key}")
                    
                    if os.path.exists(file_path):
                        print(f"    Skipping {file_name}, already exists.")
                        with open(file_path, "r") as f:
                            predictions = json.load(f)
                    else:
                        tokenizer.src_lang = src_tag
                        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_tag)
                        print(forced_bos_token_id)

                        predictions = [] 
                        
                        for i in tqdm(range(0, len(source_texts), batch_size), desc=f"  Translating {src_key}->{tgt_key} with {ID} - {checkpoint_num}", leave=False):
                            batch_sources = source_texts[i:i + batch_size]
                            batch_refs = references[i:i + batch_size]
                            
                            if kmt_tagging == False:
                                inputs_batch = batch_sources

                                tokenizer.src_lang = LANG_CODES_MAPPING[src_key]
                                tokenizer.tgt_lang = LANG_CODES_MAPPING[tgt_key]
                                encoded = tokenizer(
                                    inputs_batch,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True,
                                    max_length=128
                                )

                                encoded = {k: v.to(device) for k, v in encoded.items() if k != "token_type_ids"}

                                with torch.no_grad():
                                    batch_output_ids = model.generate(
                                        **encoded,
                                        forced_bos_token_id=forced_bos_token_id,
                                        max_new_tokens=128
                                    )

                                batch_preds = batch_output_ids.cpu().numpy()
                                batch_preds = np.where(batch_preds != -100, batch_preds, tokenizer.pad_token_id)
                                batch_preds = tokenizer.batch_decode(batch_preds, skip_special_tokens=True)
                                batch_preds = [pred.strip() for pred in batch_preds]
                            
                            else:
                                bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
                                eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
                                pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")

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

                            # print samples only from first batch
                            if i == 0:
                                print(f"    Src: {batch_sources[0]}")
                                print(f"    Tgt: {batch_refs[0]}")
                                print(f"    Pre: {batch_preds[0]}")

                            # replace dummy characters in predictions
                            replacements = {
                                '@"': '"',
                                '調"': '"',
                                '付"': '"',
                                "혼'": "'",
                                "ච'": "'",
                                "완-": "-",
                                "罪-": "-",
                                "«": "<<",
                                "»": ">>",
                                "‚": ",",
                                " ' ": "'"
                            }

                            def clean_pred(pred):
                                for dummy, real in replacements.items():
                                    pred = pred.replace(dummy, real)
                                return pred 

                            predictions.extend([
                                {"source": src, "prediction": clean_pred(pred)}
                                for src, pred in zip(batch_sources, batch_preds)
                            ])

                        with open(file_path, "w", encoding="utf-8") as f:
                                json.dump(predictions, f, ensure_ascii=False, indent=2)
                        
                    pred_only = [p["prediction"] for p in predictions]
                    score_all = chrf.compute(predictions=pred_only, references=references)["score"]
                    results.append({
                        "language": langpair,
                        "direction": f"{src_key}->{tgt_key}",
                        "model": ID,
                        "checkpoint": checkpoint_num,
                        "split": "full",
                        "chrf": score_all
                    })

                    if "split" in dataset:
                        # compute scores for our test data, kmt test data, and all test data 
                        ours_preds, ours_refs = [], []
                        kmt_preds, kmt_refs = [], []

                        for pred, ref in zip(predictions, references):
                            src = pred["source"]
                            if src in ours_sources:
                                ours_preds.append(pred["prediction"])
                                ours_refs.append(ref)
                            elif src in kmt_sources:
                                kmt_preds.append(pred["prediction"])
                                kmt_refs.append(ref)
                            else:
                                print("error sorting sentence: ", pred)
                    
                        score_ours = chrf.compute(predictions=ours_preds, references=ours_refs)["score"] if ours_preds else None
                        score_kmt = chrf.compute(predictions=kmt_preds, references=kmt_refs)["score"] if kmt_preds else None
                    
                        if score_ours is not None:
                            results.append({
                            "language": langpair,
                            "direction": f"{src_key}->{tgt_key}",
                            "model": ID,
                            "checkpoint": checkpoint_num,
                            "split": "ours",
                            "chrf": score_ours
                        })
                        if score_kmt is not None:
                            results.append({
                            "language": langpair,
                            "direction": f"{src_key}->{tgt_key}",
                            "model": ID,
                            "checkpoint": checkpoint_num,
                            "split": "kmt",
                            "chrf": score_kmt
                        })

                except Exception as e:
                    print(f"    ERROR: {e}")
                    results.append({
                        "language": langpair,
                        "direction": f"{src_key}->{tgt_key}",
                        "model": ID,
                        "checkpoint": checkpoint_num,
                        "chrf": None,
                        "error": str(e)
                    })

    print("DEBUG RESULTS", results)

    df = pd.DataFrame(results)
    # group by split and direction
    df["direction_type"] = df["direction"].apply(
        lambda x: "XX-en" if x.endswith("->eng") else "en-XX"
    )
    df_sorted = df.sort_values(by=["split", "direction_type"])
    if pov_norm_eval == False:
        extension = ""
    else:
        extension = "_pov_norm"
    df_sorted.to_csv(f"{out_dir}/results_{ID}{extension}.csv", index=False)
    
    print(df_sorted)
    print(df_sorted["chrf"].to_string(index=False))

if __name__ == "__main__":
    fire.Fire(main)
