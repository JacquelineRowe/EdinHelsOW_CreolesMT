import os
import torch
import shutil
from typing import Dict, List
from datasets import DatasetDict, concatenate_datasets, Dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES

git_repo_path = "/net/storage/pr3/plgrid/plggmultilingualnlp/creole-nllb/creolesMT"

class CreoleMTDataModule(LightningDataModule):
    def __init__(
        self,
        model: str,
        data: str,
        exp_dir: str,
        lang_pairs: List[str],
        lang_codes_mapping: Dict[str, str],
        dist_data: bool,
        pov_norm_train: bool,
        syn_cri: int,
        kmt_tagging: bool,
        max_seq_length: int,
        train_batch_size: int,
        eval_batch_size: int,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.data_path = data
        self.exp_dir = exp_dir
        self.lang_pairs = lang_pairs
        self.lang_codes_mapping = lang_codes_mapping
        self.dist_data = dist_data
        self.pov_norm_train = pov_norm_train
        self.syn_cri = syn_cri
        self.kmt_tagging = kmt_tagging
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.raw_datasets = DatasetDict()
        self.num_proc = 8

        # make lists of source and target languages passed from language pairs
        source_langs = []
        target_langs = []

        print(self.lang_pairs)
        for pair in self.lang_pairs:
            src, tgt = pair.split("-")
            source_langs.append(src)
            target_langs.append(tgt)

        self.source_langs = source_langs
        self.target_langs = target_langs

        self.all_langs = source_langs + target_langs

        # load tokenizer (with extra lang codes if nllb)
        if model == "facebook/nllb-200-distilled-600M" or model == "facebook/nllb-200-distilled-1.3B":
            self.language_codes = FAIRSEQ_LANGUAGE_CODES
            for code in self.all_langs:
                if f"{code}_Latn" not in self.language_codes:
                    self.language_codes.append(f"{code}_Latn")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model,
                                                        additional_special_tokens=self.language_codes,
                                                        use_fast=True)
        elif model == "jhu-clsp/kreyol-mt":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model,
                                                        do_lower_case=False, 
                                                        use_fast=False, 
                                                        keep_accents=True) # copying tokenizer settings from model HF repo

        tokenizer_save_path = os.path.join(self.exp_dir, "translation", "tokenizer", model.split("/")[-1])
        if not os.path.exists(tokenizer_save_path):
            os.makedirs(tokenizer_save_path, exist_ok=True)
            self.tokenizer.save_pretrained(tokenizer_save_path)

    # load data from raw data files
    def setup(self, stage: str):
        if stage == "fit":
            splits = ["train", "validation", "test"]
        else:
            raise NotImplementedError()

        for split in splits:

            if split in self.raw_datasets:
                print(f"Split {split} has been already loaded. Skipping.")
                continue

            iter_datasets = []
            for src_lang, tgt_lang in zip(self.source_langs, self.target_langs):
                if src_lang == tgt_lang:
                    print(f"error: source lang {src_lang} is target lang {tgt_lang}")
                    continue

                suffix = "_filtered" if split == "train" else ""
                
                src_file = None
                tgt_file = None
                ## when distilled data is enabled, use set5 distilled target data for train and val for pap, kea, cri and pov:
                if self.dist_data == True and split != "test":
                    distilled_data_path = f"{git_repo_path}/data/distilled/set_5"
                    if src_lang == "eng" and tgt_lang in ["pap", "pov", "kea", "cri"]:
                        src_file = os.path.join(distilled_data_path, f"{split}.eng-{tgt_lang}.{tgt_lang}{suffix}.eng")
                        tgt_file = os.path.join(distilled_data_path, f"{split}.eng-{tgt_lang}.eng{suffix}.{tgt_lang}")
                    elif tgt_lang == "eng" and src_lang in ["pap", "pov", "kea", "cri"]:
                        src_file = os.path.join(distilled_data_path, f"{split}.eng-{src_lang}.eng{suffix}.{src_lang}")
                        tgt_file = os.path.join(distilled_data_path, f"{split}.eng-{src_lang}.{src_lang}{suffix}.eng")
                if src_file == None and tgt_file == None:
                    # handle additional cri data if specified in params
                    cri_extension = ""
                    if src_lang == "cri" or tgt_lang == "cri":
                        # if using syn cri only for train not for val, should just skip this file
                        if self.syn_cri != 0 and split in ["train", "validation"]:
                            cri_extension = f"_{self.syn_cri}k"
                    if src_lang == "eng":
                        src_file = os.path.join(self.data_path, f"{split}.eng-{tgt_lang}.eng{cri_extension}{suffix}")
                        tgt_file = os.path.join(self.data_path, f"{split}.eng-{tgt_lang}.{tgt_lang}{cri_extension}{suffix}")
                    elif tgt_lang == "eng":
                        src_file = os.path.join(self.data_path, f"{split}.eng-{src_lang}.{src_lang}{cri_extension}{suffix}")
                        tgt_file = os.path.join(self.data_path, f"{split}.eng-{src_lang}.eng{cri_extension}{suffix}")

                # if using normalised pov data for training, if pov is src lang, change src file to pov norm (note: can't co-occur with distilled data)
                if self.pov_norm_train == True and src_lang == "pov":
                    src_file = os.path.join(self.data_path, f"{split}.eng-{src_lang}.{src_lang}_filtered_norm")  

                if not os.path.exists(src_file) or not os.path.exists(tgt_file):
                    print(f"Skipping missing pair: {src_file} / {tgt_file}")
                    continue              

                with open(src_file, "r", encoding="utf-8") as src_f:
                    sources = src_f.read().splitlines()
                with open(tgt_file, "r", encoding="utf-8") as tgt_f:
                    targets = tgt_f.read().splitlines()
                                    
                if len(sources) != len(targets):
                    print(f"Error: Source and target lengths do not match for {tgt_lang} in {split}")
                    raise ValueError()         

                # if distilled data AND syn_cri data is true, we need to combine the distilled data (loaded) with the synthetic data 
                syn_cri_src = None
                syn_cri_tgt = None
                if self.syn_cri != 0 and split in ["train", "validation"] and self.dist_data == True:
                    cri_extension = f"_{self.syn_cri}k"
                    if src_lang == "eng" and tgt_lang == "cri":
                        syn_cri_src = os.path.join(self.data_path, f"{split}.eng-{tgt_lang}.eng{cri_extension}{suffix}")
                        syn_cri_tgt = os.path.join(self.data_path, f"{split}.eng-{tgt_lang}.{tgt_lang}{cri_extension}{suffix}")
                    elif tgt_lang == "eng" and src_lang == "cri":
                        syn_cri_src = os.path.join(self.data_path, f"{split}.eng-{src_lang}.{src_lang}{cri_extension}{suffix}")
                        syn_cri_tgt = os.path.join(self.data_path, f"{split}.eng-{src_lang}.eng{cri_extension}{suffix}")
                    if syn_cri_src is not None and syn_cri_tgt is not None:
                        if not os.path.exists(syn_cri_src) or not os.path.exists(syn_cri_tgt):
                            print(f"Skipping missing pair: {syn_cri_src} / {syn_cri_tgt}")
                            continue 
                        else:
                            with open(syn_cri_src, "r", encoding="utf-8") as src_f:
                                sources_syn = src_f.read().splitlines()
                            with open(syn_cri_tgt, "r", encoding="utf-8") as tgt_f:
                                targets_syn = tgt_f.read().splitlines()
                            
                            # Add deduplication here if needed
                            sources = sources + sources_syn
                            targets = targets + targets_syn
                
                print(f"Creating dataset for {src_lang}-{tgt_lang} {suffix} in split: {split}")
                print(f"  source file: {src_file}")
                print(f"  target file: {tgt_file}")
                print(f"  source len: {len(sources)} | target len: {len(targets)}")
                print(f"  example source: {sources[0]}")
                print(f"  example target: {targets[0]}")

                dset = Dataset.from_dict({"source": sources, "target": targets})

                dset = dset.map(self.convert_to_features,
                                    num_proc=self.num_proc,
                                    remove_columns=["source", "target"],
                                    new_fingerprint=f"features_{split}_{src_lang}_{tgt_lang}",
                                    desc=f"convert_to_features map, {src_lang}-> {tgt_lang}",
                                    fn_kwargs={
                                        "source_key": "source",
                                        "target_key": "target",
                                        "src_lang": self.lang_codes_mapping[src_lang],
                                        "tgt_lang": self.lang_codes_mapping[tgt_lang]})

                # check tokenisation and language tags correctly implemented in data loading and conversion:
                print("SAMPLE DATASET ITEM")
                print(dset[0])                        

                iter_datasets.append(dset)

            if iter_datasets:
                self.raw_datasets[f"{split}"] = iter_datasets[0]
                for i in range(1, len(iter_datasets)):
                    self.raw_datasets[split] = concatenate_datasets(
                        [self.raw_datasets[f"{split}"], iter_datasets[i]])
                    self.raw_datasets[split].set_format(type="torch",
                                                        columns=["input_ids", "attention_mask", "labels",
                                                                    "forced_bos_token_id"])

            else:
                print("No datasets loaded for split", split)

    def train_dataloader(self):
        return DataLoader(self.raw_datasets["train"], batch_size=self.train_batch_size, shuffle=True, num_workers=8)


    def val_dataloader(self):
        return DataLoader(self.raw_datasets["validation"], batch_size=self.eval_batch_size, num_workers=8)

    def convert_to_features(self, example_batch, source_key, target_key, src_lang, tgt_lang):

        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        # if using kmt_tagging method ("{text} <\s> {tag}") then need to split each batch item and manually add tokens,
        # then tokenise without adding special tokens 
        if self.kmt_tagging == True:
            source = example_batch["source"]
            target = example_batch["target"]

            inputs_source = f"{source} </s> {src_lang}"
            inputs_target = f"{tgt_lang} {target} </s>"
            
            features = self.tokenizer(
                inputs_source, 
                text_target=inputs_target,
                max_length=self.max_seq_length, 
                padding="max_length",
                truncation=True,
                add_special_tokens=False
            )
        
        # otherwise, tokenize as normal
        else:
            features = self.tokenizer(
                example_batch[source_key],
                text_target=example_batch[target_key],
                max_length=self.max_seq_length, 
                padding="max_length",
                truncation=True,
                add_special_tokens=True
        )

        # set forced_bos_token_id for batch from target language
        # this will be used as forced_bos_token_id for nllb, and used as decoder_start_token_id for kreyolmt in model module
        features["forced_bos_token_id"] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tgt_lang)

        return features
