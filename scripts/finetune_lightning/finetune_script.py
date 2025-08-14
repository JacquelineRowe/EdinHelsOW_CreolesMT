import datetime
import os
from typing import List
import fire
import torch
from lightning_fabric import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import json

from data_module import CreoleMTDataModule
from model_module import CreoleMT_model

def main(
        # specified
        ID: str,
        base_model: str, 
        dataset: str,
        lang_pairs: str,
        por_emb: bool = False,
        dist_data: bool = False,
        pov_norm_train: bool = False,
        kmt_tagging: bool = False,
        syn_cri: int = 0,
        # default
        strategy: str = "auto",
        devices_count: int = 1,
        max_seq_length: int = 256,
        num_beams: int = 1,
        train_batch_size: int = 32,
        eval_batch_size: int = 16,
        seed: int = 42,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        warmup_ratio: float = 0.0,
        weight_decay: float = 0.01,
        adam_epsilon: float = 1e-8,
        max_epochs: int = 30,
):
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
        project_name = "kreyol-mt-finetuning"
        # mimicking default kmt settings
        max_seq_length = 60
        num_beams = 4

    elif base_model == "facebook/nllb-200-distilled-600M" or base_model == "facebook/nllb-200-distilled-1.3B":
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
        project_name = "NLLB-finetuning"

    torch.set_float32_matmul_precision('medium')
    grant_root_dir = "/net/storage/pr3/plgrid/plggmultilingualnlp"

    hpams = {
        # specified
        "model": base_model,
        "data": f"{grant_root_dir}/creole-nllb/creolesMT/data/preprocessed_lusophone/{dataset}/",
        "exp_dir": f"{grant_root_dir}/creole-nllb/{ID}",
        "lang_pairs": lang_pairs,
        "lang_codes_mapping": LANG_CODES_MAPPING,
        "dist_data": dist_data,
        "pov_norm_train": pov_norm_train,
        "syn_cri": syn_cri,
        "kmt_tagging": kmt_tagging,
        "por_emb": por_emb,
        # default
        "max_seq_length": max_seq_length,
        "num_beams": num_beams,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "seed": seed,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "adam_epsilon": adam_epsilon,
        "max_epochs": max_epochs
    }

    seed = 42
    seed_everything(seed)

    data_module = CreoleMTDataModule(**hpams)
    data_module.setup("fit")
    model = CreoleMT_model(**hpams)
    logger = WandbLogger(project=project_name, name=f"{ID}-{datetime.datetime.now()}")
    checkpoint_callback = ModelCheckpoint(
        monitor="eval/chrf",
        dirpath=os.path.join(hpams["exp_dir"], "checkpoints"),
        filename="best-{global_step}-{eval_chrf:.2f}",
        every_n_train_steps = 5000,
        save_top_k=3,
        mode="max",
        save_weights_only=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="eval/chrf",
        patience=3,
        mode="max",
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # ---- Trainer ----
    trainer = Trainer(
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback, early_stop_callback],
        max_epochs=hpams["max_epochs"],
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy=strategy,
        num_nodes=1,
        devices=devices_count,
        gradient_clip_algorithm="value",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        val_check_interval=5000,
    )

    # ---- Train ----
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    fire.Fire(main)
