import os
from typing import List, Dict
import numpy as np
import sacrebleu
from pytorch_lightning import LightningModule
from torch.distributed.fsdp.wrap import wrap
from torch.optim import AdamW
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup


class CreoleMT_model(LightningModule):
    def __init__(
        self,
        model: str,
        exp_dir: str,
        lang_pairs: List[str],
        lang_codes_mapping: Dict[str,str],
        kmt_tagging: bool,
        por_emb: bool,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.por_emb = por_emb
        self.kmt_tagging = kmt_tagging
        self.validation_step_outputs = []
        self.lang_pairs = lang_pairs
        self.lang_codes_mapping = lang_codes_mapping
        self.CreoleMT = None
        self.exp_dir = exp_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(exp_dir, "translation", "tokenizer", model.split("/")[-1]))

        self.gen_kwargs = {
            "max_length": self.hparams.max_seq_length,
            "num_beams": self.hparams.num_beams,
        }

        self.bos_id = self.tokenizer._convert_token_to_id_with_added_voc("<s>")
        self.eos_id = self.tokenizer._convert_token_to_id_with_added_voc("</s>")
        self.pad_id = self.tokenizer._convert_token_to_id_with_added_voc("<pad>")

    def configure_model(self) -> None:
        self.CreoleMT = AutoModelForSeq2SeqLM.from_pretrained(self.model)

        # only need to resize nllb
        if self.model == "facebook/nllb-200-distilled-600M" or self.model == "facebook/nllb-200-distilled-1.3B":
            self.CreoleMT.resize_token_embeddings(len(self.tokenizer))
        
        # initialise with portuguese embeddings if specified
        if self.por_emb == True:
            embeds = self.CreoleMT.model.shared.weight.data
            for lang in ["pov", "aoa", "cri", "fab", "pre", "pap", "kea"]:
                    lang_token_id = self.tokenizer.convert_tokens_to_ids(self.lang_codes_mapping[lang])
                    por_token_id = self.tokenizer.convert_tokens_to_ids(self.lang_codes_mapping["por"])
                    embeds[lang_token_id] = embeds[por_token_id]

    def forward(self, **inputs):
        output = self.CreoleMT(**inputs)
        return output

    def training_step(self, batch, batch_idx):
        # pop target lang code from inputs (not needed in training step)
        _ = batch.pop("forced_bos_token_id")

        output = self(**batch)
        loss = output.loss

        metrics = {
            "train/loss": loss,
        }
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        self.log("global_step", self.global_step, prog_bar=False)
        self.log("train/epoch", self.current_epoch, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # pop target lang code from inputs and return it
        forced_bos_token_id = batch.pop("forced_bos_token_id").cpu().numpy().tolist()

        output = self(**batch)
        loss = output.loss

        labels = batch.pop("labels").cpu().numpy()

        if self.kmt_tagging == True:
            preds = self.CreoleMT.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                # following kreyol-mt implementation
                use_cache=True,
                min_length=1,
                early_stopping=True,
                pad_token_id=self.pad_id,
                bos_token_id=self.bos_id,
                eos_token_id=self.eos_id,
                decoder_start_token_id=forced_bos_token_id,
                **self.gen_kwargs
            )
        else:
            preds = self.CreoleMT.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                forced_bos_token_id=forced_bos_token_id,
                **self.gen_kwargs
        )

        preds = preds.cpu().numpy()
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        # check tokenisation and language tags correctly implemented at validation by decoding with special tokens 
        if batch_idx == 1:
            print("SAMPLE BATCH ITEM IN VALIDATION")
            val_sample_src = self.tokenizer.batch_decode(batch["input_ids"][0], skip_special_tokens=False)
            val_sample_pred = self.tokenizer.batch_decode(preds[0], skip_special_tokens=False)
            val_sample_label = self.tokenizer.batch_decode(labels[0], skip_special_tokens=False)
            print("SRC: ", val_sample_src)
            print("PRED: ", val_sample_pred)
            print("LABEL: ", val_sample_label)
            
        preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]

        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        labels = [label.strip() for label in labels]

        bleu_score = sacrebleu.corpus_bleu(preds, [labels]).score
        chrf_score = sacrebleu.corpus_chrf(preds, [labels]).score

        if batch_idx == 1:
            print(preds[0])
            print(labels[0])
            print(bleu_score)
            print(chrf_score)

        metrics = {
            "eval/bleu": bleu_score,
            "eval/loss": loss,
            "eval/chrf": chrf_score,

        }
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        self.log("global_step", self.global_step, prog_bar=False)
        self.log("train/epoch", self.current_epoch, prog_bar=True)

        return metrics

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        # Optimizer
        model = self.CreoleMT
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        print("Setting learning rate:", self.hparams.learning_rate)
        print("Estimated steps", self.trainer.estimated_stepping_batches)
        assert self.hparams.warmup_ratio == 0 or self.hparams.warmup_steps == 0, \
            "'warmup_ratio' and 'warmup_steps' cannot be both set to non-zero value"

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=(self.hparams.warmup_steps
                              if self.hparams.warmup_ratio == 0 else
                              self.hparams.warmup_ratio * self.trainer.estimated_stepping_batches),
            num_training_steps=self.trainer.estimated_stepping_batches)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
