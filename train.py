from __future__ import annotations

import argparse
import inspect
import json
import os
from functools import partial
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

from data import build_id_maps, load_ner_dataset
from models import load_token_classifier, load_tokenizer
from ner_utils import compute_ner_metrics, tokenize_and_align_labels


# ---------------------------------------------------------------------------
# Custom Trainer with class-weighted cross-entropy loss.
# This is the key fix for DISEASE=0: the "O" label dominates ~85-90% of all
# tokens, so the model learns to predict O for everything. Weighting rare
# entity classes (DISEASE, CHEMICAL) higher forces the model to pay attention.
# ---------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (batch, seq_len, num_labels)

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
        else:
            weights = None

        loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        # logits: (B, T, C) -> (B*T, C); labels: (B, T) -> (B*T,)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(train_dataset, num_labels: int) -> torch.Tensor:
    """
    Compute inverse-frequency weights for each label from the training set.
    Label -100 (padding/subword) is ignored.
    """
    counts = np.zeros(num_labels, dtype=np.float64)
    for example in train_dataset:
        for lbl in example["labels"]:
            if lbl != -100:
                counts[int(lbl)] += 1

    # Avoid division by zero for unseen labels
    counts = np.where(counts == 0, 1, counts)
    weights = counts.sum() / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune biomedical NER model.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="dmis-lab/biobert-base-cased-v1.2",
        choices=[
            "dmis-lab/biobert-base-cased-v1.2",
            "allenai/scibert_scivocab_cased",
        ],
    )
    parser.add_argument("--dataset_name", type=str, default="bigbio/n2c2_2018_track2")
    parser.add_argument("--fallback_dataset", type=str, default="bigbio/bc5cdr")
    parser.add_argument("--output_dir", type=str, default="outputs/model")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Fraction of training steps used for LR warmup.")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true",
                        help="Enable mixed-precision training (faster on GPU).")
    parser.add_argument("--no_class_weights", action="store_true",
                        help="Disable inverse-frequency class weighting (not recommended).")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    bundle = load_ner_dataset(args.dataset_name, args.fallback_dataset)
    label2id, id2label = build_id_maps(bundle.label_list)

    tokenizer = load_tokenizer(args.model_name)
    model = load_token_classifier(args.model_name, label2id, id2label)

    tokenized = bundle.dataset.map(
        partial(tokenize_and_align_labels, tokenizer=tokenizer, label2id=label2id),
        batched=True,
        desc="Tokenizing dataset",
    )

    train_ds = tokenized["train"]
    eval_ds = tokenized["validation"]

    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    # Compute class weights from the (possibly truncated) training set
    class_weights = None
    if not args.no_class_weights:
        print("Computing class weights to handle label imbalance...")
        class_weights = compute_class_weights(train_ds, num_labels=len(bundle.label_list))
        for i, (label, w) in enumerate(zip(bundle.label_list, class_weights)):
            print(f"  {label}: {w:.4f}")

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    metrics_fn = partial(compute_ner_metrics, label_list=bundle.label_list)

    # fp16 causes grad_norm=nan on Apple MPS — use bf16 instead on MPS, fp16 only on CUDA
    use_fp16 = args.fp16 and torch.cuda.is_available()
    use_bf16 = args.fp16 and not torch.cuda.is_available()
    if args.fp16:
        if use_fp16:
            print("Mixed precision: fp16 (CUDA)")
        elif use_bf16:
            print("Mixed precision: bf16 (MPS/CPU fallback — fp16 disabled to avoid grad_norm=nan)")
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs: Dict = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": 50,
        "seed": args.seed,
        "fp16": use_fp16,
        "bf16": use_bf16,
    }
    if "evaluation_strategy" in ta_params:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in ta_params:
        kwargs["eval_strategy"] = "epoch"
    if "save_strategy" in ta_params:
        kwargs["save_strategy"] = "epoch"
    if "logging_strategy" in ta_params:
        kwargs["logging_strategy"] = "steps"
    if "load_best_model_at_end" in ta_params:
        kwargs["load_best_model_at_end"] = True
    if "metric_for_best_model" in ta_params:
        kwargs["metric_for_best_model"] = "f1"
    if "greater_is_better" in ta_params:
        kwargs["greater_is_better"] = True
    if "report_to" in ta_params:
        kwargs["report_to"] = "none"

    training_args = TrainingArguments(**kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": data_collator,
        "compute_metrics": metrics_fn,
        "class_weights": class_weights,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = WeightedTrainer(**trainer_kwargs)

    trainer.train()
    eval_metrics = trainer.evaluate()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "dataset_used": bundle.dataset_name,
        "model_name": args.model_name,
        "labels": bundle.label_list,
        "class_weights_used": not args.no_class_weights,
        "metrics": eval_metrics,
    }
    with open(os.path.join(args.output_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
