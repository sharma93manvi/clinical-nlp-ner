from __future__ import annotations

import argparse
import inspect
import json
import os
from functools import partial

from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

from data import build_id_maps, load_ner_dataset
from models import load_token_classifier, load_tokenizer
from ner_utils import compute_ner_metrics, tokenize_and_align_labels


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
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
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

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    metrics_fn = partial(compute_ner_metrics, label_list=bundle.label_list)

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "logging_steps": 50,
        "seed": args.seed,
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
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    eval_metrics = trainer.evaluate()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "dataset_used": bundle.dataset_name,
        "model_name": args.model_name,
        "labels": bundle.label_list,
        "metrics": eval_metrics,
    }
    with open(os.path.join(args.output_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
