from __future__ import annotations

import argparse
import json
import os
import inspect
from functools import partial

from transformers import DataCollatorForTokenClassification, Trainer

from data import load_ner_dataset
from models import load_token_classifier, load_tokenizer
from ner_utils import aggregate_entity_metrics, compute_ner_metrics, tokenize_and_align_labels


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned NER model.")
    parser.add_argument("--model_dir", type=str, default="outputs/model")
    parser.add_argument("--dataset_name", type=str, default="bigbio/n2c2_2018_track2")
    parser.add_argument("--fallback_dataset", type=str, default="bigbio/bc5cdr")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_metrics", type=str, default="outputs/eval_metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_metrics), exist_ok=True)

    bundle = load_ner_dataset(args.dataset_name, args.fallback_dataset)
    label2id = {label: i for i, label in enumerate(bundle.label_list)}
    id2label = {i: label for i, label in enumerate(bundle.label_list)}

    tokenizer = load_tokenizer(args.model_dir)
    model = load_token_classifier(args.model_dir, label2id, id2label)

    tokenized = bundle.dataset.map(
        partial(
            tokenize_and_align_labels,
            tokenizer=tokenizer,
            label2id=label2id,
        ),
        batched=True,
        desc="Tokenizing dataset",
    )

    metrics_fn = partial(compute_ner_metrics, label_list=bundle.label_list)
    trainer_kwargs = {
        "model": model,
        "data_collator": DataCollatorForTokenClassification(tokenizer=tokenizer),
        "compute_metrics": metrics_fn,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    metrics = trainer.evaluate(tokenized[args.split], metric_key_prefix=args.split)
    expanded = {
        "dataset_used": bundle.dataset_name,
        "split": args.split,
        "overall": {
            "precision": metrics.get(f"{args.split}_precision"),
            "recall": metrics.get(f"{args.split}_recall"),
            "f1": metrics.get(f"{args.split}_f1"),
        },
        "per_entity": aggregate_entity_metrics(
            {k.removeprefix(f"{args.split}_"): v for k, v in metrics.items()}
        ),
    }

    with open(args.output_metrics, "w", encoding="utf-8") as f:
        json.dump(expanded, f, indent=2)

    print(json.dumps(expanded, indent=2))


if __name__ == "__main__":
    main()
