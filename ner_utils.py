from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


def tokenize_and_align_labels(
    examples, tokenizer, label_all_tokens: bool = False, label2id: Optional[Dict[str, int]] = None
):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=256,
    )

    labels = []
    for i, label_ids in enumerate(examples["ner_tags"]):
        if label_ids and isinstance(label_ids[0], str):
            if label2id is None:
                raise ValueError("label2id is required when ner_tags are strings.")
            label_ids = [label2id[x] for x in label_ids]
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        aligned = []

        for word_idx in word_ids:
            if word_idx is None:
                aligned.append(-100)
            elif word_idx != previous_word_idx:
                aligned.append(label_ids[word_idx])
            else:
                aligned.append(label_ids[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(aligned)

    tokenized["labels"] = labels
    return tokenized


def compute_ner_metrics(eval_pred, label_list: List[str]) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=2)

    true_predictions = []
    true_labels = []
    for pred_row, label_row in zip(predictions, labels):
        pred_tags = []
        label_tags = []
        for p, l in zip(pred_row, label_row):
            if l != -100:
                pred_tags.append(label_list[p])
                label_tags.append(label_list[l])
        true_predictions.append(pred_tags)
        true_labels.append(label_tags)

    # zero_division=0 suppresses UndefinedMetricWarning when a class has no predictions
    metrics: Dict[str, float] = {
        "precision": precision_score(true_labels, true_predictions, zero_division=0),
        "recall": recall_score(true_labels, true_predictions, zero_division=0),
        "f1": f1_score(true_labels, true_predictions, zero_division=0),
    }

    report = classification_report(true_labels, true_predictions, output_dict=True, zero_division=0)
    for key, val in report.items():
        if isinstance(val, dict) and "f1-score" in val:
            safe = key.replace(" ", "_")
            metrics[f"{safe}_precision"] = val.get("precision", 0.0)
            metrics[f"{safe}_recall"] = val.get("recall", 0.0)
            metrics[f"{safe}_f1"] = val.get("f1-score", 0.0)
    return metrics


def get_entity_type(tag: str) -> str:
    if tag == "O":
        return "O"
    if "-" in tag:
        return tag.split("-", 1)[1]
    return tag


def extract_entities_from_tags(
    tokens: List[str], tags: List[str], scores: List[float]
) -> List[Dict]:
    entities = []
    current_tokens = []
    current_scores = []
    current_type = None
    start_idx = 0

    for i, (tok, tag, score) in enumerate(zip(tokens, tags, scores)):
        if tag == "O":
            if current_tokens:
                entities.append(
                    {
                        "entity_type": current_type,
                        "text": " ".join(current_tokens),
                        "start_token": start_idx,
                        "end_token": i - 1,
                        "confidence": float(np.mean(current_scores)),
                    }
                )
                current_tokens, current_scores, current_type = [], [], None
            continue

        prefix = "I"
        ent_type = tag
        if "-" in tag:
            prefix, ent_type = tag.split("-", 1)

        if (
            prefix == "B"
            or current_type != ent_type
            or (prefix == "I" and not current_tokens)
        ):
            if current_tokens:
                entities.append(
                    {
                        "entity_type": current_type,
                        "text": " ".join(current_tokens),
                        "start_token": start_idx,
                        "end_token": i - 1,
                        "confidence": float(np.mean(current_scores)),
                    }
                )
            current_tokens = [tok]
            current_scores = [score]
            current_type = ent_type
            start_idx = i
        else:
            current_tokens.append(tok)
            current_scores.append(score)

    if current_tokens:
        entities.append(
            {
                "entity_type": current_type,
                "text": " ".join(current_tokens),
                "start_token": start_idx,
                "end_token": len(tokens) - 1,
                "confidence": float(np.mean(current_scores)),
            }
        )
    return entities


def aggregate_entity_metrics(metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    per_class: Dict[str, Dict[str, float]] = defaultdict(dict)
    for key, value in metrics.items():
        if key.count("_") < 1:
            continue
        suffixes = ("_precision", "_recall", "_f1")
        matched = next((s for s in suffixes if key.endswith(s)), None)
        if matched is None:
            continue
        entity = key[: -len(matched)]
        metric_name = matched.lstrip("_")
        per_class[entity][metric_name] = round(float(value), 4)
    return dict(per_class)
