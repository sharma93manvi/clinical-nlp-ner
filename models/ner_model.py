from __future__ import annotations

from typing import Dict

from transformers import AutoModelForTokenClassification, AutoTokenizer


def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def load_token_classifier(
    model_name: str, label2id: Dict[str, int], id2label: Dict[int, str]
):
    return AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )
