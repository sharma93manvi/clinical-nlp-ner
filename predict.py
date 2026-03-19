from __future__ import annotations

import argparse
import json
import os
import re
from typing import List

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ner_utils import extract_entities_from_tags


def parse_args():
    parser = argparse.ArgumentParser(description="Run NER inference on clinical text.")
    parser.add_argument("--model_dir", type=str, default="outputs/model")
    parser.add_argument("--text", type=str, required=True, help="Raw clinical text input.")
    parser.add_argument("--output_file", type=str, default="outputs/predictions.json")
    return parser.parse_args()


def simple_tokenize(text: str) -> List[str]:
    # Keeps punctuation as independent tokens for better span readability.
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.eval()

    tokens = simple_tokenize(args.text)
    encoded = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_ids = probs.argmax(dim=-1).tolist()
        confidences = probs.max(dim=-1).values.tolist()

    word_ids = encoded.word_ids(batch_index=0)
    word_preds = {}
    for tok_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx in word_preds:
            continue
        label = model.config.id2label[pred_ids[tok_idx]]
        score = float(confidences[tok_idx])
        word_preds[word_idx] = (label, score)

    final_tags = []
    final_scores = []
    for i in range(len(tokens)):
        label, score = word_preds.get(i, ("O", 0.0))
        final_tags.append(label)
        final_scores.append(score)

    entities = extract_entities_from_tags(tokens, final_tags, final_scores)
    payload = {"text": args.text, "tokens": tokens, "entities": entities}

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
