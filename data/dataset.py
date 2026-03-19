from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import re

from datasets import Dataset, DatasetDict, get_dataset_config_names, load_dataset


@dataclass
class DatasetBundle:
    dataset: DatasetDict
    label_list: List[str]
    dataset_name: str


def _infer_label_list(dataset: DatasetDict) -> List[str]:
    train_features = dataset["train"].features
    if "ner_tags" in train_features:
        feature = train_features["ner_tags"].feature
        if hasattr(feature, "names") and feature.names:
            return list(feature.names)

    labels = set()
    for split in dataset.keys():
        if "ner_tags" not in dataset[split].column_names:
            continue
        for row in dataset[split]:
            labels.update(row["ner_tags"])
    labels = {str(x) for x in labels}
    ordered = ["O"] + sorted(x for x in labels if x != "O")
    return ordered


def _tokenize_with_offsets(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    tokens: List[str] = []
    offsets: List[Tuple[int, int]] = []
    for m in re.finditer(r"\S+", text):
        tokens.append(m.group(0))
        offsets.append((m.start(), m.end()))
    return tokens, offsets


def _convert_bigbio_source_to_token_tags(dataset: DatasetDict) -> DatasetDict:
    converted = DatasetDict()

    for split, split_ds in dataset.items():
        out = {"tokens": [], "ner_tags": []}

        for row in split_ds:
            passages = row.get("passages", [])
            for passage in passages:
                text = passage.get("text", "")
                if not text:
                    continue
                tokens, token_offsets = _tokenize_with_offsets(text)
                if not tokens:
                    continue

                tags = ["O"] * len(tokens)
                entities = passage.get("entities", [])
                for ent in entities:
                    ent_type = ent.get("type", "ENTITY").upper()
                    for span in ent.get("offsets", []):
                        if not span or len(span) < 2:
                            continue
                        char_start, char_end = span[0], span[1]
                        hit_indices = [
                            i
                            for i, (tok_start, tok_end) in enumerate(token_offsets)
                            if not (tok_end <= char_start or tok_start >= char_end)
                        ]
                        for j, idx in enumerate(hit_indices):
                            prefix = "B" if j == 0 else "I"
                            tags[idx] = f"{prefix}-{ent_type}"

                out["tokens"].append(tokens)
                out["ner_tags"].append(tags)

        converted[split] = Dataset.from_dict(out)

    return converted


def load_ner_dataset(
    primary_dataset: str = "bigbio/n2c2_2018_track2",
    fallback_dataset: str = "bigbio/bc5cdr",
) -> DatasetBundle:
    """
    Loads a token-level NER dataset from HuggingFace datasets with fallback.
    """
    selected = primary_dataset
    last_err: Optional[Exception] = None
    dataset: Optional[DatasetDict] = None

    for candidate in [primary_dataset, fallback_dataset]:
        attempts = [
            {},
            {"trust_remote_code": True},
        ]

        # Some BigBio datasets require an explicit config name.
        try:
            configs = get_dataset_config_names(candidate, trust_remote_code=True)
            if configs:
                attempts.append({"name": configs[0], "trust_remote_code": True})
        except Exception:
            pass

        for kwargs in attempts:
            try:
                dataset = load_dataset(candidate, **kwargs)
                selected = candidate
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        if dataset is not None:
            break

    if dataset is None:
        raise RuntimeError(
            f"Failed loading both '{primary_dataset}' and '{fallback_dataset}'."
        ) from last_err

    if not isinstance(dataset, DatasetDict):
        # convert if a single split comes back for a dataset config
        dataset = DatasetDict({"train": dataset})

    # Normalize BigBio source schema to token-tag format.
    if "train" in dataset and {"tokens", "ner_tags"} - set(dataset["train"].column_names):
        if "passages" in dataset["train"].column_names:
            dataset = _convert_bigbio_source_to_token_tags(dataset)

    # If no validation split exists, create one from train.
    if "validation" not in dataset and "train" in dataset:
        split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    if "test" not in dataset and "validation" in dataset:
        dataset["test"] = dataset["validation"]

    required_cols = {"tokens", "ner_tags"}
    for split in dataset.keys():
        missing = required_cols - set(dataset[split].column_names)
        if missing:
            raise ValueError(
                f"Dataset '{selected}' split '{split}' is missing columns: {missing}. "
                "This project expects token-level fields `tokens` and `ner_tags`."
            )

    label_list = _infer_label_list(dataset)
    return DatasetBundle(dataset=dataset, label_list=label_list, dataset_name=selected)


def build_id_maps(label_list: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label
