from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from datasets import DatasetDict, load_dataset


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
    labels = sorted(int(x) for x in labels)
    return [str(x) for x in labels]


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
        try:
            dataset = load_dataset(candidate)
            selected = candidate
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc

    if dataset is None:
        raise RuntimeError(
            f"Failed loading both '{primary_dataset}' and '{fallback_dataset}'."
        ) from last_err

    if not isinstance(dataset, DatasetDict):
        # convert if a single split comes back for a dataset config
        dataset = DatasetDict({"train": dataset})

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
