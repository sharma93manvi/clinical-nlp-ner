# Clinical NLP NER and Information Extraction

This project fine-tunes a biomedical transformer for named entity recognition (NER) and information extraction on medical text, with a primary target dataset of `bigbio/n2c2_2018_track2` and automatic fallback to `bigbio/bc5cdr` when needed.

## Why Clinical NLP Matters

Clinical notes contain high-value information about diseases, medications, symptoms, and procedures, but most of it is unstructured text. Reliable NER helps:

- improve patient cohort discovery and clinical research workflows
- reduce manual chart review effort
- support downstream decision support systems
- speed up medical coding and quality reporting

## Project Structure

```text
clinical-nlp-ner/
├── data/               # data loading utilities
├── models/             # model wrappers
├── notebooks/          # exploratory analysis
├── outputs/            # saved predictions, metrics
├── train.py            # fine-tuning script
├── predict.py          # inference on raw text
├── evaluate.py         # evaluation metrics
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

BioBERT (default):

```bash
python train.py \
  --model_name dmis-lab/biobert-base-cased-v1.2 \
  --dataset_name bigbio/n2c2_2018_track2 \
  --fallback_dataset bigbio/bc5cdr \
  --output_dir outputs/model
```

SciBERT alternative:

```bash
python train.py --model_name allenai/scibert_scivocab_cased
```

## Evaluate

```bash
python evaluate.py \
  --model_dir outputs/model \
  --dataset_name bigbio/n2c2_2018_track2 \
  --fallback_dataset bigbio/bc5cdr \
  --split test \
  --output_metrics outputs/eval_metrics.json
```

## Inference on New Clinical Text

```bash
python predict.py \
  --model_dir outputs/model \
  --text "The patient with diabetes reports chest pain and was started on metformin after MRI procedure." \
  --output_file outputs/predictions.json
```

The output JSON includes extracted entities with confidence scores.

## Example Input and Output

Input:

```text
The patient with diabetes reports chest pain and was started on metformin after MRI procedure.
```

Example extracted entities:

```json
[
  {"entity_type": "DISEASE", "text": "diabetes", "confidence": 0.94},
  {"entity_type": "SYMPTOM", "text": "chest pain", "confidence": 0.89},
  {"entity_type": "DRUG", "text": "metformin", "confidence": 0.96},
  {"entity_type": "PROCEDURE", "text": "MRI procedure", "confidence": 0.91}
]
```

## Results (Fill After Training)

After running `evaluate.py`, copy values from `outputs/eval_metrics.json`.

| Entity Type | Precision | Recall | F1 |
|---|---:|---:|---:|
| DISEASE | - | - | - |
| DRUG / MEDICATION | - | - | - |
| SYMPTOM | - | - | - |
| PROCEDURE | - | - | - |
| Overall | - | - | - |

## Notebook

See `notebooks/exploration.ipynb` for:

- dataset sampling and preview
- label distribution analysis
- example model predictions with highlighted entities
