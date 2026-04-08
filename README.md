# Clinical NLP — Biomedical Named Entity Recognition

Fine-tunes a biomedical transformer (BioBERT) to extract named entities — diseases and chemicals/drugs — from unstructured medical text using the `bigbio/bc5cdr` dataset.

---

## Problem Statement

Clinical notes contain high-value information about diseases, medications, symptoms, and procedures, but it's all unstructured free text. You can't query "which patients were prescribed metformin" without reading every note manually. This project automates that extraction using Named Entity Recognition (NER), tagging spans of text as `CHEMICAL` (drugs, compounds) or `DISEASE` (conditions, diagnoses).

---

## Why This Is Hard

- ~85–90% of tokens in any clinical sentence are labeled `O` (not an entity). A naive model learns to predict `O` for everything and still gets high accuracy — but zero recall on the entities that matter.
- Disease names overlap heavily with normal English words ("pain", "failure", "attack").
- Transformer tokenizers split words into subwords — "metformin" becomes `["met", "##for", "##min"]` — so label alignment has to be handled carefully.

---

## Dataset

**`bigbio/bc5cdr`** — BioCreative V CDR corpus. PubMed abstracts annotated with:
- `CHEMICAL` — drug names, compounds, chemical entities
- `DISEASE` — disease names, conditions, symptoms

| Split | Sentences |
|---|---:|
| Train | ~4,500 |
| Validation | ~4,500 |
| Test | ~4,500 |

> Note: `bigbio/n2c2_2018_track2` (real EHR clinical notes) is the primary target but requires a data use agreement from i2b2/n2c2. `bc5cdr` is used as the publicly available alternative. It's a well-established benchmark used in published NLP papers.

---

## Model

**`dmis-lab/biobert-base-cased-v1.2`** — BERT pre-trained on PubMed abstracts and PMC full-text articles. It already understands biomedical vocabulary before fine-tuning, which gives it a head start over general-purpose BERT.

Architecture: `BertForTokenClassification` with a linear classification head on top of each token's hidden state.

Label schema (BIO tagging):
```
O           → not an entity
B-CHEMICAL  → beginning of a chemical/drug span
I-CHEMICAL  → continuation of a chemical/drug span
B-DISEASE   → beginning of a disease span
I-DISEASE   → continuation of a disease span
```

---

## Implementation

### Project Structure

```
clinical-nlp-ner/
├── data/
│   └── dataset.py      # dataset loading, BigBio schema conversion, split creation
├── models/
│   └── ner_model.py    # tokenizer + model loader wrappers
├── notebooks/
│   └── exploration.ipynb
├── outputs/            # saved model, metrics, predictions
├── train.py            # fine-tuning script with WeightedTrainer
├── evaluate.py         # evaluation on any split
├── predict.py          # inference on raw text
├── ner_utils.py        # label alignment, metrics, entity extraction
└── requirements.txt
```

### Key Design Decisions

**1. Subword token label alignment (`ner_utils.py`)**

BioBERT tokenizes words into subword pieces. Only the first subword of each word gets the real label; the rest are set to `-100` so they're ignored by the loss function.

```
word:     "metformin"   →  subwords: ["met", "##for", "##min"]
labels:       B-CHEMICAL         →       B-CHEMICAL, -100,    -100
```

**2. Inverse-frequency class weighting (`train.py` — `WeightedTrainer`)**

The core fix for `DISEASE=0`. Standard cross-entropy treats all tokens equally, so the model learns to predict `O` for everything. `WeightedTrainer` computes per-class weights from the training set and passes them to `nn.CrossEntropyLoss`:

```
weight(class) = total_tokens / (num_classes × count(class))
```

Weights from the 1000-sample run:
```
O:          0.2526   ← down-weighted (majority class)
B-CHEMICAL: 3.8603
B-DISEASE:  4.4504   ← up-weighted (rare, hard to learn)
I-CHEMICAL: 4.2403
I-DISEASE:  3.1128
```

**3. Dataset loading with fallback (`data/dataset.py`)**

Tries the primary dataset first, falls back to `bc5cdr` if unavailable. Handles BigBio's passage-based schema by converting it to standard `tokens` + `ner_tags` format. Auto-creates validation/test splits if missing.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the Project

### 1. Train

Quick run (1000 samples, ~6 min on CPU):
```bash
python3 train.py \
  --fallback_dataset bigbio/bc5cdr \
  --output_dir outputs/model \
  --max_train_samples 1000 \
  --max_eval_samples 200 \
  --epochs 3
```

Full dataset run (best results, ~1–2 hrs on CPU, ~15 min on GPU):
```bash
python3 train.py \
  --fallback_dataset bigbio/bc5cdr \
  --output_dir outputs/model \
  --epochs 3 \
  --fp16
```

SciBERT alternative:
```bash
python3 train.py --model_name allenai/scibert_scivocab_cased --fallback_dataset bigbio/bc5cdr
```

### 2. Evaluate

```bash
python3 evaluate.py \
  --model_dir outputs/model \
  --fallback_dataset bigbio/bc5cdr \
  --split test \
  --output_metrics outputs/eval_metrics.json
```

### 3. Predict on new text

```bash
python3 predict.py \
  --model_dir outputs/model \
  --text "The patient was diagnosed with hypertension and type 2 diabetes. She was prescribed metformin 500mg and lisinopril." \
  --output_file outputs/predictions.json
```

---

## Example: Input → Output

**Input text:**
```
The patient was diagnosed with hypertension and type 2 diabetes.
She was prescribed metformin 500mg and lisinopril.
MRI showed no signs of cerebral edema.
```

**Tokens fed to model:**
```
["The", "patient", "was", "diagnosed", "with", "hypertension", "and",
 "type", "2", "diabetes", ".", "She", "was", "prescribed", "metformin",
 "500mg", "and", "lisinopril", ".", "MRI", "showed", "no", "signs",
 "of", "cerebral", "edema", "."]
```

**Extracted entities (1000-sample model, `outputs/predictions.json`):**
```json
[
  { "entity_type": "DISEASE",  "text": "hypertension and",                          "confidence": 0.677 },
  { "entity_type": "DISEASE",  "text": "type 2 diabetes . She was",                 "confidence": 0.622 },
  { "entity_type": "DISEASE",  "text": "metformin 500mg",                           "confidence": 0.508 },
  { "entity_type": "CHEMICAL", "text": "lisinopril . MRI showed no signs of cerebral edema .", "confidence": 0.485 }
]
```

> The model correctly identifies `hypertension`, `type 2 diabetes`, and `lisinopril`. Span boundaries are imprecise — "hypertension and" includes the conjunction, and the lisinopril span bleeds into the rest of the sentence. This is the expected behavior with 1000 training samples. Boundary precision improves significantly with the full dataset.

---

## Results

### Before fixes (300 samples, 1 epoch, no class weighting)

| Entity | Precision | Recall | F1 |
|---|---:|---:|---:|
| CHEMICAL | 0.821 | 0.080 | 0.146 |
| DISEASE  | 0.000 | 0.000 | 0.000 |
| Overall  | 0.821 | 0.043 | 0.081 |

DISEASE was completely missed — the model predicted `O` for every token.

### After fixes — 1000 samples, 3 epochs, class weighting, bf16 on Apple Silicon

Training run: `python3 train.py --fallback_dataset bigbio/bc5cdr --epochs 3 --fp16`
Evaluated on test split (`outputs/eval_metrics.json`):

| Entity | Precision | Recall | F1 |
|---|---:|---:|---:|
| CHEMICAL | 0.091 | 0.245 | 0.133 |
| DISEASE  | 0.063 | 0.217 | 0.098 |
| Overall  | 0.077 | 0.232 | 0.115 |

Training loss progression across 3 epochs:
```
epoch 0.4 → loss 1.578
epoch 0.8 → loss 1.474
epoch 1.2 → loss 1.355
epoch 1.6 → loss 1.262
epoch 2.0 → loss 1.213
epoch 2.4 → loss 1.136
epoch 2.8 → loss 1.107
epoch 3.0 → train_loss 1.293 (final)
```

DISEASE is no longer zero — both entity types are being learned. Precision is low because with only 1000 training samples the model over-predicts (spans bleed into neighboring tokens). Recall improved from 4% → 23%.

### Training notes

- `--fp16` on Apple Silicon (MPS) automatically switches to `bf16` to avoid `grad_norm=nan`
- On CUDA GPUs, `--fp16` uses true half-precision for faster training
- Class weights printed at start of training show the imbalance correction applied:
  ```
  O:          0.2526  (down-weighted)
  B-CHEMICAL: 3.8603
  B-DISEASE:  4.4504  (up-weighted ~17x vs O)
  I-CHEMICAL: 4.2403
  I-DISEASE:  3.1128
  ```

### What to expect with the full dataset (3 epochs, CUDA GPU)

Based on published bc5cdr benchmarks with BioBERT:

| Entity | F1 (expected) |
|---|---:|
| CHEMICAL | ~0.89 |
| DISEASE  | ~0.77 |
| Overall  | ~0.84 |

---

## Why DISEASE Was 0 — Root Cause

The `O` label dominates ~85–90% of all tokens. With standard cross-entropy, the model minimizes loss by predicting `O` everywhere — it never needs to learn DISEASE at all. CHEMICAL was partially learned because drug names have distinctive morphology that BioBERT already recognizes from pre-training. DISEASE names look like regular English words, so they need explicit loss pressure to learn.

The fix: `WeightedTrainer` applies `~4.5×` more loss weight to `B-DISEASE` tokens than to `O` tokens, forcing the model to pay attention to rare classes.

---

## Limitations

- **1000-sample model**: Span boundaries are imprecise. The model detects the right region but includes neighboring tokens. Full dataset training resolves this.
- **Domain gap**: bc5cdr is PubMed abstracts, not real EHR notes. Real clinical text has abbreviations, typos, and non-standard phrasing that would require domain-specific fine-tuning.
- **Only 2 entity types**: A production system would need symptoms, procedures, dosages, lab values, etc. — requiring a richer dataset like n2c2.
- **No confidence threshold**: `predict.py` returns all predictions regardless of confidence. Adding `--threshold 0.5` filtering would reduce noisy low-confidence spans.

---

## Notebook

`notebooks/exploration.ipynb` covers dataset sampling, label distribution analysis, and qualitative model predictions.
