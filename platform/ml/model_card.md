# Model Card — {model_name} v{model_version}

## Overview
- **Task:** Binary classification — UCI Bank Marketing (predicts term-deposit subscription).
- **Model family:** {model_family}
- **Trained:** {trained_at}
- **Decision threshold:** {threshold:.4f} (highest threshold where validation recall ≥ {min_recall:.2f}).
- **Registered alias:** `{alias}`

## Dataset
- Source: UCI Bank Marketing — `bank-additional-full.csv` (semicolon-separated).
- Rows: train={n_train}, val={n_val}, test={n_test}; positive rate {positive_rate:.4f}.
- Stratified 60/20/20 split, `random_state=42`.
- Dataset hash (sha256 of cleaned frame): `{dataset_hash}`.

## Data prep (locked by `tests/test_data.py`)
- `duration` dropped (target leakage, per UCI data card).
- `y` mapped to int `target` (yes→1, no→0).
- `"unknown"` retained as a literal categorical level (not imputed).
- `pdays == 999` encoded as `was_previously_contacted` (0/1) and `pdays_clean` (0 when sentinel); original `pdays` dropped.

## Validation metrics (held-in, used to pick the threshold)
- Accuracy: {val_accuracy:.4f}
- Macro F1: {val_f1_macro:.4f}
- ROC-AUC:  {val_roc_auc:.4f}
- Precision: {val_precision:.4f}
- Recall:    {val_recall:.4f}

## Test metrics (held-out, single-shot at the locked threshold)
- Accuracy: {test_accuracy:.4f}
- Macro F1: {test_f1_macro:.4f}
- ROC-AUC:  {test_roc_auc:.4f}
- Precision: {test_precision:.4f}
- Recall:    {test_recall:.4f}

## Intended use & limitations
- Internal Drift Triage Co-Pilot project — not for production banking decisions.
- Trained on a single Portuguese bank's calls; will not generalise to other markets.
- `class_weight="balanced"` for logreg/RF; gradient boosting relies on threshold tuning to manage imbalance.

## Environment fingerprint
{env_fingerprint_md}
