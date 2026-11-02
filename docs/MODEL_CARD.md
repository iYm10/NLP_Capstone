# Maqsad Model Card

## Overview

Maqsad predicts sarcasm and sentiment in Arabic text. The main model fine-tunes `UBC-NLP/MARBERT` on ArSarcasm-v2. Its checkpoint is hosted at [yahya-ai10/maqsad-marbert](https://huggingface.co/yahya-ai10/maqsad-marbert).

The application supports the joint MARBERT model and a lightweight TF-IDF baseline. It analyzes single texts or batches and exports predictions as CSV.

## Architecture and training

MARBERT's CLS representation passes through dropout and two linear heads: two sarcasm classes and three sentiment classes. Joint training minimizes the sum of the two cross-entropy losses. A sentiment-only control uses the same encoder and excludes the sarcasm loss; its untrained sarcasm head is not used for inference or reported in evaluation.

Training uses three epochs, batch size 8, maximum length 128, AdamW learning rate 2e-5 and seed 42. Checkpoints are selected by validation sentiment macro-F1.

The baseline uses character 2–5-gram TF-IDF with independent class-balanced logistic regression classifiers. Features are fitted only on training text. Validation selects C=2 for both tasks.

## Evaluation

| Model | Sentiment macro-F1 | Sarcastic-class F1 |
|---|---:|---:|
| TF-IDF | 56.75% | 54.04% |
| MARBERT sentiment-only | 65.71% | — |
| MARBERT joint | 65.57% | 58.56% |

Results use the official 3,000-example test set after filtering overlapping texts from training. Joint learning does not improve sentiment over the sentiment-only control in this single-seed experiment. See the [experiment report](../reports/EXPERIMENT_REPORT.md) for details and dialect-level results.

## Intended use

Educational research and exploratory analysis of Arabic text. Outputs are model predictions rather than definitive judgments about the author's intent. The model does not generate explanations or classify dialects.

## Limitations

Sarcasm can depend on missing conversational context. Training data consists of historical tweets, dialect representation is uneven, and performance may change across domains. Scores are uncalibrated. Multiple seeds and additional domain evaluations are needed to assess robustness.

## Inference

The Streamlit application downloads the hosted checkpoint on demand and caches one loaded model. A complete local checkpoint takes precedence. Model artifacts should come from a trusted source. Available RAM must accommodate the full model; hosting resource requirements have not been benchmarked.

Offline tests use a small synthetic BERT model to verify software behavior. They do not measure the quality of the full trained checkpoint.
