# Experiment Report

## Objective

Evaluate whether joint sarcasm and sentiment learning improves Arabic sentiment classification over sentiment-only fine-tuning, using a character TF-IDF model as a baseline.

## Data preparation

ArSarcasm-v2 provides 12,548 training examples and 3,000 test examples. Normalization replaces URLs and mentions, removes Arabic diacritics and tatweel, and collapses whitespace while retaining negation, punctuation and emoji.

Four training examples overlap with normalized test text. After removing them and 67 additional training duplicates, the remaining examples are split into 10,605 training and 1,872 validation examples. The official test set is preserved, including six repeated normalized texts. Validation is stratified by the joint label where possible, using seed 42.

This filtering protocol should be considered when comparing results with published benchmark scores.

## Experimental setup

The baseline uses up to 80,000 character 2–5-gram TF-IDF features and class-balanced logistic regression. C is selected from 0.5, 2 and 8 using validation F1; both tasks select C=2. Features are fitted on training data only, without refitting on validation.

The neural models use MARBERT revision `88e1fa192dd723cf0b3563500aec46209762eb22`. Both are trained on Tesla T4 with the following settings:

| Setting | Value |
|---|---|
| Epochs | 3 |
| Batch size | 8 |
| Maximum sequence length | 128 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Seed | 42 |
| Checkpoint selection | Validation sentiment macro-F1 |

The sentiment-only condition minimizes sentiment cross-entropy. Joint learning minimizes the sum of sentiment and sarcasm cross-entropy. The selected checkpoint is epoch 1 for sentiment-only and epoch 2 for joint learning.

## Test results

| Model | Sentiment macro-F1 | Sentiment accuracy | Sarcastic-class F1 |
|---|---:|---:|---:|
| Training-majority classifier | 13.30% | 24.93% | 0.00% |
| Character TF-IDF + logistic regression | 56.75% | 63.60% | 54.04% |
| MARBERT sentiment-only | 65.71% | 71.90% | — |
| MARBERT joint | 65.57% | 70.00% | 58.56% |

Both neural models improve sentiment performance over the baseline. Joint sarcasm F1 also improves by approximately 4.52 percentage points over the baseline.

The joint model's sentiment macro-F1 is approximately 0.15 percentage points lower than the sentiment-only model. This run therefore does not support the hypothesis that joint supervision improves sentiment classification. Aggregate scores from one seed do not establish statistical significance.

## Dialect analysis

| Dialect | Test examples | Sentiment-only macro-F1 | Joint sentiment macro-F1 |
|---|---:|---:|---:|
| Egyptian | 306 | 48.16% | 47.72% |
| Gulf | 322 | 58.15% | 60.04% |
| Levantine | 47 | 65.03% | 56.41% |
| Maghrebi | 2 | 0.00% | 22.22% |
| Modern Standard Arabic | 2,323 | 68.95% | 68.19% |

Performance varies across groups. The very small Maghrebi and Levantine samples limit interpretation, while the test set is dominated by Modern Standard Arabic. Gulf labels span multiple countries and do not establish Saudi-specific performance.

## Reproducibility

The notebooks contain experiment outputs. Aggregate metrics are stored in `baseline_metrics.json` and `colab/marbert-*-metrics.json`. The neural run configuration, GPU and source revision are recorded in `colab/run.json`; baseline software versions are recorded in `environment.json`.

Software tests use a small local BERT to verify training, gradients, checkpoint serialization and inference. These tests check implementation behavior, while the reported model-quality results come from the experiment artifacts. Hosted deployment performance is not measured by these experiments.

## Future work

Repeat both neural conditions with matched seeds and report the mean and standard deviation. Retain paired test predictions for uncertainty analysis, investigate neutral-class errors, and evaluate on an independently collected domain before drawing broader conclusions.
