# Model card

## Available evaluated model

The Streamlit app uses character TF-IDF and independent class-balanced logistic regression classifiers. Features are fit on training only, up to 80,000 character 2–5-grams, with minimum document frequency three. Validation selects C independently from 0.5, 2 and 8. The sentiment classifier uses one-versus-rest. This is not a fine-tuned language model.

The local `artifacts/baseline.joblib` is reproducible with `python -m maqsad.baseline`. Only load joblib files you trust, since deserialization can execute Python code.

## Neural model implementation

MARBERT's CLS representation feeds a dropout layer and two linear heads: two sarcasm classes and three sentiment classes. Joint training minimizes the sum of the two cross-entropy losses. The sentiment-only control has the same encoder and sentiment head but excludes sarcasm loss. Its untrained sarcasm head is not reported or used for inference.

Checkpoint selection uses validation sentiment macro-F1. Full training and matched ablation are not yet executed in this environment; no neural quality metrics are claimed. Offline tiny-BERT tests validate software mechanics only.

## Intended use and limitations

Educational research and exploratory Arabic-text analysis. Predictions are not reliable determinations of a person's intent. Scores are uncalibrated, conversational context is absent, and dialect coverage is unequal. Historical tweets are not representative of every modern domain. The model does not generate explanations or predict dialects.

See `reports/EXPERIMENT_REPORT.md` for current measured results, and the executed notebook for confusion matrices and subgroup metrics.
