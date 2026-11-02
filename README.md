# Maqsad — Arabic Sarcasm & Sentiment Analysis

Understand sentiment behind Arabic sarcasm with a reproducible NLP capstone.

> **Execution status:** The CPU notebook trains and evaluates a TF-IDF baseline on ArSarcasm-v2. MARBERT joint fine-tuning and its sentiment-only control are implemented, but full MARBERT training has **not** been executed: this environment has no GPU and its network policy blocks Hugging Face downloads. See [the experiment report](reports/EXPERIMENT_REPORT.md) for measured results and exact limitations.

## Research question

**Does jointly learning sarcasm improve Arabic sentiment classification compared with sentiment-only fine-tuning?**

An Arabic comment can contain positive words while expressing frustration. Maqsad predicts both sarcasm and sentiment, and evaluates performance across dialect groups. The current baseline establishes a reference; it does not prove the joint-learning hypothesis.

## Included

- An **executed notebook** with dataset audit, full CPU baseline training, held-out metrics, plots and inference.
- A **GPU notebook** for full MARBERT joint learning and the matched sentiment-only experiment.
- A shared transformer encoder with separate sarcasm and sentiment heads.
- Validation-based checkpoint selection, local checkpoint inference, and reproducible data splitting.
- A Streamlit demo using the measured CPU baseline.
- Unit and offline end-to-end training tests; GitHub Actions runs tests automatically.
- Dataset documentation, model limitations, citations and experiment methodology in English.

## Quick start

Python 3.10+ is required; the recorded run used Python 3.12. Create an isolated environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,app]"
python scripts/download_data.py
python -m maqsad.baseline
streamlit run app.py
```

On Windows activate with `.venv\Scripts\activate`. The baseline does not require a GPU or a Hugging Face account. Data is downloaded from an immutable upstream revision with SHA-256 verification. Raw data and trained artifacts are generated locally and excluded from Git.

To reuse the main package versions from the recorded run, add `-c constraints-tested.txt` to the installation command. This file is not a complete transitive lockfile.

To reproduce the executed notebook:

```bash
python -m ipykernel install --sys-prefix --name python3
python scripts/execute_notebook.py
```

The runner saves outputs into `notebooks/01_data_and_baseline.ipynb` and fails on cell errors. Its training step can take several minutes on a CPU. Open the notebook directly on GitHub to inspect the saved outputs.

## Full fine-tuning

[Open the training notebook in Google Colab](https://colab.research.google.com/github/iYm10/NLP_Capstone/blob/main/notebooks/02_marbert_finetuning.ipynb). Select a GPU runtime and run the cells in order. The notebook automatically clones this repository and asks you to authorize Google Drive storage. Checkpoints are written directly to a timestamped `MyDrive/Maqsad/artifacts` directory. At completion, a small metrics archive is exported for GitHub; large weights remain in Drive. Save the executed notebook separately. These Colab-specific steps are prepared but have not been executed from this environment.

Use a GPU runtime with access to Hugging Face. Install the PyTorch build appropriate to that runtime, then:

```bash
python -m pip install -e ".[train,dev]"
python -m maqsad.train --task sentiment --output artifacts/marbert-sentiment
python -m maqsad.train --task joint --output artifacts/marbert-joint
```

For strict comparisons use the same `--revision <Hugging-Face-commit>`, `--seed`, epochs, batch size and sequence length for both commands. `notebooks/02_marbert_finetuning.ipynb` resolves one immutable model revision and runs both experiments. Default settings: three epochs, batch size eight, maximum length 128, AdamW learning rate 2e-5. Memory requirements depend on the GPU; reduce batch size when necessary. CPU fine-tuning is supported by the CLI but can be very slow.

Each run saves `weights.pt`, tokenizer files, encoder configuration and `metrics.json` in its output folder. Separate output folders are required for separate tasks/seeds. Metrics include the resolved model revision when available. No fine-tuned weights or MARBERT scores are published in this version.

```python
from maqsad.inference import predict_transformer
results = predict_transformer(["يا سلام على الخدمة، ساعتين وأنا أنتظر!"])
```

## Evaluation protocol

1. Keep the official 3,000-row test set intact.
2. Remove training texts overlapping normalized test text, and deduplicate remaining training text.
3. Reserve 15% of the remaining training data for stratified validation, seed 42.
4. Fit preprocessing features on training only. Select baseline C on validation.
5. Select each MARBERT checkpoint by validation sentiment macro-F1 for a matched primary comparison.
6. Evaluate the selected model once on test. Report sarcastic-class F1, sentiment macro-F1, class-level and dialect-level results.
7. Repeat neural experiments with at least three matched seeds before claiming consistent improvement.

The official test set may itself contain repeated text; counts are reported rather than silently changing the benchmark. Minority dialect results need cautious interpretation. The Maghrebi test group contains only two tweets.

## Repository layout

```text
app.py                         Streamlit baseline demo
src/maqsad/data.py             Pinned download, validation and splitting
src/maqsad/baseline.py         CPU training and inference
src/maqsad/model.py            Shared encoder and two task heads
src/maqsad/train.py            Neural fine-tuning and evaluation
src/maqsad/inference.py        Local neural checkpoint inference
src/maqsad/evaluation.py       Aggregate and dialect metrics
notebooks/                    Executed baseline and GPU experiment
reports/                      Measured results and plots
scripts/                      Dataset and notebook entry points
tests/                        Unit and integration tests
docs/                         Dataset and model documentation
```

## Validation

```bash
python -m pip install -e ".[train,dev,app]"
python -m pytest -q
```

Neural tests use a tiny locally constructed BERT for offline software validation. They test gradient flow, a training epoch, checkpoint serialization and inference; they do **not** demonstrate MARBERT performance.

## Data and responsible interpretation

ArSarcasm-v2 contains **15,548 tweets**: 12,548 training and 3,000 test examples, with sarcasm, sentiment and dialect annotations. See [DATASET.md](docs/DATASET.md), [MODEL_CARD.md](docs/MODEL_CARD.md) and [the upstream repository](https://github.com/iabufarha/ArSarcasm-v2).

The data is historical social-media language, not a representative customer-service benchmark. Predictions can be wrong and model scores are uncalibrated. Do not interpret sarcasm labels as definitive judgments about a person. No explanation-generation or dialect-classification capability is claimed.

## Citation

Abu Farha, Ibrahim; Zaghouani, Wajdi; and Magdy, Walid. 2021. [Overview of the WANLP 2021 Shared Task on Sarcasm and Sentiment Detection in Arabic](https://aclanthology.org/2021.wanlp-1.36/).

Abdul-Mageed, Muhammad; Elmadany, AbdelRahim; and Nagoudi, El Moatez Billah. 2021. [ARBERT & MARBERT: Deep Bidirectional Transformers for Arabic](https://aclanthology.org/2021.acl-long.551/).

The upstream dataset repository provides an MIT license; its notice is retained in `docs/ARSARCASM_LICENSE.txt`. Raw tweets are not redistributed in this repository. The pretrained model has its own model card and terms.
