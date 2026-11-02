# Maqsad | Arabic Sarcasm and Sentiment Analysis

Maqsad is an NLP capstone project that explores how sarcasm affects sentiment analysis in Arabic. It compares a character-based TF-IDF baseline with two fine-tuned MARBERT models: a sentiment-only model and a joint model that predicts both sarcasm and sentiment.

The project includes reproducible experiments, dialect-level evaluation, and an interactive Streamlit application for analyzing Arabic text.

[Live demo](https://nlpcapstone-ya.streamlit.app/) · [Trained model](https://huggingface.co/yahya-ai10/maqsad-marbert) · [Training notebook](notebooks/02_marbert_finetuning.ipynb) · [Experiment report](reports/EXPERIMENT_REPORT.md)

## Research question

**Does learning sarcasm and sentiment together improve Arabic sentiment classification?**

Consider the phrase:

> يا سلام على الخدمة، ساعتين وأنا أنتظر!

Positive wording can express a negative opinion. This project investigates whether additional sarcasm supervision helps a model recognize that distinction.

## Dataset

The experiments use [ArSarcasm-v2](https://github.com/iabufarha/ArSarcasm-v2), which contains **15,548 Arabic tweets** annotated for sarcasm, sentiment, and dialect.

| Split | Examples |
|---|---:|
| Training, after filtering and validation split | 10,605 |
| Validation | 1,872 |
| Official test set | 3,000 |

Four training examples overlap with normalized test text, and 67 additional training duplicates are removed before splitting. The official test set remains unchanged. Dialect labels are used for evaluation, not as model inputs.

See the [dataset card](docs/DATASET.md) for label definitions, preprocessing, and data integrity checks.

## Methodology

Three models are compared:

1. **TF-IDF baseline:** character 2–5-grams with separate class-balanced logistic regression classifiers.
2. **MARBERT sentiment-only:** a shared pretrained encoder with a sentiment classification head.
3. **MARBERT joint:** the same encoder with separate sentiment and sarcasm heads, trained using the sum of their cross-entropy losses.

Both neural experiments use three epochs, batch size 8, sequence length 128, learning rate 2e-5, and seed 42. The best checkpoint is selected using validation sentiment macro-F1. Training was performed on a Tesla T4 in Google Colab.

## Results

| Model | Sentiment macro-F1 | Sarcasm F1 |
|---|---:|---:|
| TF-IDF baseline | 56.75% | 54.04% |
| MARBERT sentiment-only | **65.71%** | — |
| MARBERT joint | 65.57% | **58.56%** |

Sarcasm F1 refers to the sarcastic class. The sentiment-only model is not trained for sarcasm detection.

Both MARBERT models outperform the baseline on sentiment. However, joint learning is **0.15 percentage points below** sentiment-only training in this experiment, so the results do not demonstrate an improvement from joint supervision. Further runs with multiple seeds are needed to assess whether the difference is consistent.

Full metrics and dialect-level results are available in [reports/](reports/) and the [experiment report](reports/EXPERIMENT_REPORT.md).

## Application

Try the application: **[Maqsad on Streamlit](https://nlpcapstone-ya.streamlit.app/)**.

The Streamlit interface provides:

- Single-text analysis with Arabic input and example phrases.
- Batch analysis for up to 16 texts.
- Sarcasm and sentiment predictions with model scores.
- CSV export and model comparison charts.

### Run locally

Python 3.12 is recommended. From the repository root:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows Command Prompt
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies and start the application:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

MARBERT downloads from [Hugging Face](https://huggingface.co/yahya-ai10/maqsad-marbert) on the first analysis. The checkpoint and loaded model are cached for subsequent requests. The initial download requires internet access and can take several minutes. CPU inference is supported, but the full model requires sufficient memory.

To use the lightweight baseline, select **TF-IDF baseline** and click **Prepare baseline model**, or run:

```bash
python -m maqsad.baseline
```

### Deployment

For Streamlit Community Cloud, select repository `iYm10/NLP_Capstone`, branch `main`, entry point `app.py`, and Python 3.12. Dependencies are installed from `requirements.txt`. Available hosting memory must be sufficient for the model; deployment resource requirements have not been benchmarked.

Optional settings:

| Variable | Purpose | Default |
|---|---|---|
| `MAQSAD_MODEL_REPO` | Hosted checkpoint repository | `yahya-ai10/maqsad-marbert` |
| `MAQSAD_MODEL_REVISION` | Model revision; use a commit for reproducibility | `main` |
| `MAQSAD_CHECKPOINT` | Complete local checkpoint directory | `artifacts/marbert-joint` |

A complete local checkpoint takes precedence over downloading. Large model weights are stored separately from the source code.

## Reproduce the experiments

Install the development dependencies:

```bash
python -m pip install -e ".[train,dev,app]"
```

**Baseline notebook:**

```bash
python -m ipykernel install --sys-prefix --name python3
python scripts/execute_notebook.py
```

**MARBERT experiments:** open the [training notebook in Colab](https://colab.research.google.com/github/iYm10/NLP_Capstone/blob/main/notebooks/02_marbert_finetuning.ipynb) with a GPU runtime, or run:

```bash
python -m maqsad.train --task sentiment --revision 88e1fa192dd723cf0b3563500aec46209762eb22 --output artifacts/marbert-sentiment
python -m maqsad.train --task joint --revision 88e1fa192dd723cf0b3563500aec46209762eb22 --output artifacts/marbert-joint
```

Use matching seeds and settings for both conditions and separate output directories for each run. Data files are downloaded from a pinned revision and checked against SHA-256 hashes. The baseline package versions are recorded in `constraints-tested.txt`; this is not a complete dependency lockfile. Neural run metadata is in `reports/colab/run.json`.

## Project structure

```text
app.py                  Interactive Streamlit application
src/maqsad/             Data preparation, models, training and inference
notebooks/              Baseline and MARBERT experiments
reports/                Evaluation metrics, figures and experiment report
docs/                   Dataset and model cards
scripts/                Data download and notebook execution
tests/                  Pipeline, checkpoint and interface tests
requirements.txt        Application dependencies
```

Run tests with:

```bash
python -m pytest -q
```

Tests cover preprocessing, evaluation, gradient flow, checkpoint loading, offline training, and interface interactions. Small synthetic models validate the code independently of the full MARBERT checkpoint.

## Limitations

- Results are based on one training seed.
- Historical tweets may not generalize to current customer reviews or other domains.
- Sarcasm can require context beyond a single text.
- Dialect representation is uneven; the Maghrebi test group has only two examples.
- Model scores are uncalibrated and should not be interpreted as certainty.

## References

- Abu Farha, I., Zaghouani, W., & Magdy, W. (2021). [Overview of the WANLP 2021 Shared Task on Sarcasm and Sentiment Detection in Arabic](https://aclanthology.org/2021.wanlp-1.36/).
- Abdul-Mageed, M., Elmadany, A., & Nagoudi, E. M. B. (2021). [ARBERT & MARBERT: Deep Bidirectional Transformers for Arabic](https://aclanthology.org/2021.acl-long.551/).

The dataset repository's license notice is retained in [docs/ARSARCASM_LICENSE.txt](docs/ARSARCASM_LICENSE.txt). Dataset and pretrained model usage remain subject to their respective terms.

## Gradio on Hugging Face Spaces

The Gradio interface uses the same trained model and shared source package as Streamlit. It includes interactive examples, a live text counter, result cards, expandable score distributions, batch summaries, CSV downloads, and experiment results. CPU inference is supported; on ZeroGPU, GPU work runs inside a `spaces.GPU` function with one inference request at a time. The checkpoint loads before the timed GPU request.

To run locally:

```bash
python -m pip install -r deployment/gradio/requirements.txt
python gradio_app.py
```

To deploy to `yahya-ai10/maqsad-demo`, authenticate to Hugging Face with write access and run `python scripts/deploy_space.py`. The script uploads the interface, shared Python package, and Space configuration; it does not upload model weights. Space build and ZeroGPU execution must be checked on the hosting service after deployment.
