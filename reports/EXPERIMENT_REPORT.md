# Updated Colab experiment results

User-supplied `maqsad-results.zip` contains aggregate metrics and run metadata for both three-epoch MARBERT runs on Tesla T4, seed 42. The reported split audit matches the local baseline. These reports have been inspected but predictions have not been independently regenerated: weights and the executed GPU notebook were not supplied.

| Model | Sentiment macro-F1 | Sarcastic-class F1 |
|---|---:|---:|
| TF-IDF baseline | 56.75% | 54.04% |
| MARBERT sentiment-only | 65.71% | Not trained |
| MARBERT joint | 65.57% | 58.56% |

Joint learning is lower than sentiment-only by approximately **0.15 percentage points** in this single-seed sentiment comparison. It does not support a claim that joint training improves sentiment. Both neural runs improve over the baseline in sentiment, while joint sarcasm F1 is about 4.52 points above baseline. Statistical significance cannot be inferred from aggregate single-run scores. The selected validation checkpoint is epoch 1 for sentiment-only and epoch 2 for joint. Further matched seeds and paired prediction analysis are needed for stronger conclusions.

Source reports: `reports/colab/`. Training run: `20261102T192207Z`; model revision: `88e1fa192dd723cf0b3563500aec46209762eb22`. No weights are committed to Git.

---

## Historical local execution report

The following section records the original CPU execution before the Colab results were supplied. Its pending-training statements describe that earlier environment.

# Experiment report

## Execution status

The baseline notebook executed successfully end-to-end on CPU. All reported baseline results below come from the executed run, not estimates. MARBERT downloading was blocked by an HTTP 403 from the managed network proxy. Full MARBERT training and the joint-versus-sentiment comparison remain unexecuted. No GPU was available.

## Dataset audit

```json
{
  "seed": 42,
  "original_train": 12548,
  "test": 3000,
  "train_test_overlap_removed": 4,
  "train_duplicates_removed": 67,
  "test_duplicate_texts": 6,
  "train": 10605,
  "validation": 1872
}
```

The official test set is preserved. Filtering changes the effective training split, so comparisons to published benchmark numbers require accounting for this protocol.

## Held-out test results

| Model | Sarcastic-class F1 | Sentiment macro-F1 | Sentiment accuracy |
|---|---:|---:|---:|
| Training-majority | 0.0000 | 0.1330 | 0.2493 |
| Character TF-IDF + logistic regression | 0.5404 | 0.5675 | 0.6360 |
| MARBERT sentiment-only | Not executed | Not executed | Not executed |
| MARBERT joint learning | Not executed | Not executed | Not executed |

The baseline improves on the majority reference, but its errors remain substantial. These results do not establish that joint learning helps; that hypothesis requires the two matched MARBERT runs.

## Validation and model selection

C was selected from 0.5, 2 and 8 on validation only. Selection scores and per-class/dialect metrics are in `baseline_metrics.json`. Only training data fit the TF-IDF vocabulary. The final held-out evaluation uses the validation-selected models without refitting.

## Verified software behavior

- The baseline notebook completed every code cell and saved outputs.
- Unit tests verified normalization, schema validation and missing-class metrics.
- A tiny local BERT test verified both task gradients, sentiment-only loss isolation and checkpoint round-trip equality.
- An offline integration test executed one complete training epoch, saved a checkpoint and performed inference. It uses synthetic data and a tiny random BERT; it is not MARBERT evaluation.
- The Streamlit app was tested with an Arabic input and rendered both classification outputs.

## Reproduction and remaining work

See the README commands and `environment.json` for tested package versions. `constraints-tested.txt` records the main tested versions; it is not a complete transitive lockfile. Raw tweets and model artifacts are intentionally regenerated outside Git.

Enable Hugging Face access in the environment settings, or use a GPU notebook runtime with model access. Run `02_marbert_finetuning.ipynb` for both neural conditions, then repeat with three matched seeds. Keep test data out of parameter selection. Report the mean and standard deviation and review subgroup sample sizes before drawing conclusions.
