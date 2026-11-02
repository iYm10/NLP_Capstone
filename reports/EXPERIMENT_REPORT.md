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
