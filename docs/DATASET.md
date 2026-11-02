# Dataset card

**Source:** https://github.com/iabufarha/ArSarcasm-v2

**Pinned revision:** `eb85513e133cfc67e30aca4968fadbe2594fcb42`

The upstream repository reports 15,548 annotated tweets: 12,548 training and 3,000 test rows. Fields are `tweet`, `sarcasm`, `sentiment`, and `dialect`.

| Field | Values |
|---|---|
| Sarcasm | TRUE, FALSE |
| Sentiment | NEG, NEU, POS |
| Dialect | msa, egypt, gulf, levant, magreb |

Numeric sentiment labels are NEG=0, NEU=1, POS=2. Sarcasm labels are FALSE=0 and TRUE=1. Dialect is used for subgroup evaluation, not as an input feature or training target.

## Original distributions

| Split | Not sarcastic | Sarcastic | Negative | Neutral | Positive |
|---|---:|---:|---:|---:|---:|
| Train | 10,380 | 2,168 | 4,621 | 5,747 | 2,180 |
| Test | 2,179 | 821 | 1,677 | 748 | 575 |

The labels are imbalanced and the train/test label proportions differ. The `gulf` group includes several countries and cannot establish Saudi-specific performance.

## Preparation

Normalization replaces URLs and mentions, strips Arabic diacritics and tatweel, and collapses whitespace. Negation, emoji, punctuation and Arabic letter distinctions are preserved. Matching normalized text in train and test is excluded from training. Remaining training duplicates retain their first occurrence; this rule does not adjudicate conflicting labels. Validation is sampled only after filtering, using joint-label stratification where possible.

The full audit, including removed rows and test duplicates, is stored in `reports/baseline_metrics.json` and the executed notebook. Test labels are used only for final evaluation, never model selection.

## Integrity

- training_data.csv: `1da727eb763459f436c4e61f52c6a0bc63d5300af8fcc99b5675ae4616dae04a`
- testing_data.csv: `0f0a77ba8a4a0a9370846a42a93a63244092f1fb30a909595b70e17e299bcf6c`

The downloader fails if existing or downloaded content does not match. It does not overwrite a mismatching file automatically.

## Limitations

Annotations reflect human judgments and sarcasm can require context unavailable in a single tweet. Historical social-media text may include offensive content, named people and outdated language. Raw tweets remain outside version control. The upstream repository has an MIT license (notice retained), but downstream users should also consider applicable source-platform and data-use terms.
