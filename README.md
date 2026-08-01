# Replication Package

This package accompanies the paper *Can a Language Model Judge Commit-Message Faithfulness? A
Measurement-Validity Study of Automatic Evaluation*. It contains the human labels, the frozen model
predictions, the analysis code, the hash-seal record, and the figures, so that the reported
results can be verified and regenerated.

## Quick verification (no GPU)

The headline numbers are reproduced from the released predictions and human labels:

```
pip install -r requirements.txt
python scripts/analyze.py
```

This recomputes the held-out correlations, confidence intervals, inter-rater reliability, the
judge-vs-baseline comparison, and the prompt ablation reported in the paper.

The overlap metrics in `scripts/analyze.py` are scored against the diff (the object of faithfulness).
The paper also reports the metrics in their standard form, scored against the developer's reference
message, on the generated candidates (RQ1). That check is reproduced by:

```
python scripts/score_reference_metrics_vs_gold.py
```

which uses `data/heldout_gold_references.jsonl` as the reference and confirms these metrics are also
weak (ROUGE-L Kendall tau = 0.15; all far below the judge).

## Contents

```
data/
  heldout_items.jsonl            held-out evaluation items (diff, message, source, language)
  heldout_key.csv                item -> message source (for analysis, hidden from raters)
  heldout_gold_references.jsonl  developer reference message per item (for the gold-reference check)
  dev_human_labels.jsonl         development-set human labels (used to develop the judge)
  heldout_final_result.json      the reported headline result
  human_labels/
    rater_A.xlsx, rater_B.xlsx   two primary raters, all 160 held-out items
    adjudicator.csv              third rater, disagreement items only
  predictions/
    judge_32b.jsonl              LLM judge (32B), frozen before human labeling
    judge_14b.jsonl              LLM judge (14B), robustness
    judge_generic_prompt.jsonl   judge with a generic prompt (ablation)
    reference_metrics.jsonl      BLEU/ROUGE/CHRF++/METEOR/BM25/BERTScore/raw NLI, held-out
    dev_reference_metrics.jsonl  the same metrics on the development set
    code_nli.jsonl               code-adapted NLI score, held-out
preregistration/
  protocol.md                    the protocol fixed before labeling
  seal.txt                       hash record of the frozen artifacts, internally timestamped
  DEVIATIONS.md                  where the executed study departed from the protocol
MODEL_PINS.md                    models used, with configuration hashes
scripts/
  analyze.py                     verification from released data (no GPU)
  score_reference_metrics_vs_gold.py  standard gold-reference metric check (no GPU)
  build_heldout_sample.py        construct the held-out sample from the corpus
  run_llm_judge.py               run the LLM judge (requires the model and a GPU)
  score_reference_metrics.py     compute the reference/NLI metrics (requires a GPU)
  make_figures.py                regenerate the figures
figures/
  *.pdf                          the figures used in the paper
```

## Regenerating predictions (GPU)

The scripts in `scripts/` other than `analyze.py` regenerate the model predictions from the
inputs. They require the judge and scoring models (see `MODEL_PINS.md`) and a GPU, and paths should
be adjusted to the local environment. The judge model, prompt, and decoding are frozen; the
frozen predictions in `data/predictions/` were produced before any held-out human label was
collected (see `preregistration/`, including `DEVIATIONS.md`).

## Human labels

Faithfulness is rated on a three-point scale (2 = accurate, 1 = partial/vague, 0 = wrong or
unsupported): whether the candidate message accurately describes the main change shown in the diff.
Raters were blind to the message source and to the model scores. Rater identities are anonymized.

## Notes

- Confidence intervals are non-parametric bootstrap (5,000 resamples).
- The quantized model checkpoints are community re-quantizations; exact reproduction should use the
  same checkpoints (configuration hashes in `MODEL_PINS.md`).
