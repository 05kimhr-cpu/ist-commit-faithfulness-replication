# MODEL PINS — reproducibility

Written 2026-07-20 for the frozen judge and the baseline models. Extended 2026-08-02
with the four models used only for the cross-family and in-sample robustness
comparisons, which the first version omitted.

For each model: the local path it was loaded from, the quantization, and, where the
snapshot still exists, the SHA-256 of its `config.json` as a provenance check.

## Frozen judge and baselines

| Role | Model | model_type | Quantization | config.json SHA-256 (first 12) |
|---|---|---|---|---|
| judge 32B (headline) | Qwen2.5-Coder-32B-Instruct-AWQ | qwen2 | AWQ 4-bit | `bd070b51e60a` |
| judge 14B (robustness) | Qwen2.5-Coder-14B-Instruct-AWQ | qwen2 | AWQ 4-bit | `af62852325fc` |
| code-adapted NLI | Llama-3.1-8B-Instruct | llama | bitsandbytes 4-bit | `29e4c210b0d6` |
| raw NLI | facebook/bart-large-mnli | bart | none | `a0f9bcb245b6` |
| BERTScore backbone | roberta-large | roberta | none | `82ba49810e64` |

All five were loaded from `/cephfs/lab/models/`. The generic-prompt ablation
(`judge_generic_prompt.jsonl`) reuses the 32B judge checkpoint above; only the prompt
differs.

## Cross-family and in-sample sweep models

These four produce the Codestral row of Table 4 and the four non-Qwen bars of Figure 3.
Their local snapshots have since been deleted from disk, so no `config.json` hash is
available. What follows is taken from the job logs.

| Model (as recorded in the outputs) | Used for | Quantization | Prompt script | Job | Date | Items scored |
|---|---|---|---|---|---|---|
| Codestral-22B-AWQ | held-out cross-family judge (Table 4) | AWQ 4-bit | `19_judge_consequence.py` (task-specific) | 6194, 6195 | 2026-07-20 | 160 held-out |
| QwQ-32B-AWQ | in-sample sweep (Figure 3) | AWQ 4-bit | `19_judge_consequence.py` (task-specific) | 5924 | 2026-07-17 | 120 development |
| CodeLlama-34B-Instruct-AWQ | in-sample sweep (Figure 3) | AWQ 4-bit | `19_judge_consequence.py` (task-specific) | 5927 | 2026-07-17 | 120 development |
| DeepSeek-Coder-V2-Lite-Instruct-AWQ | in-sample sweep (Figure 3) | AWQ 4-bit | `11_judge_infer_v2.py` (generic) | 5921 | 2026-07-17 | 120 development |

Codestral-22B-AWQ and QwQ-32B-AWQ were loaded from
`/cephfs/lab/users/2025810023/models/`, the other two from `/cephfs/lab/models/`.

The development set has 120 labeled items, of which 40 are natural (diff, message)
pairs and 80 are polarity-controlled constructions. The sweep scored all 120; Figure 3
reports Kendall tau on the 40 natural items only, matching the in-sample column of
Table 1. DeepSeek-Coder-V2-Lite is the model marked with a dagger in the Figure 3
caption as scored with the generic prompt; the log above is the record of that.

### Upstream provenance for these four

The snapshots are gone, so the upstream repository cannot be confirmed from the files
themselves. The Hugging Face download cache on the machine that ran the jobs still
holds an entry for three of them, and those entries record:

```
Qwen/QwQ-32B-AWQ                                dc9f21221581580ccfa51b74077db6056b56cb69
TechxGenus/Codestral-22B-v0.1-AWQ               a45d426d2752b0325e5b74d31c15c929be94c5ae
TechxGenus/DeepSeek-Coder-V2-Lite-Instruct-AWQ  7cf1b67809a5bcc7b9ba9f854ccdf2858298484d
```

These are cache records, not a verified match against the deleted snapshots, so they
should be read as the most likely provenance rather than as a pin. For
CodeLlama-34B-Instruct-AWQ no cache entry remains and the upstream repository is not
recorded; only the local path and the name above are.

This limitation is one reason the manuscript reports the sweep as exploratory and
treats cross-family generalization as not yet established.

## Environment

- vLLM 0.15.0 (the analysis environment), transformers, bitsandbytes 0.49.2 (used for
  the code-adapted NLI).
- Judge decoding: temperature 0, max_tokens 450, single sample (frozen).
- Execution: a single RTX 4090, `enforce_eager` for the 32B AWQ judge.
- Seeds: the sampling and bootstrap seeds are fixed inside the scripts.

## Reproducibility limits

- The AWQ checkpoints are community re-quantizations. The exact upstream git revision
  of the original Qwen2.5-Coder release was not recorded with the local copies. To
  reproduce, use the same AWQ snapshot and check it against the `config.json` hashes
  above.
- The four models in the second table cannot be pinned at all, for the reason given in
  that section.
