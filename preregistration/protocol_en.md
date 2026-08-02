# Held-out human evaluation — pre-registration protocol (English translation)

**This file is a translation, not the sealed artifact.** The sealed protocol is
`protocol.md`, written in Korean on 2026-07-18 and hash-recorded in `seal.txt` as
`HELDOUT_PROTOCOL.md`:

```
52842da45c33af648ca2d10e6738b0099daa8dd432b3da7d5826a00498b20a35  HELDOUT_PROTOCOL.md
```

`protocol.md` is byte-identical to the sealed original and is the authoritative
record. This translation was written on 2026-08-02, after the study, so that
readers who do not read Korean can check the pre-specification claims made in the
manuscript. It is outside the seal and adds nothing to it. Where the two differ,
`protocol.md` governs.

A terminology note mapping the protocol's internal names to the manuscript's
appears at the end.

---

Written 2026-07-18. Purpose: measure how the consequence-aware LLM judge correlates
with human faithfulness judgment on **held-out items that were not used to design
the prompt**, with **the judge's decisions and the analysis frozen before any human
label exists**. This defends against circularity (attack B) and sampling (attack A)
at the same time. **Report the outcome regardless of direction** (publish either way).

---

## 1. Hypothesis / endpoints

- **H1 (primary)**: the frozen consequence judge's decision (`judge_corr`) is
  **positively correlated** (Kendall tau-b) with the human correctness consensus on
  held-out natural messages, and exceeds the existing automatic metrics
  (BLEU / ROUGE / BM25 / BERTScore / DGF / code-aware).
- **Primary endpoint**: tau-b (`judge_corr` vs three-rater consensus correctness),
  160 held-out items.
- **Secondary endpoints (exploratory, low power, stated as such)**: tau by source
  (gold / generated), tau by language.

## 2. Frozen artifacts (fixed **before** human labels are collected)

| Item | Value | Frozen |
|---|---|---|
| judge model | Qwen2.5-Coder-32B-Instruct-AWQ (`/cephfs/lab/models/`) | yes |
| prompt | consequence-aware, 3 stages (`재실험/19_judge_consequence.py`, SYS+USER) | yes, no edits after this point |
| decoding | temperature 0, max_tokens 450, single sample | yes |
| diff cleaning | `30_build_heldout.py` `clean()` (format verified identical to the gate) | yes |
| held-out decisions | `heldout_judge.jsonl` — **computed before human labels** | yes (job 6083) |
| secondary judge | Qwen2.5-Coder-14B (to check that the result reproduces, robustness) | optional |

**Key point**: the prompt was designed on the original 40 items (the gate), with a
researcher in the loop. This held-out set consists of new diffs that were not used in
that design, and **the prompt is not revised after seeing the held-out set** (frozen).
So the headline tau is design-independent for the first time.

## 3. Held-out sample (`30_build_heldout.py`)

- **N=160 natural, real messages**. Drawn from the master pool of 1,598 diffs after
  **excluding the 80 `diff_id`s used in the original human evaluation** (this
  guarantees held-out status; zero overlap was verified), leaving a pool of 1,515.
- Balance: **40 per source** (gold, codellama, qwen, deepseek) times **20 per
  language** (8 languages). One diff yields one message, to avoid repeated anchoring.
  Source is hidden and the order shuffled (`heldout_packet.jsonl`).
- Power: at n=160 the tau confidence interval has a half-width of about 0.10 (scaling
  the current n=40 bootstrap SE of 0.098 by 1/sqrt(n)). That is enough to establish
  significance over code-aware (0.18).

## 4. Rater design

| Role | Count | Task | Load |
|---|---|---|---|
| primary | 2 | double-code all 160 | 160 each (about 5 h) |
| adjudicator | 1 | adjudicate the two raters' disagreements, plus a fixed 50-item overlap (3-way IRR) | about 80 to 90 |

- The authors **do not take part** in labeling or adjudication (neutrality). All raters
  are **blind** to source and to the judge's scores.
- The adjudicator resolves disagreements, giving a three-rater consensus label
  (median / majority).

## 5. Rater instructions (**independence is what prevents circularity**)

- Scale: **correctness 0/1/2**, with the **same neutral wording** as the earlier human
  evaluation: *"Does this message accurately describe the change in this diff?
  0 = wrong, 1 = partial, 2 = accurate."*
- **The consequence prompt's wording ("credit the effect / purpose / intent") must
  never be included.** Raters judge naturally. (Giving the raters the same instructions
  as the judge would make the agreement artificial.)
- Additional fields: direction 0/1/2/NA (secondary), notes (optional). Source, model,
  and judge scores are not disclosed.

## 6. Reliability

- Both raters, all items: weighted Cohen's kappa and Krippendorff's alpha.
- Three-rater 50-item overlap: Fleiss' kappa.
- Intra-rater: one re-rating of 15 items after a washout of at least one day.
- Disagreements go to the adjudicator, producing the consensus label.

## 7. Analysis (fixed **before** human labels)

1. Primary: tau-b (`judge_corr` vs consensus) with a **bootstrap 5,000-resample 95% CI**,
   on the 160 held-out items.
2. Baselines: tau for BLEU / ROUGE / BM25 / BERTScore / DGF / code-aware on the same
   160 items (script `07` / full metric recomputation).
3. Ceiling: two-rater and three-rater inter-rater tau (noting the asymmetry explicitly).
4. Exploratory: tau by source and by language (marked as low power).
5. Robustness: the same analysis for the 14B judge.

## 8. Pre-registered success criteria (reported regardless of outcome)

- **Primary success**: held-out tau with a **bootstrap CI lower bound > 0.30** **and**
  above the code-aware point estimate (about 0.18). This would establish that a strong
  code LLM with a consequence prompt correlates moderately or better with humans on
  held-out data and exceeds the existing metrics.
- **Partial**: point estimate above code-aware but CI lower bound at or below 0.30.
  Report as promising but limited in precision.
- **Negative**: point estimate at or below code-aware. Report honestly that the
  development-set result does not hold on held-out data, and write the paper as a
  critique with a negative result.
- In every case **the held-out tau is the final headline** (the development-set value
  of 0.694 is a reference and an upper bound).

## 9. Remaining execution order

1. Done: held-out sample and frozen judge decisions (job 6083).
2. Score all metrics (BLEU through code-aware) on the 160 held-out items. GPU work,
   independent of the humans.
3. Produce the rater packet in a human-friendly format (xlsx) plus the guide, and
   distribute.
4. Collect labels, compute reliability, form the consensus, run the primary analysis.
5. Promote the result to the headline in `00_FINDINGS_CONSOLIDATED.md`
   (development set to held-out).

## Appendix — files

Sample: `heldout_items.jsonl` (judge and metric input), `heldout_key.csv` (source key),
`heldout_packet.jsonl` (for raters, source hidden). Frozen decisions:
`heldout_judge.jsonl`.

---

## Terminology note (added with this translation, not part of the protocol)

The protocol uses working names that the manuscript renamed for clarity. The
quantities are the same.

| Protocol | Manuscript |
|---|---|
| DGF | raw NLI score |
| code-aware | code-adapted NLI baseline |
| consequence judge / consequence-aware prompt | the LLM judge with the task-specific prompt |
| gate, original 40 items | the in-sample prompt-development set |
| correctness | diff-grounded faithfulness label (0/1/2) |

File names also changed when the artifacts were packaged for release:
`heldout_judge.jsonl` is released as `data/predictions/judge_32b.jsonl`,
`heldout_judge_14b.jsonl` as `judge_14b.jsonl`, `heldout_metrics.jsonl` as
`reference_metrics.jsonl`, and `heldout_codeaware/codeaware_scores.jsonl` as
`code_nli.jsonl`. `seal.txt` records the original names.
