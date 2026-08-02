# Deviations from the protocol

For transparency, this records where the executed study departed from the sealed
protocol (`protocol.md`, `seal.txt`).

## Nature of the seal
The seal is an internally time-stamped SHA-256 record of the frozen artifacts,
committed before any held-out human label was collected. It is a self-hosted
record, not a public pre-registration: the ordering is auditable from the hashes
but not independently certified.

## Confirmatory vs. post hoc
- Confirmatory: the pre-specified primary endpoint alone, that is the held-out
  Kendall tau_b of the frozen 32B judge (`judge_32b.jsonl`) against the human
  consensus, together with the success criterion defined in `protocol.md` section 8.
- Also pre-specified and frozen before labeling, but secondary: the automatic
  baselines the primary endpoint is compared against (`reference_metrics.jsonl`,
  `code_nli.jsonl`; protocol section 7 item 2) and the same-family 14B robustness
  analysis (`judge_14b.jsonl`; protocol section 2 and section 7 item 5). The protocol
  lists the 14B analysis as a robustness check, not as a second headline.
- Post hoc (computed after the labels were collected, reported as exploratory): the
  generic-prompt ablation (`judge_generic_prompt.jsonl`) and the cross-family
  Codestral check (`judge_codestral.jsonl`).

## Reliability analyses not carried out
The protocol planned a 50-item three-way overlap (Fleiss kappa) and a 15-item
intra-rater re-rating. These were not carried out. Reported reliability is
two-rater agreement (quadratic-weighted Cohen's kappa, Krippendorff's alpha) and
third-rater adjudication of the 28 primary-rater disagreements.

## Frozen-artifact hash divergence
The judge script was edited after sealing, in its non-chat-template fallback branch
only, to run the post-hoc Codestral check; its hash therefore no longer matches the
seal. That branch is not used by the primary (Qwen) judge, and the primary judge
predictions (`judge_32b.jsonl`) still match their sealed hash. The original frozen
script was not under version control and is not byte-recoverable; the sealed hash is
kept as the historical record.

A second sealed artifact, the baseline score file `reference_metrics.jsonl` (sealed as
`heldout_metrics.jsonl`), also no longer matches its sealed hash. After labeling we found that BM25
had been fitted per item rather than over the corpus, which degenerates its IDF term, and we
recomputed that one score at the corpus level. The correction changes the `bm25` field only: all
other fields in the file (BLEU-4, ROUGE-L, CHRF++, METEOR, BERTScore, and the raw NLI score) are
byte-identical to the sealed version, and the three prediction files behind the pre-specified
analyses (`judge_32b.jsonl`, `judge_14b.jsonl`, `code_nli.jsonl`) still verify against their sealed
hashes. BM25's held-out Kendall tau_b moves from -0.015 to +0.027, that is, from near zero to near
zero; it was not the strongest baseline before or after the correction, and the pre-specified
success criterion, which is defined against the code-adapted NLI baseline, is unaffected. We report
the corrected value because it is the correct one, and record the divergence here because the
sealed hash no longer verifies.
