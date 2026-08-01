# Deviations from the protocol

For transparency, this records where the executed study departed from the sealed
protocol (`protocol.md`, `seal.txt`).

## Nature of the seal
The seal is an internally time-stamped SHA-256 record of the frozen artifacts,
committed before any held-out human label was collected. It is a self-hosted
record, not a public pre-registration: the ordering is auditable from the hashes
but not independently certified.

## Confirmatory vs. post hoc
- Confirmatory (frozen before labeling; predictions verify against the seal): the
  primary judge (32B) and the same-family 14B judge.
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
