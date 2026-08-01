#!/usr/bin/env python3
"""
Recompute the standard reference-overlap metrics in their conventional form (candidate message
vs the developer's reference message) on the generated held-out candidates, and correlate them
with the human faithfulness consensus. This complements scripts/analyze.py, which scores overlap
against the diff; here overlap is scored against the gold developer message.

A distinct reference exists only for the generated candidates (for gold candidates the reference
is the message itself, a degenerate self-comparison), so the correlation is computed on the 120
generated held-out items. Reproduces the "standard gold-reference metric" paragraph in RQ1.

No GPU required. Requires: sacrebleu, rouge-score, nltk (wordnet), scipy, openpyxl.
NLTK wordnet: python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

Run from the package root:  python scripts/score_reference_metrics_vs_gold.py
"""
import json, os, csv, random, collections
random.seed(20260720)
from scipy.stats import kendalltau
import openpyxl
import sacrebleu
from rouge_score import rouge_scorer as Rg
from nltk.translate.meteor_score import meteor_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")

# ---- human labels: two primary raters + adjudicator (same consensus as analyze.py) ----
def load_xlsx(p):
    ws = openpyxl.load_workbook(p)["ratings"]
    return {r[0]: int(r[4]) for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None}
def load_csv(p):
    out = {}
    for row in csv.DictReader(open(p, encoding="utf-8-sig")):
        out[row["item_id"]] = int(row["correctness(0/1/2)"])
    return out
A = load_xlsx(f"{D}/human_labels/rater_A.xlsx")
B = load_xlsx(f"{D}/human_labels/rater_B.xlsx")
J = load_csv(f"{D}/human_labels/adjudicator.csv")
def consensus(i):
    if A[i] == B[i]:
        return A[i]
    counts = collections.Counter([A[i], B[i], J[i]])
    top, n = counts.most_common(1)[0]
    return top if n >= 2 else J[i]

# ---- candidates (held-out messages) + gold references ----
items = {json.loads(l)["item_id"]: json.loads(l) for l in open(f"{D}/heldout_items.jsonl")}
gold = {json.loads(l)["item_id"]: json.loads(l)["gold_reference"]
        for l in open(f"{D}/heldout_gold_references.jsonl")}

# generated candidates only (a distinct reference exists)
gen_ids = [i for i, it in items.items() if it.get("source") != "gold"
           and (gold.get(i) or "").strip()]

_r = Rg.RougeScorer(["rougeL"], use_stemmer=False)
def tok(t): return t.lower().split()

rows = {"BLEU-4": [], "ROUGE-L": [], "CHRF++": [], "METEOR": []}
labels = []
for i in gen_ids:
    cand = items[i]["message"].strip(); ref = gold[i].strip()
    rows["BLEU-4"].append(sacrebleu.sentence_bleu(cand, [ref]).score / 100)
    rows["ROUGE-L"].append(_r.score(ref, cand)["rougeL"].fmeasure)
    rows["CHRF++"].append(sacrebleu.sentence_chrf(cand, [ref], word_order=2).score / 100)
    try: mt = meteor_score([tok(ref)], tok(cand))
    except Exception: mt = None
    rows["METEOR"].append(mt)
    labels.append(consensus(i))

def boot(xs, ys, R=5000):
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None]
    out = []
    for _ in range(R):
        s = [pts[random.randrange(len(pts))] for _ in pts]
        t = kendalltau([a for a, _ in s], [b for _, b in s]).correlation
        out.append(t if t == t else 0.0)
    out.sort(); return out[int(.025 * R)], out[int(.975 * R)]

print(f"Candidate-vs-gold reference metrics on {len(gen_ids)} generated held-out items\n")
print(f"{'Metric (candidate vs gold)':26} {'tau':>8}   {'95% CI':>18}   sig")
print("-" * 62)
for name in ["BLEU-4", "ROUGE-L", "CHRF++", "METEOR"]:
    xs = rows[name]
    pairs = [(x, y) for x, y in zip(xs, labels) if x is not None]
    t = kendalltau([a for a, _ in pairs], [b for _, b in pairs]).correlation
    lo, hi = boot(xs, labels)
    sig = "sig" if not (lo <= 0 <= hi) else "ns"
    print(f"{name:26} {t:>+8.3f}   [{lo:+.3f}, {hi:+.3f}]   {sig}")
print("\nAll values remain far below the LLM judge (tau = 0.599; see scripts/analyze.py).")
