#!/usr/bin/env python3
"""
Reproduce the paper's headline numbers from the released human labels and model predictions.
No GPU required: this reads the frozen predictions in data/predictions/ and the human labels in
data/human_labels/ and recomputes the correlations, confidence intervals, and reliability
statistics reported in the paper.

Run from the package root:  python scripts/analyze.py
"""
import json, os, csv, random, collections
random.seed(20260720)
from scipy.stats import kendalltau
import openpyxl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")

# ---- human labels: two primary raters (all items) + adjudicator (disagreements) ----
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
ids = sorted(A)

def consensus(i):
    if A[i] == B[i]:
        return A[i]
    counts = collections.Counter([A[i], B[i], J[i]])
    top, n = counts.most_common(1)[0]
    return top if n >= 2 else J[i]      # majority, else adjudicator
cons = {i: consensus(i) for i in ids}
disagree = [i for i in ids if A[i] != B[i]]

# ---- predictions ----
def jl(p): return {json.loads(l)["item_id"]: json.loads(l) for l in open(p)}
judge32 = {k: v["judge_corr"] for k, v in jl(f"{D}/predictions/judge_32b.jsonl").items()}
judge14 = {k: v["judge_corr"] for k, v in jl(f"{D}/predictions/judge_14b.jsonl").items()}
generic = {k: v["judge_corr"] for k, v in jl(f"{D}/predictions/judge_generic_prompt.jsonl").items()}
met = jl(f"{D}/predictions/reference_metrics.jsonl")
code_nli = {json.loads(l)["item_id"]: json.loads(l)["dgf_codeaware"]
            for l in open(f"{D}/predictions/code_nli.jsonl")}

def tau(getter):
    xs, ys = [], []
    for i in ids:
        v = getter(i)
        if v is None: continue
        xs.append(v); ys.append(cons[i])
    return kendalltau(xs, ys).correlation
def boot(getter, R=5000):
    pts = [(getter(i), cons[i]) for i in ids if getter(i) is not None]
    out = []
    for _ in range(R):
        s = [pts[random.randrange(len(pts))] for _ in pts]
        t = kendalltau([a for a, _ in s], [b for _, b in s]).correlation
        out.append(t if t == t else 0.0)
    out.sort(); return out[int(.025 * R)], out[int(.975 * R)]

n = len(ids)
kappa_exact = sum(1 for i in ids if A[i] == B[i]) / n
human_tau_ref = kendalltau([A[i] for i in ids], [B[i] for i in ids]).correlation

def _wkappa(X, Y, K=3):
    O = [[0]*K for _ in range(K)]
    for i in ids: O[X[i]][Y[i]] += 1
    r = [sum(O[k]) for k in range(K)]; c = [sum(O[x][k] for x in range(K)) for k in range(K)]
    W = [[((x-y)**2)/((K-1)**2) for y in range(K)] for x in range(K)]
    num = sum(W[x][y]*O[x][y] for x in range(K) for y in range(K))
    den = sum(W[x][y]*r[x]*c[y]/n for x in range(K) for y in range(K))
    return 1 - num/den if den else float('nan')

def _kripp_alpha(X, Y):
    Do = sum((X[i]-Y[i])**2 for i in ids) * 2
    allv = [X[i] for i in ids] + [Y[i] for i in ids]; N = len(allv)
    De = sum((a-b)**2 for a in allv for b in allv) / (N*(N-1))
    return 1 - (Do/(n*2))/De if De else float('nan')

print(f"Held-out items: {n}  |  primary-rater disagreements adjudicated: {len(disagree)}")
print(f"Human-human agreement: exact {kappa_exact:.3f}, weighted kappa {_wkappa(A, B):.3f}, "
      f"Krippendorff alpha {_kripp_alpha(A, B):.3f}, Kendall tau-b reference {human_tau_ref:.3f}\n")

print(f"{'Score':26} {'held-out tau-b':>14}")
print("-" * 42)
for name, g in [("BLEU-4", lambda i: met[i]["bleu4"]),
                ("ROUGE-L", lambda i: met[i]["rouge_l"]),
                ("CHRF++", lambda i: met[i]["chrf"]),
                ("METEOR", lambda i: met[i]["meteor"]),
                ("BM25", lambda i: met[i]["bm25"]),
                ("BERTScore", lambda i: met[i]["bertscore"]),
                ("raw BART-MNLI", lambda i: met[i]["dgf"]),
                ("code-adapted Llama NLI", lambda i: code_nli.get(i)),
                ("LLM judge (14B)", lambda i: judge14.get(i)),
                ("LLM judge (32B)", lambda i: judge32.get(i))]:
    print(f"{name:26} {tau(g):>+14.3f}")

lo, hi = boot(lambda i: judge32.get(i))
print(f"\nHeadline: LLM judge (32B) tau-b = {tau(lambda i: judge32.get(i)):+.3f}, "
      f"95% CI [{lo:+.3f}, {hi:+.3f}]")
print(f"          = {tau(lambda i: judge32.get(i)) / human_tau_ref:.2f} of the human-human agreement reference")

# paired: judge vs strongest baseline (code-adapted Llama NLI)
pts = [(judge32[i], code_nli.get(i), cons[i]) for i in ids if code_nli.get(i) is not None]
d = []
for _ in range(5000):
    s = [pts[random.randrange(len(pts))] for _ in pts]
    t1 = kendalltau([a for a, _, _ in s], [c for _, _, c in s]).correlation
    t2 = kendalltau([b for _, b, _ in s], [c for _, _, c in s]).correlation
    d.append(t1 - t2)
d.sort()
print(f"Judge minus code-adapted Llama NLI: mean {sum(d)/len(d):+.3f}, 95% CI [{d[125]:+.3f}, {d[4875]:+.3f}]")

# prompt ablation: task-specific vs generic
pts = [(judge32[i], generic[i], cons[i]) for i in ids]
d = []
for _ in range(5000):
    s = [pts[random.randrange(len(pts))] for _ in pts]
    t1 = kendalltau([a for a, _, _ in s], [c for _, _, c in s]).correlation
    t2 = kendalltau([b for _, b, _ in s], [c for _, _, c in s]).correlation
    d.append(t1 - t2)
d.sort()
print(f"Task-specific minus generic prompt: mean {sum(d)/len(d):+.3f}, "
      f"95% CI [{d[125]:+.3f}, {d[4875]:+.3f}] (not significant if interval spans 0)")

# ---- judge vs human as AGREEMENT (not just rank correlation): weighted kappa, MAE, confusion ----
def weighted_kappa(y1, y2, K=3):
    n = len(y1)
    O = [[0] * K for _ in range(K)]
    for a, b in zip(y1, y2): O[a][b] += 1
    r = [sum(O[i]) for i in range(K)]
    c = [sum(O[i][j] for i in range(K)) for j in range(K)]
    num = den = 0.0
    for i in range(K):
        for j in range(K):
            w = (i - j) ** 2
            num += w * O[i][j]; den += w * (r[i] * c[j] / n)
    return 1 - num / den if den else 1.0

ji = [i for i in ids if judge32.get(i) is not None]
h = [cons[i] for i in ji]; jj = [judge32[i] for i in ji]
exact_jh = sum(1 for a, b in zip(h, jj) if a == b) / len(ji)
mae_jh = sum(abs(a - b) for a, b in zip(h, jj)) / len(ji)
print(f"\nJudge vs human agreement: exact {exact_jh:.3f}, weighted kappa {weighted_kappa(h, jj):.3f}, "
      f"MAE {mae_jh:.3f}")
conf = [[0] * 3 for _ in range(3)]
for a, b in zip(h, jj): conf[a][b] += 1
print("Confusion (rows = human 0/1/2, cols = judge 0/1/2):")
for lab, row in enumerate(conf): print(f"  human={lab}: {row}")

# ---- bootstrap CIs for the six text-similarity baselines (reseeded so each is order-independent) ----
print("\nText-similarity baselines, held-out tau-b with 95% bootstrap CI:")
for name, field in [("BLEU-4", "bleu4"), ("ROUGE-L", "rouge_l"), ("CHRF++", "chrf"),
                    ("METEOR", "meteor"), ("BM25", "bm25"), ("BERTScore", "bertscore")]:
    random.seed(20260720)
    g = (lambda f: (lambda i: met[i][f]))(field)
    lo_b, hi_b = boot(g)
    print(f"  {name:10} {tau(g):+.3f}  [{lo_b:+.3f}, {hi_b:+.3f}]")
