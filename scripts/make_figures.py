#!/usr/bin/env python3
"""
Regenerate the metric-vs-human figure from the released human labels and model predictions.
Run from the package root:  python scripts/make_figures.py
Writes figures/metric_vs_human.pdf. No GPU required.
"""
import json, os, csv, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
import openpyxl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data"); FIG = os.path.join(ROOT, "figures")
# colorblind-safe (Okabe-Ito)
C = {"blue": "#0072B2", "verm": "#D55E00", "green": "#009E73", "gray": "#999999", "black": "#000000"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

def load_xlsx(p):
    ws = openpyxl.load_workbook(p)["ratings"]
    return {r[0]: int(r[4]) for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None}
def load_csv(p):
    return {r["item_id"]: int(r["correctness(0/1/2)"]) for r in csv.DictReader(open(p, encoding="utf-8-sig"))}
A = load_xlsx(f"{D}/human_labels/rater_A.xlsx"); B = load_xlsx(f"{D}/human_labels/rater_B.xlsx"); J = load_csv(f"{D}/human_labels/adjudicator.csv")
ids = sorted(A)
def cons_of(i):
    if A[i] == B[i]: return A[i]
    c = collections.Counter([A[i], B[i], J[i]]); t, n = c.most_common(1)[0]
    return t if n >= 2 else J[i]
cons = {i: cons_of(i) for i in ids}
def jl(p): return {json.loads(l)["item_id"]: json.loads(l) for l in open(p)}
met = jl(f"{D}/predictions/reference_metrics.jsonl")
code_nli = {json.loads(l)["item_id"]: json.loads(l)["dgf_codeaware"] for l in open(f"{D}/predictions/code_nli.jsonl")}
judge = {k: v["judge_corr"] for k, v in jl(f"{D}/predictions/judge_32b.jsonl").items()}
def tau(g):
    xs = [g(i) for i in ids if g(i) is not None]; ys = [cons[i] for i in ids if g(i) is not None]
    return kendalltau(xs, ys).correlation
ceil = kendalltau([A[i] for i in ids], [B[i] for i in ids]).correlation

rows = [("BLEU-4", tau(lambda i: met[i]["bleu4"])), ("ROUGE-L", tau(lambda i: met[i]["rouge_l"])),
        ("CHRF++", tau(lambda i: met[i]["chrf"])), ("METEOR", tau(lambda i: met[i]["meteor"])),
        ("BM25", tau(lambda i: met[i]["bm25"])), ("BERTScore", tau(lambda i: met[i]["bertscore"])),
        ("raw NLI", tau(lambda i: met[i]["dgf"])), ("code-adapted NLI", tau(lambda i: code_nli.get(i))),
        ("LLM judge", tau(lambda i: judge.get(i)))]
rows.sort(key=lambda r: r[1]); labels = [r[0] for r in rows]; vals = [r[1] for r in rows]
colors = [C["blue"] if "judge" in l else (C["verm"] if v < 0 else C["gray"]) for l, v in zip(labels, vals)]
fig, ax = plt.subplots(figsize=(7, 4.2)); y = range(len(rows))
ax.barh(list(y), vals, color=colors, height=0.62, zorder=3); ax.axvline(0, color=C["black"], lw=0.8)
ax.axvline(ceil, color=C["green"], lw=1.6, ls="--"); ax.text(ceil - 0.018, len(rows) / 2.0 - 0.5, f"inter-rater ceiling {ceil:.2f}", color=C["green"], rotation=90, va="center", ha="center", fontsize=8.5)
ax.set_yticks(list(y)); ax.set_yticklabels(labels)
for i, v in enumerate(vals): ax.text(v + (0.012 if v >= 0 else -0.012), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=9)
ax.set_xlim(-0.2, 0.80); ax.set_xlabel("Kendall tau vs human faithfulness (held-out, n=160)")
ax.set_title("Automatic metrics do not track human faithfulness; a code-LLM judge does", fontsize=11)
os.makedirs(FIG, exist_ok=True)
fig.savefig(os.path.join(FIG, "metric_vs_human.pdf")); print("wrote figures/metric_vs_human.pdf")
