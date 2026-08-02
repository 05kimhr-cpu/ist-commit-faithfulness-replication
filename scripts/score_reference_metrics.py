#!/usr/bin/env python3
"""
Compute the automatic baseline scores for each (diff, message) pair, with the message as candidate
and the diff as reference: BLEU-4, ROUGE-L, CHRF++, METEOR, BM25, BERTScore, and a raw signed
BART-MNLI score (entailment minus contradiction). Requires a GPU for BERTScore and the NLI model.
Usage: ITEMS_FILE=heldout_items.jsonl OUT=heldout_metrics.jsonl python score_reference_metrics.py
"""
import json, os
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
HERE=os.path.dirname(os.path.abspath(__file__))
ITEMS=os.path.join(HERE, os.environ.get("ITEMS_FILE","heldout_items.jsonl"))
OUT=os.path.join(HERE, os.environ.get("OUT","heldout_metrics.jsonl"))
BART=os.environ.get("BART_MNLI_MODEL","/cephfs/lab/models/facebook--bart-large-mnli")
ROBERTA=os.environ.get("ROBERTA_MODEL","/cephfs/lab/models/roberta-large")

import sacrebleu
from rouge_score import rouge_scorer as Rg
from rank_bm25 import BM25Okapi
import nltk
from nltk.translate.meteor_score import meteor_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def diff_of(it):
    # held-out items store cleaned 'diff'; 120 items_master store 'diff_tok'
    return it.get("diff") or it.get("diff_tok") or ""

items=[json.loads(l) for l in open(ITEMS)]
msgs=[it["message"] for it in items]
diffs=[diff_of(it) for it in items]
n=len(items); print(f"scoring {n} items from {ITEMS}", flush=True)

_r=Rg.RougeScorer(["rougeL"], use_stemmer=False)
def tok(t): return t.lower().split()

bleu=[sacrebleu.sentence_bleu(m,[d]).score/100 for m,d in zip(msgs,diffs)]
rouge=[_r.score(d,m)["rougeL"].fmeasure for d,m in zip(diffs,msgs)]
chrf=[sacrebleu.sentence_chrf(m,[d],word_order=2).score/100 for m,d in zip(msgs,diffs)]  # chrF++
def _meteor(d,m):
    try: return meteor_score([tok(d)], tok(m))
    except Exception: return None
meteor=[_meteor(d,m) for d,m in zip(diffs,msgs)]
_bm25_corpus=BM25Okapi([tok(d) for d in diffs])  # corpus-level IDF over all diffs (not per-example)
bm25=[float(_bm25_corpus.get_scores(tok(m))[i]) for i,m in enumerate(msgs)]
print("lexical(+chrf++,meteor) done", flush=True)

from bert_score import score as bs
_,_,F=bs(cands=msgs, refs=[d[:1500] for d in diffs], model_type=ROBERTA, num_layers=17,
         batch_size=32, device="cuda" if torch.cuda.is_available() else "cpu", verbose=False)
bertscore=[float(x) for x in F.tolist()]
print("bertscore done", flush=True)

# raw signed DGF = P_ent - P_con from BART-MNLI, premise=diff, hyp=message
dev="cuda" if torch.cuda.is_available() else "cpu"
btok=AutoTokenizer.from_pretrained(BART); bmdl=AutoModelForSequenceClassification.from_pretrained(BART).to(dev).eval()
dgf=[]
for i in range(0,n,32):
    p=[d[:2000] for d in diffs[i:i+32]]; h=msgs[i:i+32]
    inp=btok(p,h,return_tensors="pt",truncation=True,max_length=512,padding=True).to(dev)
    with torch.no_grad(): pr=torch.softmax(bmdl(**inp).logits,-1).tolist()
    dgf+=[x[2]-x[0] for x in pr]   # entail - contradict
print("raw DGF done", flush=True)

with open(OUT,"w") as f:
    for i,it in enumerate(items):
        f.write(json.dumps({"item_id":it["item_id"],"bleu4":bleu[i],"rouge_l":rouge[i],
                            "chrf":chrf[i],"meteor":meteor[i],"bm25":bm25[i],
                            "bertscore":bertscore[i],"dgf":dgf[i]})+"\n")
print(f"wrote {OUT}", flush=True)
