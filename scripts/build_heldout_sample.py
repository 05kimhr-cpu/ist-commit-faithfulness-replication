#!/usr/bin/env python3
"""
Construct the held-out evaluation sample (N=160). Diffs are disjoint from the development set,
balanced across eight languages and four message sources (one message per diff). Diffs are
normalized to a readable unified-diff format; the message source is hidden in the rater packet.
This script runs against the commit corpus in the original environment; paths must be adjusted.
Outputs: heldout_items.jsonl (judge/metric input), heldout_key.csv (source key),
         heldout_packet.jsonl (rater-facing, source hidden, shuffled).
"""
import json, os, re, csv, random, collections
random.seed(20260718)
ROOT=os.environ.get("CORPUS_ROOT",".")  # set to the corpus root
HE=f"{ROOT}/research/human_eval/packet"
OUT=os.path.dirname(os.path.abspath(__file__))
SRC=f"{ROOT}/research/data/merged/master_pool.jsonl"
SOURCES=[("gold","gold"),("codellama","gen_codellama"),("qwen","gen_qwen"),("deepseek","gen_deepseek")]
PER_CELL=5   # per (language x source)

def raw_to_rendered(s):
    """master_pool raw ('mmm a/.. <nl> ppp b/.. <nl> + line ..') -> diff_rendered style.
    Matches the reference diff_rendered: <nl>->newline, mmm/ppp->---/+++, join ' / '->'/', drop blank lines."""
    s=(s or "").replace(" <nl> ","\n").replace("<nl>","\n")
    out=[]
    for ln in s.split("\n"):
        ln=ln.strip()
        if not ln: continue                       # diff_rendered drops blank context lines
        if ln.startswith("mmm "): ln="--- "+ln[4:]
        elif ln.startswith("ppp "): ln="+++ "+ln[4:]
        ln=ln.replace(" / ","/")                  # join path slashes (matches reference)
        out.append(ln)
    return "\n".join(out)

def clean_diff(s):
    out=[]
    for ln in s.split("\n"):
        marker=""
        for mk in ("--- ","+++ ","@@ ","- ","+ "):
            if ln.startswith(mk): marker=mk; ln=ln[len(mk):]; break
        else:
            for mk in ("---","+++","-","+"):
                if ln.startswith(mk): marker=mk+" "; ln=ln[len(mk):].lstrip(); break
        b=ln
        b=re.sub(r"\s+:\s+:\s+","::",b); b=re.sub(r"\s*::\s*","::",b)
        b=re.sub(r"\s+\.\s+",".",b); b=re.sub(r"\s+([,;)\]}])",r"\1",b)
        b=re.sub(r"([(\[{])\s+",r"\1",b); b=re.sub(r"\s+->\s+","->",b)
        b=re.sub(r"\s+:\s+",": ",b); b=re.sub(r"\s{2,}"," ",b).rstrip()
        out.append((marker+b).rstrip())
    return "\n".join(out)

def clean(s): return clean_diff(raw_to_rendered(s))

# ---- exclude original-120 diff_ids (held-out guarantee) ----
he_diffids=set(r["diff_id"] for r in csv.DictReader(open(f"{HE}/hidden_key.csv")))
mp=[json.loads(l) for l in open(SRC)]
avail=[r for r in mp if str(r["diff_id"]) not in he_diffids]
print(f"master_pool={len(mp)}  human_diff_ids={len(he_diffids)}  held-out available={len(avail)}")

# group by language
bylang=collections.defaultdict(list)
for r in avail: bylang[r["language"]].append(r)
langs=sorted(bylang)
print("languages:", {L:len(bylang[L]) for L in langs})

# ---- sample: for each language, 4 sources x 5 distinct diffs (no diff reused) ----
items=[]; used=set(); iid=0
for L in langs:
    pool=[r for r in bylang[L] if str(r["diff_id"]) not in used]
    random.shuffle(pool)
    it=iter(pool)
    for srcname,key in SOURCES:
        got=0
        while got<PER_CELL:
            try: r=next(it)
            except StopIteration:
                print(f"  WARN: {L}/{srcname} ran out"); break
            msg=(r.get(key) or "").strip()
            if not msg or str(r["diff_id"]) in used: continue
            used.add(str(r["diff_id"])); got+=1; iid+=1
            items.append({"item_id":f"H{iid:03d}","diff_id":str(r["diff_id"]),
                          "language":L,"source":srcname,
                          "diff":clean(r["diff"])[:6000],"message":msg})
print(f"\nbuilt {len(items)} items")
print("source balance:", dict(collections.Counter(x['source'] for x in items)))
print("lang balance:", dict(collections.Counter(x['language'] for x in items)))
# held-out check
assert not (set(x["diff_id"] for x in items) & he_diffids), "HELD-OUT VIOLATION"
print("held-out check: 0 overlap with human diff_ids  [OK]")

# write judge/metric input + key
with open(f"{OUT}/heldout_items.jsonl","w") as f:
    for it in items: f.write(json.dumps(it,ensure_ascii=False)+"\n")
with open(f"{OUT}/heldout_key.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["item_id","diff_id","language","source"])
    for it in items: w.writerow([it["item_id"],it["diff_id"],it["language"],it["source"]])
# rater packet: source hidden, shuffled
pk=[{"item_id":it["item_id"],"language":it["language"],"diff":it["diff"],"message":it["message"]} for it in items]
random.shuffle(pk)
with open(f"{OUT}/heldout_packet.jsonl","w") as f:
    for it in pk: f.write(json.dumps(it,ensure_ascii=False)+"\n")
print(f"wrote heldout_items.jsonl, heldout_key.csv, heldout_packet.jsonl")

# ---- verify cleaning matches reference format on an OVERLAPPING diff (sanity) ----
print("\n=== cleaning sanity: my-cleaned master_pool vs reference diff_rendered (shared diff_id) ===")
items_master={json.loads(l)["item_id"]:json.loads(l) for l in open(f"{HE}/items_master.jsonl")}
ref_by_diffid={}
for it in items_master.values(): ref_by_diffid.setdefault(str(it["diff_id"]),it)
for r in mp:
    did=str(r["diff_id"])
    if did in ref_by_diffid:
        mine=clean(r["diff"])
        reference=clean_diff(ref_by_diffid[did].get("diff_rendered",""))
        print(f"diff_id={did}")
        print(" MINE  :", repr(mine[:150]))
        print(" REFERENCE  :", repr(reference[:150]))
        break
print("\nsample held-out item:", json.dumps({k:(v[:100] if k=='diff' else v) for k,v in items[0].items()},ensure_ascii=False))
