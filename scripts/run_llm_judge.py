#!/usr/bin/env python3
"""
Run the LLM judge (GPU). A code-specialized instruction model rates each (diff, message) pair on
the 0/1/2 correctness scale, reasoning step by step about what the change accomplishes before
scoring. Greedy decoding. The model, prompt, and decoding are frozen for the held-out evaluation.
Paths must be adjusted to the local environment.
"""
import json, os, re
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
HERE=os.path.dirname(os.path.abspath(__file__))
MODEL=os.environ.get("JUDGE_MODEL","/cephfs/lab/models/Qwen2.5-Coder-32B-Instruct-AWQ")
TAG=os.environ.get("JUDGE_TAG","Qwen2.5-Coder-32B-AWQ-conseq")
ITEMS=os.path.join(HERE,os.environ.get("ITEMS_FILE","items.jsonl"))
OUT=os.path.join(HERE,f"{os.environ.get('OUT_PREFIX','judge_scores')}{os.environ.get('OUT_SUFFIX','')}.jsonl")

SYS=("You are a senior engineer judging whether a commit message faithfully describes a diff.\n"
     "Crucially: a commit message is CORRECT if it describes EITHER the literal code change OR "
     "what that change ACCOMPLISHES (its effect, purpose, optimization, or bug fix). Real "
     "developers describe intent and effect, not just syntax. Reason about consequences.\n"
     "- '+' = added, '-' = removed.\n"
     "- Credit an accurate high-level/abstract description of the effect (e.g. 'reduce lookups', "
     "'avoid crash', 'speed up X') when the diff plausibly produces that effect.\n"
     "- Do NOT credit claims the diff does not support (wrong entity, wrong direction, invented change).")
USER=("[DIFF]\n{diff}\n\n[MESSAGE] {message}\n\n"
      "Reason in 3 steps:\n"
      "1. What does the diff LITERALLY change?\n"
      "2. What does that change ACCOMPLISH (effect / purpose / fix / optimization)?\n"
      "3. Does the message accurately describe the literal change OR its accomplishment? "
      "(abstract/effect description = accurate; wrong or unsupported = not)\n"
      "End with exactly one line: SCORE: <0,1,2>  (2=accurate  1=partial/vague  0=wrong/unsupported)")

def parse(t):
    m=re.findall(r"SCORE:\s*([012])",t)
    if m: return int(m[-1])
    m=re.findall(r"\b([012])\b",t); return int(m[-1]) if m else None

def main():
    from vllm import LLM, SamplingParams
    items=[json.loads(l) for l in open(ITEMS)]
    print(f"judge={TAG} items={len(items)}",flush=True)
    llm=LLM(model=MODEL,quantization="awq",dtype="float16",max_model_len=4096,
            gpu_memory_utilization=0.92,enforce_eager=True,trust_remote_code=True)
    tok=llm.get_tokenizer(); sp=SamplingParams(temperature=0.0,max_tokens=int(os.environ.get("MAX_TOK","450")))
    prompts=[]
    for it in items:
        u=USER.format(diff=it["diff"][:6000],message=it["message"])
        msgs=[{"role":"system","content":SYS},{"role":"user","content":u}]
        try:
            p=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        except TypeError:
            p=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        except Exception:
            # models without a chat_template (e.g. CodeLlama-Instruct) -> manual Llama-2 [INST] format
            p=f"<s>[INST] <<SYS>>\n{SYS}\n<</SYS>>\n\n{u} [/INST]"
        prompts.append(p)
    outs=llm.generate(prompts,sp); bad=0
    with open(OUT,"w") as f:
        for it,o in zip(items,outs):
            sc=parse(o.outputs[0].text); bad+=sc is None
            f.write(json.dumps({"item_id":it["item_id"],"block":it.get("block",""),"source":it.get("source",""),
                                "language":it.get("language",""),
                                "judge":TAG,"judge_corr":sc,"raw":o.outputs[0].text.strip()[-150:]})+"\n")
    print(f"wrote {OUT} (unparsed={bad})",flush=True)
if __name__=="__main__": main()
