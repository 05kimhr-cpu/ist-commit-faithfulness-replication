# MODEL PINS — reproducibility (frozen)

작성 2026-07-20. 각 모델의 로컬 경로·config 핵심필드·config.json SHA256(provenance). vLLM 0.15, AWQ 4bit(judge), bnb 4bit(code-aware).

| 역할 | 모델 | model_type | quant | config SHA256(12) |
|---|---|---|---|---|
| judge 32B (headline) | Qwen2.5-Coder-32B-Instruct-AWQ | qwen2 | awq | `bd070b51e60a` |
| judge 14B (robustness) | Qwen2.5-Coder-14B-Instruct-AWQ | qwen2 | awq | `af62852325fc` |
| code-aware DGF | Llama-3.1-8B-Instruct | llama | - | `29e4c210b0d6` |
| raw NLI DGF | facebook--bart-large-mnli | bart | - | `a0f9bcb245b6` |
| BERTScore backbone | roberta-large | roberta | - | `82ba49810e64` |

## 환경
- vLLM 0.15.0 (the analysis environment), transformers, bitsandbytes 0.49.2 (code-aware).
- 디코딩: judge temperature 0, max_tokens 450, single sample (frozen).
- 실행: 단일 RTX 4090, enforce_eager (AWQ 32B).
- seeds: 표본/부트스트랩 random.seed 고정(스크립트 내).

## 주의 (재현성 한계)
- AWQ 모델은 커뮤니티 재양자화본; 원 Qwen2.5-Coder 릴리스의 정확 git revision hash는 로컬 사본에 미기록.
  재현 시 동일 AWQ 스냅샷 사용 권장(위 config SHA256로 대조).
