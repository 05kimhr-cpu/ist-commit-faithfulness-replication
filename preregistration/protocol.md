# Held-out 사람평가 — 사전등록 프로토콜 (pre-registration)

작성 2026-07-18. 목적: consequence-aware LLM judge의 사람 충실성 판단 상관을, **프롬프트 설계에 쓰지 않은 held-out 항목**에서 **판단·분석을 사람 라벨 전에 동결**하고 측정. 순환(공격 B)·표본(공격 A) 동시 방어. **결과와 무관하게 보고**(publish either way).

---

## 1. 가설 / 엔드포인트

- **H1 (주)**: frozen consequence judge의 판정(judge_corr)은 held-out 자연 메시지에서 사람 correctness 합의와 **양의 상관**(Kendall τ-b)이며, 기존 자동지표(BLEU/ROUGE/BM25/BERTScore/DGF/code-aware)를 상회한다.
- **주 엔드포인트**: τ-b(judge_corr, 3인 합의 correctness), held-out 160항목.
- **부 엔드포인트(exploratory, 검정력 낮음 명시)**: source별(gold/gen) τ, language별.

## 2. 동결 산출물 (사람 라벨 수집 **전** 확정)

| 항목 | 값 | 동결 |
|---|---|---|
| judge 모델 | Qwen2.5-Coder-32B-Instruct-AWQ (`/cephfs/lab/models/`) | ✅ |
| 프롬프트 | consequence-aware 3단계 (`재실험/19_judge_consequence.py`, SYS+USER) | ✅ 이후 수정 금지 |
| 디코딩 | temperature 0, max_tokens 450, single sample | ✅ |
| diff 정제 | `30_build_heldout.py` clean()(gate와 동일 포맷 검증됨) | ✅ |
| held-out 판정 | `heldout_judge.jsonl` — **사람 라벨 전 산출** | ✅ (job 6083) |
| 보조 judge | Qwen2.5-Coder-14B(robustness 재현 확인용) | 선택 |

**핵심**: 프롬프트는 원 40항목(gate)서 설계됨(researcher-in-the-loop). 본 held-out은 그 설계에 쓰이지 않은 새 diff이며, **프롬프트를 held-out 보고 다시 고치지 않음**(동결). → τ 헤드라인이 처음으로 설계-독립.

## 3. Held-out 표본 (`30_build_heldout.py`)

- **N=160 자연/실제 메시지**. master_pool 1598 diff에서 **원 사람평가 80 diff_id 제외**(held-out 보장, 0 겹침 검증됨) → 1515 pool서 추출.
- 균형: **source 40씩**(gold, codellama, qwen, deepseek) × **언어 20씩**(8lang). 1 diff = 1 메시지(반복 anchoring 방지). source 숨김·셔플(`heldout_packet.jsonl`).
- 파워: n=160 → τ CI 반폭 ~0.10(현 n=40 bootstrap SE 0.098를 1/√n 스케일). code-aware(0.18) 상회 유의에 충분.

## 4. Rater 설계

| 역할 | 인원 | 과업 | 부담 |
|---|---|---|---|
| primary | 2 | 전 160 double-coding | 각 160 (~5h) |
| adjudicator | 1 | 2인 불일치분 판정 + 고정 50 overlap(3-way IRR) | ~80–90 |

- 저자는 라벨·판정에 **불참**(중립성). 모두 source·judge점수 **blind**.
- adjudicator가 불일치 판정 → 3인 합의(median/다수결) 라벨.

## 5. Rater 지침 (**독립성 = 순환 방지 핵심**)

- 척도: **correctness 0/1/2** — 기존 사람평가와 **동일 중립 문구**: *"이 메시지가 이 diff의 변경을 정확히 설명하는가? 0=틀림, 1=부분, 2=정확."*
- ⚠️ **consequence 프롬프트 문구("effect/purpose/의도를 인정하라") 절대 미포함.** rater가 자연스럽게 판단. (judge와 지침을 같게 하면 일치가 인위적.)
- 부가: direction 0/1/2/NA(보조), notes(선택). source·모델·judge 점수 비공개.

## 6. 신뢰도

- 2인 전체: Cohen κ(가중) + Krippendorff α.
- 3인 overlap 50: Fleiss κ.
- intra-rater: 15항목 washout(≥1일) 재평가 1회.
- 불일치 → adjudicator 판정 → 합의 라벨.

## 7. 분석 (사람 라벨 **전** 고정)

1. 주: τ-b(judge_corr vs 합의) + **bootstrap 5000 95%CI**, held-out 160.
2. baseline: 동일 160에 BLEU/ROUGE/BM25/BERTScore/DGF/code-aware τ (스크립트 `07`/전지표 재계산).
3. ceiling: 2인·3인 IRR τ (비대칭 주의 명시).
4. exploratory: source별·language별 τ (검정력 낮음 표기).
5. robustness: 14B judge 동일 분석.

## 8. 사전등록 성공기준 (결과 무관 보고)

- **주 성공**: held-out τ **bootstrap CI 하한 > 0.30** **및** > code-aware 점추정(≈0.18). → "강 코드-LLM+consequence judge가 held-out서 사람과 moderate+ 상관, 기존지표 상회" 확정.
- **부분**: 점추정 > code-aware이나 CI 하한 ≤ 0.30 → "유망하나 정밀도 제한" 보고.
- **음성**: 점추정 ≤ code-aware → "설계셋 결과가 held-out서 유지 안 됨"(정직 보고, 논문은 critique+음성으로).
- 어느 경우든 **held-out τ가 최종 헤드라인**(설계셋 0.694는 참고·상한).

## 9. 남은 실행 순서

1. ✅ held-out 표본·frozen judge 판정 (6083).
2. held-out 160에 전지표 스코어(BLEU~code-aware) — GPU, 사람과 독립.
3. rater packet 사람친화 포맷(xlsx)+지침서 → 배포.
4. 라벨 회수 → 신뢰도 → 합의 → 주 분석.
5. 결과를 `00_FINDINGS_CONSOLIDATED.md` 헤드라인으로 승격(설계셋→held-out).

## 부록 — 파일
표본 `heldout_items.jsonl`(judge/metric 입력), `heldout_key.csv`(source 정답), `heldout_packet.jsonl`(rater용, source 숨김). frozen 판정 `heldout_judge.jsonl`.
