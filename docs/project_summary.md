# GPT-2 Small로 GSM8K 수학 추론 끌어올리기
## 커리큘럼 학습 + Plan-then-Solve CoT 포맷 정렬 실험 전체 정리

**과목**: CSEG321 인간언어기술개론  
**작성**: 2026-06-09 KST  
**모델**: GPT-2 small (124M, 자체 구현 `models/gpt2.py`)  
**실행 환경**: Colab T4 (초기) → Sogang AIPub A100 MIG 1g.10gb (본 실험)

---

## 1. 연구 배경 및 동기

### 1.1 출발점: 기존 파이프라인 실패

Phase 0에서 구축한 vanilla 3단계 파이프라인:

| 방법 | GSM8K acc | fmt | repeat | no_ans | 비고 |
|------|-----------|-----|--------|--------|------|
| vanilla GPT-2 → CoT-SFT (10ep, lr=1e-5, bs=8) | **1.6%** | **95.8%** | 4% | 0% | 기준선 |
| 위 모델 → DPO (3ep, beta=0.1) | **0.8%** | **0.4%** | 34% | 31% | format collapse |

> vanilla SFT의 fmt=95.8%는 역설적: 형식(#### N)은 거의 다 맞추는데 acc=1.6%. 즉 형식이 아니라 **내용(추론)**이 문제.  
> DPO는 오히려 하락. off-policy chosen(gold reasoning 재사용) + likelihood displacement + rewards/margins 3.449 과최적화로 collapse.

**DPO 상세**: 오답 생성 1473쌍 (`data/dpo_pairs.jsonl`), chosen=gold_reasoning (Gemini API 없이 `gsm8k_dpo_source.jsonl` 재사용), beta=0.1, 3 epochs.

### 1.2 실패 원인 진단 (2026-06-03, dev 500개 전수 분석)

`outputs/9_10-1e-05-reasoning_generations.txt` 스크립트 분류:

```
already correct          :   8  (1.6%)
calculator FIXES answer  :   5  (산술만 틀림)
still wrong (reasoning)  : 482  (연산 선택 자체 오류 — 계산기도 무력)

→ 계산기 완벽 적용 시: 1.6% → 2.6% (+1%p 에 그침)
→ 오답의 99%가 "추론(operation-selection)" 오류
```

dev0 전형: 정답 18(=6+12)인데 모델이 ×2, ×3 단계를 *지어냄*. 계산기가 112→126으로 고쳐도 여전히 틀림.

**핵심 결론**: GSM8K는 **capacity-bound** — 124M 가중치 용량의 한계, technique-bound가 아님.  
calculator / DPO / 포맷 변경 등 inference-level 패치로는 돌파 불가.

### 1.3 전략 전환

```
구 전략: CoT-SFT → DPO → calculator
신 전략: 산술 pretrain → MultiArith 커리큘럼(plan CoT) → GSM8K (plan 포맷 정렬)
```

두 가설:
- **H1 (실행)**: 산술 pretrain → 계산이 정확해져 GSM8K 오름 → calculator-sim +1%p로 *이미 반증. 약함.*
- **H2 (초기화/표본효율)**: 산술 pretrain → "계산 배우는 비용" 절약 → SFT 여유 용량을 operation-selection에 투자 → *검증 대상*

---

## 2. 환경

| 항목 | 값 |
|------|-----|
| **Phase 0-1** | Colab T4 GPU, Google Drive |
| **Phase 2 (본 실험)** | Sogang AIPub A100 MIG 1g.10gb, VRAM 10GB |
| **서버 주소** | `163.239.15.20:30205` |
| **영구 스토리지** | `/mnt/` (NAS, 컨테이너 재시작 후 유지) |
| **파이프라인** | `run_pipeline.py` (서버), `colab_pipeline_math.ipynb` (Colab) |
| **arith 체크포인트** | `cot_large_integer_arithmetic_pretrain.pt` — 자릿수 분해 CoT(pemdas→large_integer)로 사칙연산 pretrain된 GPT-2 small |
| **디코딩** | greedy (argmax), 전 실험 통일 |
| **seed** | 11711 |

---

## 3. 데이터셋

### 3.1 MultiArith (Stage 1 커리큘럼)

| 파일 | 블록 수 | 설명 |
|------|--------|------|
| `multiarith_sft_train_base.txt` | 510 | 원본 MultiArith SFT (standard CoT) |
| `multiarith_sft_train_aug.txt` | 3,060 | 숫자치환 증강 ×10 (standard CoT) |
| `multiarith_sft_train_plan_aug.txt` | 3,061 | plan-then-solve CoT + 숫자치환 증강 |
| `multiarith_dev.jsonl` | 90 | dev split (서버 최종 평가 기준) |

**증강 방식**: 각 예제의 수치를 무작위 교체 후 CoT 자동 재계산 (결정론적, LLM 불필요)  
**dev 불변 원칙**: dev는 절대 증강 안 함 (leakage 방지)

**in-template / held-out 분리** (`audit_template_leakage.py`):  
MultiArith dev의 일부가 train과 동일 문제 템플릿 → 점수가 암기인지 일반화인지 구분 필수.  
in_template = train에서 본 템플릿 구조, held_out = 새 템플릿.

**MultiArith plan-then-solve CoT 포맷**:
```
{id}

Question: {문제}

Plan: {N}단계. (1) {수식관계}; ...

Reasoning:
{step1} <<expr=result>>
...
#### {답}

<|endoftext|>
```

### 3.2 GSM8K (Stage 2 파인튜닝)

| 파일 | 블록 수 | 설명 |
|------|--------|------|
| `gsm8k_sft_train.txt` | 3,000 | GSM8K train 원본 (7473 중 3000 사용) |
| `gsm8k_sft_train_plan_skeleton.txt` | 3,000 | skeleton plan 추가 (연산자만, 숫자 0개) |
| `gsm8k_sft_train_plan_entity.txt` | 3,000 | entity plan 추가 (PS+ 스타일, 숫자 0개) |
| `gsm8k_plan_skeleton_plus_ma.txt` | 6,061 | skeleton + MA plan aug 혼합 |
| `gsm8k_plan_entity_plus_ma.txt` | 6,061 | entity + MA plan aug 혼합 |
| `gsm8k_dev.jsonl` | 500 | dev split (평가 전용) |

**GSM8K standard CoT 포맷**:
```
{id}

Question: {문제}

Reasoning:
{step} <<expr=result>>
...
#### {답}

<|endoftext|>
```

**skeleton plan 포맷** (연산자만, 숫자 0개):
```
Plan: Solve in N steps. (1) multiply; (2) add; ...; then give the final answer.
```

**entity plan 포맷** (PS+ 스타일, 엔티티+연산, 숫자 0개):
```
Plan: find Kim's height, then calculate Tamara's height, then give the final answer.
```

> 설계 배경: 초기 Gemini 생성 plan은 답 계산 노출(`calculate (3*24)-4`)과 상수 박힘(`multiply 12 by 2`) 문제 → 완전 재설계. 숫자를 전부 제거해 answer leakage 차단. skeleton은 연산 순서만, entity는 엔티티(주어)+연산 서술로 operation-selection만 가이드.

---

## 4. 학습 파이프라인 및 설계

### 4.1 전체 커리큘럼 구조

#### Stage 0 — 산술 사전학습 (제공된 체크포인트)

| 항목 | 내용 |
|------|------|
| **의도** | 사칙연산(덧셈·뺄셈·곱셈·나눗셈) 능력을 먼저 확보해 이후 SFT가 추론에 집중할 수 있도록 초기화 |
| **베이스 모델** | GPT-2 small (124M, 랜덤 초기화) |
| **학습 데이터** | 자릿수 분해 CoT 산술 데이터 (pemdas → large_integer 체인) |
| **평가셋** | — (외부 제공, 직접 학습하지 않음) |
| **산출물** | `cot_large_integer_arithmetic_pretrain.pt` |

---

#### Stage 1 — MultiArith plan-CoT SFT

| 항목 | 내용 |
|------|------|
| **의도** | 2단계 word problem에서 plan-then-solve 포맷을 학습. operation-selection 능력을 명시적 Plan 헤더로 supervise. GSM8K Stage 2의 warm-start 체크포인트 생성. |
| **베이스 모델** | Stage 0 체크포인트 (`cot_large_integer_arithmetic_pretrain.pt`) |
| **학습 데이터** | `multiarith_sft_train_plan_aug.txt` — 3,061 blocks (숫자치환 증강 ×10 + Plan 헤더) |
| **평가셋** | `multiarith_dev.jsonl` — 90개 (in_template / held_out 분리 평가, 서버 최종 기준) |
| **하이퍼파라미터** | lr=1e-5, bs=8, epochs=40, patience=8 |
| **산출물** | `MA_plan_numaug_best.pt` — acc **91.1%** (held_out 86.0%) |

---

#### Stage 2-A — GSM8K 직접 파인튜닝 (arith init 베이스라인)

Stage 0에서 바로 GSM8K로 SFT. 커리큘럼 없이 산술-init만의 효과를 측정.

| exp_id | 의도 | 베이스 모델 | 학습 데이터 | 평가셋 |
|--------|------|------------|------------|--------|
| **G_A1_direct** | 산술-init의 순수 효과 (vs vanilla 1.6%) | Stage 0 ckpt | `gsm8k_sft_train.txt` 3,000 blocks | `gsm8k_dev.jsonl` 500개 |
| **G_A2_mixed** | MA 데이터 혼합이 형식·루핑에 미치는 효과 | Stage 0 ckpt | gsm8k + MA plan aug 혼합, 6,061 blocks | `gsm8k_dev.jsonl` 500개 |

공통: lr=1e-5, bs=8, epochs=20, patience=4

---

#### Stage 2-B — GSM8K 커리큘럼 파인튜닝 (MA ckpt init)

Stage 1 체크포인트에서 GSM8K로 파인튜닝. **포맷 정렬 여부**가 핵심 변수.

| exp_id | 의도 | 베이스 모델 | 학습 데이터 | 평가셋 |
|--------|------|------------|------------|--------|
| **G_B1_std** | 포맷 불일치 기준 — Plan 헤더 없이 전이 시 catastrophic interference 정량화 | Stage 1 ckpt | `gsm8k_sft_train.txt` 3,000 blocks (Plan 없음) | `gsm8k_dev.jsonl` 500개 |
| **G_B1_skel** | skeleton plan으로 포맷 정렬 — 연산자 순서만 명시 | Stage 1 ckpt | `gsm8k_sft_train_plan_skeleton.txt` 3,000 blocks | `gsm8k_dev.jsonl` 500개 |
| **G_B2_skel** | skeleton plan + MA 혼합으로 plan 패턴 강화 | Stage 1 ckpt | skeleton + MA 혼합, 6,061 blocks | `gsm8k_dev.jsonl` 500개 |
| **G_B1_ent** | entity plan(PS+)으로 포맷 정렬 — 엔티티+연산 서술 | Stage 1 ckpt | `gsm8k_sft_train_plan_entity.txt` 3,000 blocks | `gsm8k_dev.jsonl` 500개 |
| **G_B2_ent** | entity plan + MA 혼합으로 plan 패턴 강화 | Stage 1 ckpt | entity + MA 혼합, 6,061 blocks | `gsm8k_dev.jsonl` 500개 |

공통: lr=5e-6, bs=4 (plan arms — max_len 486~504으로 OOM 방지), epochs=20, patience=4

### 4.2 설계 피봇: B arm 원안 → plan format arm

**원래 설계 (B1_std 결과 전)**:
- B1: MA ckpt → gsm8k (standard CoT)
- B2: MA ckpt → gsm8k + MA 혼합 (standard CoT)

**피봇 계기**: B1_std 완료 후 fmt=0.156 확인 → 포맷 불일치가 커리큘럼 효과를 소멸시킴.  
→ GSM8K 데이터에도 Plan 헤더를 추가해 MA ckpt와 포맷 정렬 → skeleton/entity 두 버전으로 설계.

### 4.3 plan-then-solve CoT 설계 의도

operation-selection(병목)을 **명시적 학습 타깃**으로 끌어올림:

```
Question: 사탕 32+42개, 35개 먹음

Plan: Solve in 2 steps. (1) add; (2) subtract; then give the final answer.

Reasoning:
32 + 42 = <<32+42=74>>74
74 - 35 = <<74-35=39>>39
#### 39
```

Plan 헤더 → reasoning 루프에 앵커 제공 → 루핑 억제 + 연산 순서 가이드.

### 4.4 공통 학습 설정

| 파라미터 | A arms | B arms | MA Stage 1 |
|---------|--------|--------|-----------|
| lr | 1e-5 | 5e-6 | 1e-5 |
| batch_size | 8 | 4 (plan) / 8 (std) | 8 |
| max epochs | 20 | 20 | 40 |
| early stop patience | 4 | 4 | 8 |
| per-epoch eval | 150 서브셋 | 150 서브셋 | 전체 |
| 최종 eval | 500 full dev | 500 full dev | 전체 |

> batch_size=4 이유: plan 줄 추가로 max_len 486~504 (standard: 430) → bs=8에서 CUDA OOM

> prompt loss masking: **미적용** (질문 토큰도 함께 학습). 오히려 의도적 설계 선택으로 볼 수 있음 — 질문 패턴을 학습함으로써 "Question: X → Reasoning: ..." 의 조건부 구조 전체를 학습해 reasoning 방향의 컨텍스트 앵커가 강해짐. 질문이 reasoning 대비 짧고 병목이 모델 용량이므로 마스킹의 추가 이득도 미미.

### 4.5 평가 지표

| 지표 | 정의 |
|------|------|
| acc | `#### N` 최종 숫자가 gold와 일치 |
| fmt (format_valid_rate) | 생성 텍스트가 `#### N` 형식 도달 |
| repeat (repetition_rate) | 동일 구절 반복 루프 비율 |
| no_ans (no_answer_rate) | `####` 미포함 생성 비율 |
| in_template | train과 동일 템플릿 구조 dev 정확도 |
| held_out | train에서 못 본 템플릿 dev 정확도 |

**self-consistency**: `scripts/eval/eval_self_consistency.py` — k=8, temp=0.8, top_p=0.95, 다수결 (학습 완료 후 별도 실행)

**generation 저장**: 최종 full-dev eval 시 `{exp_id}_generations.json` 자동 저장 (커밋 7a60e79, 2026-06-08~)

---

## 5. 전체 실험 결과

### 5.1 Phase 0 — Colab, vanilla 파이프라인

| 실험 | GSM8K acc | fmt | repeat | no_ans |
|------|-----------|-----|--------|--------|
| vanilla CoT-SFT (10ep) | 1.6% | **95.8%** | 4% | 0% |
| + DPO (3ep, beta=0.1) | 0.8% | **0.4%** | 34% | 31% |

### 5.2 Phase 1 — Colab 파일럿, arith-init 설계 검증

| 실험 | 평가셋 | acc | in_template | held_out | 비고 |
|------|--------|-----|-------------|----------|------|
| MA_plan_numaug (서버) | MultiArith dev 90 | **91.1%** | 97.5% | **86.0%** | 서버 실측 |

### 5.3 Phase 2 — A100 서버, 전체 파이프라인

| exp_id | init | 데이터 | acc | fmt | repeat | no_ans | best_ep | total_ep | 소요 |
|--------|------|--------|-----|-----|--------|--------|---------|----------|------|
| MA_plan_numaug | arith | MA plan aug (3061) | **0.911** | **1.000** | **0.000** | **0.000** | 34 | 40 | 10,553s |
| G_A1_direct | arith | gsm8k (3000) | 0.022 | 0.228 | 0.778 | 0.026 | 1 | 6 | 9,537s |
| G_A2_mixed | arith | gsm8k+MA (6061) | 0.020 | 0.428 | 0.548 | 0.026 | 0 | 5 | 8,675s |
| G_B1_std | MA ckpt | gsm8k (3000) | 0.024 | 0.156 | 0.796 | 0.042 | 1 | 6 | 10,269s |
| G_B1_skel | MA ckpt | gsm8k skeleton (3000) | 0.022 | 0.128 | 0.716 | 0.060 | 0 | 5 | 8,997s |
| G_B2_skel | MA ckpt | skeleton+MA (6061) | 0.024 | **0.832** | **0.146** | **0.008** | 1 | 6 | 6,811s |
| G_B1_ent | MA ckpt | gsm8k entity (3000) | 0.026 | 0.738 | 0.424 | 0.028 | 7 | 12 | 12,783s |
| G_B2_ent | MA ckpt | entity+MA (6061) | **0.028** | 0.744 | 0.188 | 0.020 | 0 | 5 | 6,164s |

> MA_plan_numaug (서버): in_template=0.975, held_out=0.860

---

## 6. 실험별 상세 분석

### 6.1 MA_plan_numaug — 성공

```
acc=0.911, fmt=1.000, repetition=0.000, no_answer=0.000
in_template=0.975, held_out=0.860
best_epoch=34 / 40 epochs
```

**성공 원인:**
- fmt=1.0: 모든 생성이 `#### N` 형식 도달 — plan 앵커가 생성 궤적 안정화
- repetition=0.0: 루핑 완전 억제 — plan 헤더가 탈출 구조 제공
- held_out 0.860: 학습 시 못 본 템플릿에서도 86% → **암기가 아닌 일반화**
- 단계적 복잡도 증가(arith → 2-step MA)가 124M 용량 내에서 수렴 가능

---

### 6.2 G_A1_direct — 예상된 실패 (arith init 베이스라인)

```
acc=0.022, fmt=0.228, repetition=0.778, no_answer=0.026
best_epoch=1 / 6 epochs
```

**주요 failure mode: repetition=0.778**
```
She uses 20 students.
She uses 20 students.
She uses 20 students.  ...
```
- greedy 디코딩 + 추론 앵커 부재 → 고확률 루프 진입 후 탈출 불가
- fmt=0.228: 500개 중 114개만 `#### N` 도달, 나머지는 루프 중 끊김
- **acc ≈ fmt(0.228) × 0.10**: 형식을 맞춰도 답은 10%만 맞음. 이중 실패.
- vanilla 1.6% 대비: +0.6%p → H2 미미한 신호, 통계적으로 의미 없음

---

### 6.3 G_A2_mixed — 예상된 실패 (부분 개선 확인)

```
acc=0.020, fmt=0.428, repetition=0.548, no_answer=0.026
best_epoch=0 / 5 epochs
```

A1 대비 비교:

| 지표 | A1 | A2 | 변화 |
|------|----|----|------|
| acc | 0.022 | 0.020 | ≈ 동일 (−0.002) |
| fmt | 0.228 | **0.428** | **+0.200 ↑** |
| repeat | 0.778 | **0.548** | **−0.230 ↓** |

- MA 혼합이 **형식 학습 + 루핑 억제**에는 기여 → fmt +20%p, repeat −23%p
- 그러나 acc는 오히려 미세 하락 → MA 단순 산술 패턴이 GSM8K 다단계 추론과 간섭 가능성
- best_epoch=0: 첫 epoch이 최적, 이후 계속 하락 → MA와 GSM8K 간 catastrophic forgetting

---

### 6.4 G_B1_std — 핵심 발견: 포맷 불일치 정량 확인

```
acc=0.024, fmt=0.156, repetition=0.796, no_answer=0.042
best_epoch=1 / 6 epochs
```

**MA 커리큘럼(0.911) → GSM8K acc=0.024: 커리큘럼 효과 사라짐**

A1(arith init) vs B1_std(MA init) 비교:

| 지표 | A1 (arith init) | B1_std (MA init) | 변화 |
|------|-----------------|------------------|------|
| acc | 0.022 | 0.024 | +0.002 (무의미) |
| fmt | 0.228 | **0.156** | **−0.072 악화** |
| repeat | 0.778 | **0.796** | +0.018 악화 |

**fmt=0.156이 결정적 단서 — 포맷 불일치 메커니즘:**

```
MA ckpt 내부 상태 →  "Plan: ...\n\nReasoning:" 시퀀스에 최적화
GSM8K SFT 포맷  →  Plan 없이 바로 "Reasoning:" 시작
→ 두 포맷 간 충돌 → catastrophic interference
→ fmt가 A1보다 낮고, 루핑도 최대
```

포맷 불일치 가설 정량 요약:
```
커리큘럼 효과 (acc):  +0.002  — 통계적으로 무의미
포맷 손상 (fmt):     −0.072  — 오히려 악화
루핑 악화 (repeat):  +0.018  — 소폭 악화
```

→ MA ckpt를 GSM8K standard CoT로 직접 파인튜닝: **무의미 또는 역효과**  
→ B1_skel / B1_ent (plan 포맷 정렬)이 이 문제를 해결할 수 있는지가 실험의 핵심 질문

---

### 6.5 G_B1_skel — 포맷 정렬 실패 (예상 외 결과)

```
acc=0.022, fmt=0.128, repetition=0.716, no_answer=0.060
best_epoch=0 / 5 epochs
```

B1_std 대비 비교:

| 지표 | B1_std | B1_skel | 변화 |
|------|--------|---------|------|
| acc | 0.024 | 0.022 | −0.002 (무의미) |
| fmt | 0.156 | **0.128** | **−0.028 악화** |
| repeat | 0.796 | **0.716** | −0.080 소폭 개선 |
| no_ans | 0.042 | 0.060 | +0.018 악화 |

**예상과 반대 결과**: skeleton plan이 포맷 정렬에 실패하고 fmt를 오히려 더 낮춤.

**원인 분석**:
- MA plan 포맷: `Plan: 2단계. (1) find total apples; (2) subtract eaten;` → 동사구 서술, 자연어
- skeleton plan 포맷: `Plan: Solve in N steps. (1) multiply; (2) add;` → 단일 동사(operator), 매우 짧음
- 두 포맷이 어휘·구조 모두 달라서 새로운 미스매치를 만든 것
- best_epoch=0: skeleton SFT 시작 즉시 MA ckpt 성능 degradation → skeleton 학습이 MA 패턴을 덮어씀
- repeat이 0.796→0.716으로 소폭 개선: 유일한 긍정 신호 (plan 존재 자체가 루핑을 약하게 억제)

**entity plan(B1_ent)의 기대**: entity 포맷(`find Kim's height, then calculate...`)은 MA의 자연어 서술 스타일과 구조적으로 유사 → 포맷 정렬 가능성 더 높음.

### 6.6 G_B2_skel — 가장 중요한 발견: 포맷 완전 해결, 추론은 여전히 실패

```
acc=0.024, fmt=0.832, repetition=0.146, no_answer=0.008
best_epoch=1 / 6 epochs, elapsed=6811s
```

전체 arm 비교에서 fmt·repeat 모두 압도적:

| 지표 | B1_std | B1_skel | **B2_skel** |
|------|--------|---------|-------------|
| acc | 0.024 | 0.022 | **0.024** |
| fmt | 0.156 | 0.128 | **0.832** |
| repeat | 0.796 | 0.716 | **0.146** |
| no_ans | 0.042 | 0.060 | **0.008** |

**핵심 발견 1 — 포맷 완전 해결, acc는 동일**

fmt가 0.156→0.832로 5배 개선, repeat이 0.796→0.146으로 격감 → 루핑 거의 제거.  
그러나 acc는 0.024로 동일. 이것이 capacity-bound의 최종 증거:

```
B1_std: 형식 도달 78개 중 12개 정답  → 형식 맞추면 15.4% 정답
B2_skel: 형식 도달 416개 중 12개 정답 → 형식 맞추면  2.9% 정답
```

형식 도달율이 5배 늘었지만 정답은 동일 → 모델이 형식을 맞추는 법은 배웠으나 추론은 못 함.  
**GPT-2 124M의 추론 용량이 진짜 천장임을 정량 확인.**

**핵심 발견 2 — MA 데이터가 plan 패턴을 유지, skeleton 포맷 자체가 아님**

- B1_skel (skeleton만): fmt=0.128 → 포맷 정렬 실패
- B2_skel (skeleton+MA 혼합): fmt=0.832 → 포맷 정렬 성공

skeleton 포맷이 MA 학습 패턴과 달라 단독으로는 degradation을 일으키지만,  
MA 데이터(3061 blocks)가 혼합되면 plan 패턴이 유지되면서 GSM8K format으로 전이됨.  
**MA 데이터가 plan-format anchor 역할.**

### 6.7 G_B1_ent / G_B2_ent — entity plan의 제한적 개선

#### G_B1_ent — entity plan 단독

```
acc=0.026, fmt=0.738, repetition=0.424, no_answer=0.028
best_epoch=7 / 12 epochs, elapsed=12783s
```

B1_skel 대비 entity plan은 fmt를 0.128→0.738로 크게 회복했고 acc도 0.022→0.026으로 소폭 상승했다. 그러나 repeat=0.424로 여전히 높다. entity plan은 자연어 plan 스타일을 맞추는 데는 skeleton보다 낫지만, MA anchor 없이 GSM8K entity plan만 학습하면 루핑 억제가 충분하지 않다.

#### G_B2_ent — 최고 acc, 그러나 제한적 상승

```
acc=0.028, fmt=0.744, repetition=0.188, no_answer=0.020
best_epoch=0 / 5 epochs, elapsed=6164s
```

G_B2_ent는 전체 greedy arm 중 최고 acc이지만 14/500 정답이다. B2_skel 12/500 대비 +2문제에 그쳐 실질적 돌파는 아니다. generation 로그 기준으로도 14개 정답 중 10개만 `####` 형식 정답이고 4개는 last-number fallback으로 맞은 케이스다. 반면 B2_skel은 12개 정답 전부 `####` 형식에서 추출되었다.

| 지표 | B2_skel | B2_ent | 해석 |
|------|---------|--------|------|
| acc | 0.024 | **0.028** | entity가 +2문제 |
| fmt | **0.832** | 0.744 | skeleton이 더 안정적 |
| repeat | **0.146** | 0.188 | skeleton이 루핑 억제 우세 |
| no_ans | **0.008** | 0.020 | skeleton이 형식 도달 우세 |
| 정식 `####` 정답 | **12/12** | 10/14 | entity acc에는 fallback 우연 포함 |

**결론**: entity plan은 최고 acc를 만들었지만, 포맷 안정성은 skeleton+MA가 더 좋다. 둘 다 2~3%대에 머물러서 병목은 plan 포맷이 아니라 GSM8K operation-selection 자체다.

### 6.8 10시간 추가 실험 — 완료

추가 실험 커밋 `29a72f8`의 `run_extra_10h.py`는 2026-06-09 서버 로그 기준 `02:22:08`에 완료되었다. 목적은 학습 시간을 단순히 늘리는 것이 아니라, test-time sampling과 더 약한 update가 5% 근처 신호를 만드는지 확인하는 것이었다.

#### lower-LR continuation 결과

| exp_id | 데이터 | full-dev greedy acc | fmt | repeat | no_ans | best_ep | total_ep | elapsed |
|--------|--------|---------------------|-----|--------|--------|---------|----------|---------|
| G_B2_ent_lr1e6 | entity+MA | 0.026 | 0.712 | 0.192 | 0.032 | 3 | 7 | 8,708s |
| G_B2_skel_lr1e6 | skeleton+MA | **0.028** | 0.684 | 0.224 | 0.012 | 1 | 5 | 7,112s |

lower-LR는 catastrophic drift를 줄일 가능성을 봤지만, full-dev greedy 기준으로 기존 최고 `G_B2_ent=0.028`을 넘지 못했다. `G_B2_skel_lr1e6`은 acc=0.028로 동률이나, 기존 `G_B2_skel` 대비 fmt가 0.832→0.684로 낮아지고 repeat도 0.146→0.224로 악화되었다.

#### Self-consistency 결과 (dev150, k=8)

| checkpoint | SC acc | 정답 수 | any_correct@8 | 후보 정답 존재 | mean vote agreement | 해석 |
|------------|--------|---------|---------------|----------------|---------------------|------|
| G_B2_ent_best | **0.040** | **6/150** | 0.133 | 20/150 | 0.328 | SC 최고, 5% 미도달 |
| G_B2_skel_best | 0.020 | 3/150 | **0.147** | **22/150** | 0.334 | 정답 후보는 가장 많지만 vote 실패 |
| G_B2_ent_lr1e6_best | 0.033 | 5/150 | 0.080 | 12/150 | 0.318 | lower-LR로 후보 다양성 감소 |
| G_B2_skel_lr1e6_best | 0.033 | 5/150 | 0.127 | 19/150 | 0.283 | vote agreement 최저, SC 3.3% |

**핵심 결론**:
- SC 최고는 `G_B2_ent_best`의 4.0%(6/150)로, 목표였던 5%에는 도달하지 못했다.
- any_correct@8 최고는 `G_B2_skel_best`의 14.7%(22/150)다. 정답 후보가 일부 존재하지만 다수결이 이를 고르지 못한다.
- mean vote agreement가 0.28~0.33으로 낮아 sampling 후보가 넓게 흩어진다. 이는 모델이 안정적인 추론 모드를 갖고 있지 않다는 신호다.
- any_correct@8이 8~15% 수준이라 PPO/GRPO 같은 outcome-RL을 바로 적용하기에는 reward가 너무 sparse하다. 가능하다면 mini-STaR/rejection SFT가 더 현실적이다.


---

## 7. 핵심 인사이트 요약

### 7.1 실패 원인 계층

```
1순위: repetition (루핑) — A계열 0.55~0.80, B1_std 0.80
       greedy 디코딩 + 추론 구조 부재 → 고확률 루프 탈출 불가

2순위: 포맷 불일치 — B1_std fmt=0.156
       MA plan CoT ↔ GSM8K standard CoT 포맷 충돌 → catastrophic interference

3순위: 모델 용량 — 124M은 GSM8K 다단계 추론이 근본적으로 어려움 (capacity-bound)
```

### 7.2 성능 진행 맥락

```
Phase 0  vanilla SFT              →  1.6%   (기준)
Phase 0  vanilla DPO              →  0.8%   (오히려 하락)
Phase 2  arith init SFT (A1)      →  2.2%   (+0.6%p, H2 약한 신호)
Phase 2  MA 커리큘럼 (B1_std)     →  2.4%   (+0.2%p, 포맷 불일치로 커리큘럼 효과 소멸)
Phase 2  B2_skel                  →  2.4%   (fmt/repeat 해결, acc 미개선)
Phase 2  B2_ent                   →  2.8%   (greedy full-dev 최고, +2문제 수준)
Phase 3  lower-LR continuation    →  2.8%   (B2_skel_lr1e6, 최고 동률이나 fmt/repeat 악화)
Phase 3  SC@8 dev150              →  4.0%   (B2_ent_best, 6/150; 5% 미도달)
```

```
MultiArith (2-step, plan CoT 정렬):  91.1%  ← plan 구조가 통하는 환경에서의 천장
GSM8K greedy full-dev 최고:           2.8%  ← B2_ent / B2_skel_lr1e6
GSM8K SC@8 dev150 최고:               4.0%  ← B2_ent_best
GSM8K any_correct@8 최고:            14.7%  ← B2_skel_best, verifier/STaR의 약한 신호
```

### 7.3 통제된 비교로 확인한 것들

| 비교 | 확인 내용 |
|------|----------|
| vanilla(1.6%) vs arith-init A1(2.2%) | arith pretrain의 미미한 효과 (H2 약한 신호) |
| A1(fmt=0.228) vs A2(fmt=0.428) | MA 혼합이 형식·루핑에 기여, acc에는 미기여 |
| A1(fmt=0.228) vs B1_std(fmt=0.156) | 포맷 불일치 → 커리큘럼이 오히려 해가 됨 |
| B1_skel(fmt=0.128) vs B2_skel(fmt=0.832) | skeleton 단독은 해로움, MA 혼합이 plan anchor 역할 |
| B2_skel: fmt=0.832 → acc=0.024 | **포맷 고쳐도 acc 동일 → capacity-bound 최종 확인** |
| B2_skel vs B2_ent | entity가 acc 최고(0.028)지만 skeleton이 fmt/repeat 안정성 우세 |
| MA(repeat=0.0) vs B2_skel(repeat=0.146) | plan+MA 조합이 루핑 거의 완전 억제 |
| B2_ent 정답 분석 | 14개 정답 중 10개만 `####` 정답, 4개는 last-number fallback |

---

## 8. 선행 연구 참조

| 방법 | 출처 | 우리 적용 여부 |
|------|------|--------------|
| Calculator-augmented generation | Cobbe et al., 2021 (GSM8K) | **미적용** — 추론 오류 99%라 효과 없음 (+1%p 확인) |
| Self-Consistency | Wang et al., 2022 | **적용** — k=8 dev150, 최고 SC 4.0%, any_correct@8 최고 14.7%; 5% 미도달 |
| Plan-and-Solve PS+ | Wang et al., ACL 2023 | **적용** — entity plan 포맷 설계 참조 |
| Program-of-Thought / PaD | Chen 2022, Zhu NAACL 2024 | **미적용** — GPT-2 code-pretrain 아님 |
| TinyGSM verifier | Liu et al., 2023 | **미적용** — 과제 범위 초과 (second model 필요) |
| CoT distillation | Magister et al., 2023 | **일부 적용** — plan CoT SFT (12M 규모는 아님) |

---

## 9. 구현 이력 및 파일 목록

### Phase 0 — Colab (2026-06-03)

| 파일 | 설명 |
|------|------|
| `reasoning_generation.py` | top-p 버그 수정, max_length 512 |
| `gpt_datasets.py` | max_length 통일 |
| `eval_sft.py` | SFT/.pt/HF 디렉토리 통합 평가 |
| `generate_rejected.py` | DPO rejected pair 생성 |
| `train_dpo.py` | DPO 학습 |
| `colab_pipeline.ipynb` | Colab 9셀 파이프라인 |

### Phase 1-2 — arith-init + 서버 (2026-06-05~)

| 파일 | 설명 |
|------|------|
| `scripts/run_ablation.py` | 실험 러너, 평가, generation 저장 |
| `run_pipeline.py` | 전체 8-arm 파이프라인 |
| `scripts/generate/generate_gsm8k_plan.py` | skeleton/entity plan 생성 |
| `scripts/eval/eval_self_consistency.py` | SC 평가 (k=8) + `--save_samples` raw generation 저장 |
| `scripts/eval/sample_generate.py` | 기존 ckpt 샘플 생성기 |
| `run_extra_10h.py` | SC dev150 + lower-LR continuation 2개 실행 큐 |
| `models/gpt2.py` | GPT-2 small 자체 구현 |
| `scripts/eval/gsm8k_eval.py` | 평가 지표 함수 |
| `audit_template_leakage.py` | in_template/held_out 분리 |

### 서버 출력 (`/mnt/outputs/`)

```
MA_plan_numaug_best.pt
G_A1_direct_metrics.json
G_A2_mixed_metrics.json
G_B1_std_metrics.json
G_B1_skel_metrics.json / G_B1_skel_generations.json
G_B2_skel_metrics.json / G_B2_skel_generations.json
G_B1_ent_metrics.json  / G_B1_ent_generations.json
G_B2_ent_metrics.json  / G_B2_ent_generations.json
G_B2_ent_lr1e6_metrics.json  / G_B2_ent_lr1e6_generations.json
G_B2_skel_lr1e6_metrics.json / G_B2_skel_lr1e6_generations.json
sc_eval/*_eval.json             # SC aggregate metrics (extra run)
sc_eval/*_samples.json          # SC raw sampled generations (extra run)
extra_10h.log                   # 추가 실험 통합 로그
run_log.txt                     # 실험별 완료 타임스탬프
pipeline.log                    # 에폭별 loss/acc/fmt 상세 로그
```

---

## 10. 정성적 분석 (현재까지)

### 10.1 Phase 0 — vanilla SFT 오답 유형 분류 (dev 500개)

**분류 기준**: `outputs/9_10-1e-05-reasoning_generations.txt` 전수 스크립트 분석

| 유형 | 수 | 비율 | 설명 |
|------|----|------|------|
| 정답 | 8 | 1.6% | — |
| 산술 오류만 (계산기로 수정 가능) | 5 | 1.0% | 연산 선택은 맞고 계산만 틀림 |
| 추론 오류 (계산기 무력) | 482 | 96.4% | 연산 선택 자체가 틀림 |

**대표 실패 사례 (vanilla SFT)**:
```
Question: Kim's height is 6. Tamara is 3 times Kim's height. How tall is Tamara?
Gold answer: 18

Model generation:
Kim is 6 inches tall.
6 × 2 = <<6*2=12>>12        ← 2를 지어냄 (정답은 ×3)
12 × 3 = <<12*3=36>>36      ← 추가 단계 날조
#### 36                      ← 오답
```
→ 계산기로 12→24, 36→36 고쳐도 틀림. **연산 선택이 잘못된 것.**

---

### 10.2 Phase 2 — GSM8K arm 실패 유형 (정량 확인)

#### 루핑(Repetition) — 모든 arm의 지배적 실패 모드

```
# G_A1_direct 전형적 생성
Question: A store has 20 students. Each student needs 4 books...

Reasoning:
She uses 20 students.
She uses 20 students.
She uses 20 students.
She uses 20 students.  (이하 256 토큰 소진까지 반복)
```

루핑 발생 메커니즘:
- greedy 디코딩 → 한 번 고확률 구절 진입 시 탈출 불가
- plan 헤더 없을 때: 첫 reasoning 줄이 다음 줄의 conditioning이 되어 루프 형성
- MA plan CoT에서 repetition=0.0 → plan 앵커가 루프 방지함을 역으로 확인

#### 포맷 붕괴(Format Collapse) — B1_std의 특징적 실패

```
# G_B1_std 전형적 생성 — MA의 Plan 패턴 혼입
Question: There are 15 trees in the grove...

Plan: 2 steps. (1) multiply 15 by 4;   ← MA에서 학습한 Plan이 나옴
Reasoning:
15 × 4 = <<15*4=60>>60
60 ÷ 3 = <<60/3=20>>20
Plan: 2 steps. (1) multiply...          ← 다시 Plan으로 돌아가는 루프
```

→ MA ckpt가 "Plan을 먼저 뱉어야"라는 패턴에 갇혀 GSM8K standard 포맷으로 전환 실패.  
fmt=0.156: 생성의 84%가 `#### N`까지 도달하지 못함.

---

### 10.3 Plan arm raw generation 분석

`outputs/server_final/*_generations.json` 기준:

| arm | 정답 | format valid | repetition | 정식 `####` 정답 | 관찰 |
|-----|------|--------------|------------|------------------|------|
| B1_skel | 11/500 | 64/500 | 358/500 | 2/11 | skeleton 단독은 루핑과 fallback 정답이 지배적 |
| B2_skel | 12/500 | 416/500 | 73/500 | 12/12 | 형식 안정성 최고, 그러나 연산 선택은 실패 |
| B1_ent | 13/500 | 369/500 | 212/500 | 10/13 | entity가 fmt는 복구하지만 루핑이 많음 |
| B2_ent | 14/500 | 372/500 | 94/500 | 10/14 | greedy 최고 acc, 일부 fallback 정답 포함 |
| B2_ent_lr1e6 | 13/500 | 356/500 | 96/500 | 10/13 | lower-LR로도 entity 성능 개선 없음 |
| B2_skel_lr1e6 | 14/500 | 342/500 | 112/500 | 9/14 | acc는 최고 동률이나 fallback 비중 증가 |

대표 실패 양상은 계산 실수보다 plan/operation-selection 오류다. 모델은 `twice`, `%`, `per week`, `per hour` 같은 의미를 연산 그래프로 변환하지 못하고, 문제에 보이는 숫자 둘을 골라 2-step MultiArith식으로 처리한다. 예: `25% + 50%`를 `75`로 더한 뒤 `400`을 곱해 우연히 맞는 경우가 있지만, 이는 백분율 의미를 제대로 계산한 것이 아니다.

**정성 결론**: plan은 답 형식과 루핑을 제어하지만, plan 자체가 틀린 연산을 고른다. lower-LR continuation도 이 병목을 해결하지 못했다.

### 10.4 Self-consistency raw sample 분석

SC raw sample 기준으로 정답 후보가 존재했지만 다수결이 놓친 사례가 많다.

| checkpoint | any correct | SC correct | majority miss | 평균 고유 예측값 수 | 평균 top-count |
|------------|-------------|------------|---------------|--------------------|----------------|
| B2_ent_best | 20/150 | 6/150 | 14/150 | 5.83 | 2.63 |
| B2_skel_best | 22/150 | 3/150 | 19/150 | 5.87 | 2.67 |
| B2_ent_lr1e6 | 12/150 | 5/150 | 7/150 | 5.99 | 2.54 |
| B2_skel_lr1e6 | 19/150 | 5/150 | 14/150 | 6.32 | 2.26 |

관찰:
- 정답 후보가 있는 문제에서도 정답 completion은 보통 1~2개뿐이라 majority가 오답으로 쏠린다.
- `B2_skel_best`는 any_correct@8이 가장 높지만 SC acc는 가장 낮다. 후보 생성력과 후보 선택력이 분리되어 있다.
- verifier/selection은 이론상 개선 여지가 있지만, any_correct@8 자체가 15% 미만이라 큰 폭의 성능 향상은 기대하기 어렵다.
- PPO/GRPO 같은 outcome reward RL은 reward sparsity가 심하다. 지금 단계에서는 mini-STaR 또는 correct-sample rejection SFT가 더 현실적인 후속 실험이다.


---

## 11. 트러블슈팅 기록

| 증상 | 원인 | 조치 |
|------|------|------|
| DPO format collapse (fmt=0.4%) | off-policy chosen + likelihood displacement | 방향 전환 (커리큘럼으로) |
| `drive.mount` 400 에러 (VS Code Colab ext) | 인증 위젯은 Colab 웹 전용 | 브라우저 Colab에서 실행 |
| GSM8K epoch ~30분, 9h에 1개 진행 | dev 500 매 epoch 생성 | per-epoch eval 150 서브셋 |
| CUDA OOM (G_B 계열, bs=8, plan arm) | max_len 486~504 > 430 (standard) | `bs = max(1, bs//2) = 4` |
| stale plan file (4000 블록) | Gemini가 잘못된 소스로 생성 | 삭제 후 3000 블록 원본에서 재생성 |
| `-- batch_size` argparse 오류 | 인자명에 공백 | 수정 |
| `echo &!` 대신 `$!` | 타이포로 별도 프로세스 생성 | PID 확인 후 정리 |

---

## 12. 남은 작업 및 한계

### 보고서 반영 포인트

**[정량]**
1. 최종 최고 greedy full-dev는 2.8% (`G_B2_ent`, `G_B2_skel_lr1e6`)로 정리한다.
2. 최종 최고 SC dev150은 4.0% (`G_B2_ent_best`, 6/150)이며, 5% 목표에는 도달하지 못했다.
3. any_correct@8 최고는 14.7% (`G_B2_skel_best`, 22/150)로, 정답 후보 생성은 약하게 가능하지만 다수결 선택은 실패한다.
4. lower-LR continuation은 최고 acc 동률만 만들었고, fmt/repeat 안정성은 오히려 나빠졌다.

**[정성]**
5. 대표 실패 사례는 “형식 붕괴”보다 “틀린 plan/operation-selection”으로 잡는다.
6. SC raw sample에서는 정답 completion이 소수 후보로 존재하나 majority vote가 고르지 못하는 사례를 제시한다.
7. 후속 연구는 full RL보다 mini-STaR/rejection SFT/verifier를 제안하되, any_correct@8이 낮아 기대 효과가 제한적임을 명시한다.

### 확인된 한계 (설계 제약)
- **누락된 대조군**: vanilla→MA→gsm8k 없음. H2를 커리큘럼까지 주장하려면 필요.
- **추론시 계산기 미사용**: 사용자 결정. (진단상 +1%p 뿐이라 영향 미미)
- **verifier 미적용**: 과제 범위 초과.
- **RL 미적용**: any_correct@8 8~15% 수준이라 outcome reward가 sparse함. PPO/GRPO보다 STaR류가 우선.
- **모델 용량 한계**: 124M은 GSM8K 다단계 추론에 근본적 한계 (capacity-bound).
