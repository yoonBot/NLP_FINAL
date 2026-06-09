# 실험 전체 정리

## 1. 환경

| 항목 | 값 |
|------|-----|
| 서버 | Sogang AIPub GPU 클러스터 (`163.239.15.20:30205`) |
| GPU | A100 MIG 1g.10gb (VRAM 10GB) |
| 모델 | GPT-2 small (124M) — 자체 구현 (`models/gpt2.py`) |
| Python | 3.9, miniconda (`/mnt/miniconda/bin/`) |
| 영구 스토리지 | `/mnt/` NAS |
| 파이프라인 | `run_pipeline.py`, 추가 실험 `run_extra_10h.py` |
| seed | 11711 |

---

## 2. 데이터셋

### MultiArith

| 파일 | 블록 수 | 설명 |
|------|--------|------|
| `multiarith_sft_train_base.txt` | 510 | 원본 MultiArith SFT |
| `multiarith_sft_train_aug.txt` | 3,060 | 숫자치환 증강 |
| `multiarith_sft_train_plan_aug.txt` | 3,061 | 숫자치환 + plan-then-solve CoT |
| `multiarith_dev.jsonl` | 90 | 서버 최종 평가 기준 dev split |

### GSM8K

| 파일 | 블록 수 | 설명 |
|------|--------|------|
| `gsm8k_sft_train.txt` | 3,000 | GSM8K train subset |
| `gsm8k_sft_train_plan_skeleton.txt` | 3,000 | skeleton plan 추가, 숫자 없음 |
| `gsm8k_sft_train_plan_entity.txt` | 3,000 | entity plan 추가, 숫자 없음 |
| `gsm8k_plan_skeleton_plus_ma.txt` | 6,061 | skeleton + MA plan aug 혼합 |
| `gsm8k_plan_entity_plus_ma.txt` | 6,061 | entity + MA plan aug 혼합 |
| `gsm8k_dev.jsonl` | 500 | 평가 전용 dev |

---

## 3. 실험 설계

```text
arith pretrain
├─ Stage 1: MultiArith plan-CoT SFT → MA_plan_numaug
│  └─ Stage 2-B: GSM8K curriculum arms
│     ├─ G_B1_std       MA ckpt → GSM8K standard
│     ├─ G_B1_skel      MA ckpt → GSM8K skeleton plan
│     ├─ G_B2_skel      MA ckpt → skeleton+MA mixed
│     ├─ G_B1_ent       MA ckpt → GSM8K entity plan
│     └─ G_B2_ent       MA ckpt → entity+MA mixed
└─ Stage 2-A: direct baseline arms
   ├─ G_A1_direct       arith ckpt → GSM8K standard
   └─ G_A2_mixed        arith ckpt → GSM8K+MA mixed
```

추가 10시간 실험:

```text
G_B2_ent_best, G_B2_skel_best SC@8 dev150
G_B2_ent_lr1e6, G_B2_skel_lr1e6 lower-LR continuation
lower-LR checkpoints SC@8 dev150
```

---

## 4. 최종 Greedy 결과

| exp_id | init | 데이터 | acc | 정답 수 | fmt | repeat | no_ans | best_ep | total_ep |
|--------|------|--------|-----|---------|-----|--------|--------|---------|----------|
| MA_plan_numaug | arith | MA plan aug | **0.911** | 82/90 | **1.000** | **0.000** | **0.000** | 34 | 40 |
| G_A1_direct | arith | GSM8K | 0.022 | 11/500 | 0.228 | 0.778 | 0.026 | 1 | 6 |
| G_A2_mixed | arith | GSM8K+MA | 0.020 | 10/500 | 0.428 | 0.548 | 0.026 | 0 | 5 |
| G_B1_std | MA ckpt | GSM8K standard | 0.024 | 12/500 | 0.156 | 0.796 | 0.042 | 1 | 6 |
| G_B1_skel | MA ckpt | GSM8K skeleton | 0.022 | 11/500 | 0.128 | 0.716 | 0.060 | 0 | 5 |
| G_B2_skel | MA ckpt | skeleton+MA | 0.024 | 12/500 | **0.832** | **0.146** | **0.008** | 1 | 6 |
| G_B1_ent | MA ckpt | GSM8K entity | 0.026 | 13/500 | 0.738 | 0.424 | 0.028 | 7 | 12 |
| G_B2_ent | MA ckpt | entity+MA | **0.028** | **14/500** | 0.744 | 0.188 | 0.020 | 0 | 5 |
| G_B2_ent_lr1e6 | MA ckpt | entity+MA, lr=1e-6 | 0.026 | 13/500 | 0.712 | 0.192 | 0.032 | 3 | 7 |
| G_B2_skel_lr1e6 | MA ckpt | skeleton+MA, lr=1e-6 | **0.028** | **14/500** | 0.684 | 0.224 | 0.012 | 1 | 5 |

결론: greedy full-dev 최고는 2.8%로, `G_B2_ent`와 `G_B2_skel_lr1e6`이 동률이다. 그러나 lower-LR skeleton은 format/repetition이 기존 `G_B2_skel`보다 나빠져 실질적 개선으로 보기 어렵다.

---

## 5. Self-Consistency 결과

설정: dev150, k=8, temperature=0.8, top_p=0.95, `--save_samples` raw generation 저장.

| checkpoint | SC acc | 정답 수 | any_correct@8 | 후보 정답 존재 | mean vote agreement | raw samples |
|------------|--------|---------|---------------|----------------|---------------------|-------------|
| G_B2_ent_best | **0.040** | **6/150** | 0.133 | 20/150 | 0.328 | 있음 |
| G_B2_skel_best | 0.020 | 3/150 | **0.147** | **22/150** | 0.334 | 있음 |
| G_B2_ent_lr1e6_best | 0.033 | 5/150 | 0.080 | 12/150 | 0.318 | 있음 |
| G_B2_skel_lr1e6_best | 0.033 | 5/150 | 0.127 | 19/150 | 0.283 | 있음 |

해석:
- SC 최고는 `G_B2_ent_best`의 4.0%로 5%에는 도달하지 못했다.
- any_correct@8 최고는 `G_B2_skel_best`의 14.7%다. 즉 정답 후보가 가끔 나오지만 다수결이 이를 고르지 못한다.
- mean vote agreement가 낮아 후보 예측이 넓게 흩어진다. 안정적인 reasoning 모드가 형성되지 않았다.

---

## 6. Raw Generation 분석

| arm | 정답 | format valid | repetition | 정식 `####` 정답 | 관찰 |
|-----|------|--------------|------------|------------------|------|
| B1_skel | 11/500 | 64/500 | 358/500 | 2/11 | skeleton 단독은 루핑과 fallback 정답이 지배적 |
| B2_skel | 12/500 | 416/500 | 73/500 | 12/12 | 형식 안정성 최고, 추론은 실패 |
| B1_ent | 13/500 | 369/500 | 212/500 | 10/13 | entity가 fmt는 복구하지만 루핑이 많음 |
| B2_ent | 14/500 | 372/500 | 94/500 | 10/14 | greedy 최고 acc, 일부 fallback 정답 포함 |
| B2_ent_lr1e6 | 13/500 | 356/500 | 96/500 | 10/13 | lower-LR 개선 없음 |
| B2_skel_lr1e6 | 14/500 | 342/500 | 112/500 | 9/14 | acc 동률이나 fallback 비중 증가 |

SC raw sample 집계:

| checkpoint | any correct | SC correct | majority miss | 평균 고유 예측값 수 | 평균 top-count |
|------------|-------------|------------|---------------|--------------------|----------------|
| B2_ent_best | 20/150 | 6/150 | 14/150 | 5.83 | 2.63 |
| B2_skel_best | 22/150 | 3/150 | 19/150 | 5.87 | 2.67 |
| B2_ent_lr1e6 | 12/150 | 5/150 | 7/150 | 5.99 | 2.54 |
| B2_skel_lr1e6 | 19/150 | 5/150 | 14/150 | 6.32 | 2.26 |

정성적 실패 패턴:
- `twice`, `%`, `per week`, `remaining`, `less than` 같은 의미를 안정적인 연산 그래프로 바꾸지 못한다.
- Plan 헤더가 있어도 plan 자체가 틀린 연산을 고른다.
- 정답 후보가 sampling 중 나와도 소수 후보라 majority vote에서 밀린다.

### 계산 trace 신뢰성

`<<expr=result>>` 태그 내부를 재계산 가능한 경우만 검사했을 때도 오류가 많았다.

| arm | 재계산 가능 태그 | 잘못된 태그 | 잘못된 태그 포함 row | 정답 row 중 태그 오류 |
|-----|----------------:|------------:|---------------------:|---------------------:|
| B1_ent | 398 | 185 | 123/500 | 2/13 |
| B1_skel | 215 | 90 | 31/500 | 1/11 |
| B2_ent | 771 | 298 | 204/500 | 4/14 |
| B2_ent_lr1e6 | 719 | 240 | 182/500 | 5/13 |
| B2_skel | 849 | 333 | 227/500 | 6/12 |
| B2_skel_lr1e6 | 727 | 212 | 153/500 | 1/14 |

예시는 `25+30=65`, `200*0=200`, `75*400=750`, `6+Razel=12` 같은 형태다. 즉 모델은 `####` 형식을 학습했지만 계산 trace 자체를 검증 가능한 프로그램처럼 생성하지는 못했다. 맞은 문제도 정상 reasoning이라기보다 숫자 패턴/우연/fallback이 섞여 있다.

---

## 7. 핵심 결론

1. MultiArith 2-step 문제에서는 plan-CoT 커리큘럼이 성공했다: 91.1%, held-out 86.0%.
2. GSM8K에서는 형식 안정화와 정답률 개선이 분리된다. `G_B2_skel`은 fmt 83.2%, repeat 14.6%까지 개선했지만 acc는 2.4%다.
3. 최고 greedy acc는 2.8%로 5%에 도달하지 못했다.
4. SC도 최고 4.0%에 그쳤다. any_correct@8은 최대 14.7%라 verifier/STaR 신호는 있으나 약하다.
5. 병목은 산술 계산보다 operation-selection이다. 이는 Phase 0 계산기 시뮬레이션과 Phase 2 plan-arm 결과가 같은 방향으로 지지한다.
6. GPT-2 small 124M + GSM8K 조합에서는 capacity-bound 결론이 가장 방어 가능하다.

---

## 8. 보고서용 문장

- “Plan-then-solve CoT는 작은 모델에서도 형식 안정화와 반복 억제에는 효과적이었다.”
- “그러나 GSM8K 정확도는 거의 오르지 않았고, 형식 도달률이 5배 오른 `B2_skel`에서도 정답 수는 12/500에 머물렀다.”
- “Self-consistency는 후보 다양성을 늘렸지만, 정답 후보가 majority가 될 만큼 자주 생성되지 않았다.”
- “따라서 본 실험의 주요 기여는 GPT-2 small에서 GSM8K 실패가 단순 포맷 문제가 아니라 operation-selection/capacity bottleneck임을 여러 통제 실험으로 분리해 보인 것이다.”

---

## 9. 산출물 위치

```text
outputs/server_final/*_metrics.json
outputs/server_final/*_generations.json
outputs/server_final/sc_eval/*_eval.json
outputs/server_final/sc_eval/*_samples.json
outputs/server_final/extra_10h.log
outputs/server_final/run_log.txt
outputs/server_final/pipeline.log
```
