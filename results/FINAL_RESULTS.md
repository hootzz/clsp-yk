# FINAL RESULTS — Full Result Dashboard

작성일: 2026-08-03  
최종 후보: **B2 Target-routed** — Valence는 frozen Context-only, Arousal과 Cognitive load는 target-specific direct Context+PPG, 개인화는 선택적 EB output offset.

> 이 파일은 결과 확인용 상세 정본이다. 실제 논문에 붙일 압축 표와 문단은 별도 파일 `PAPER_RESULTS_SECTION.tex`에 둔다.

## 0. 평가 조건과 baseline

| 표기 | 입력·구조 | 역할 |
|---|---|---|
| **PPG-only** | frozen PaPaGEI-P embedding만 사용 | 생리 modality 단독 진단 기준선 |
| **B0 Context-only** | title-level Context만 사용 | **B1/B2의 공식 matched population baseline** |
| **B1 Uniform Context+PPG** | V/A/C 모두 residual-free direct Context+PPG | 모든 타깃에 PPG를 넣는 균일 융합 대조군 |
| **B2 Target-routed** | V=고정 Context-only, A/C=target-specific direct Context+PPG | 최종 아키텍처 후보 |
| **M0 Base** | 개인화하지 않은 B2 query prediction | 개인화 비교 기준선 |
| **M2 EB-personalized** | B2 prediction + target-wise shrunk EB offset | K-shot inference-time 개인화 |

### 데이터셋과 평가 역할

| 구분 | 데이터셋 | 타깃 | 참가자 | 역할 |
|---|---|---|---:|---|
| Source held-out | CASE | V/A | 11 | participant-disjoint source evaluation |
| Source held-out | EEVR | V/A | 14 | participant-disjoint source evaluation |
| Source held-out | MAUS | C | 8 | participant-disjoint source evaluation |
| Strict external zero-shot | WESAD (E4) | V/A | 15 | unseen affect dataset |
| Strict external zero-shot | CogWear (E4/Galaxy) | C | 18 | unseen objective rest-vs-Stroop dataset |
| External diagnostic | EmoWear (seated/walk) | V/A | 48 | motion/device diagnostic; walk는 seated와 동일 trial SAM 사용 |
| External diagnostic | VRFS (flat-screen/VR) | V/A | 33 | condition-level proxy labels; 개인 SAM과 PPG를 직접 join할 수 없음 |

절대 성능의 CCC는 3개 seed의 mean±population SD이며 Accuracy·BA·Macro-F1은 seed 평균이다. Source와 affect 데이터의 분류 임계값은 0.625, CogWear는 0.5다. Paired effect는 참가자 단위 bootstrap mean과 95% CI(2,000 resamples)다. 굵은 절대값은 동일 데이터셋·타깃에서 가장 높은 값이며, 굵은 차이값은 CI가 0을 제외한다.

---

## 1. Source held-out — 공식 통합 결과

V/A는 CASE와 EEVR 평가 행을 합친 값이고, C는 MAUS 값이다. `known-content, unseen-participant` 평가이며 완전한 외부 zero-shot은 아니다.

| Evaluation data | Target | Condition | CCC mean±SD | Accuracy | BA | Macro-F1 |
|---|---|---|---:|---:|---:|---:|
| CASE+EEVR | Valence | PPG-only | 0.076±0.010 | 0.599 | 0.514 | 0.484 |
|  |  | B0 Context-only | 0.509±0.029 | **0.704** | **0.686** | **0.671** |
|  |  | B1 Uniform Context+PPG | **0.532±0.071** | 0.688 | 0.656 | 0.653 |
|  |  | **B2 Target-routed** | 0.509±0.029 | **0.704** | **0.686** | **0.671** |
| CASE+EEVR | Arousal | PPG-only | 0.063±0.080 | 0.636 | 0.538 | 0.536 |
|  |  | B0 Context-only | **0.501±0.024** | **0.746** | 0.708 | 0.687 |
|  |  | B1 Uniform Context+PPG | 0.440±0.033 | 0.705 | 0.672 | 0.639 |
|  |  | **B2 Target-routed** | 0.456±0.038 | 0.743 | **0.720** | **0.702** |
| MAUS | Cognitive load | PPG-only | 0.151±0.140 | **0.852†** | 0.500 | 0.459 |
|  |  | B0 Context-only | **0.647±0.093** | 0.778 | **0.850** | **0.680** |
|  |  | B1 Uniform Context+PPG | 0.485±0.166 | 0.778 | 0.605 | 0.517 |
|  |  | **B2 Target-routed** | 0.492±0.207 | 0.815 | 0.782 | 0.624 |

† PPG-only C의 Accuracy 0.852는 BA 0.500과 함께 나타난 다수 클래스 예측의 영향이므로 성능 우위로 해석하지 않는다.

### 1.1 Source 데이터셋별 절대 성능

| Dataset·Target | Condition | CCC mean±SD | Accuracy | BA | Macro-F1 |
|---|---|---:|---:|---:|---:|
| CASE · A | PPG-only | 0.037±0.028 | 0.682 | 0.547 | 0.545 |
|  | B0 Context-only | **0.740±0.023** | **0.853** | **0.830** | **0.806** |
|  | B1 Uniform Context+PPG | 0.652±0.039 | 0.820 | 0.794 | 0.767 |
|  | **B2 Target-routed** | 0.634±0.054 | 0.827 | 0.810 | 0.777 |
| CASE · V | PPG-only | 0.065±0.026 | 0.643 | 0.513 | 0.491 |
|  | B0 Context-only | 0.649±0.043 | **0.755** | **0.736** | **0.711** |
|  | B1 Uniform Context+PPG | **0.657±0.040** | 0.738 | 0.679 | 0.675 |
|  | **B2 Target-routed** | 0.649±0.043 | **0.755** | **0.736** | **0.711** |
| EEVR · A | PPG-only | 0.090±0.145 | 0.590 | 0.529 | 0.528 |
|  | B0 Context-only | 0.262±0.067 | 0.639 | 0.587 | 0.569 |
|  | B1 Uniform Context+PPG | 0.228±0.096 | 0.590 | 0.550 | 0.510 |
|  | **B2 Target-routed** | **0.279±0.130** | **0.660** | **0.631** | **0.627** |
| EEVR · V | PPG-only | 0.088±0.028 | 0.556 | 0.515 | 0.478 |
|  | B0 Context-only | 0.370±0.037 | **0.653** | **0.635** | **0.632** |
|  | B1 Uniform Context+PPG | **0.407±0.114** | 0.639 | 0.634 | 0.630 |
|  | **B2 Target-routed** | 0.370±0.037 | **0.653** | **0.635** | **0.632** |
| MAUS · C | PPG-only | 0.151±0.140 | **0.852†** | 0.500 | 0.459 |
|  | B0 Context-only | **0.647±0.093** | 0.778 | **0.850** | **0.680** |
|  | B1 Uniform Context+PPG | 0.485±0.166 | 0.778 | 0.605 | 0.517 |
|  | **B2 Target-routed** | 0.492±0.207 | 0.815 | 0.782 | 0.624 |

### 1.2 Source paired effect — B2 minus B0

| Dataset·Target | n | ΔCCC [95% CI] | ΔAccuracy [95% CI] | ΔBA [95% CI] | ΔMacro-F1 [95% CI] |
|---|---:|---:|---:|---:|---:|
| CASE · A | 11 | **−0.103 [−0.140, −0.073]** | **−0.030 [−0.057, −0.005]** | **−0.030 [−0.059, −0.006]** | **−0.036 [−0.062, −0.009]** |
| CASE · V | 11 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| EEVR · A | 14 | **−0.065 [−0.123, −0.009]** | +0.022 [−0.067, 0.103] | +0.012 [−0.051, 0.078] | +0.007 [−0.092, 0.098] |
| EEVR · V | 14 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| MAUS · C | 8 | −0.142 [−0.327, 0.011] | 0.000 [−0.146, 0.146] | −0.060 [−0.229, 0.073] | −0.090 [−0.302, 0.075] |

Source에서는 B2가 B0의 A/C CCC를 이기지 못했다. Valence의 0 차이는 성능 추정이 아니라 B2가 B0의 frozen Valence 경로를 그대로 사용한다는 구조적 불변식이다.

---

## 2. Strict external zero-shot — WESAD·CogWear

모든 checkpoint는 frozen이며 외부 라벨을 이용한 업데이트·threshold 선택·checkpoint 선택·fusion 선택이 없다.

| Dataset·Target | Condition | CCC mean±SD | Accuracy | BA | Macro-F1 |
|---|---|---:|---:|---:|---:|
| WESAD E4 · A | PPG-only | 0.100±0.058 | **0.750** | 0.507 | 0.472 |
|  | B0 Context-only | 0.048±0.061 | 0.711 | 0.463 | 0.424 |
|  | B1 Uniform Context+PPG | 0.113±0.070 | 0.683 | **0.557** | **0.498** |
|  | **B2 Target-routed** | **0.136±0.101** | **0.750** | 0.497 | 0.461 |
| WESAD E4 · V | PPG-only | **0.070±0.049** | **0.606** | **0.539** | **0.482** |
|  | B0 Context-only | 0.009±0.106 | 0.339 | 0.443 | 0.249 |
|  | B1 Uniform Context+PPG | −0.100±0.265 | 0.350 | 0.464 | 0.340 |
|  | **B2 Target-routed** | 0.009±0.106 | 0.339 | 0.443 | 0.249 |
| CogWear E4 · C | PPG-only | **0.169±0.127** | **0.551** | **0.551** | 0.445 |
|  | B0 Context-only | −0.081±0.006 | 0.500 | 0.500 | 0.333 |
|  | B1 Uniform Context+PPG | −0.036±0.196 | 0.435 | 0.435 | 0.349 |
|  | **B2 Target-routed** | 0.087±0.143 | 0.543 | 0.543 | **0.459** |
| CogWear Galaxy · C | PPG-only | **0.125±0.134** | **0.565** | **0.565** | **0.456** |
|  | B0 Context-only | −0.081±0.006 | 0.500 | 0.500 | 0.333 |
|  | B1 Uniform Context+PPG | −0.135±0.164 | 0.391 | 0.391 | 0.324 |
|  | **B2 Target-routed** | 0.006±0.137 | 0.486 | 0.486 | 0.415 |

### 2.1 External paired effect — B2 minus B0

| Dataset·Target | n | ΔCCC [95% CI] | ΔAccuracy [95% CI] | ΔBA [95% CI] | ΔMacro-F1 [95% CI] |
|---|---:|---:|---:|---:|---:|
| WESAD E4 · A | 15 | **+0.061 [0.001, 0.118]** | **+0.039 [0.017, 0.061]** | **+0.039 [0.020, 0.059]** | **+0.020 [0.009, 0.034]** |
| WESAD E4 · V | 15 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| CogWear E4 · C | 18 | **+0.189 [0.126, 0.258]** | **+0.069 [0.014, 0.130]** | **+0.069 [0.009, 0.130]** | **+0.118 [0.048, 0.185]** |
| CogWear Galaxy · C | 18 | **+0.078 [0.034, 0.127]** | −0.019 [−0.074, 0.032] | −0.019 [−0.074, 0.032] | +0.027 [−0.023, 0.080] |

WESAD A와 CogWear E4 C에서는 B2의 paired 증분이 네 지표에서 모두 양수였고 CI가 0을 제외했다. CogWear Galaxy C에서는 CCC 증분만 유의했다. 외부 절대 CCC는 낮으므로 이를 높은 배포 정확도로 표현하지 않는다.

---

## 3. External diagnostic — EmoWear

EmoWear walk는 seated와 동일한 trial-level SAM label을 사용하므로 독립 affect ground truth가 아니라 motion/device diagnostic이다. 또한 개발 과정에서 관찰된 데이터이므로 순수 확증 결과로 사용하지 않는다.

| Domain·Target | Condition | CCC mean±SD | Accuracy | BA | Macro-F1 |
|---|---|---:|---:|---:|---:|
| Seated · A | PPG-only | **0.014±0.032** | 0.611 | 0.500 | **0.479** |
|  | B0 Context-only | −0.002±0.003 | **0.656** | 0.492 | 0.417 |
|  | B1 Uniform Context+PPG | 0.001±0.009 | 0.522 | 0.493 | 0.385 |
|  | **B2 Target-routed** | 0.009±0.027 | 0.543 | **0.501** | 0.478 |
| Seated · V | PPG-only | −0.003±0.015 | 0.486 | **0.510** | **0.477** |
|  | B0 Context-only | −0.002±0.008 | **0.582** | 0.500 | 0.368 |
|  | B1 Uniform Context+PPG | **0.007±0.015** | 0.572 | 0.504 | 0.424 |
|  | **B2 Target-routed** | −0.002±0.008 | **0.582** | 0.500 | 0.368 |
| Walk · A | PPG-only | 0.005±0.002 | 0.602 | 0.505 | **0.492** |
|  | B0 Context-only | 0.000±0.000 | **0.679** | 0.500 | 0.404 |
|  | B1 Uniform Context+PPG | **0.011±0.004** | 0.658 | **0.513** | 0.475 |
|  | **B2 Target-routed** | **0.011±0.004** | 0.624 | 0.505 | 0.486 |
| Walk · V | PPG-only | −0.009±0.006 | 0.508 | 0.495 | **0.489** |
|  | B0 Context-only | **0.000±0.000** | 0.527 | **0.500** | 0.343 |
|  | B1 Uniform Context+PPG | −0.001±0.010 | **0.535** | 0.497 | 0.482 |
|  | **B2 Target-routed** | **0.000±0.000** | 0.527 | **0.500** | 0.343 |

### 3.1 EmoWear paired effect — B2 minus B0

| Domain·Target | n | ΔCCC [95% CI] | ΔAccuracy [95% CI] | ΔBA [95% CI] | ΔMacro-F1 [95% CI] |
|---|---:|---:|---:|---:|---:|
| Seated · A | 48 | +0.007 [−0.010, 0.023] | **−0.109 [−0.147, −0.071]** | −0.007 [−0.037, 0.018] | **+0.039 [0.013, 0.064]** |
| Seated · V | 48 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Walk · A | 48 | +0.014 [−0.008, 0.037] | **−0.054 [−0.078, −0.031]** | −0.008 [−0.029, 0.012] | **+0.064 [0.044, 0.083]** |
| Walk · V | 48 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

EmoWear는 Arousal CCC의 일관된 증분을 보이지 않았고 Accuracy와 Macro-F1의 방향도 충돌했다.

---

## 4. External diagnostic — VRFS

VRFS는 생리 session과 개인별 SAM row를 직접 대응할 수 없다. 아래 두 label policy 모두 condition-level proxy이며 논문의 확증 표가 아니라 민감도 진단으로만 사용한다.

### 4.1 Planned-quadrant labels

| Display·Target | Condition | CCC mean±SD | Accuracy | BA | Macro-F1 |
|---|---|---:|---:|---:|---:|
| Flat-screen · A | PPG-only | 0.039±0.067 | 0.505 | 0.501 | 0.465 |
|  | B0 Context-only | 0.091±0.109 | 0.586 | 0.588 | 0.471 |
|  | B1 Uniform Context+PPG | −0.013±0.055 | 0.505 | 0.502 | 0.381 |
|  | **B2 Target-routed** | **0.113±0.056** | **0.596** | **0.597** | **0.549** |
| Flat-screen · V | PPG-only | −0.070±0.100 | 0.434 | 0.445 | 0.327 |
|  | B0 Context-only | **0.044±0.019** | **0.485** | **0.500** | 0.327 |
|  | B1 Uniform Context+PPG | −0.026±0.084 | 0.455 | 0.467 | **0.339** |
|  | **B2 Target-routed** | **0.044±0.019** | **0.485** | **0.500** | 0.327 |
| VR · A | PPG-only | 0.012±0.067 | 0.485 | 0.481 | 0.414 |
|  | B0 Context-only | **0.123±0.112** | **0.586** | **0.588** | 0.471 |
|  | B1 Uniform Context+PPG | −0.030±0.058 | 0.465 | 0.461 | 0.332 |
|  | **B2 Target-routed** | 0.077±0.046 | 0.556 | 0.558 | **0.489** |
| VR · V | PPG-only | **0.072±0.036** | **0.505** | **0.513** | **0.390** |
|  | B0 Context-only | 0.033±0.032 | 0.485 | 0.500 | 0.327 |
|  | B1 Uniform Context+PPG | 0.004±0.039 | 0.475 | 0.486 | 0.368 |
|  | **B2 Target-routed** | 0.033±0.032 | 0.485 | 0.500 | 0.327 |

### 4.2 SAM game×display group-mean labels

| Display·Target | Condition | CCC mean±SD | Accuracy | BA | Macro-F1 |
|---|---|---:|---:|---:|---:|
| Flat-screen · A | PPG-only | 0.046±0.086 | **0.737** | **0.576** | **0.531** |
|  | B0 Context-only | 0.133±0.150 | 0.727 | 0.500 | 0.421 |
|  | B1 Uniform Context+PPG | 0.048±0.073 | 0.717 | 0.493 | 0.418 |
|  | **B2 Target-routed** | **0.150±0.081** | 0.707 | 0.486 | 0.414 |
| Flat-screen · V | PPG-only | **−0.004±0.079** | **0.495** | 0.496 | **0.451** |
|  | B0 Context-only | −0.034±0.069 | **0.495** | **0.500** | 0.331 |
|  | B1 Uniform Context+PPG | −0.233±0.118 | 0.374 | 0.371 | 0.308 |
|  | **B2 Target-routed** | −0.034±0.069 | **0.495** | **0.500** | 0.331 |
| VR · A | PPG-only | 0.001±0.029 | **0.333** | **0.542** | **0.296** |
|  | B0 Context-only | **0.088±0.056** | 0.273 | 0.500 | 0.214 |
|  | B1 Uniform Context+PPG | −0.018±0.054 | 0.273 | 0.500 | 0.214 |
|  | **B2 Target-routed** | 0.041±0.047 | 0.293 | 0.514 | 0.243 |
| VR · V | PPG-only | 0.066±0.088 | **0.576** | **0.575** | **0.509** |
|  | B0 Context-only | **0.115±0.087** | 0.495 | 0.500 | 0.331 |
|  | B1 Uniform Context+PPG | 0.078±0.130 | 0.525 | 0.525 | 0.449 |
|  | **B2 Target-routed** | **0.115±0.087** | 0.495 | 0.500 | 0.331 |

두 proxy policy의 순위가 달라지므로 VRFS에서 아키텍처 우위를 주장하지 않는다.

---

## 5. Chronological EB personalization

M0와 M2는 각 참가자의 동일 query row에서 평가한다. support는 첫 K개 시간순 label만 사용하고, variance component는 validation participant에서만 추정한다. 절대값은 seed-level query metric이고 ΔCCC CI는 participant-level paired bootstrap이므로 표시된 절대 CCC 차이와 bootstrap mean lift가 정확히 같을 필요는 없다.

| Dataset | Target | K | CCC M0→M2 | Accuracy M0→M2 | BA M0→M2 | Macro-F1 M0→M2 | participant-mean ΔCCC [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|
| EEVR | A | 3 | 0.322→0.361 | 0.700→0.678 | 0.688→0.638 | 0.683→0.637 | **+0.034 [0.007, 0.064]** |
| EEVR | A | 4 | 0.391→0.407 | 0.736→0.653 | 0.737→0.622 | 0.728→0.598 | **+0.074 [0.024, 0.127]** |
| CASE | A | 4 | 0.654→0.672 | 0.808→0.830 | 0.799→0.813 | 0.747→0.769 | +0.021 [−0.011, 0.058] |
| MAUS | C | 2 | 0.441→0.526 | 0.722→0.750 | 0.753→0.770 | 0.589→0.615 | +0.043 [−0.008, 0.092] |
| CASE | V | 4 | 0.636→0.639 | 0.796→0.758 | 0.796→0.730 | 0.754→0.704 | **+0.003 [0.001, 0.007]** |
| EEVR | V | 4 | 0.324→0.341 | 0.694→0.681 | 0.631→0.620 | 0.631→0.618 | −0.000 [−0.009, 0.008] |

주 개인화 결과는 **EEVR Arousal K=4의 continuous CCC lift**다. 같은 조건의 Accuracy·BA·Macro-F1은 개선되지 않았으므로 개인화가 모든 지표나 타깃을 높인다고 주장하지 않는다. CASE V의 유의한 수치는 +0.003으로 매우 작고 분류 지표가 감소했다.

---

## 6. 전체 판독

1. 논문의 공식 baseline은 **B0 Context-only**다. PPG-only는 modality-only 진단 기준선이고 M0는 개인화하지 않은 B2 query prediction이다.
2. Source에서는 B0가 A/C CCC에서 가장 높다. 따라서 B2를 source 성능 최적 모델로 주장할 수 없다.
3. Strict external에서는 B2가 WESAD A와 CogWear C에서 B0 대비 유의한 paired 증분을 보인다. 이는 domain shift 아래의 complementary robustness 근거이지 보편적 우월성의 근거는 아니다.
4. PPG-only가 CogWear의 절대 CCC에서 B2보다 높다. 생리 신호가 외부 C에 유용하다는 근거는 되지만 현재 fusion이 PPG 정보를 최적으로 활용한다고 말할 수는 없다.
5. Valence는 B2에서 B0 경로를 그대로 고정하므로 PPG 증분이 구조적으로 0이다. 완전한 외부 zero-shot에서 Valence 절대 성능도 낮다.
6. 개인화는 target-·K-·metric-selective하다. 확실한 주 결과는 EEVR A의 CCC lift이며 전 타깃 개선으로 일반화하지 않는다.
7. EmoWear와 VRFS는 diagnostic/non-confirmatory 결과로만 둔다.

## 7. 수치 정본

| 결과 | 원본 파일 |
|---|---|
| PPG-only source | `model_v3/personalization_v2/reports/source/metrics_per_seed.csv` (`ready_nsegment`, `ppg_only`, K=0) |
| B0/B1/B2 source | `target_routed_title_final/reports/b0_b1_b2/{target_metrics_summary.csv,metrics_summary.csv}` |
| Source paired CI | `target_routed_title_final/reports/b0_b1_b2/participant_bootstrap_pairwise.csv` |
| PPG-only external | `model_v3/personalization_v2/reports/external_zero_shot/metrics_summary.csv` (`ready_nsegment`, `ppg_only`) |
| B0/B1/B2 external | `target_routed_title_final/reports/b0_b1_b2/external_zero_shot/metrics_summary.csv` |
| External paired CI | `target_routed_title_final/reports/b0_b1_b2/external_zero_shot/participant_bootstrap.csv` |
| EB absolute metrics | `target_routed_title_final/reports/b0_b1_b2/metrics_summary.csv` |
| EB paired CI | `target_routed_title_final/reports/b0_b1_b2/participant_bootstrap_ri_lift.csv` |
