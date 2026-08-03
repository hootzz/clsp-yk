# Final results — target-routed cognitive-emotional estimator

확정 추정량: 절대 성능 = 3-seed **mean±SD**; 증분 = participant paired **bootstrap [95% CI]** (2000회).
이진지표(Acc/BA/F1)는 1–3 vs 4–5 (**0.625**) 경계. 조건: **P**=PPG-only, **C/B0**=Context-only, **U/B1**=uniform Context+PPG, **R/B2**=target-routed(제안).

---

## 1. Source held-out (K=0, CASE+EEVR+MAUS, participant-disjoint)

### 1a. CCC (주지표)
| Target | P (PPG-only) | C (B0) | U (B1) | R (B2, 제안) |
|---|---|---|---|---|
| Valence | 0.076 ± 0.010 | 0.509 ± 0.029 | **0.532 ± 0.071** | 0.509 ± 0.029 (frozen=C) |
| Arousal | 0.063 ± 0.080 | **0.501 ± 0.024** | 0.440 ± 0.033 | 0.456 ± 0.038 |
| Cognitive load | 0.151 ± 0.140 | **0.647 ± 0.093** | 0.485 ± 0.166 | 0.492 ± 0.207 |

### 1b. Accuracy / Balanced-Acc / Macro-F1 (C·U·R; PPG-only는 CCC만 보고)
| Target | C (B0) Acc/BA/F1 | U (B1) Acc/BA/F1 | R (B2) Acc/BA/F1 |
|---|---|---|---|
| Valence | 0.704 / 0.686 / 0.671 | 0.688 / 0.656 / 0.653 | 0.704 / 0.686 / 0.671 |
| Arousal | 0.746 / 0.708 / 0.687 | 0.705 / 0.672 / 0.639 | 0.743 / **0.720** / **0.702** |
| Cognitive load | 0.778 / **0.850** / 0.680 | 0.778 / 0.605 / 0.517 | **0.815** / 0.782 / 0.624 |

> 읽기: in-distribution에선 **C(Context-only)가 최고**, PPG 추가(U/R)는 A/C CCC를 못 올림(content prior 강함). PPG-only는 세 타깃 CCC 0.06–0.15로 chance 근처. PPG-only는 **matched-row**(C/U/R와 동일 3,270 평가행); zrecording 계보 .090/.072/.153은 **본표 금지**(supplementary "legacy preprocessing diagnostic"만).

---

## 2. External zero-shot — PPG 증분 (R over C, B2−B0), participant bootstrap ΔCCC [95% CI]
학습에 없는 외부셋, 무업데이트·무선택. Valence는 고정 Context-only 경로라 Δ=0.

| Dataset (device) | Target | ΔCCC [95% CI] | 유의 | 참고(B2 절대 CCC) |
|---|---|---|---|---|
| **CogWear (E4)** | cognitive load | **+0.189 [0.126, 0.258]** | ✅ (Acc/BA/F1 CI도 0 제외) | 0.087 |
| **CogWear (Galaxy)** | cognitive load | **+0.078 [0.034, 0.127]** | ✅ (CCC만) | 0.006 |
| **WESAD (E4)** | arousal | **+0.061 [0.001, 0.118]** | ✅ (경계) | 0.136* |
| EmoWear (seated/walk) | arousal | CI 0 포함 | ✗ | ~0.01 |
| (전 external) | valence | 0.000 (고정경로) | — | ~0 |

*WESAD arousal Acc 0.750인데 BA 0.497(≈chance) — 불균형 착시. **Acc 단독 인용 금지, CCC/BA로 해석.**
> 읽기: 콘텐츠 prior가 전이 안 되는 배포에서 **PPG가 arousal·cognitive load에 유의한 상대적 증분**을 준다(regime flip). 절대 CCC는 낮음 → 상대 강건성 진단으로 서술.

---

## 3. Inference-time personalization — EB 개인화 lift (M2−M0), participant bootstrap ΔCCC [95% CI]
같은 query row, K개 등록라벨만(누수 없음), σ²/τ²=validation only.

| Dataset | Target | K | lift ΔCCC [95% CI] | 유의 |
|---|---|---|---|---|
| **EEVR** | arousal | 4 | **+0.074 [0.024, 0.127]** | ✅ |
| EEVR | arousal | 3 | +0.034 [0.007, 0.064] | ✅ |
| CASE | arousal | 4 | +0.021 [−0.011, 0.058] | ✗ |
| MAUS | cognitive load | 2 | +0.043 [−0.008, 0.092] | ✗ |
| CASE/EEVR | valence | 1–4 | tiny / mixed | ✗ |

> 읽기: 개인화는 **arousal(개인 baseline 분산 큰 축)에서만 유의**, valence는 τ²≈0라 사실상 없음(개인화가 라우팅 서사를 강화). 분류지표 lift는 비유의(연속 CCC 개선). K=0이면 zero-shot population fallback.
> ※ pooled(+0.017)·participant-macro-combined(+0.052)는 **다른 추정량 — 혼용 금지**. 본표는 per-dataset participant-bootstrap.

---

## 표 배치 (논문)
- **§3.1.3 / H1:** 위 §1 (source, PPG-only/C/U/R) — component validation.
- **§2 external:** 배포 전이 근거. (이 MEM-HQS 논문에 넣을지 vs supplementary/별도 estimator 논문은 정책 결정 — "known-content" caveat과 함께.)
- **§3 personalization:** §3.1.3.6 방법 + 이 lift 표.

## 원본 수치 소스 (검증됨)
- source: `target_metrics_summary.csv` (+ matched-row PPG-only report)
- external: `external_zero_shot/participant_bootstrap.csv`
- personalization: `participant_bootstrap_ri_lift.csv`
- 확정 규칙: `target_routed_title_final/VERIFICATION_REPORT.md`
