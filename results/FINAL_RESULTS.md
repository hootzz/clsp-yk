# FINAL RESULTS — Full Result Dashboard

Date: 2026-08-03  
Final candidate: **B2 Target-routed**
— Valence uses a frozen Context-only path; Arousal and Cognitive load use target-specific direct Context+PPG; personalization is an optional EB output offset.

## 0. Evaluation conditions and baselines

| Label | Input / architecture | Role |
|---|---|---|
| **PPG-only** | frozen PaPaGEI-P embeddings only | physiological-modality-only diagnostic baseline |
| **B0 Context-only** | title-level Context only | **official matched population baseline for B1/B2** |
| **B1 Uniform Context+PPG** | residual-free direct Context+PPG for V/A/C | uniform-fusion control that adds PPG to all targets |
| **B2 Target-routed** | V=fixed Context-only, A/C=target-specific direct Context+PPG | final architecture candidate |
| **M0 Base** | non-personalized B2 query prediction | personalization comparison baseline |
| **M2 EB-personalized** | B2 prediction + target-wise shrunk EB offset | K-shot inference-time personalization |

### Datasets and evaluation roles

| Category | Dataset | Target | Participants | Role |
|---|---|---|---:|---|
| Source held-out | CASE | V/A | 11 | participant-disjoint source evaluation |
| Source held-out | EEVR | V/A | 14 | participant-disjoint source evaluation |
| Source held-out | MAUS | C | 8 | participant-disjoint source evaluation |
| Strict external zero-shot | WESAD (E4) | V/A | 15 | unseen affect dataset |
| Strict external zero-shot | CogWear (E4/Galaxy) | C | 18 | unseen objective rest-vs-Stroop dataset |
| External diagnostic | EmoWear (seated/walk) | V/A | 48 | motion/device diagnostic; walk uses the same trial-level SAM labels as seated |
| External diagnostic | VRFS (flat-screen/VR) | V/A | 33 | condition-level proxy labels; participant-level SAM cannot be directly joined with PPG |

Absolute CCC values are reported as mean±population SD across three seeds, while Accuracy, BA, and Macro-F1 are seed averages. The classification threshold is 0.625 for source and affect datasets and 0.5 for CogWear. Paired effects are participant-level bootstrap means with 95% CIs (2,000 resamples). Bold absolute values indicate the highest value for the same dataset and target; bold differences indicate CIs that exclude zero.

---

## 1. Source held-out — official aggregate results

V/A aggregates the CASE and EEVR evaluation rows, while C is evaluated on MAUS. This is a `known-content, unseen-participant` evaluation and is not a fully external zero-shot setting.

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

† The PPG-only C Accuracy of 0.852 occurs alongside BA=0.500 and is driven by majority-class prediction; it should not be interpreted as superior performance.

### 1.1 Absolute performance by source dataset

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

On the source datasets, B2 did not outperform B0 on A/C CCC. The zero Valence difference is a structural invariant, not a performance estimate, because B2 uses the frozen B0 Valence path unchanged.

---

## 2. Strict external zero-shot — WESAD·CogWear

All checkpoints are frozen, with no updates, threshold selection, checkpoint selection, or fusion selection using external labels.

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

For WESAD A and CogWear E4 C, B2 showed positive paired gains across all four metrics, with CIs excluding zero. For CogWear Galaxy C, only the CCC gain was significant. Because absolute external CCC values remain low, these results should not be presented as high deployment accuracy.

---

## 3. External diagnostic — EmoWear

EmoWear walk uses the same trial-level SAM labels as seated and therefore serves as a motion/device diagnostic rather than an independent affect ground truth. It was also observed during development and is not treated as a purely confirmatory result.

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

EmoWear did not show a consistent Arousal CCC gain, and the directions of Accuracy and Macro-F1 were also inconsistent.

---

## 4. External diagnostic — VRFS

VRFS does not allow direct matching between physiological sessions and participant-level SAM rows. Both label policies below are condition-level proxies and are used only as sensitivity diagnostics rather than as confirmatory paper results.

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

Because the rankings differ across the two proxy policies, no architecture superiority is claimed on VRFS.

---

## 5. Chronological EB personalization

M0 and M2 are evaluated on identical query rows for each participant. Support uses only the first K chronologically ordered labels, and variance components are estimated from validation participants only. Absolute values are seed-level query metrics, whereas ΔCCC CIs come from participant-level paired bootstrap estimates; therefore, the displayed absolute CCC difference and bootstrap mean lift need not be identical.

| Dataset | Target | K | CCC M0→M2 | Accuracy M0→M2 | BA M0→M2 | Macro-F1 M0→M2 | participant-mean ΔCCC [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|
| EEVR | A | 3 | 0.322→0.361 | 0.700→0.678 | 0.688→0.638 | 0.683→0.637 | **+0.034 [0.007, 0.064]** |
| EEVR | A | 4 | 0.391→0.407 | 0.736→0.653 | 0.737→0.622 | 0.728→0.598 | **+0.074 [0.024, 0.127]** |
| CASE | A | 4 | 0.654→0.672 | 0.808→0.830 | 0.799→0.813 | 0.747→0.769 | +0.021 [−0.011, 0.058] |
| MAUS | C | 2 | 0.441→0.526 | 0.722→0.750 | 0.753→0.770 | 0.589→0.615 | +0.043 [−0.008, 0.092] |
| CASE | V | 4 | 0.636→0.639 | 0.796→0.758 | 0.796→0.730 | 0.754→0.704 | **+0.003 [0.001, 0.007]** |
| EEVR | V | 4 | 0.324→0.341 | 0.694→0.681 | 0.631→0.620 | 0.631→0.618 | −0.000 [−0.009, 0.008] |

The primary personalization result is the **continuous CCC lift for EEVR Arousal at K=4**. Accuracy, BA, and Macro-F1 do not improve under the same condition, so personalization is not claimed to improve every metric or target. The significant CASE V effect is only +0.003, and its classification metrics decrease.

---
