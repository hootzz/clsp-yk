# Target-routed title-context final candidate

## Population architecture

| Target | Population predictor | PPG path | Inference-time personalization |
|---|---|---|---|
| Valence | Fixed title-level Context-only predictor | None | Context prediction + target-wise EB offset |
| Arousal | Residual-free, target-specific direct Context+PPG predictor | Direct fusion | Base prediction + target-wise EB offset |
| Cognitive load | Residual-free, target-specific direct Context+PPG predictor | Direct fusion | Base prediction + target-wise EB offset |

The title-level context contract is:

- CASE: official content title.
- EEVR: official scene/content identity.
- MAUS: n-back level.

Valence has no PPG gate, PPG residual, or PPG concatenation. The matched Context-only model is trained first. The final model strictly loads its full state, then freezes the shared text encoder, the Valence context trunk, and the Valence head. Only the Arousal/Cognitive-load direct-fusion paths and PPG projection can adapt. Evaluation asserts that Context-only and final-model Valence predictions are equal within `1e-7` on both validation and test rows.

## Matched comparison

Both conditions use participant-disjoint validation/test splits, seeds 42/43/44, the same title context, normalization, sample rows, training budget, and target labels:

1. `title_context_only_fixed_base`: Context-only for all three targets.
2. `title_context_ac_direct_ppg`: frozen Context-only Valence; direct Context+PPG for Arousal and Cognitive load.

This comparison measures the Arousal/Cognitive-load incremental value of PPG without a residual parameterization. Prior residual/direct/gated results are retained only as a reference and are not used to select this architecture from test results.

## Personalization contract

Personalization is a target-wise empirical-Bayes random intercept applied after the population prediction. For each test participant, the first chronological `K` labeled enrollment trials form support and only later trials form the query set. Variance/shrinkage components are estimated from validation participants; test labels never choose the shrinkage strength. M0 and M2 are reported on identical query rows.

Outputs include CCC, Macro-F1, Accuracy, and Balanced Accuracy at seed, target, dataset, and participant level, plus participant-bootstrap confidence intervals.

## GPU commands

Run a one-epoch contract smoke test:

```bat
run_target_routed_final.cmd smoke 0
```

Run full training, held-out evaluation, EB personalization, frozen external
zero-shot, and reports:

```bat
run_target_routed_final.cmd all 0
```

Use another GPU by changing the second argument. Add `force` as the third argument only when intentionally retraining existing completed runs:

```bat
run_target_routed_final.cmd all 1 force
```

Training only and evaluation only are also available:

```bat
run_target_routed_final.cmd train 0
run_target_routed_final.cmd evaluate 0
run_target_routed_final.cmd external 0
```

## Main outputs

- `reports/TARGET_ROUTED_FINAL_RESULTS.md`: concise population/personalization result report.
- `reports/metrics_per_participant.csv`: actual CCC/F1/Accuracy/BA for every participant and condition.
- `reports/target_metrics_summary.csv`: V/A/C target summaries across seeds.
- `reports/population_incremental_summary.csv`: direct multimodal minus matched Context-only effect.
- `reports/same_query_personalization_summary.csv`: M2 minus M0 on identical queries.
- `reports/participant_bootstrap_population_fusion.csv`: participant-bootstrap fusion effects.
- `reports/participant_bootstrap_ri_lift.csv`: participant-bootstrap EB effects.
- `reports/frozen_valence_invariant.csv`: hard Valence-isolation assertion.
- `reports/title_context_audit.csv`: title/scene/level context cardinality audit.
- `reports/TARGET_ROUTED_FINAL_AUDIT.json`: split, support/query, checkpoint, freeze, and context contract audit.
- `reports/external_zero_shot/TARGET_ROUTED_EXTERNAL_ZERO_SHOT_RESULTS.md`: frozen WESAD V/A, CogWear C, EmoWear V/A, and condition-level VRFS results.
- `reports/external_zero_shot/metrics_per_participant.csv`: actual external participant/session metrics.
- `reports/external_zero_shot/TARGET_ROUTED_EXTERNAL_ZERO_SHOT_AUDIT.json`: no-update, no-selection, routing, and Valence-invariance audit.

## Promotion rule

The requested structure becomes the final model only if held-out Arousal and Cognitive-load performance improves over the matched Context-only control without changing Valence. EB is accepted separately per target based on identical-query and participant-bootstrap results; it is not assumed to help every target.

## Uniform-fusion follow-up

The isolated [B0/B1/B2 follow-up](B0_B1_B2_FOLLOWUP.md) adds the historical all-target direct-concat B1 control after the current B0/B2 command completes. It reuses B0/B2 and trains only B1.
