"""Held-out and chronological EB evaluation for the target-routed candidate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
PERSONALIZATION_DIR = MODEL_DIR / "personalization_v2"
if str(PERSONALIZATION_DIR) not in sys.path:
    sys.path.insert(0, str(PERSONALIZATION_DIR))

import evaluate_personalization as core  # noqa: E402


SEEDS = (42, 43, 44)
BASE_CONDITION = "title_context_only_fixed_base"
FINAL_CONDITION = "title_context_ac_direct_ppg"
CONDITIONS = (BASE_CONDITION, FINAL_CONDITION)
MODEL_VERSION = "ready_nsegment_target_routed_title"
FREEZE_PREFIXES = (
    "text_branch",
    "target_context_trunks.valence",
    "target_routed_heads.valence",
)
EXPECTED_CONTEXT_COUNTS = {"case": 8, "eevr": 18, "maus": 3}


def _markdown(frame: pd.DataFrame) -> str:
    return core._markdown(frame)  # noqa: SLF001


def _prediction_path(root: Path, condition: str, seed: int, split: str) -> Path:
    evaluation = "evaluation" if split == "test" else "evaluation_validation"
    return (
        root
        / "runs"
        / "participant_holdout"
        / condition
        / f"seed_{seed}"
        / evaluation
        / "predictions.csv"
    )


def _load_predictions(
    root: Path, source_root: Path, split: str
) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            path = _prediction_path(root, condition, seed, split)
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            frame["condition_id"] = condition
            frame["source_file"] = str(path.resolve())
            rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    result["model_version"] = MODEL_VERSION
    result["evaluation_split"] = split
    result["dataset"] = result["dataset"].astype(str).str.lower()
    result["participant_id"] = result["participant_id"].astype(str)
    return core._attach_units(result, source_root)  # noqa: SLF001


def _participant_overlap(
    validation: pd.DataFrame, test: pd.DataFrame
) -> list[dict[str, Any]]:
    errors = []
    keys = ["condition_id", "seed", "dataset"]
    for values, dev in validation.groupby(keys, sort=True):
        heldout = test
        for column, value in zip(keys, values):
            heldout = heldout[heldout[column].eq(value)]
        overlap = sorted(
            set(dev["participant_id"]).intersection(
                set(heldout["participant_id"])
            )
        )
        if overlap:
            errors.append({"group": list(values), "participants": overlap})
    return errors


def _target_tables(
    per_seed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [
        "model_version",
        "condition_id",
        "seed",
        "target",
        "K",
        "method",
    ]
    rows = []
    for values, group in per_seed.groupby(keys, sort=True):
        row = {
            **dict(zip(keys, values)),
            "dataset_tasks": int(group["dataset"].nunique()),
        }
        for metric in core.METRICS:
            row[metric] = float(group[metric].mean())
        rows.append(row)
    target_seed = pd.DataFrame(rows)
    summary_keys = [key for key in keys if key != "seed"]
    summary_rows = []
    for values, group in target_seed.groupby(summary_keys, sort=True):
        row = {
            **dict(zip(summary_keys, values)),
            "seeds": int(group["seed"].nunique()),
            "dataset_tasks": float(group["dataset_tasks"].mean()),
        }
        for metric in core.METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        summary_rows.append(row)
    return target_seed, pd.DataFrame(summary_rows)


def _population_incremental(target_seed: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_version", "seed", "target", "K", "method"]
    base = target_seed[
        target_seed["condition_id"].eq(BASE_CONDITION)
        & target_seed["K"].eq(0)
        & target_seed["method"].eq("M0_base")
    ]
    final = target_seed[
        target_seed["condition_id"].eq(FINAL_CONDITION)
        & target_seed["K"].eq(0)
        & target_seed["method"].eq("M0_base")
    ]
    paired = base.merge(
        final,
        on=keys,
        suffixes=("_context", "_multimodal"),
        validate="one_to_one",
    )
    rows = []
    for target, group in paired.groupby("target", sort=True):
        row = {"target": target, "seeds": int(group["seed"].nunique())}
        for metric in core.METRICS:
            context = group[f"{metric}_context"].to_numpy(float)
            multimodal = group[f"{metric}_multimodal"].to_numpy(float)
            row[f"{metric}_context_mean"] = float(context.mean())
            row[f"{metric}_multimodal_mean"] = float(multimodal.mean())
            row[f"{metric}_delta_mean"] = float((multimodal - context).mean())
            row[f"{metric}_delta_std"] = float(
                (multimodal - context).std(ddof=0)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _same_query_personalization(target_summary: pd.DataFrame) -> pd.DataFrame:
    selected = target_summary[
        target_summary["condition_id"].eq(FINAL_CONDITION)
        & target_summary["K"].gt(0)
        & target_summary["method"].isin(
            ["M0_base", "M2_EB_shrunk_RI"]
        )
    ]
    base = selected[selected["method"].eq("M0_base")]
    m2 = selected[selected["method"].eq("M2_EB_shrunk_RI")]
    keys = ["model_version", "condition_id", "target", "K"]
    paired = base.merge(
        m2,
        on=keys,
        suffixes=("_base", "_m2"),
        validate="one_to_one",
    )
    rows = []
    for _, value in paired.iterrows():
        row = {key: value[key] for key in keys}
        for metric in core.METRICS:
            row[f"{metric}_base"] = float(value[f"{metric}_mean_base"])
            row[f"{metric}_m2"] = float(value[f"{metric}_mean_m2"])
            row[f"{metric}_delta"] = float(
                value[f"{metric}_mean_m2"]
                - value[f"{metric}_mean_base"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _participant_fusion_bootstrap(
    participant: pd.DataFrame,
    *,
    iterations: int = 2000,
    seed: int = 20260805,
) -> pd.DataFrame:
    selected = participant[
        participant["K"].eq(0) & participant["method"].eq("M0_base")
    ]
    keys = ["seed", "dataset", "target", "participant_id"]
    base = selected[selected["condition_id"].eq(BASE_CONDITION)]
    final = selected[selected["condition_id"].eq(FINAL_CONDITION)]
    paired = base.merge(
        final,
        on=keys,
        suffixes=("_context", "_multimodal"),
        validate="one_to_one",
    )
    rng = np.random.default_rng(seed)
    rows = []
    for (dataset, target), group in paired.groupby(
        ["dataset", "target"], sort=True
    ):
        for metric in ("ccc", "macro_f1", "accuracy", "balanced_accuracy"):
            user = pd.DataFrame(
                {
                    "participant_id": group["participant_id"],
                    "delta": (
                        group[f"{metric}_multimodal"]
                        - group[f"{metric}_context"]
                    ),
                }
            ).groupby("participant_id", as_index=False)["delta"].mean()
            values = user["delta"].to_numpy(float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            boot = np.asarray(
                [
                    rng.choice(values, len(values), replace=True).mean()
                    for _ in range(iterations)
                ]
            )
            rows.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "metric": metric,
                    "participants": int(len(values)),
                    "mean_delta": float(values.mean()),
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                    "iterations": iterations,
                }
            )
    return pd.DataFrame(rows)


def _valence_invariant(
    validation: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    keys = ["seed", "dataset", "participant_id", "sample_id", "target"]
    for split, frame in (("validation", validation), ("test", test)):
        selected = frame[frame["target"].eq("valence")]
        base = selected[selected["condition_id"].eq(BASE_CONDITION)]
        final = selected[selected["condition_id"].eq(FINAL_CONDITION)]
        paired = base.merge(
            final,
            on=keys,
            suffixes=("_context", "_multimodal"),
            validate="one_to_one",
        )
        maximum = float(
            np.abs(
                paired["prediction_context"]
                - paired["prediction_multimodal"]
            ).max()
        )
        rows.append(
            {
                "split": split,
                "rows": int(len(paired)),
                "max_abs_prediction_delta": maximum,
                "tolerance": 1e-7,
                "status": "pass" if maximum <= 1e-7 else "fail",
            }
        )
    result = pd.DataFrame(rows)
    if not result["status"].eq("pass").all():
        raise RuntimeError("Frozen Valence prediction invariant failed")
    return result


def _training_contract_audit(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_rows = []
    context_rows = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            run = (
                root
                / "runs"
                / "participant_holdout"
                / condition
                / f"seed_{seed}"
            )
            metadata = json.loads(
                (run / "run_metadata.json").read_text(encoding="utf-8")
            )
            metrics = json.loads(
                (run / "model_v3" / "training_metrics.json").read_text(
                    encoding="utf-8"
                )
            )["summary"]
            warm = metrics.get("ppg_backbone_warm_start") or {}
            frozen = metrics.get("frozen_parameter_contract") or {}
            full = metrics.get("full_model_warm_start") or {}
            if condition == FINAL_CONDITION:
                ppg_projection_only = bool(warm.get("loaded")) and all(
                    name.startswith("ppg_branch.")
                    for name in warm.get("loaded", [])
                )
                frozen_prefixes = list(frozen.get("prefixes", []))
                status = (
                    metadata.get("fusion_mode") == "target_routed_direct"
                    and bool(full.get("strict"))
                    and ppg_projection_only
                    and frozen_prefixes == list(FREEZE_PREFIXES)
                )
            else:
                ppg_projection_only = True
                frozen_prefixes = []
                status = metadata.get("fusion_mode") == "target_routed_direct"
            if not status:
                raise RuntimeError(
                    f"training contract failed: {condition}/seed_{seed}"
                )
            training_rows.append(
                {
                    "condition_id": condition,
                    "seed": seed,
                    "fusion_mode": metadata.get("fusion_mode"),
                    "full_init_checkpoint": metadata.get(
                        "full_init_checkpoint"
                    ),
                    "ppg_projection_only": ppg_projection_only,
                    "freeze_prefixes": frozen_prefixes,
                    "status": "pass",
                }
            )

            manifest = pd.read_csv(run / "training_manifest.csv")
            for dataset, expected in EXPECTED_CONTEXT_COUNTS.items():
                group = manifest[
                    manifest["dataset"].astype(str).str.lower().eq(dataset)
                ]
                observed = int(group["text"].astype(str).nunique())
                context_status = observed == expected
                if not context_status:
                    raise RuntimeError(
                        f"title context audit failed: {condition}/seed_{seed}/"
                        f"{dataset}: {observed} != {expected}"
                    )
                context_rows.append(
                    {
                        "condition_id": condition,
                        "seed": seed,
                        "dataset": dataset,
                        "expected_unique_contexts": expected,
                        "observed_unique_contexts": observed,
                        "status": "pass",
                    }
                )
    return pd.DataFrame(training_rows), pd.DataFrame(context_rows)


def _prior_reference() -> pd.DataFrame:
    path = (
        MODEL_DIR
        / "fusion_personalization_control"
        / "reports"
        / "target_metrics_summary.csv"
    )
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    return frame[
        frame["model_version"].eq("ready_nsegment")
        & frame["condition_id"].isin(
            [
                "residual_context_only",
                "direct_context_only",
                "direct_context_ppg",
                "direct_validation_gated",
            ]
        )
        & frame["K"].eq(0)
        & frame["method"].eq("M0_base")
    ][
        [
            "condition_id",
            "target",
            "ccc_mean",
            "macro_f1_mean",
            "accuracy_mean",
            "balanced_accuracy_mean",
        ]
    ].copy()


def _write_report(
    reports: Path,
    target_summary: pd.DataFrame,
    population_incremental: pd.DataFrame,
    same_query: pd.DataFrame,
    ri_lift: pd.DataFrame,
    fusion_bootstrap: pd.DataFrame,
    invariant: pd.DataFrame,
    context_audit: pd.DataFrame,
    prior: pd.DataFrame,
) -> None:
    population = target_summary[
        target_summary["K"].eq(0)
        & target_summary["method"].eq("M0_base")
    ][
        [
            "condition_id",
            "target",
            "ccc_mean",
            "macro_f1_mean",
            "accuracy_mean",
            "balanced_accuracy_mean",
        ]
    ]
    delta = population_incremental[
        [
            "target",
            "ccc_context_mean",
            "ccc_multimodal_mean",
            "ccc_delta_mean",
            "macro_f1_delta_mean",
            "accuracy_delta_mean",
            "balanced_accuracy_delta_mean",
        ]
    ]
    personal = same_query[
        [
            "target",
            "K",
            "ccc_base",
            "ccc_m2",
            "ccc_delta",
            "macro_f1_delta",
            "accuracy_delta",
            "balanced_accuracy_delta",
        ]
    ]
    ri_ccc = ri_lift[
        ri_lift["condition_id"].eq(FINAL_CONDITION)
        & ri_lift["metric"].eq("ccc")
    ][
        [
            "dataset",
            "target",
            "K",
            "participants",
            "mean_lift",
            "ci95_low",
            "ci95_high",
        ]
    ]
    fusion_ccc = fusion_bootstrap[
        fusion_bootstrap["metric"].eq("ccc")
    ]
    context_summary = (
        context_audit.groupby("dataset", as_index=False)
        .agg(
            expected_unique_contexts=(
                "expected_unique_contexts", "first"
            ),
            observed_unique_contexts=(
                "observed_unique_contexts", "min"
            ),
            audited_runs=("status", "count"),
        )
    )
    lines = [
        "# Target-routed title-context final-candidate results",
        "",
        "## Fixed contract",
        "",
        "- Valence is a frozen Context-only population predictor.",
        "- PPG has no Valence forward path and cannot update the text/Valence parameters during A/C training.",
        "- Arousal and Cognitive load use independent residual-free direct Context+PPG trunks.",
        "- Context includes CASE title, EEVR scene identity, and MAUS n-back level.",
        "- M2 personalization is evaluated for V/A/C against M0 on the identical query rows.",
        "- This is a new hypothesis; prior held-out results are references, not training-selection input.",
        "",
        "## Title-level context audit",
        "",
        _markdown(context_summary),
        "",
        "## Frozen Valence invariant",
        "",
        _markdown(invariant),
        "",
        "## K=0 held-out metrics",
        "",
        _markdown(population),
        "",
        "## A/C PPG incremental effect within the routed family",
        "",
        _markdown(delta),
        "",
        "## Participant-bootstrap population fusion CCC",
        "",
        _markdown(fusion_ccc),
        "",
        "## Same-query M2 minus M0",
        "",
        _markdown(personal),
        "",
        "## Participant-bootstrap M2 CCC lift",
        "",
        _markdown(ri_ccc),
        "",
        "## Prior controlled-result reference",
        "",
        _markdown(prior),
        "",
        "## Decision rule",
        "",
        "The requested target routing is promoted only if Arousal and Cognitive-load multimodal predictions improve their matched Context-only controls without changing Valence, and EB personalization is interpreted target-by-target from identical-query and participant-bootstrap results.",
    ]
    (reports / "TARGET_ROUTED_FINAL_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def evaluate(root: Path, source_root: Path, reports: Path) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    validation = _load_predictions(root, source_root, "validation")
    test = _load_predictions(root, source_root, "test")
    overlap = _participant_overlap(validation, test)
    if overlap:
        raise RuntimeError(f"validation/test participant leakage: {overlap}")

    variance = core._variance_components(validation)  # noqa: SLF001
    personalized, support_audit = core._personalize(test, variance)  # noqa: SLF001
    per_seed, summary, participant = core._metric_tables(personalized)  # noqa: SLF001
    diagnostics = core._diagnostics(test)  # noqa: SLF001
    ri_lift = core._participant_lift(participant)  # noqa: SLF001
    target_seed, target_summary = _target_tables(per_seed)
    population_incremental = _population_incremental(target_seed)
    same_query = _same_query_personalization(target_summary)
    fusion_bootstrap = _participant_fusion_bootstrap(participant)
    invariant = _valence_invariant(validation, test)
    training_audit, context_audit = _training_contract_audit(root)
    prior = _prior_reference()

    validation.to_csv(reports / "validation_predictions.csv", index=False)
    test.to_csv(reports / "heldout_predictions.csv", index=False)
    variance.to_csv(reports / "eb_variance_components.csv", index=False)
    personalized.to_csv(
        reports / "personalized_query_predictions.csv", index=False
    )
    support_audit.to_json(
        reports / "chronological_support_query_audit.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    per_seed.to_csv(reports / "metrics_per_seed.csv", index=False)
    summary.to_csv(reports / "metrics_summary.csv", index=False)
    participant.to_csv(reports / "metrics_per_participant.csv", index=False)
    diagnostics.to_csv(reports / "ri_diagnostics.csv", index=False)
    ri_lift.to_csv(reports / "participant_bootstrap_ri_lift.csv", index=False)
    target_seed.to_csv(reports / "target_metrics_per_seed.csv", index=False)
    target_summary.to_csv(reports / "target_metrics_summary.csv", index=False)
    population_incremental.to_csv(
        reports / "population_incremental_summary.csv", index=False
    )
    same_query.to_csv(
        reports / "same_query_personalization_summary.csv", index=False
    )
    fusion_bootstrap.to_csv(
        reports / "participant_bootstrap_population_fusion.csv", index=False
    )
    invariant.to_csv(reports / "frozen_valence_invariant.csv", index=False)
    training_audit.to_csv(reports / "training_contract_audit.csv", index=False)
    context_audit.to_csv(reports / "title_context_audit.csv", index=False)

    audit = {
        "schema_version": 1,
        "status": "pass",
        "model_version": MODEL_VERSION,
        "conditions": list(CONDITIONS),
        "validation_test_participant_overlap": overlap,
        "support_query_overlap_total": int(
            support_audit["support_query_overlap"].sum()
        ),
        "support_precedes_query": bool(
            (
                support_audit["support_max_order"]
                < support_audit["query_min_order"]
            ).all()
        ),
        "test_metrics_used_for_variance_components": False,
        "model_weight_updates_during_personalization": 0,
        "frozen_valence_invariant": invariant.to_dict(orient="records"),
        "training_contract": training_audit.to_dict(orient="records"),
        "title_context": context_audit.to_dict(orient="records"),
        "report": str((reports / "TARGET_ROUTED_FINAL_RESULTS.md").resolve()),
    }
    (reports / "TARGET_ROUTED_FINAL_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_report(
        reports,
        target_summary,
        population_incremental,
        same_query,
        ri_lift,
        fusion_bootstrap,
        invariant,
        context_audit,
        prior,
    )
    print(
        f"[DONE] target-routed report -> "
        f"{reports / 'TARGET_ROUTED_FINAL_RESULTS.md'}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reports", type=Path, default=HERE / "reports")
    args = parser.parse_args()
    evaluate(
        args.root.resolve(),
        args.source_root.resolve(),
        args.reports.resolve(),
    )


if __name__ == "__main__":
    main()
