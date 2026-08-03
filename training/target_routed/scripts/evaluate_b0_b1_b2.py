"""Complete held-out and EB comparison for B0, B1, and B2."""
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
for path in (HERE, PERSONALIZATION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_personalization as core  # noqa: E402
import evaluate_target_routed_final as routed  # noqa: E402


B0 = "title_context_only_fixed_base"
B1 = "title_context_ppg_uniform_direct"
B2 = "title_context_ac_direct_ppg"
CONDITIONS = (B0, B1, B2)
SEEDS = (42, 43, 44)
MODEL_VERSION = "title_target_routed_b0_b1_b2"
EXPECTED_CONTEXT_COUNTS = {"case": 8, "eevr": 18, "maus": 3}
COMPARISONS = (
    ("B1_minus_B0", B1, B0),
    ("B2_minus_B0", B2, B0),
    ("B2_minus_B1", B2, B1),
)


def _prediction_path(root: Path, condition: str, seed: int, split: str) -> Path:
    evaluation = "evaluation" if split == "test" else "evaluation_validation"
    return (
        root / "runs" / "participant_holdout" / condition
        / f"seed_{seed}" / evaluation / "predictions.csv"
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


def _comparison_tables(
    target_seed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = target_seed[
        target_seed["K"].eq(0) & target_seed["method"].eq("M0_base")
    ]
    keys = ["model_version", "seed", "target", "K", "method"]
    rows = []
    for label, candidate_id, reference_id in COMPARISONS:
        candidate = selected[
            selected["condition_id"].eq(candidate_id)
        ]
        reference = selected[
            selected["condition_id"].eq(reference_id)
        ]
        paired = reference.merge(
            candidate,
            on=keys,
            suffixes=("_reference", "_candidate"),
            validate="one_to_one",
        )
        for _, value in paired.iterrows():
            row = {
                "comparison": label,
                "candidate_condition": candidate_id,
                "reference_condition": reference_id,
                "seed": int(value["seed"]),
                "target": value["target"],
            }
            for metric in core.METRICS:
                reference_value = float(value[f"{metric}_reference"])
                candidate_value = float(value[f"{metric}_candidate"])
                row[f"{metric}_reference"] = reference_value
                row[f"{metric}_candidate"] = candidate_value
                row[f"{metric}_delta"] = candidate_value - reference_value
            rows.append(row)
    per_seed = pd.DataFrame(rows)
    summary_rows = []
    group_keys = [
        "comparison",
        "candidate_condition",
        "reference_condition",
        "target",
    ]
    for values, group in per_seed.groupby(group_keys, sort=True):
        row = {**dict(zip(group_keys, values)), "seeds": int(len(group))}
        for metric in core.METRICS:
            for suffix in ("reference", "candidate", "delta"):
                column = f"{metric}_{suffix}"
                row[f"{column}_mean"] = float(group[column].mean())
                row[f"{column}_std"] = float(group[column].std(ddof=0))
        summary_rows.append(row)
    return per_seed, pd.DataFrame(summary_rows)


def _participant_bootstrap(
    participant: pd.DataFrame,
    *,
    iterations: int = 2000,
    seed: int = 20260807,
) -> pd.DataFrame:
    selected = participant[
        participant["K"].eq(0) & participant["method"].eq("M0_base")
    ]
    keys = ["seed", "dataset", "target", "participant_id"]
    rng = np.random.default_rng(seed)
    rows = []
    for label, candidate_id, reference_id in COMPARISONS:
        candidate = selected[selected["condition_id"].eq(candidate_id)]
        reference = selected[selected["condition_id"].eq(reference_id)]
        paired = reference.merge(
            candidate,
            on=keys,
            suffixes=("_reference", "_candidate"),
            validate="one_to_one",
        )
        for (dataset, target), group in paired.groupby(
            ["dataset", "target"], sort=True
        ):
            for metric in (
                "ccc",
                "macro_f1",
                "accuracy",
                "balanced_accuracy",
            ):
                user = pd.DataFrame(
                    {
                        "participant_id": group["participant_id"],
                        "delta": (
                            group[f"{metric}_candidate"]
                            - group[f"{metric}_reference"]
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
                        "comparison": label,
                        "candidate_condition": candidate_id,
                        "reference_condition": reference_id,
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


def _training_audit(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    training, contexts = routed._training_contract_audit(root)  # noqa: SLF001
    training_rows = training.to_dict(orient="records")
    context_rows = contexts.to_dict(orient="records")
    for seed in SEEDS:
        run = (
            root / "runs" / "participant_holdout" / B1 / f"seed_{seed}"
        )
        metadata = json.loads(
            (run / "run_metadata.json").read_text(encoding="utf-8")
        )
        summary = json.loads(
            (run / "model_v3" / "training_metrics.json").read_text(
                encoding="utf-8"
            )
        )["summary"]
        warm = summary.get("ppg_backbone_warm_start") or {}
        projection_only = bool(warm.get("loaded")) and all(
            name.startswith("ppg_branch.") for name in warm.get("loaded", [])
        )
        status = (
            metadata.get("fusion_mode") == "concat"
            and projection_only
            and summary.get("full_model_warm_start") is None
            and summary.get("frozen_parameter_contract") is None
        )
        if not status:
            raise RuntimeError(f"B1 training contract failed for seed {seed}")
        training_rows.append(
            {
                "condition_id": B1,
                "seed": seed,
                "fusion_mode": "concat",
                "full_init_checkpoint": None,
                "ppg_projection_only": projection_only,
                "freeze_prefixes": [],
                "status": "pass",
            }
        )
        manifest = pd.read_csv(run / "training_manifest.csv")
        for dataset, expected in EXPECTED_CONTEXT_COUNTS.items():
            group = manifest[
                manifest["dataset"].astype(str).str.lower().eq(dataset)
            ]
            observed = int(group["text"].astype(str).nunique())
            if observed != expected:
                raise RuntimeError(
                    f"B1 title context failed: {dataset} {observed}!={expected}"
                )
            context_rows.append(
                {
                    "condition_id": B1,
                    "seed": seed,
                    "dataset": dataset,
                    "expected_unique_contexts": expected,
                    "observed_unique_contexts": observed,
                    "status": "pass",
                }
            )
    return pd.DataFrame(training_rows), pd.DataFrame(context_rows)


def _write_report(
    reports: Path,
    target_summary: pd.DataFrame,
    comparison_summary: pd.DataFrame,
    same_query: pd.DataFrame,
    bootstrap: pd.DataFrame,
    invariant: pd.DataFrame,
) -> None:
    actual = target_summary[
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
    deltas = comparison_summary[
        [
            "comparison",
            "target",
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
    bootstrap_ccc = bootstrap[bootstrap["metric"].eq("ccc")]
    lines = [
        "# B0/B1/B2 target-routing results",
        "",
        "## Conditions",
        "",
        "- B0: title-level Context-only for V/A/C.",
        "- B1: historical uniform direct concat Context+PPG for V/A/C.",
        "- B2: frozen Context-only Valence and target-specific direct Context+PPG for A/C.",
        "- B0 and B2 are reused from the completed target-routed run; only B1 is newly trained.",
        "",
        "## K=0 actual held-out metrics",
        "",
        routed._markdown(actual),  # noqa: SLF001
        "",
        "## Pairwise deltas",
        "",
        routed._markdown(deltas),  # noqa: SLF001
        "",
        "## Participant-bootstrap CCC deltas",
        "",
        routed._markdown(bootstrap_ccc),  # noqa: SLF001
        "",
        "## B2 chronological EB: identical-query M2 minus M0",
        "",
        routed._markdown(personal),  # noqa: SLF001
        "",
        "## B2 frozen-Valence invariant",
        "",
        routed._markdown(invariant),  # noqa: SLF001
        "",
        "## Interpretation",
        "",
        "B1-B0 estimates the historical all-target PPG package. B2-B0 estimates the proposed decoupled package. B2-B1 tests whether removing PPG from Valence while retaining A/C multimodality is preferable to uniform fusion. Architecture promotion remains target-specific and is not based on a post-hoc single best metric.",
    ]
    (reports / "B0_B1_B2_RESULTS.md").write_text(
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
    personalized, support = core._personalize(test, variance)  # noqa: SLF001
    per_seed, summary, participant = core._metric_tables(personalized)  # noqa: SLF001
    diagnostics = core._diagnostics(test)  # noqa: SLF001
    ri_lift = core._participant_lift(participant)  # noqa: SLF001
    target_seed, target_summary = routed._target_tables(per_seed)  # noqa: SLF001
    comparison_seed, comparison_summary = _comparison_tables(target_seed)
    same_query = routed._same_query_personalization(  # noqa: SLF001
        target_summary
    )
    bootstrap = _participant_bootstrap(participant)
    invariant = routed._valence_invariant(validation, test)  # noqa: SLF001
    training, contexts = _training_audit(root)

    validation.to_csv(reports / "validation_predictions.csv", index=False)
    test.to_csv(reports / "heldout_predictions.csv", index=False)
    variance.to_csv(reports / "eb_variance_components.csv", index=False)
    personalized.to_csv(
        reports / "personalized_query_predictions.csv", index=False
    )
    support.to_json(
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
    comparison_seed.to_csv(
        reports / "pairwise_comparisons_per_seed.csv", index=False
    )
    comparison_summary.to_csv(
        reports / "pairwise_comparisons_summary.csv", index=False
    )
    same_query.to_csv(
        reports / "b2_same_query_personalization.csv", index=False
    )
    bootstrap.to_csv(
        reports / "participant_bootstrap_pairwise.csv", index=False
    )
    invariant.to_csv(reports / "b2_frozen_valence_invariant.csv", index=False)
    training.to_csv(reports / "training_contract_audit.csv", index=False)
    contexts.to_csv(reports / "title_context_audit.csv", index=False)

    support_ok = bool(
        int(support["support_query_overlap"].sum()) == 0
        and (support["support_max_order"] < support["query_min_order"]).all()
    )
    if not support_ok:
        raise RuntimeError("chronological support/query audit failed")
    audit = {
        "schema_version": 1,
        "status": "pass",
        "conditions": list(CONDITIONS),
        "seeds": list(SEEDS),
        "validation_test_participant_overlap": overlap,
        "support_query_overlap_total": int(
            support["support_query_overlap"].sum()
        ),
        "support_precedes_query": True,
        "test_metrics_used_for_architecture": False,
        "model_weight_updates_during_personalization": 0,
        "b2_frozen_valence": invariant.to_dict(orient="records"),
        "training_contract": training.to_dict(orient="records"),
        "report": str((reports / "B0_B1_B2_RESULTS.md").resolve()),
    }
    (reports / "B0_B1_B2_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_report(
        reports,
        target_summary,
        comparison_summary,
        same_query,
        bootstrap,
        invariant,
    )
    print(
        f"[DONE] B0/B1/B2 report -> {reports / 'B0_B1_B2_RESULTS.md'}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    evaluate(
        args.root.resolve(),
        args.source_root.resolve(),
        args.reports.resolve(),
    )


if __name__ == "__main__":
    main()
