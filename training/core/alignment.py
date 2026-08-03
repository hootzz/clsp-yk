"""Cross-modal alignment diagnostics for CLSP text/PPG embeddings."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def cross_modal_alignment_metrics(
    text_z: torch.Tensor,
    ppg_z: torch.Tensor,
    context_sig: list[str],
    valid: torch.Tensor | None = None,
) -> dict[str, float | int]:
    """Measure matching-context cosine separation and prototype retrieval."""
    if valid is None:
        valid_mask = torch.ones(
            text_z.size(0), dtype=torch.bool, device=text_z.device
        )
    else:
        valid_mask = valid.to(text_z.device).bool()
    if int(valid_mask.sum()) < 2:
        return {
            "n": int(valid_mask.sum()),
            "n_contexts": 0,
            "cos_same": float("nan"),
            "cos_different": float("nan"),
            "cos_gap": float("nan"),
            "prototype_r1": float("nan"),
            "prototype_r5": float("nan"),
            "prototype_mrr": float("nan"),
            "chance_r1": float("nan"),
        }

    text = F.normalize(text_z[valid_mask], dim=-1)
    ppg = F.normalize(ppg_z[valid_mask], dim=-1)
    keep = valid_mask.detach().cpu().tolist()
    signatures = np.asarray(
        [str(sig) for sig, is_valid in zip(context_sig, keep) if is_valid],
        dtype=object,
    )
    similarity = (text @ ppg.T).detach().cpu().numpy()
    same = signatures[:, None] == signatures[None, :]

    same_by_anchor = np.asarray(
        [similarity[i, same[i]].mean() for i in range(len(signatures))],
        dtype=float,
    )
    different_by_anchor = np.asarray(
        [
            similarity[i, ~same[i]].mean() if (~same[i]).any() else np.nan
            for i in range(len(signatures))
        ],
        dtype=float,
    )

    contexts = sorted(set(signatures.tolist()))
    text_prototypes = []
    ppg_prototypes = []
    for context in contexts:
        indices = np.flatnonzero(signatures == context)
        text_prototypes.append(F.normalize(text[indices].mean(dim=0), dim=0))
        ppg_prototypes.append(F.normalize(ppg[indices].mean(dim=0), dim=0))
    text_proto = torch.stack(text_prototypes)
    ppg_proto = torch.stack(ppg_prototypes)
    prototype_similarity = (text_proto @ ppg_proto.T).detach().cpu().numpy()
    order = np.argsort(-prototype_similarity, axis=1)
    ranks = np.asarray(
        [int(np.flatnonzero(order[i] == i)[0]) + 1 for i in range(len(contexts))],
        dtype=int,
    )
    top5 = min(5, len(contexts))

    return {
        "n": int(len(signatures)),
        "n_contexts": int(len(contexts)),
        "cos_same": float(np.nanmean(same_by_anchor)),
        "cos_different": float(np.nanmean(different_by_anchor)),
        "cos_gap": float(np.nanmean(same_by_anchor - different_by_anchor)),
        "prototype_r1": float(np.mean(ranks == 1)),
        "prototype_r5": float(np.mean(ranks <= top5)),
        "prototype_mrr": float(np.mean(1.0 / ranks)),
        "chance_r1": float(1.0 / len(contexts)),
    }


@torch.no_grad()
def cross_modal_domain_retrieval_metrics(
    text_z: torch.Tensor,
    ppg_z: torch.Tensor,
    datasets: list[str],
    participant_ids: list[str],
    valid: torch.Tensor | None = None,
) -> dict[str, float | int]:
    """Quantify domain retrieval after excluding the paired participant.

    A high domain-match rate relative to candidate-distribution chance means
    cross-modal representations retain dataset identity. This is diagnostic,
    not a target performance metric.
    """
    if valid is None:
        valid_mask = torch.ones(
            text_z.size(0), dtype=torch.bool, device=text_z.device
        )
    else:
        valid_mask = valid.to(text_z.device).bool()
    keep = valid_mask.detach().cpu().tolist()
    if int(valid_mask.sum()) < 2:
        return {
            "n_anchors": 0,
            "domain_match_r1": float("nan"),
            "candidate_chance": float("nan"),
            "excess_over_chance": float("nan"),
        }

    text = F.normalize(text_z[valid_mask], dim=-1)
    ppg = F.normalize(ppg_z[valid_mask], dim=-1)
    kept_datasets = np.asarray(
        [str(value) for value, flag in zip(datasets, keep) if flag],
        dtype=object,
    )
    kept_participants = np.asarray(
        [str(value) for value, flag in zip(participant_ids, keep) if flag],
        dtype=object,
    )
    similarity = (text @ ppg.T).detach().cpu().numpy()

    matches = []
    chances = []
    for anchor in range(len(kept_datasets)):
        candidates = kept_participants != kept_participants[anchor]
        if not candidates.any():
            continue
        candidate_indices = np.flatnonzero(candidates)
        nearest = candidate_indices[
            int(np.argmax(similarity[anchor, candidate_indices]))
        ]
        matches.append(
            float(kept_datasets[nearest] == kept_datasets[anchor])
        )
        chances.append(
            float(
                np.mean(
                    kept_datasets[candidate_indices]
                    == kept_datasets[anchor]
                )
            )
        )
    if not matches:
        return {
            "n_anchors": 0,
            "domain_match_r1": float("nan"),
            "candidate_chance": float("nan"),
            "excess_over_chance": float("nan"),
        }
    match_rate = float(np.mean(matches))
    chance = float(np.mean(chances))
    return {
        "n_anchors": int(len(matches)),
        "domain_match_r1": match_rate,
        "candidate_chance": chance,
        "excess_over_chance": float(match_rate - chance),
    }


def identity_shortcut_recovery_metrics(
    embeddings: torch.Tensor | np.ndarray,
    dataset_ids: list[str],
    participant_ids: list[str],
    session_ids: list[str],
    valid: torch.Tensor | np.ndarray | None = None,
) -> dict[str, dict[str, float | int | str]]:
    """Nearest-neighbor recovery audit for dataset/participant/session IDs.

    This is diagnostic only and is never a checkpoint-selection metric.
    Participant and session recovery are evaluated within dataset so trivial
    dataset separation cannot count as identity recovery.
    """
    values = (
        embeddings.detach().cpu().numpy()
        if isinstance(embeddings, torch.Tensor)
        else np.asarray(embeddings)
    )
    if valid is None:
        keep = np.ones(len(values), dtype=bool)
    elif isinstance(valid, torch.Tensor):
        keep = valid.detach().cpu().numpy().astype(bool)
    else:
        keep = np.asarray(valid).astype(bool)
    if len(keep) != len(values):
        raise ValueError("identity audit valid mask length mismatch")

    values = values[keep]
    datasets = np.asarray(dataset_ids, dtype=object)[keep]
    participants = np.asarray(participant_ids, dtype=object)[keep]
    sessions = np.asarray(session_ids, dtype=object)[keep]
    if len(values) < 2:
        empty = {
            "n": int(len(values)),
            "eligible": 0,
            "accuracy": float("nan"),
            "chance": float("nan"),
            "excess_over_chance": float("nan"),
            "status": "not_estimable",
        }
        return {
            "dataset_id": dict(empty),
            "participant_id": dict(empty),
            "session_id": dict(empty),
        }

    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, 1e-12)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)

    def recover(
        labels: np.ndarray,
        *,
        within_dataset: bool,
    ) -> dict[str, float | int | str]:
        hits: list[float] = []
        chances: list[float] = []
        for index in range(len(values)):
            candidates = np.arange(len(values)) != index
            if within_dataset:
                candidates &= datasets == datasets[index]
            candidate_indices = np.where(candidates)[0]
            if len(candidate_indices) == 0:
                continue
            same_label = labels[candidate_indices] == labels[index]
            if int(same_label.sum()) == 0:
                # Singleton identities/sessions have no recoverable peer.
                continue
            nearest = candidate_indices[
                int(np.argmax(similarity[index, candidate_indices]))
            ]
            hits.append(float(labels[nearest] == labels[index]))
            chances.append(float(same_label.mean()))
        if not hits:
            return {
                "n": int(len(values)),
                "eligible": 0,
                "accuracy": float("nan"),
                "chance": float("nan"),
                "excess_over_chance": float("nan"),
                "status": "not_estimable_no_repeated_group",
            }
        accuracy = float(np.mean(hits))
        chance = float(np.mean(chances))
        return {
            "n": int(len(values)),
            "eligible": int(len(hits)),
            "accuracy": accuracy,
            "chance": chance,
            "excess_over_chance": accuracy - chance,
            "status": "diagnostic_only",
        }

    return {
        "dataset_id": recover(datasets, within_dataset=False),
        "participant_id": recover(
            participants, within_dataset=True
        ),
        "session_id": recover(sessions, within_dataset=True),
    }
