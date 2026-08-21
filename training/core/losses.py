"""Partial-label masked MSE and supervised-contrastive CLSP losses.

partial_masked_mse:
  Computes MSE only for labeled target/sample pairs. NaN labels are excluded
  by mask=0, allowing datasets with different target availability in one batch.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from model import TARGETS


def partial_masked_mse(
    preds: dict[str, torch.Tensor],   # {target: [B]}
    targets: dict[str, torch.Tensor], # {target: [B]} missing values excluded by mask
    masks: dict[str, torch.Tensor],   # {target: [B]} 1=labeled
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    total = preds[TARGETS[0]].new_zeros(())
    n_terms = 0
    logs: dict[str, float] = {}
    if sample_weights is None:
        sample_weights = torch.ones_like(masks[TARGETS[0]])
    sample_weights = sample_weights.to(
        device=preds[TARGETS[0]].device,
        dtype=preds[TARGETS[0]].dtype,
    )
    if torch.any(~torch.isfinite(sample_weights)) or torch.any(
        sample_weights < 0
    ):
        raise ValueError("sample_weights must be finite and non-negative")
    for t in TARGETS:
        m = masks[t].to(sample_weights.dtype)
        effective_weight = m * sample_weights
        denom = effective_weight.sum()
        if denom > 0:
            se = ((preds[t] - targets[t]) ** 2) * effective_weight
            l = se.sum() / denom
            total = total + l
            n_terms += 1
            logs[f"mse_{t}"] = float(l.detach())
        else:
            logs[f"mse_{t}"] = float("nan")
    if n_terms > 0:
        total = total / n_terms
    return total, logs


def dataset_balanced_mse(
    preds,
    targets,
    masks,
    datasets,
    dataset_weights: dict[str, float] | None = None,
    sample_weights: torch.Tensor | None = None,
):
    """Average partial-label masked MSE across datasets present in the batch.

    This prevents datasets with more rows from dominating the loss.
    ``dataset_weights`` applies explicit weights only to dataset-level loss
    terms; sampling probabilities are not modified here.
    """
    ds = np.asarray(datasets)
    dev = preds[TARGETS[0]].device
    terms = []
    weights = []
    configured = {
        str(key).lower(): float(value)
        for key, value in (dataset_weights or {}).items()
    }
    for d in np.unique(ds):
        sel = torch.tensor(ds == d, device=dev)
        if sel.sum() == 0:
            continue
        sp = {t: preds[t][sel] for t in TARGETS}
        sy = {t: targets[t][sel] for t in TARGETS}
        sm = {t: masks[t][sel] for t in TARGETS}
        sw = sample_weights[sel] if sample_weights is not None else None
        if sum(float(sm[t].sum()) for t in TARGETS) == 0:
            continue
        l, _ = partial_masked_mse(
            sp,
            sy,
            sm,
            sample_weights=sw,
        )
        weight = configured.get(str(d).lower(), 1.0)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"dataset weight must be finite and non-negative: {d}={weight}"
            )
        if weight == 0.0:
            continue
        terms.append(l)
        weights.append(weight)
    if not terms:
        return preds[TARGETS[0]].new_zeros(())
    weight_tensor = torch.as_tensor(
        weights,
        device=dev,
        dtype=terms[0].dtype,
    )
    return (torch.stack(terms) * weight_tensor).sum() / weight_tensor.sum()


def supcon_clsp(
    text_z: torch.Tensor,       # [B,256]
    ppg_z: torch.Tensor,        # [B,256]
    context_sig: list[str],     # [B]
    valid: torch.Tensor,        # [B] 1=valid PPG
    temperature: float = 0.07,
) -> torch.Tensor:
    device = text_z.device
    vmask = valid.to(device).bool()
    if vmask.sum() < 2:
        return text_z.new_zeros(())

    t = F.normalize(text_z[vmask], dim=-1)
    p = F.normalize(ppg_z[vmask], dim=-1)
    sigs = [s for s, keep in zip(context_sig, vmask.tolist()) if keep]

    logits = t @ p.T / temperature                          # [V,V] text→ppg
    B = t.size(0)
    pos = torch.zeros(B, B, device=device)
    for i in range(B):
        for j in range(B):
            if sigs[i] == sigs[j]:
                pos[i, j] = 1.0
    pos = pos / pos.sum(dim=1, keepdim=True).clamp_min(1e-8)  # soft-positive distribution

    loss_t = (-pos * F.log_softmax(logits, dim=-1)).sum(dim=1)
    loss_p = (-pos.T * F.log_softmax(logits.T, dim=-1)).sum(dim=1)
    return ((loss_t + loss_p) / 2).mean()


def _equality_mask(values: list[str], device: torch.device) -> torch.Tensor:
    array = np.asarray([str(value) for value in values], dtype=object)
    return torch.as_tensor(array[:, None] == array[None, :], device=device)


def domain_safe_clsp(
    text_z: torch.Tensor,
    ppg_z: torch.Tensor,
    pair_ids: list[str],
    negative_mask_sigs: list[str],
    semantic_sigs: list[str],
    datasets: list[str],
    valid: torch.Tensor,
    temperature: float = 0.07,
    semantic_positive_weight: float = 0.0,
    participant_ids: list[str] | None = None,
    session_ids: list[str] | None = None,
    label_group_sigs: list[str] | None = None,
    window_indices: list[int] | None = None,
    adjacent_radii: list[int] | None = None,
    loss_groups: list[str] | None = None,
) -> torch.Tensor:
    """Dataset-balanced CLSP without trivial cross-dataset negatives.

    Candidate negatives are restricted to the anchor's dataset. Exact pairs
    are the only primary positives. Samples sharing the same teacher sentence
    are removed from one another's denominator rather than pulled together as
    positives. An explicitly curated semantic signature may add weak
    positives, but unmatched cross-dataset samples never become negatives.

    Rows sharing participant, session, label group, or an adjacent window
    are excluded from the negative denominator. Each declared teacher/loss
    group is averaged separately.
    """
    if not 0.0 <= semantic_positive_weight <= 1.0:
        raise ValueError("semantic_positive_weight must be in [0, 1]")

    device = text_z.device
    valid_mask = valid.to(device).bool()
    if int(valid_mask.sum()) < 2:
        return text_z.new_zeros(())

    keep = valid_mask.detach().cpu().tolist()
    text = F.normalize(text_z[valid_mask], dim=-1)
    ppg = F.normalize(ppg_z[valid_mask], dim=-1)
    kept_pairs = [value for value, flag in zip(pair_ids, keep) if flag]
    kept_negative_mask = [
        value for value, flag in zip(negative_mask_sigs, keep) if flag
    ]
    kept_semantic = [
        value for value, flag in zip(semantic_sigs, keep) if flag
    ]
    kept_datasets = [
        value for value, flag in zip(datasets, keep) if flag
    ]
    row_count = len(pair_ids)

    def kept_optional(
        values: list | None,
        default,
    ) -> list:
        source = values if values is not None else [default] * row_count
        if len(source) != row_count:
            raise ValueError(
                "CLSP group metadata length must match pair_ids"
            )
        return [value for value, flag in zip(source, keep) if flag]

    kept_participants = kept_optional(participant_ids, "")
    kept_sessions = kept_optional(session_ids, "")
    kept_label_groups = kept_optional(label_group_sigs, "")
    kept_window_indices = [
        int(value) for value in kept_optional(window_indices, -1)
    ]
    kept_adjacent_radii = [
        max(0, int(value)) for value in kept_optional(adjacent_radii, 1)
    ]
    kept_loss_groups = [
        str(group).strip() or str(dataset)
        for group, dataset in zip(
            kept_optional(loss_groups, ""),
            kept_datasets,
        )
    ]

    same_dataset = _equality_mask(kept_datasets, device)
    same_pair = _equality_mask(kept_pairs, device)
    same_teacher = _equality_mask(kept_negative_mask, device)
    teacher_nonempty = torch.as_tensor(
        [bool(str(value).strip()) for value in kept_negative_mask],
        device=device,
        dtype=torch.bool,
    )
    same_teacher = (
        same_teacher
        & teacher_nonempty[:, None]
        & teacher_nonempty[None, :]
    )
    primary_positive = same_dataset & same_pair

    def nonempty_equality(values: list[str]) -> torch.Tensor:
        nonempty = torch.as_tensor(
            [bool(str(value).strip()) for value in values],
            device=device,
            dtype=torch.bool,
        )
        return (
            _equality_mask(values, device)
            & nonempty[:, None]
            & nonempty[None, :]
        )

    same_participant = nonempty_equality(kept_participants)
    same_session = nonempty_equality(kept_sessions)
    same_label_group = nonempty_equality(kept_label_groups)
    indices = torch.as_tensor(
        kept_window_indices,
        device=device,
        dtype=torch.long,
    )
    radii = torch.as_tensor(
        kept_adjacent_radii,
        device=device,
        dtype=torch.long,
    )
    indexed = indices.ge(0)
    adjacent_window = (
        same_session
        & indexed[:, None]
        & indexed[None, :]
        & (
            (indices[:, None] - indices[None, :]).abs()
            <= torch.maximum(radii[:, None], radii[None, :])
        )
    )

    semantic_nonempty = torch.as_tensor(
        [bool(str(value).strip()) for value in kept_semantic],
        device=device,
        dtype=torch.bool,
    )
    same_semantic = (
        _equality_mask(kept_semantic, device)
        & semantic_nonempty[:, None]
        & semantic_nonempty[None, :]
    )

    # Repeated identical teachers are ambiguous supervision: neither a
    # positive equivalence claim nor a valid negative.
    group_ambiguous = (
        same_teacher
        | same_participant
        | same_session
        | same_label_group
        | adjacent_window
    )
    candidate = same_dataset & ~(group_ambiguous & ~same_pair)
    candidate |= primary_positive
    if semantic_positive_weight > 0.0:
        candidate |= same_dataset & same_semantic

    positive_weight = primary_positive.to(text.dtype)
    if semantic_positive_weight > 0.0:
        weak_only = same_semantic & ~primary_positive
        positive_weight = positive_weight + (
            weak_only.to(text.dtype) * semantic_positive_weight
        )
    positive_weight = positive_weight / positive_weight.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-8)

    logits = text @ ppg.T / temperature

    def directional_loss(
        scores: torch.Tensor,
        candidates: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        masked_scores = scores.masked_fill(~candidates, float("-inf"))
        log_prob = F.log_softmax(masked_scores, dim=-1)
        terms = torch.where(
            weights > 0,
            -weights * log_prob,
            torch.zeros_like(log_prob),
        )
        return terms.sum(dim=1)

    text_loss = directional_loss(logits, candidate, positive_weight)
    ppg_loss = directional_loss(
        logits.T,
        candidate.T,
        positive_weight.T
        / positive_weight.T.sum(dim=1, keepdim=True).clamp_min(1e-8),
    )
    per_anchor = (text_loss + ppg_loss) / 2.0

    # Equal teacher-type/loss-group contribution even when their row counts
    # differ. Empty historical groups fall back to dataset identity.
    loss_group_array = np.asarray(kept_loss_groups, dtype=object)
    group_terms = []
    for loss_group in sorted(set(kept_loss_groups)):
        selection = torch.as_tensor(
            loss_group_array == loss_group,
            device=device,
            dtype=torch.bool,
        )
        if int(selection.sum()) > 0:
            group_terms.append(per_anchor[selection].mean())
    return (
        torch.stack(group_terms).mean()
        if group_terms
        else text_z.new_zeros(())
    )
