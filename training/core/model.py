"""v3 model — MultimodalStateEstimator (self-contained).

논문 §3.1.3 구조를 보존하되 v3 확장:
  - per-target heads (valence / arousal / cognitive_load) — partial-label 마스킹과 정합
  - modality dropout (학습 시 text 또는 ppg 임베딩 확률적 제거) — modality dominance 완화 (THEORETICAL_REVIEW §5.3)
  - ppg_mask 입력 — PPG 없는 샘플(SWELL 등)은 ppg_z=0 강제

Text : DistilBERT[CLS]768 → ProjectionHead(GELU+residual) → 256
PPG  : frozen offline PaPaGEI-P512 → trainable projection → 256

Fusion modes:
  - concat (legacy/default): strict backward compatibility for old checkpoints
  - context_prior_quality_gated_ppg_residual (rescue):
      prediction_t = context_prior_t + ppg_confidence × ppg_residual_t
  - target_routed_direct:
      valence = context-only target branch
      arousal/cognitive_load = residual-free direct [context, PPG] branches
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

TARGETS = ("valence", "arousal", "cognitive_load")
POSTURE_FILM_CATEGORIES = ("lying", "reclining", "standing")


class ProjectionHead(nn.Module):
    """논문 Text Projection: 2×Linear, GELU, dropout, LayerNorm + residual."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        z = self.fc2(self.act(h))
        z = self.drop(z)
        return self.norm(z + h)


class TextBranch(nn.Module):
    def __init__(
        self,
        model_name: str,
        projection_dim: int,
        dropout: float = 0.1,
        init_ckpt: str | None = None,
        max_length: int = 96,
    ):
        super().__init__()
        self.projection_dim = int(projection_dim)
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        # 7-slot 사전학습 인코더 이식 (context7 best.pt의 text_encoder_state)
        if init_ckpt:
            import torch as _t
            obj = _t.load(init_ckpt, map_location="cpu", weights_only=False)
            enc_sd = obj["text_encoder_state"] if isinstance(obj, dict) and "text_encoder_state" in obj else obj
            miss, unexp = self.encoder.load_state_dict(enc_sd, strict=False)
            print(f"[TextBranch] 7-slot 인코더 이식: missing={len(miss)} unexpected={len(unexp)}")
        self.proj = ProjectionHead(self.encoder.config.hidden_size, projection_dim, dropout)

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)
        cls = self.encoder(**tok).last_hidden_state[:, 0, :]
        return self.proj(cls)


class PPGBranch(nn.Module):
    """논문 §3.1.3.5: PaPaGEI(512) → Linear→ReLU→Dropout→Linear→LayerNorm → 256."""

    def __init__(self, input_dim: int, projection_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualPostureFiLM(nn.Module):
    """Small posture-conditioned residual adapter on frozen PaPaGEI features.

    ``sitting`` and unknown postures are the identity reference. ``walking`` is
    conservatively mapped to the available upright ``standing`` adapter.
    Parameters are bounded by tanh and scaled so the adapter cannot replace the
    physiological representation.
    """

    def __init__(self, input_dim: int, residual_scale: float = 0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.residual_scale = float(residual_scale)
        self.gamma = nn.Parameter(
            torch.zeros(len(POSTURE_FILM_CATEGORIES), self.input_dim)
        )
        self.beta = nn.Parameter(
            torch.zeros(len(POSTURE_FILM_CATEGORIES), self.input_dim)
        )

    @staticmethod
    def _category_index(posture: str) -> int:
        normalized = str(posture).strip().lower()
        if normalized in {"walking", "walk", "upright"}:
            normalized = "standing"
        try:
            # Zero is reserved for the identity reference.
            return POSTURE_FILM_CATEGORIES.index(normalized) + 1
        except ValueError:
            return 0

    def forward(
        self,
        features: torch.Tensor,
        postures: list[str] | tuple[str, ...] | None,
    ) -> torch.Tensor:
        if not postures:
            return features
        if len(postures) != features.shape[0]:
            raise ValueError(
                "posture count must match PPG batch size: "
                f"{len(postures)} != {features.shape[0]}"
            )
        indices = torch.as_tensor(
            [self._category_index(value) for value in postures],
            device=features.device,
            dtype=torch.long,
        )
        # [B,4] -> discard identity column -> [B,3].
        weights = F.one_hot(
            indices, num_classes=len(POSTURE_FILM_CATEGORIES) + 1
        )[:, 1:].to(features.dtype)
        gamma = weights @ self.gamma
        beta = weights @ self.beta
        scale = self.residual_scale
        return features * (1.0 + scale * torch.tanh(gamma)) + scale * beta


class MultimodalStateEstimator(nn.Module):
    def __init__(
        self,
        text_model_name: str = "distilbert-base-uncased",
        ppg_input_dim: int = 512,
        projection_dim: int = 256,
        projection_dropout: float = 0.1,
        ppg_hidden_dim: int = 128,
        fusion_hidden_dim: int = 256,
        text_init_ckpt: str | None = None,
        text_max_length: int = 96,
        posture_film_enabled: bool = False,
        posture_film_residual_scale: float = 0.1,
        fusion_mode: str = "concat",
    ):
        super().__init__()
        self.projection_dim = int(projection_dim)
        self.fusion_mode = str(fusion_mode).strip().lower()
        if self.fusion_mode not in {
            "concat",
            "context_prior_quality_gated_ppg_residual",
            "target_routed_direct",
        }:
            raise ValueError(
                "fusion_mode must be 'concat', "
                "'context_prior_quality_gated_ppg_residual', or "
                "'target_routed_direct'"
            )
        self.text_branch = TextBranch(
            text_model_name,
            projection_dim,
            projection_dropout,
            init_ckpt=text_init_ckpt,
            max_length=text_max_length,
        )
        self.ppg_branch = PPGBranch(ppg_input_dim, projection_dim, ppg_hidden_dim, projection_dropout)
        self.posture_film = (
            ResidualPostureFiLM(
                ppg_input_dim,
                residual_scale=posture_film_residual_scale,
            )
            if posture_film_enabled
            else None
        )

        if self.fusion_mode == "concat":
            self.trunk = nn.Sequential(
                nn.Linear(projection_dim * 2, fusion_hidden_dim),
                nn.ReLU(inplace=True),
            )
            # per-target heads (partial-label 마스킹과 정합)
            self.heads = nn.ModuleDict(
                {t: nn.Linear(fusion_hidden_dim, 1) for t in TARGETS}
            )
            self.context_prior_heads = None
            self.ppg_residual_trunk = None
            self.ppg_residual_heads = None
            self.target_context_trunks = None
            self.target_multimodal_trunks = None
            self.target_routed_heads = None
        elif self.fusion_mode == "target_routed_direct":
            # Hard target routing: PPG cannot enter the Valence computation.
            # Arousal and Cognitive load each receive an independent direct
            # fusion trunk so one target cannot use another target's fusion
            # representation.
            self.trunk = None
            self.heads = None
            self.context_prior_heads = None
            self.ppg_residual_trunk = None
            self.ppg_residual_heads = None
            self.target_context_trunks = nn.ModuleDict(
                {
                    "valence": nn.Sequential(
                        nn.Linear(projection_dim, fusion_hidden_dim),
                        nn.ReLU(inplace=True),
                    )
                }
            )
            self.target_multimodal_trunks = nn.ModuleDict(
                {
                    target: nn.Sequential(
                        nn.Linear(projection_dim * 2, fusion_hidden_dim),
                        nn.ReLU(inplace=True),
                    )
                    for target in ("arousal", "cognitive_load")
                }
            )
            self.target_routed_heads = nn.ModuleDict(
                {t: nn.Linear(fusion_hidden_dim, 1) for t in TARGETS}
            )
        else:
            # Context predicts the prior. PPG is restricted to an additive
            # residual whose contribution is multiplied by measured quality.
            self.trunk = None
            self.heads = None
            self.context_prior_heads = nn.ModuleDict(
                {t: nn.Linear(projection_dim, 1) for t in TARGETS}
            )
            self.ppg_residual_trunk = nn.Sequential(
                nn.Linear(projection_dim, fusion_hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.ppg_residual_heads = nn.ModuleDict(
                {t: nn.Linear(fusion_hidden_dim, 1) for t in TARGETS}
            )
            self.target_context_trunks = None
            self.target_multimodal_trunks = None
            self.target_routed_heads = None
            # A PPG-only checkpoint sees an all-zero text vector. Starting the
            # context weights at zero lets that checkpoint initialize fusion
            # without a random context perturbation.
            for head in self.context_prior_heads.values():
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def ppg_backbone_named_parameters(self):
        """Yield the reusable PPG-side parameters for the active fusion mode.

        Residual fusion can reuse its PPG predictor and scalar intercepts.
        Direct concat fusion has no modality-specific prediction head, so its
        common warm start is deliberately limited to ``ppg_branch``.
        """
        for name, parameter in self.ppg_branch.named_parameters():
            yield f"ppg_branch.{name}", parameter
        if self.ppg_residual_trunk is not None:
            for name, parameter in self.ppg_residual_trunk.named_parameters():
                yield f"ppg_residual_trunk.{name}", parameter
        if self.ppg_residual_heads is not None:
            for name, parameter in self.ppg_residual_heads.named_parameters():
                yield f"ppg_residual_heads.{name}", parameter
        if self.context_prior_heads is not None:
            for target, head in self.context_prior_heads.items():
                yield f"context_prior_heads.{target}.bias", head.bias

    def load_ppg_backbone_state(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, list[str]]:
        """Load the selected PPG predictor without importing its text branch."""
        expected = {
            name for name, _ in self.ppg_backbone_named_parameters()
        }
        own = self.state_dict()
        missing = sorted(expected - set(state_dict))
        incompatible = sorted(
            name
            for name in expected & set(state_dict)
            if tuple(own[name].shape) != tuple(state_dict[name].shape)
        )
        if missing or incompatible:
            raise ValueError(
                "PPG backbone checkpoint is incompatible: "
                f"missing={missing}, incompatible={incompatible}"
            )
        selected = {name: state_dict[name] for name in expected}
        result = self.load_state_dict(selected, strict=False)
        return {
            "loaded": sorted(selected),
            "missing_outside_backbone": sorted(result.missing_keys),
        }

    def forward(
        self,
        texts: list[str],
        ppg_features: torch.Tensor,
        device: torch.device,
        ppg_mask: torch.Tensor | None = None,   # [B] 1=PPG 있음, 0=없음(SWELL 등)
        modality_dropout: float = 0.0,           # 학습 시 text/ppg 확률적 제거
        mute_text: bool = False,                 # ablation: text 브랜치 0
        mute_ppg: bool = False,                  # ablation: ppg 브랜치 0
        postures: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        # CLSP consumes clean projected embeddings. Modality dropout and
        # ablation switches are applied only to separate fusion tensors.
        ppg_features_clean = ppg_features.to(device)
        if self.posture_film is not None:
            ppg_features_clean = self.posture_film(
                ppg_features_clean,
                postures,
            )
        batch_size = ppg_features_clean.shape[0]
        # In a true single-modality baseline the muted branch cannot
        # contribute to fusion or CLSP.  Avoid evaluating that encoder while
        # preserving exactly the same zero tensor seen by the fusion trunk.
        text_z_clean = (
            torch.zeros(
                batch_size,
                self.projection_dim,
                device=device,
                dtype=ppg_features_clean.dtype,
            )
            if mute_text
            else self.text_branch(texts, device)
        )
        ppg_z_projected = (
            torch.zeros(
                batch_size,
                self.projection_dim,
                device=device,
                dtype=ppg_features_clean.dtype,
            )
            if mute_ppg
            else self.ppg_branch(ppg_features_clean)
        )

        quality_gate = torch.ones(
            batch_size,
            device=device,
            dtype=ppg_features_clean.dtype,
        )
        # PPG 결측 샘플은 임베딩 0
        if ppg_mask is not None:
            quality_gate = ppg_mask.to(
                device=device,
                dtype=ppg_features_clean.dtype,
            ).view(-1).clamp(0.0, 1.0)
        # Keep the historical quality-masked tensor for the shared CLSP path.
        # Residual fusion uses the unscaled projection and applies quality
        # exactly once at the scalar residual output.
        ppg_z_clean = (
            ppg_z_projected * quality_gate.view(-1, 1)
        )

        text_z = text_z_clean
        ppg_z = (
            ppg_z_projected
            if self.fusion_mode
            == "context_prior_quality_gated_ppg_residual"
            else ppg_z_clean
        )

        # ablation 스위치 (PPG/text 기여도 측정용)
        if mute_text:
            text_z = text_z * 0.0
        if mute_ppg:
            ppg_z = ppg_z * 0.0
            quality_gate = quality_gate * 0.0

        # modality dropout (둘 다 죽이지는 않음)
        if self.training and modality_dropout > 0.0:
            B = text_z.size(0)
            drop = torch.rand(B, device=device)
            drop_text = (drop < modality_dropout / 2).view(-1, 1)
            drop_ppg = ((drop >= modality_dropout / 2) & (drop < modality_dropout)).view(-1, 1)
            text_z = text_z * (~drop_text)
            ppg_z = ppg_z * (~drop_ppg)
            quality_gate = quality_gate * (~drop_ppg.view(-1)).to(
                quality_gate.dtype
            )

        target_routes = None
        if self.fusion_mode == "concat":
            fused = torch.cat([text_z, ppg_z], dim=-1)     # [B,512]
            h = self.trunk(fused)
            preds = {
                t: self.heads[t](h).squeeze(-1) for t in TARGETS
            }
            context_prior = None
            ppg_residual = None
        elif self.fusion_mode == "target_routed_direct":
            valence_h = self.target_context_trunks["valence"](text_z)
            fused = torch.cat([text_z, ppg_z], dim=-1)
            multimodal_h = {
                target: self.target_multimodal_trunks[target](fused)
                for target in ("arousal", "cognitive_load")
            }
            preds = {
                "valence": self.target_routed_heads["valence"](
                    valence_h
                ).squeeze(-1),
                "arousal": self.target_routed_heads["arousal"](
                    multimodal_h["arousal"]
                ).squeeze(-1),
                "cognitive_load": self.target_routed_heads[
                    "cognitive_load"
                ](multimodal_h["cognitive_load"]).squeeze(-1),
            }
            context_prior = None
            ppg_residual = None
            target_routes = {
                "valence": "context_only",
                "arousal": "direct_context_ppg",
                "cognitive_load": "direct_context_ppg",
            }
        else:
            residual_h = self.ppg_residual_trunk(ppg_z)
            context_prior = {
                target: self.context_prior_heads[target](
                    text_z
                ).squeeze(-1)
                for target in TARGETS
            }
            ppg_residual = {
                target: self.ppg_residual_heads[target](
                    residual_h
                ).squeeze(-1)
                for target in TARGETS
            }
            preds = {
                target: (
                    context_prior[target]
                    + quality_gate * ppg_residual[target]
                )
                for target in TARGETS
            }
        return {
            "text_z": text_z,
            "ppg_z": ppg_z,
            "text_z_clean": text_z_clean,
            "ppg_z_clean": ppg_z_clean,
            "ppg_features_adapted": ppg_features_clean,
            "ppg_quality_gate": quality_gate,
            "context_prior": context_prior,
            "ppg_residual": ppg_residual,
            "target_routes": target_routes,
            "fusion_mode": self.fusion_mode,
            "preds": preds,
        }
