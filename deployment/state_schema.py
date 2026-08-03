from __future__ import annotations

from dataclasses import dataclass, field


ALLOWED_STATE_VALUES: dict[str, list[str]] = {
    "posture": ["standing", "sitting", "reclining", "lying"],
    "movement": ["vigorous", "moderate", "light", "sedentary", "sleep"],
    "social_engagement": ["low", "middle", "high"],
    "interpersonal_density": ["0", "1", "2", "3"],
    "device_interaction_behavior": [
        "no_interaction",
        "passive_viewing",
        "short_taps",
        "continuous_scrolling",
        "active_input",
    ],
    "environment": ["indoor", "outdoor", "dynamic", "quiet", "crowded"],
    "temporal": ["brief", "intermittent", "sustained", "continuous"],
}


@dataclass
class UserSessionState:
    posture: str
    movement: str
    social_engagement: str
    interpersonal_density: str
    device_interaction_behavior: str
    environment: str
    temporal: str
    digital_summary: str = ""  # LLM이 생성한 디지털 행동 요약 (선택적)


def validate_state(state: dict[str, str]) -> None:
    for k, allowed in ALLOWED_STATE_VALUES.items():
        if k not in state:
            raise ValueError(f"[MISSING] {k}")
        v = str(state[k]).strip()
        if v not in allowed:
            raise ValueError(f"[INVALID] {k}: {v}")
    # digital_summary는 자유 텍스트라 값 검증 안 함


def state_signature(state: dict[str, str]) -> tuple[str, ...]:
    validate_state(state)
    return (
        state["posture"],
        state["movement"],
        state["social_engagement"],
        state["interpersonal_density"],
        state["device_interaction_behavior"],
        state["environment"],
        state["temporal"],
    )