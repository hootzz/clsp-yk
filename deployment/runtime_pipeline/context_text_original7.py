"""
Seven-slot context schema used by the HQS runtime pipeline:

    posture / energy_expenditure / social_engagement / interpersonal_density /
    app_window / duration / event_type

Only observable fields are used. Unknown fields remain `not_recorded`.
"""
from __future__ import annotations

import re
from typing import Any

SLOT_NAMES = (
    "posture",
    "energy_expenditure",
    "social_engagement",
    "interpersonal_density",
    "app_window",
    "duration",
    "event_type",
)

# Target leakage guard.
FORBIDDEN_LABEL_HINTS = (
    "arousal", "valence", "cognitive load", "mental workload", "hvha", "hvla",
    "lvha", "lvla", "stress", "stressed", "anxious", "overloaded",
    "mental state", "emotion",
)

_ENERGY = {"vigorous", "moderate", "light", "sedentary", "sleep", "not_recorded"}
_DUR_BUCKETS = (
    (0, 60, "under_one_minute"),
    (60, 300, "one_to_five_minutes"),
    (300, 1800, "five_to_thirty_minutes"),
    (1800, 10**9, "over_thirty_minutes"),
)


def _norm(v: Any) -> str:
    s = "" if v is None else str(v).strip()
    return s if s else "not_recorded"


def duration_bucket(seconds: Any) -> str:
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "not_recorded"
    for lo, hi, name in _DUR_BUCKETS:
        if lo <= s < hi:
            return name
    return "not_recorded"


def user_state_to_slots(user_state: dict[str, Any], duration_seconds: Any = None) -> dict[str, str]:
    """Map runtime user_state to seven-slot context values.

    Only observable fields are used; unknowns remain not_recorded.
    """
    phys = user_state.get("physical", {}) or {}
    soc = user_state.get("social", {}) or {}
    dig = user_state.get("digital", {}) or {}

    posture = _norm(phys.get("posture"))
    # Use movement as a fallback for energy expenditure.
    energy = _norm(phys.get("energy_expenditure", phys.get("movement")))
    if energy not in _ENERGY:
        energy = energy  # Preserve unrecognized observable values.
    social = _norm(soc.get("engagement", soc.get("social_engagement")))
    density = _norm(soc.get("interpersonal_density"))

    # Map observable device usage fields.
    device = _norm(dig.get("device_type"))
    app_window = _norm(dig.get("app_window"))
    if app_window == "not_recorded" and device != "not_recorded":
        app_window = f"{device}_application_window"
    duration = duration_bucket(duration_seconds) if duration_seconds is not None else _norm(dig.get("duration"))
    event_type = _norm(dig.get("event_type"))

    return {
        "posture": posture,
        "energy_expenditure": energy,
        "social_engagement": social,
        "interpersonal_density": density,
        "app_window": app_window,
        "duration": duration,
        "event_type": event_type,
    }


def validate_slots(context: dict[str, Any]) -> dict[str, str]:
    missing = [k for k in SLOT_NAMES if k not in context]
    if missing:
        raise ValueError(f"Missing original-7 fields: {missing}")
    return {k: _norm(context[k]) for k in SLOT_NAMES}


def validate_text(text: str) -> list[str]:
    low = " ".join(str(text).split()).lower()
    return [f"target leakage term: {t}" for t in FORBIDDEN_LABEL_HINTS if t in low]


def render(context: dict[str, Any], observable_detail: str = "") -> str:
    """Render seven-slot context in the model's training sentence style."""
    s = validate_slots(context)
    posture = s["posture"].replace("_", " ")
    energy = s["energy_expenditure"].replace("_", " ")
    app_window = s["app_window"].replace("_", " ")
    duration = s["duration"].replace("_", " ")
    event = s["event_type"].replace("_", " ")
    social = s["social_engagement"].replace("_", " ")
    density = s["interpersonal_density"]

    posture_clause = "Posture not recorded" if posture == "not recorded" else posture.capitalize()
    energy_clause = f"with {energy} energy expenditure" if energy != "not recorded" else "with energy expenditure not recorded"
    if social == "not recorded":
        social_clause = "social engagement not recorded"
    else:
        social_clause = f"social engagement {social}"
    density_clause = "interpersonal density not recorded" if density == "not_recorded" else f"interpersonal density {density}"
    app_clause = "The App/Window is not recorded" if app_window == "not recorded" else f"The App/Window is {app_window}"
    dur_clause = "an interval not recorded" if duration == "not recorded" else f"an interval of {duration}"
    event_clause = "an event not recorded" if event == "not recorded" else f"a {event} event"
    detail = " ".join(str(observable_detail).split()).strip(" .")
    detail_clause = f" The observed activity is: {detail}." if detail else ""

    text = (
        f"{posture_clause} {energy_clause}. "
        f"The social context has {social_clause} and {density_clause}. "
        f"{app_clause} for {dur_clause}, during {event_clause}.{detail_clause}"
    )
    issues = validate_text(text)
    if issues:
        raise ValueError(f"Generated original-7 context failed validation: {issues}")
    return re.sub(r"\s+", " ", text).strip()


def build_context_text(user_state: dict[str, Any], observable_detail: str = "",
                       duration_seconds: Any = None) -> str:
    slots = user_state_to_slots(user_state, duration_seconds=duration_seconds)
    return render(slots, observable_detail=observable_detail)


def context_signature(context: dict[str, Any]) -> str:
    s = validate_slots(context)
    return "|".join(s[k] for k in SLOT_NAMES)
