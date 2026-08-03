"""Leakage-safe natural-language rendering of observed seven-slot context."""
from __future__ import annotations

import json
import os
import re
from typing import Any


SLOT_NAMES = (
    "posture",
    "movement",
    "social_engagement",
    "interpersonal_density",
    "device_interaction_behavior",
    "environment",
    "temporal",
)

FORBIDDEN_LABEL_HINTS = (
    "arousal",
    "valence",
    "cognitive load",
    "mental workload",
    "hvha",
    "hvla",
    "lvha",
    "lvla",
    "stress",
    "stressed",
    "anxious",
    "overloaded",
    "mental state",
    "emotion",
)

DEVICE_TEXT = {
    "no_interaction": "not interacting with a digital device",
    "passive_viewing": "passively viewing digital content",
    "short_taps": "using short taps on a device",
    "continuous_scrolling": "continuously scrolling on a device",
    "active_input": "actively entering input on a device",
    "not_recorded": "with device interaction not recorded",
}

DENSITY_TEXT = {
    "0": "with no one nearby",
    "1": "with one person nearby",
    "2": "with two people nearby",
    "3": "with three or more people nearby",
    "not_recorded": "with nearby-person density not recorded",
}


def validate_slots(context: dict[str, Any]) -> dict[str, str]:
    missing = [key for key in SLOT_NAMES if key not in context]
    if missing:
        raise ValueError(f"Missing seven-slot fields: {missing}")
    normalized = {key: str(context[key]).strip() for key in SLOT_NAMES}
    empty = [key for key, value in normalized.items() if not value]
    if empty:
        raise ValueError(f"Empty seven-slot fields: {empty}")
    return normalized


def validate_text(text: str) -> list[str]:
    issues: list[str] = []
    clean = " ".join(str(text).split())
    if len(clean.split()) < 12:
        issues.append("context text is too short")
    low = clean.lower()
    for term in FORBIDDEN_LABEL_HINTS:
        if term in low:
            issues.append(f"target leakage term: {term}")
    return issues


def deterministic_text(context: dict[str, Any], observable_detail: str = "") -> str:
    slots = validate_slots(context)
    posture = slots["posture"].replace("_", " ")
    movement = slots["movement"].replace("_", " ")
    environment = slots["environment"].replace("_", " ")
    temporal = slots["temporal"].replace("_", " ")
    social = slots["social_engagement"].replace("_", " ")
    density = DENSITY_TEXT.get(
        slots["interpersonal_density"],
        f"with interpersonal density {slots['interpersonal_density']}",
    )
    device = DEVICE_TEXT.get(
        slots["device_interaction_behavior"],
        slots["device_interaction_behavior"].replace("_", " "),
    )
    detail = " ".join(str(observable_detail).split()).strip(" .")
    activity = f" The observed activity is: {detail}." if detail else ""
    if social == "not recorded":
        social_clause = f"Social engagement is not recorded, {density}."
    else:
        social_clause = f"Social engagement is {social}, {density}."
    article = "an" if environment[:1].lower() in "aeiou" else "a"
    if posture == "not recorded":
        physical_clause = (
            f"Posture is not recorded and movement is {movement}"
            if movement != "not recorded"
            else "Posture and movement are not recorded"
        )
    else:
        physical_clause = (
            f"The user is {posture} with movement not recorded"
            if movement == "not recorded"
            else f"The user is {posture} with {movement} movement"
        )
    text = (
        f"{physical_clause} in {article} {environment} setting, {device}, "
        f"over a {temporal} interval. {social_clause}{activity}"
    )
    issues = validate_text(text)
    if issues:
        raise ValueError(f"Generated context failed validation: {issues}")
    return text


def _openai_text(
    context: dict[str, Any],
    observable_detail: str,
    model: str,
) -> str:
    from openai import OpenAI

    slots = validate_slots(context)
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for --context-backend openai")
    prompt = {
        "seven_slot_context": slots,
        "observable_detail": observable_detail,
        "requirements": {
            "all_slots": list(SLOT_NAMES),
            "missing_values": "Keep not_recorded explicit; do not infer alone, zero, or low.",
            "forbidden": list(FORBIDDEN_LABEL_HINTS),
            "maximum_words": 60,
        },
    }
    response = OpenAI(api_key=key).chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Write one concise factual English description of observed user context. "
                    "Represent all seven fields, preserve the observable detail, and never infer "
                    "an emotion or target state. Return JSON as {\"text\": \"...\"}."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    )
    obj = json.loads(response.choices[0].message.content or "{}")
    text = " ".join(str(obj["text"]).split())
    issues = validate_text(text)
    if issues:
        raise ValueError(f"GPT context failed validation: {issues}")
    if "not_recorded" in slots.values() and "not recorded" not in text.lower():
        raise ValueError("GPT context fabricated or omitted a not_recorded field")
    return text


def build_context_text(
    context: dict[str, Any],
    observable_detail: str = "",
    backend: str = "deterministic",
    model: str = "gpt-4o-mini",
) -> str:
    """Render context without changing any supplied slot value."""
    if backend == "deterministic":
        return deterministic_text(context, observable_detail)
    if backend == "openai":
        return _openai_text(context, observable_detail, model)
    raise ValueError("backend must be 'deterministic' or 'openai'")


def context_signature(context: dict[str, Any]) -> str:
    slots = validate_slots(context)
    return "|".join(slots[key] for key in SLOT_NAMES)
