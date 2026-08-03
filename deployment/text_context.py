from __future__ import annotations

import re
from dataclasses import dataclass

from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from state_schema import ALLOWED_STATE_VALUES

for _env in [
    Path(__file__).parent.parent / ".env",
    Path.home() / "ieeevr" / ".env",
]:
    if _env.exists():
        load_dotenv(_env)
        break

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


_SYSTEM_PROMPT = (
    "You describe a user's current situational context for an XR personalization system. "
    "Given structured context fields, write ONE natural English sentence (15-30 words) "
    "describing the user's situation. "
    "Rules: "
    "1. Do NOT mention emotions, feelings, stress, arousal, valence, cognitive load, or mental state. "
    "2. Focus on observable situation: what the user is doing, where they are, who is around. "
    "3. Be concise and factual. "
    "4. Output the sentence only, no extra text."
)

_DEVICE_LABEL: dict[str, str] = {
    "no_interaction":       "not interacting with any device",
    "passive_viewing":      "passively viewing content",
    "short_taps":           "using short taps on a device",
    "continuous_scrolling": "continuously scrolling on a device",
    "active_input":         "actively inputting on a device",
}

_DENSITY_LABEL: dict[str, str] = {
    "0": "alone with no one nearby",
    "1": "with one person nearby",
    "2": "with two people nearby",
    "3": "with three or more people nearby",
}


@dataclass
class ContextFields:
    posture:                    str = "sitting"
    movement:                   str = "sedentary"
    social_engagement:          str = "low"
    interpersonal_density:      str = "0"
    device_interaction_behavior: str = "passive_viewing"
    environment:                str = "indoor"
    temporal:                   str = "intermittent"
    digital_summary:            str = ""  


def validate_context(context: dict[str, str]) -> None:
    for key, allowed in ALLOWED_STATE_VALUES.items():
        if key not in context:
            raise ValueError(f"[MISSING] {key}")
        v = str(context[key]).strip()
        if v not in allowed:
            raise ValueError(f"[INVALID] {key}: {v!r}  (allowed: {allowed})")


def build_text(context: dict[str, str]) -> str:
    validate_context(context)

    device_desc = _DEVICE_LABEL.get(
        context["device_interaction_behavior"],
        context["device_interaction_behavior"],
    )
    density_desc = _DENSITY_LABEL.get(context["interpersonal_density"], context["interpersonal_density"])

    user_prompt = (
        f"Context fields:\n"
        f"- Posture: {context['posture']}\n"
        f"- Movement: {context['movement']}\n"
        f"- Social engagement level: {context['social_engagement']}\n"
        f"- Nearby people: {density_desc}\n"
        f"- Device interaction: {device_desc}\n"
        f"- Environment: {context['environment']}\n"
        f"- Temporal pattern: {context['temporal']}\n"
        f"- Recent digital activity: {context.get('digital_summary', '')}\n\n"  # 추가
        f"Write one sentence describing this user's situational context."
    )

    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=100,
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def context_from_state(
    posture: str,
    movement: str,
    social_engagement: str,
    interpersonal_density: str,
    device_interaction_behavior: str,
    environment: str,
    temporal: str,
) -> ContextFields:
    context = {
        "posture": posture,
        "movement": movement,
        "social_engagement": social_engagement,
        "interpersonal_density": interpersonal_density,
        "device_interaction_behavior": device_interaction_behavior,
        "environment": environment,
        "temporal": temporal,
    }
    validate_context(context)
    return ContextFields(**context)


FORBIDDEN_LABEL_HINTS = [
    "stressed", "stress",
    "high cognitive load", "low cognitive load",
    "high arousal", "low arousal",
    "positive valence", "negative valence",
    "anxious", "overloaded", "excited", "uncomfortable",
]


def validate_context_text(text: str, min_words: int = 8) -> tuple[bool, list[str]]:
    issues: list[str] = []
    cleaned = text.strip()

    if not cleaned:
        issues.append("empty text")
        return False, issues

    word_count = len(re.findall(r"\b\w+\b", cleaned))
    if word_count < min_words:
        issues.append(f"too short ({word_count} words)")

    low = cleaned.lower()
    for bad in FORBIDDEN_LABEL_HINTS:
        if re.search(r"\b" + re.escape(bad) + r"\b", low):
            issues.append(f"label leakage: '{bad}'")

    return len(issues) == 0, issues
