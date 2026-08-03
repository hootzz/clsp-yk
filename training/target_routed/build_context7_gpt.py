"""Build leakage-audited original-seven-slot training context text.

The Stage-1 text encoder was adapted on 6,000 label-free seven-slot context
signatures (four paraphrases each).  This script closes the distribution gap in
Stage 2: every real dataset condition is represented with the same seven slots,
then converted to concise natural language.

Important boundaries:
  * HVHA/HVLA/LVHA/LVLA and target-state words are never input features.
  * Missing social fields remain ``not_recorded`` instead of being fabricated.
  * Main context omits stimulus identity and N-back level; they are shortcut
    ablations only.
  * The source manifest is never overwritten.

Default output:
  ../data/v3_manifest_512_context7.csv
  ../data/context7_gpt_cache.json

Examples:
  python build_context7_gpt.py --backend openai
  python build_context7_gpt.py --backend openai --maus-level omit \
      --output ../data/v3_manifest_512_context7_nolevel.csv
  python build_context7_gpt.py --backend deterministic --output _smoke.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
SCHEMA_VERSION = "original_context7_v2"
SLOT_VALUES: dict[str, list[str]] = {
    "posture": ["standing", "sitting", "reclining", "lying", "not_recorded"],
    "energy_expenditure": [
        "vigorous", "moderate", "light", "sedentary", "sleep", "not_recorded",
    ],
    "social_engagement": ["low", "middle", "high", "not_recorded"],
    "interpersonal_density": ["0", "1", "2", "3", "not_recorded"],
    "app_window": [
        "no_active_app_window",
        "video_media_player",
        "vr_media_player",
        "web_browser",
        "communication_app",
        "document_productivity",
        "development_tool",
        "game",
        "task_interface",
        "audio_player",
        "other",
        "not_recorded",
    ],
    "duration": [
        "ten_seconds",
        "under_one_minute",
        "one_to_five_minutes",
        "five_to_thirty_minutes",
        "over_thirty_minutes",
        "not_recorded",
    ],
    "event_type": [
        "resting_baseline",
        "passive_video_viewing",
        "immersive_video_viewing",
        "working_memory_task",
        "public_speaking_and_mental_arithmetic",
        "guided_meditation",
        "document_work",
        "web_browsing",
        "communication",
        "software_development",
        "gameplay",
        "other",
        "not_recorded",
    ],
}
SLOTS = tuple(SLOT_VALUES)
LEGACY_SCHEMA_COLUMNS = {
    "context7_movement",
    "context7_device_interaction_behavior",
    "context7_environment",
    "context7_temporal",
}

ORIGINAL_SLOT_ORDER = (
    "posture",
    "energy_expenditure",
    "social_engagement",
    "interpersonal_density",
    "app_window",
    "duration",
    "event_type",
)
if SLOTS != ORIGINAL_SLOT_ORDER:
    raise RuntimeError("Original seven-slot order changed unexpectedly")

# These may not be invented by the generator.  A word is allowed only when it
# already occurs verbatim in observable stimulus metadata (e.g. CASE's official
# title "Relaxing Music with Beach").
FORBIDDEN_STATE_TERMS = (
    "arousal",
    "valence",
    "cognitive load",
    "mental workload",
    "hvha",
    "hvla",
    "lvha",
    "lvla",
    "stressed",
    "stressful",
    "anxious",
    "anxiety",
    "highly aroused",
    "low arousal",
    "high arousal",
    "high valence",
    "low valence",
    "emotionally",
    "feels happy",
    "feels sad",
    "feels afraid",
)

DETAIL_STOPWORDS = {
    "a", "an", "and", "at", "before", "during", "for", "in", "is", "of",
    "on", "the", "to", "under", "user", "while", "with",
    "active", "baseline", "computer", "controlled", "game", "input",
    "performing", "playing", "recording", "responding", "responses",
    "scene", "screen", "session", "sitting", "stimulus", "task", "trial",
    "virtual", "reality", "vr", "watching", "working",
}

WESAD = {
    "wesad_baseline": {
        "posture": "sitting",
        "energy_expenditure": "sedentary",
        "social_engagement": "not_recorded",
        "interpersonal_density": "not_recorded",
        "app_window": "no_active_app_window",
        "duration": "five_to_thirty_minutes",
        "event_type": "resting_baseline",
        "observable_detail": "",
    },
    "wesad_stress": {
        "posture": "standing",
        "energy_expenditure": "light",
        "social_engagement": "high",
        "interpersonal_density": "3",
        "app_window": "no_active_app_window",
        "duration": "five_to_thirty_minutes",
        "event_type": "public_speaking_and_mental_arithmetic",
        "observable_detail": "",
    },
    "wesad_amusement": {
        "posture": "sitting",
        "energy_expenditure": "sedentary",
        "social_engagement": "not_recorded",
        "interpersonal_density": "not_recorded",
        "app_window": "video_media_player",
        "duration": "five_to_thirty_minutes",
        "event_type": "passive_video_viewing",
        "observable_detail": "",
    },
    "wesad_meditation": {
        "posture": "sitting",
        "energy_expenditure": "sedentary",
        "social_engagement": "not_recorded",
        "interpersonal_density": "not_recorded",
        "app_window": "audio_player",
        "duration": "five_to_thirty_minutes",
        "event_type": "guided_meditation",
        "observable_detail": "",
    },
}

SWELL = {
    "swell_N": {
        "duration": "five_to_thirty_minutes",
        "event_type": "document_work",
        "observable_detail": "",
    },
    "swell_T": {
        "duration": "five_to_thirty_minutes",
        "event_type": "document_work",
        "observable_detail": "",
    },
    "swell_I": {
        "duration": "five_to_thirty_minutes",
        "event_type": "document_work",
        "observable_detail": "",
    },
}

COGWEAR = {
    "baseline": {
        "posture": "sitting",
        "energy_expenditure": "sedentary",
        "social_engagement": "not_recorded",
        "interpersonal_density": "not_recorded",
        "app_window": "no_active_app_window",
        "duration": "five_to_thirty_minutes",
        "event_type": "resting_baseline",
        "observable_detail": "",
    },
    "cognitive_load": {
        "posture": "sitting",
        "energy_expenditure": "sedentary",
        "social_engagement": "not_recorded",
        "interpersonal_density": "not_recorded",
        "app_window": "task_interface",
        "duration": "five_to_thirty_minutes",
        "event_type": "working_memory_task",
        "observable_detail": "",
    },
}

VRFS = {
    "Dirt Rally": "playing a fast racing game in virtual reality",
    "Minecraft": "playing a sandbox building game in virtual reality",
    "Phasmophobia": "playing a first-person exploration game in virtual reality",
    "Warthunder": "playing an aerial-combat game in virtual reality",
}


def _sha(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _strip_prefix(text: str, prefix: str) -> str:
    text = str(text).strip()
    if text.lower().startswith(prefix.lower()):
        text = text[len(prefix):].strip()
    return text.rstrip(".")


def duration_bucket(duration_seconds: Any) -> str:
    """Map observed seconds to one non-overlapping duration category.

    The interval contract is deliberately half-open so boundary values are
    reproducible: [0, 60), [60, 300), [300, 1800), [1800, infinity).
    Exactly ten seconds has its own runtime-matched CASE category. Missing or
    non-finite values remain ``not_recorded``; they are never estimated.
    """
    if duration_seconds is None or pd.isna(duration_seconds):
        return "not_recorded"
    seconds = float(duration_seconds)
    if not 0.0 < seconds < float("inf"):
        raise ValueError(
            f"duration_seconds must be finite and positive, got {seconds}"
        )
    if abs(seconds - 10.0) <= 1e-6:
        return "ten_seconds"
    if seconds < 60.0:
        return "under_one_minute"
    if seconds < 300.0:
        return "one_to_five_minutes"
    if seconds < 1800.0:
        return "five_to_thirty_minutes"
    return "over_thirty_minutes"


def make_spec(
    dataset: str,
    context_sig: str,
    original_text: str = "",
    maus_level: str = "omit",
    case_title: str = "omit",
    eevr_scene: str = "omit",
    duration_seconds: Any = None,
) -> dict[str, str]:
    """Return seven observed/protocol-derived slots plus an observable detail."""
    dataset = str(dataset).lower()
    context_sig = str(context_sig)

    if dataset == "eevr":
        detail = _strip_prefix(original_text, "The user is watching 360-degree VR content:")
        return {
            "posture": "sitting",
            "energy_expenditure": "sedentary",
            "social_engagement": "not_recorded",
            "interpersonal_density": "not_recorded",
            "app_window": "vr_media_player",
            "duration": duration_bucket(duration_seconds),
            "event_type": "immersive_video_viewing",
            "observable_detail": detail if eevr_scene == "include" else "",
        }

    if dataset == "case":
        title = _strip_prefix(original_text, "The user is watching video content:")
        return {
            "posture": "sitting",
            "energy_expenditure": "sedentary",
            "social_engagement": "not_recorded",
            "interpersonal_density": "not_recorded",
            "app_window": "video_media_player",
            "duration": duration_bucket(duration_seconds),
            "event_type": "passive_video_viewing",
            "observable_detail": title if case_title == "include" else "",
        }

    if dataset == "maus":
        match = re.search(r"nback(\d+)", context_sig.lower())
        level = match.group(1) if match else ""
        task = (
            f"{level}-back task interface"
            if maus_level == "include" and level
            else ""
        )
        return {
            "posture": "sitting",
            "energy_expenditure": "sedentary",
            "social_engagement": "not_recorded",
            "interpersonal_density": "not_recorded",
            "app_window": "task_interface",
            "duration": duration_bucket(duration_seconds),
            "event_type": "working_memory_task",
            "observable_detail": task,
        }

    if dataset == "wesad":
        if context_sig not in WESAD:
            raise KeyError(f"Unknown WESAD context: {context_sig}")
        return dict(WESAD[context_sig])

    if dataset == "swell":
        if context_sig not in SWELL:
            raise KeyError(f"Unknown SWELL context: {context_sig}")
        return {
            "posture": "sitting",
            "energy_expenditure": "sedentary",
            "social_engagement": "not_recorded",
            "interpersonal_density": "not_recorded",
            "app_window": "document_productivity",
            **SWELL[context_sig],
        }

    if dataset == "cogwear":
        if context_sig not in COGWEAR:
            raise KeyError(f"Unknown CogWear context: {context_sig}")
        return dict(COGWEAR[context_sig])

    if dataset == "vrfs":
        if context_sig not in VRFS:
            raise KeyError(f"Unknown VRFS context: {context_sig}")
        return {
            "posture": "sitting",
            "energy_expenditure": "sedentary",
            "social_engagement": "not_recorded",
            "interpersonal_density": "not_recorded",
            "app_window": "game",
            "duration": "five_to_thirty_minutes",
            "event_type": "gameplay",
            "observable_detail": "",
        }

    raise KeyError(f"Unsupported dataset: {dataset}")


def deterministic_text(spec: dict[str, str]) -> str:
    """Canonical Stage-2 sentence for the original seven-slot schema."""
    posture = {
        "sitting": "Seated",
        "standing": "Standing",
        "reclining": "Reclining",
        "lying": "Lying down",
        "not_recorded": "Posture not recorded",
    }.get(spec["posture"], spec["posture"].replace("_", " ").capitalize())
    energy = {
        "sedentary": "sedentary energy expenditure",
        "light": "light energy expenditure",
        "moderate": "moderate energy expenditure",
        "vigorous": "vigorous energy expenditure",
        "sleep": "sleep-level energy expenditure",
        "not_recorded": "energy expenditure not recorded",
    }[spec["energy_expenditure"]]
    app = {
        "no_active_app_window": "no active App/Window",
        "video_media_player": "a video or media player",
        "vr_media_player": "a 360-degree VR media player",
        "web_browser": "a web browser",
        "communication_app": "a communication application",
        "document_productivity": "a document or productivity application",
        "development_tool": "a software-development tool",
        "game": "a game application",
        "task_interface": "a task interface",
        "audio_player": "an audio player",
        "other": "another application or window",
        "not_recorded": "not recorded",
    }[spec["app_window"]]
    duration = {
        "ten_seconds": "a ten-second interval",
        "under_one_minute": "an interval under one minute",
        "one_to_five_minutes": "an interval of one to five minutes",
        "five_to_thirty_minutes": "an interval of five to thirty minutes",
        "over_thirty_minutes": "an interval over thirty minutes",
        "not_recorded": "an unrecorded duration",
    }[spec["duration"]]
    event = {
        "resting_baseline": "a resting baseline event",
        "passive_video_viewing": "a passive video-viewing event",
        "immersive_video_viewing": "an immersive video-viewing event",
        "working_memory_task": "a working-memory task",
        "public_speaking_and_mental_arithmetic": (
            "a public-speaking and mental-arithmetic event"
        ),
        "guided_meditation": "a guided-meditation event",
        "document_work": "a document-work event",
        "web_browsing": "a web-browsing event",
        "communication": "a communication event",
        "software_development": "a software-development event",
        "gameplay": "a gameplay event",
        "other": "another recorded event type",
        "not_recorded": "an event type that was not recorded",
    }[spec["event_type"]]
    social = spec["social_engagement"]
    density = spec["interpersonal_density"]
    social_text = (
        "social engagement not recorded"
        if social == "not_recorded"
        else f"{social} social engagement"
    )
    density_text = {
        "0": "no one nearby",
        "1": "one person nearby",
        "2": "two people nearby",
        "3": "three or more people nearby",
        "not_recorded": "interpersonal density not recorded",
    }[density]
    detail = " ".join(str(spec.get("observable_detail", "")).split()).strip(" .")
    if detail:
        app = f"{app} ({detail})"
    return (
        f"{posture} with {energy}. The social context has {social_text} and "
        f"{density_text}. The App/Window is {app} for {duration}, during "
        f"{event}."
    )


def paraphrase_seven_slots(spec: dict[str, str]) -> list[str]:
    """Four deterministic meaning-preserving Stage-1 surface forms."""
    validate_spec(spec)
    canonical = deterministic_text(spec)
    # Clause order changes only; slot values remain verbatim in meaning.
    clauses = [part.strip() for part in canonical.split(".") if part.strip()]
    variants = [
        canonical,
        ". ".join([clauses[2], clauses[0], clauses[1]]) + ".",
        ". ".join([clauses[1], clauses[2], clauses[0]]) + ".",
        (
            f"For the recorded context, {clauses[0].lower()}. "
            f"{clauses[2]}. {clauses[1]}."
        ),
    ]
    return list(dict.fromkeys(variants))


def validate_spec(spec: dict[str, str]) -> None:
    missing = [slot for slot in SLOTS if slot not in spec]
    if missing:
        raise ValueError(f"Missing original seven-slot values: {missing}")
    for slot in SLOTS:
        value = str(spec[slot])
        if value not in SLOT_VALUES[slot]:
            raise ValueError(
                f"Invalid {slot}={value!r}; allowed={SLOT_VALUES[slot]}"
            )


def validate_text(text: str, spec: dict[str, str]) -> list[str]:
    try:
        validate_spec(spec)
    except ValueError as exc:
        return [str(exc)]
    issues: list[str] = []
    clean = " ".join(str(text).split())
    low = clean.lower()
    source = json.dumps(spec, ensure_ascii=False).lower()
    words = clean.split()
    if not 15 <= len(words) <= 75:
        issues.append(f"word_count={len(words)} outside 15..75")
    if "not_recorded" in spec.values() and "not recorded" not in low:
        issues.append("missing explicit not-recorded statement")
    if spec["social_engagement"] == "not_recorded" and "social" not in low:
        issues.append("social-engagement missingness omitted")
    if spec["interpersonal_density"] == "not_recorded" and not (
        "density" in low or "nearby" in low
    ):
        issues.append("interpersonal-density missingness omitted")
    detail_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", spec["observable_detail"].lower())
        if len(token) >= 3 and token not in DETAIL_STOPWORDS
    }
    detail_hits = {token for token in detail_tokens if token in low}
    required_hits = min(2, len(detail_tokens))
    if len(detail_hits) < required_hits:
        issues.append(
            "observable detail omitted "
            f"(need {required_hits} distinctive tokens; found {sorted(detail_hits)})"
        )
    for term in FORBIDDEN_STATE_TERMS:
        if term in low and term not in source:
            issues.append(f"inferred target/state term: {term}")
    if re.search(r"\b(i|my|me)\b", low):
        issues.append("first-person wording")
    return issues


def _prompt(spec: dict[str, str], prior_issues: list[str] | None = None) -> list[dict[str, str]]:
    system = (
        "Convert structured, partially observed user context into one concise natural-English "
        "description for a DistilBERT state-estimation encoder. Use only supplied facts. "
        "Represent all seven slots. Keep not_recorded fields explicitly unobserved; never "
        "replace them with alone, zero, or low. Do not infer feelings, stress, anxiety, "
        "valence, arousal, cognitive load, workload, or an affect quadrant. An official "
        "stimulus title or description may be repeated exactly but must not be interpreted. "
        "Preserve the observable auxiliary detail instead of replacing it with a generic "
        "activity. Use at most 60 words. Return JSON with keys text and represented_slots. "
        "represented_slots must contain the seven field NAMES, not their values; for example "
        "['posture','energy_expenditure','social_engagement',"
        "'interpersonal_density','app_window','duration','event_type']."
    )
    user: dict[str, Any] = {
        "seven_slot_context": {slot: spec[slot] for slot in SLOTS},
        "observable_auxiliary_detail": {
            k: v for k, v in spec.items() if k not in SLOTS
        },
        "represented_slots_must_equal": list(SLOTS),
    }
    if prior_issues:
        user["previous_validation_issues_to_fix"] = prior_issues
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def openai_text(
    spec: dict[str, str],
    model: str,
    temperature: float,
    attempts: int = 4,
) -> str:
    from openai import OpenAI
    from _keyutil import get_openai_key

    client = OpenAI(api_key=get_openai_key())
    issues: list[str] = []
    for attempt in range(1, attempts + 1):
        response = client.chat.completions.create(
            model=model,
            messages=_prompt(spec, issues),
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        raw = response.choices[0].message.content or "{}"
        try:
            obj = json.loads(raw)
            text = " ".join(str(obj["text"]).split())
            represented = set(obj.get("represented_slots", []))
            issues = validate_text(text, spec)
            # Older cached/API responses sometimes echo slot values rather than names.
            # Accept that serialization only when it covers every distinct supplied value;
            # the natural-language text still has to pass the independent checks above.
            supplied_values = {spec[slot] for slot in SLOTS}
            if represented != set(SLOTS) and represented != supplied_values:
                issues.append(f"represented_slots={sorted(represented)}")
            if not issues:
                return text
        except Exception as exc:
            issues = [f"invalid JSON response: {exc}"]
        if attempt < attempts:
            time.sleep(0.5 * attempt)
    raise RuntimeError(f"GPT context failed validation after {attempts} attempts: {issues}")


def _entry_id(dataset: str, context_sig: str, spec: dict[str, str]) -> str:
    return f"{dataset}::{context_sig}::{_sha(spec)[:16]}"


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "format": "context7_gpt_cache_original_v2",
            "schema_version": SCHEMA_VERSION,
            "entries": {},
        }
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("format") != "context7_gpt_cache_original_v2":
        raise ValueError(f"Unexpected cache format: {path}")
    if obj.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unexpected cache schema: {path}")
    obj.setdefault("entries", {})
    return obj


def _save_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def text_for_spec(
    dataset: str,
    context_sig: str,
    spec: dict[str, str],
    cache: dict[str, Any],
    cache_path: Path,
    backend: str,
    model: str,
    temperature: float,
    force: bool = False,
) -> tuple[str, str]:
    eid = _entry_id(dataset, context_sig, spec)
    cached = cache["entries"].get(eid)
    if cached and not force:
        issues = validate_text(cached.get("text", ""), spec)
        if not issues:
            return str(cached["text"]), str(cached.get("generator", "cache"))
    if backend == "cache":
        raise KeyError(f"No valid cached context for {dataset}/{context_sig}")
    if backend == "openai":
        text = openai_text(spec, model=model, temperature=temperature)
        generator = model
    else:
        text = deterministic_text(spec)
        issues = validate_text(text, spec)
        if issues:
            raise ValueError(f"Deterministic context failed for {dataset}/{context_sig}: {issues}")
        generator = "deterministic"
    cache["entries"][eid] = {
        "dataset": dataset,
        "context_sig": context_sig,
        "spec_sha256": _sha(spec),
        "spec": spec,
        "text": text,
        "generator": generator,
        "validation": "passed",
    }
    _save_cache(cache_path, cache)
    return text, generator


def cached_external_text(
    dataset: str,
    context_sig: str,
    cache_path: str | Path | None = None,
) -> str:
    """Return external-evaluation context from the generated cache.

    This never calls an API at inference time.  A deterministic seven-slot
    sentence is used when the external entry has not been generated yet.
    """
    path = Path(cache_path) if cache_path else DATA / "context7_gpt_cache.json"
    spec = make_spec(dataset, context_sig)
    cache = _load_cache(path)
    eid = _entry_id(dataset, context_sig, spec)
    entry = cache["entries"].get(eid)
    return str(entry["text"]) if entry else deterministic_text(spec)


def build(
    source: Path,
    output: Path,
    cache_path: Path,
    backend: str,
    model: str,
    temperature: float,
    maus_level: str,
    case_title: str,
    eevr_scene: str,
    include_external: bool,
    force: bool,
) -> pd.DataFrame:
    frame = pd.read_csv(source)
    required = {"sample_id", "dataset", "context_sig", "context_text"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Source manifest missing columns: {sorted(missing)}")
    original = frame.copy(deep=True)
    cache = _load_cache(cache_path)
    records: dict[tuple[str, str], tuple[dict[str, str], str, str]] = {}

    unique = (
        frame[["dataset", "context_sig", "context_text"]]
        .drop_duplicates(["dataset", "context_sig"])
        .sort_values(["dataset", "context_sig"], kind="stable")
    )
    for row in unique.itertuples(index=False):
        spec = make_spec(
            row.dataset,
            row.context_sig,
            row.context_text,
            maus_level,
            case_title,
            eevr_scene,
        )
        text, generator = text_for_spec(
            row.dataset,
            row.context_sig,
            spec,
            cache,
            cache_path,
            backend,
            model,
            temperature,
            force,
        )
        records[(row.dataset, row.context_sig)] = (spec, text, generator)
        print(f"[{row.dataset:8s}] {row.context_sig} -> {text}")

    if include_external:
        for dataset, signatures in (("cogwear", COGWEAR), ("vrfs", VRFS)):
            for context_sig in signatures:
                spec = make_spec(dataset, context_sig)
                text, _ = text_for_spec(
                    dataset,
                    context_sig,
                    spec,
                    cache,
                    cache_path,
                    backend,
                    model,
                    temperature,
                    force,
                )
                print(f"[external] {dataset}/{context_sig} -> {text}")

    frame["context_text_original"] = frame["context_text"]
    for legacy_column in sorted(LEGACY_SCHEMA_COLUMNS):
        if legacy_column in frame:
            frame = frame.drop(columns=[legacy_column])
    for idx, row in frame.iterrows():
        spec, text, generator = records[(row["dataset"], row["context_sig"])]
        frame.at[idx, "context_text"] = text
        for slot in SLOTS:
            frame.at[idx, f"context7_{slot}"] = spec[slot]
        frame.at[idx, "context7_observable_detail"] = spec["observable_detail"]
        frame.at[idx, "context7_generator"] = generator
        frame.at[idx, "context7_spec_sha256"] = _sha(spec)
        frame.at[idx, "context7_validation"] = "passed"
        frame.at[idx, "context7_maus_level"] = maus_level
        frame.at[idx, "context7_case_title"] = case_title
        frame.at[idx, "context7_eevr_scene"] = eevr_scene
        frame.at[idx, "context_schema_version"] = SCHEMA_VERSION
        frame.at[idx, "context_input_source"] = (
            "deterministic_original_seven_slot_v2"
        )

    # Guard against accidental changes to labels, IDs, PPG availability, or row order.
    untouched = [
        column
        for column in original.columns
        if column != "context_text"
        and column not in LEGACY_SCHEMA_COLUMNS
    ]
    pd.testing.assert_frame_equal(
        original[untouched].reset_index(drop=True),
        frame[untouched].reset_index(drop=True),
        check_dtype=False,
    )
    if frame["sample_id"].duplicated().any():
        raise ValueError("Output manifest contains duplicate sample_id values")
    if not (frame["context7_validation"] == "passed").all():
        raise ValueError("At least one generated context failed validation")

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print("\n" + "=" * 72)
    print(f"source rows: {len(original)}")
    print(f"training contexts: {len(records)}")
    print(f"datasets: {sorted(frame.dataset.unique())}")
    print(f"MAUS level: {maus_level}")
    print(f"saved manifest: {output}")
    print(f"saved/resumed cache: {cache_path}")
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DATA / "v3_manifest_512.csv")
    parser.add_argument("--output", type=Path, default=DATA / "v3_manifest_512_context7.csv")
    parser.add_argument("--cache", type=Path, default=DATA / "context7_gpt_cache.json")
    parser.add_argument("--backend", choices=("openai", "deterministic", "cache"), default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--maus-level", choices=("include", "omit"), default="omit")
    parser.add_argument("--case-title", choices=("include", "omit"), default="omit")
    parser.add_argument("--eevr-scene", choices=("include", "omit"), default="omit")
    parser.add_argument("--no-external", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(
        source=args.source.resolve(),
        output=args.output.resolve(),
        cache_path=args.cache.resolve(),
        backend=args.backend,
        model=args.model,
        temperature=args.temperature,
        maus_level=args.maus_level,
        case_title=args.case_title,
        eevr_scene=args.eevr_scene,
        include_external=not args.no_external,
        force=args.force,
    )
