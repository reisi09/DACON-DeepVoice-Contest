"""Role-specific semantic validation for shared feature NPZ records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np

WINDOW: Final = 64_600
COMMON_KEYS: Final = frozenset({"starts", "valid_samples", "padding_samples", "raw_mask"})
PROFILE_KEYS: Final = {
    "core_music_base": COMMON_KEYS | {"stem_mask", "mert_l6", "mert_l12", "mert_l24", "spectrum_stem"},
    "core_music_processed": COMMON_KEYS | {"stem_mask", "mert_l24"},
    "core_speech_base": COMMON_KEYS | {
        "anti_l12", "anti_l24", "anti_l48", "df_raw", "df_vocals", "anti_raw", "anti_vocals",
    },
    "core_speech_processed": COMMON_KEYS | {"anti_l24", "df_raw", "df_vocals", "anti_raw", "anti_vocals"},
    "swap": COMMON_KEYS | {
        "stem_mask", "presence", "mert_l24", "anti_l24", "df_raw", "df_vocals", "anti_raw", "anti_vocals",
    },
    "partial": COMMON_KEYS | {"df_raw", "df_vocals", "anti_raw", "anti_vocals"},
}
ALL_FEATURE_KEYS: Final = frozenset().union(*PROFILE_KEYS.values())


class FeatureSchemaError(RuntimeError):
    """A shared feature record is incomplete or numerically invalid."""


def feature_profiles(rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    """Assign one exhaustive extraction profile from sealed manifest semantics."""
    result: dict[str, str] = {}
    for row in rows:
        family, kind, variant = row["probe_family"], row["kind"], row["variant"]
        if family == "partial" and kind == "speech":
            profile = "partial"
        elif family == "swap" and kind == "mixed":
            profile = "swap"
        elif family == "core" and kind == "speech":
            profile = "core_speech_base" if variant == "base" else "core_speech_processed"
        elif family == "core" and kind == "mixed" and row["MUSIC_PRESENT"] == "1":
            profile = "core_music_base" if variant == "base" else "core_music_processed"
        else:
            raise FeatureSchemaError(f"No extraction role for manifest row: {row['ID']}")
        if row["ID"] in result:
            raise FeatureSchemaError(f"Duplicate manifest ID in extraction roles: {row['ID']}")
        result[row["ID"]] = profile
    return result


def required_keys(profile: str) -> frozenset[str]:
    try:
        return PROFILE_KEYS[profile]
    except KeyError as error:
        raise FeatureSchemaError(f"Unknown feature profile: {profile}") from error


def validate_feature_arrays(arrays: Mapping[str, np.ndarray], profile: str) -> int:
    """Validate exact profile membership, alignment, domains, and representation widths."""
    expected = required_keys(profile)
    if set(arrays) != expected:
        raise FeatureSchemaError(
            f"Feature profile {profile} keys differ: missing={sorted(expected.difference(arrays))}, "
            f"extra={sorted(set(arrays).difference(expected))}"
        )
    starts = np.asarray(arrays["starts"])
    count = len(starts)
    if count < 1 or starts[0] != 0 or np.any(np.diff(starts) <= 0):
        raise FeatureSchemaError("Window starts must begin at zero and increase without duplicates")
    if any(np.asarray(arrays[key]).shape[0] != count or not np.isfinite(arrays[key]).all() for key in expected):
        raise FeatureSchemaError("Feature arrays are non-finite or not aligned to shared windows")
    valid, padding = arrays["valid_samples"], arrays["padding_samples"]
    if np.any(valid < 1) or np.any(valid > WINDOW) or not np.array_equal(valid + padding, np.full(count, WINDOW)):
        raise FeatureSchemaError("Valid-sample and padding counts violate the fixed window")
    for key in expected & {"raw_mask", "stem_mask"}:
        if np.asarray(arrays[key]).shape != (count,) or not np.isin(arrays[key], (0, 1)).all():
            raise FeatureSchemaError(f"Invalid binary window mask: {key}")
    if "presence" in expected and np.asarray(arrays["presence"]).shape != (count, 2):
        raise FeatureSchemaError("Presence width changed")
    if "spectrum_stem" in expected and np.asarray(arrays["spectrum_stem"]).shape != (count, 448):
        raise FeatureSchemaError("Stem spectrum width changed")
    for layer in (6, 12, 24):
        key = f"mert_l{layer}"
        if key in expected and np.asarray(arrays[key]).shape != (count, 1024):
            raise FeatureSchemaError("MERT hidden width changed")
    for layer in (12, 24, 48):
        key = f"anti_l{layer}"
        if key in expected and np.asarray(arrays[key]).shape != (count, 1280):
            raise FeatureSchemaError("AntiDeepfake hidden width changed")
    for key in expected & {"df_raw", "df_vocals", "anti_raw", "anti_vocals"}:
        if np.asarray(arrays[key]).shape != (count,):
            raise FeatureSchemaError(f"Probability schedule shape changed: {key}")
    for key in expected & {"df_raw", "df_vocals", "anti_raw", "anti_vocals", "presence"}:
        if np.any(arrays[key] < 0) or np.any(arrays[key] > 1):
            raise FeatureSchemaError(f"Probability outside [0,1]: {key}")
    return count
