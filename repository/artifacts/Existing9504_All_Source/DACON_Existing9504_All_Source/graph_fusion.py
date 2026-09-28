"""Five file-independent fusion graphs over one shared audio-inference cache."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray


class Feature(TypedDict):
    ID: str
    starts: list[int]
    presence: list[list[float]]
    voice_fake: list[float]
    music_fake: list[float]
    raw_fake: list[float]
    stem_rms: list[float]
    duration: float


class Label(TypedDict):
    ID: str
    group_id: str
    split: Literal["dev", "holdout"]
    FILE_FAKE: int | str | None
    VOICE_FAKE: int | str | None
    MUSIC_FAKE: int | str | None
    VOICE_PRESENT: int | str | None
    MUSIC_PRESENT: int | str | None


class Prediction(TypedDict):
    ID: str
    FILE_FAKE_PROB: float
    VOICE_FAKE_PROB: float
    MUSIC_FAKE_PROB: float
    VOICE_PRESENT_PROB: float
    MUSIC_PRESENT_PROB: float


class Head(TypedDict, total=False):
    status: str
    reason: str
    n: int
    train_ids: list[str]
    coef: list[float]
    intercept: float
    mean: list[float]
    scale: list[float]


class Fusion(TypedDict):
    version: int
    feature_names: list[str]
    train_ids: list[str]
    train_groups: list[str]
    heads: dict[str, Head]


@dataclass(frozen=True, slots=True)
class CacheError(ValueError):
    detail: str

    def __str__(self) -> str:
        return self.detail


FEATURE_NAMES: Final = ["voice_max", "voice_mean", "music_max", "music_mean",
    "raw_max", "raw_mean", "voice_presence_max", "music_presence_max",
    "voice_raw_disagreement", "music_raw_disagreement", "log_windows"]
HEADS: Final = ("FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB")
FloatArray = NDArray[np.float64]
GraphPredictions = dict[str, Prediction]


def _arrays(feature: Feature) -> tuple[FloatArray, FloatArray]:
    """Reject corrupted cache probabilities and misaligned window evidence."""
    starts = np.asarray(feature["starts"])
    count = len(starts)
    if not isinstance(feature["ID"], str) or not feature["ID"] or starts.ndim != 1 or count == 0:
        raise CacheError("A nonempty ID and nonempty window starts are required")
    if not np.issubdtype(starts.dtype, np.integer) or np.any(starts < 0) or np.any(np.diff(starts) <= 0):
        raise CacheError("Window starts must be strictly increasing nonnegative integers")
    presence = np.asarray(feature["presence"], dtype=np.float64)
    fake = np.asarray([feature[key] for key in ("voice_fake", "music_fake", "raw_fake")], dtype=np.float64)
    if presence.shape != (count, 2) or fake.shape != (3, count):
        raise CacheError("Presence and fake probabilities must align with every window")
    if any(np.any(~np.isfinite(values) | (values < 0) | (values > 1)) for values in (presence, fake)):
        raise CacheError("Cached probabilities must be finite and inside [0, 1]")
    rms = np.asarray(feature["stem_rms"], dtype=np.float64)
    if rms.shape != (2,) or np.any(~np.isfinite(rms) | (rms < 0)):
        raise CacheError("Stem RMS values must be finite and nonnegative")
    if not np.isfinite(feature["duration"]) or feature["duration"] <= 0:
        raise CacheError("Audio duration must be finite and positive")
    return presence, fake


def _vector(presence: FloatArray, fake: FloatArray) -> FloatArray:
    maxima = fake.max(axis=1)
    return np.concatenate((np.column_stack((maxima, fake.mean(axis=1))).ravel(),
        presence.max(axis=0), np.abs(maxima[:2] - maxima[2]), [np.log1p(fake.shape[1])]))


def feature_vector(feature: Feature) -> FloatArray:
    """Return the fixed, file-local 11-feature input used by the learned heads."""
    return _vector(*_arrays(feature))


def _row(identifier: str, values: Sequence[float]) -> Prediction:
    file_fake, voice_fake, music_fake, voice_present, music_present = values
    return {"ID": identifier, "FILE_FAKE_PROB": round(float(file_fake), 10),
        "VOICE_FAKE_PROB": round(float(voice_fake), 10),
        "MUSIC_FAKE_PROB": round(float(music_fake), 10),
        "VOICE_PRESENT_PROB": round(float(voice_present), 10),
        "MUSIC_PRESENT_PROB": round(float(music_present), 10)}


def graph_predictions(feature: Feature, fusion: Fusion | None = None) -> GraphPredictions:
    """Apply all five frozen graphs; G4 falls back per head until fitted."""
    presence, fake = _arrays(feature)
    vp, mp = presence.max(axis=0)
    vf, mf, raw = fake.max(axis=1)
    baseline = _row(feature["ID"], [max(vf * vp, mf * mp), vf, mf, vp, mp])
    top_vf, top_mf = np.sort(fake[:2], axis=1)[:, -min(2, fake.shape[1]):].mean(axis=1)
    aligned = 1.0 - (1.0 - presence[:, 0] * fake[0]) * (1.0 - presence[:, 1] * fake[1])
    hybrid_vf, hybrid_mf = 0.75 * fake[:2].max(axis=1) + 0.25 * raw
    rows = {"G0": baseline,
        "G1": _row(feature["ID"], [max(top_vf * vp, top_mf * mp), top_vf, top_mf, vp, mp]),
        "G2": _row(feature["ID"], [aligned.max(), vf, mf, vp, mp]),
        "G3": _row(feature["ID"], [max(raw, hybrid_vf * vp, hybrid_mf * mp), hybrid_vf, hybrid_mf, vp, mp]),
        "G4": baseline.copy()}
    if fusion is not None:
        if fusion["version"] != 1 or fusion["feature_names"] != FEATURE_NAMES:
            raise CacheError("Fusion model version or feature order is incompatible")
        vector = _vector(presence, fake)
        for name in HEADS:
            head = fusion["heads"][name]
            if head["status"] == "trained":
                coef, mean, scale = (np.asarray(head[key], dtype=np.float64) for key in ("coef", "mean", "scale"))
                if any(a.shape != vector.shape or np.any(~np.isfinite(a)) for a in (coef, mean, scale)) or np.any(scale <= 0) or not np.isfinite(head["intercept"]):
                    raise CacheError("Fusion parameters must be finite with positive scales")
                logit = float(((vector - mean) / scale) @ coef + head["intercept"])
                probability = np.exp(-np.logaddexp(0.0, -logit))
                rows["G4"][name] = round(float(probability), 10)
            elif head["status"] != "fallback":
                raise CacheError("Unknown fusion head status")
    return rows


def _binary(value: int | str | None) -> int | None:
    if value is None or value == "":
        return None
    if value not in (0, 1, "0", "1"):
        raise CacheError("Ground-truth labels must be binary or missing")
    return int(value)


def fit_fusion(features: Sequence[Feature], labels: Sequence[Label]) -> Fusion:
    """Fit C=0.1 logistic heads on eligible dev rows, never held-out sources."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    by_id = {f["ID"]: f for f in features}
    if len(by_id) != len(features) or len({row["ID"] for row in labels}) != len(labels):
        raise CacheError("Feature and label IDs must each be unique")
    groups: dict[str, set[str]] = {"dev": set(), "holdout": set()}
    for row in labels:
        if row["split"] not in groups or not isinstance(row["group_id"], str) or not row["group_id"]:
            raise CacheError("Labels require a dev/holdout split and nonempty group_id")
        source_groups = row["group_id"].split("|")
        if not all(source_groups):
            raise CacheError("Mixtures require nonempty constituent source groups")
        groups[row["split"]].update(source_groups)
        for key in ("FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE", "VOICE_PRESENT", "MUSIC_PRESENT"):
            _binary(row.get(key))
    if groups["dev"] & groups["holdout"]:
        raise CacheError("Source group overlap between dev and holdout")
    development = sorted((row for row in labels if row["split"] == "dev" and row["ID"] in by_id), key=lambda row: row["ID"])
    vectors = {row["ID"]: feature_vector(by_id[row["ID"]]) for row in development}
    result: Fusion = {"version": 1, "feature_names": FEATURE_NAMES.copy(),
        "train_ids": [row["ID"] for row in development],
        "train_groups": sorted({group for row in development for group in row["group_id"].split("|")}), "heads": {}}
    for name, mask in zip(HEADS, (None, "VOICE_PRESENT", "MUSIC_PRESENT")):
        target = name.removesuffix("_PROB")
        eligible = [row for row in development if _binary(row.get(target)) is not None and (mask is None or _binary(row.get(mask)) == 1)]
        y = np.asarray([_binary(row[target]) for row in eligible], dtype=np.int64)
        head: Head = {"status": "fallback", "reason": "fewer_than_two_classes",
            "n": len(eligible), "train_ids": [row["ID"] for row in eligible]}
        if np.unique(y).size == 2:
            x = np.stack([vectors[row["ID"]] for row in eligible])
            scaler = StandardScaler().fit(x)
            model = LogisticRegression(C=0.1, max_iter=1000, random_state=42).fit(scaler.transform(x), y)
            if np.any(model.n_iter_ >= model.max_iter):
                raise CacheError(f"Fusion head {name} failed to converge")
            head.update(status="trained", reason="", coef=model.coef_[0].tolist(),
                intercept=float(model.intercept_[0]), mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist())
        result["heads"][name] = head
    return result
