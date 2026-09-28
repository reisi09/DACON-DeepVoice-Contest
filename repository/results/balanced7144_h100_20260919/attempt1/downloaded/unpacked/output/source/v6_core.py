"""Numerical contract shared by final-model training and offline inference."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import expit

VERSION = "v6-xlsraasist-la-mert95-aligned-1"
COLUMNS = ("FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
           "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB")


def fusion_features(presence: np.ndarray, voice: np.ndarray,
                    music: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Multiply corresponding windows before reducing; retain raw disagreement."""
    if (presence.ndim != 2 or presence.shape != (len(voice), 2) or not len(voice)
            or music.shape != voice.shape or raw.shape != voice.shape):
        raise ValueError("Unaligned or empty window scores")
    for value in (presence, voice, music, raw):
        if not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
            raise ValueError("Scores must be finite probabilities")
    return np.array([(presence[:, 0] * voice).max(),
                     (presence[:, 1] * music).max(), raw.max(),
                     np.abs(raw - voice).max(), np.abs(raw - music).max()])


@dataclass(frozen=True, slots=True)
class Heads:
    mean: np.ndarray
    scale: np.ndarray
    music_weight: np.ndarray
    music_bias: np.ndarray
    fusion_weight: np.ndarray
    fusion_bias: np.ndarray

    def music(self, features: np.ndarray, audible: np.ndarray) -> np.ndarray:
        logits = ((features - self.mean) / self.scale) @ self.music_weight + self.music_bias
        return expit(logits).reshape(-1) * audible

    def predict(self, cached: dict[str, np.ndarray]) -> np.ndarray:
        music = self.music(cached["music"], cached["music_audible"])
        voice, raw, presence = (cached[k] for k in ("voice", "raw", "presence"))
        feature = fusion_features(presence, voice, music, raw)
        file_probability = expit(feature @ self.fusion_weight + self.fusion_bias).item()
        return np.array([file_probability, voice.max(), music.max(),
                         presence[:, 0].max(), presence[:, 1].max()])

    def save(self, path: Path) -> None:
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, **{k: getattr(self, k) for k in self.__slots__})
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Heads":
        with np.load(path, allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in cls.__slots__}
        d = len(arrays["mean"])
        expected = {"mean": (d,), "scale": (d,), "music_weight": (d,),
                    "music_bias": (), "fusion_weight": (5,), "fusion_bias": ()}
        if d != 768 or any(a.shape != expected[k] or not np.isfinite(a).all()
                           for k, a in arrays.items()) or (arrays["scale"] <= 0).any():
            raise ValueError("Invalid trained head arrays")
        return cls(**arrays)
