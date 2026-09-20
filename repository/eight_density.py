"""REAL-only fixed-polarity density baseline for E8."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class DensityContractError(ValueError):
    """REAL-only density inputs violate the experiment contract."""


class DensityStateError(RuntimeError):
    """Density model was used before a valid fit."""


@dataclass(slots=True)  # noqa: MUTABLE_OK - estimator stores fitted REAL-only parameters.
class RealOnlyGaussian:
    """Diagonal shrinkage Gaussian; larger distance always means more fake-like."""

    shrinkage: float = 0.1
    mean_: np.ndarray | None = None
    variance_: np.ndarray | None = None
    fit_ids_: tuple[str, ...] = ()

    def fit(self, features: np.ndarray, labels: np.ndarray, identifiers: list[str]) -> RealOnlyGaussian:
        matrix = np.asarray(features, dtype=np.float64)
        target = np.asarray(labels, dtype=np.int64)
        if matrix.ndim != 2 or len(matrix) != len(target) or len(target) != len(identifiers):
            raise DensityContractError("REAL-only fit inputs must be aligned")
        if np.any(target != 0):
            raise DensityContractError("REAL-only density fitting rejected fake contamination")
        if len(matrix) < 2 or not np.isfinite(matrix).all() or not 0 <= self.shrinkage <= 1:
            raise DensityContractError("REAL-only fit requires two finite rows and shrinkage in [0,1]")
        mean = matrix.mean(axis=0)
        variance = matrix.var(axis=0, ddof=1)
        common = float(np.mean(variance))
        floor = max(common * 1e-6, np.finfo(np.float64).eps)
        self.mean_ = mean
        self.variance_ = np.maximum((1 - self.shrinkage) * variance + self.shrinkage * common, floor)
        self.fit_ids_ = tuple(identifiers)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """Mean diagonal Mahalanobis distance, fixed high-is-fake polarity."""
        if self.mean_ is None or self.variance_ is None:
            raise DensityStateError("REAL-only model is not fitted")
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1:] != self.mean_.shape or not np.isfinite(matrix).all():
            raise DensityContractError("density score features are invalid")
        return np.mean(np.square(matrix - self.mean_) / self.variance_, axis=1)

    def save(self, path: Path) -> None:
        if self.mean_ is None or self.variance_ is None:
            raise DensityStateError("REAL-only model is not fitted")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, mean=self.mean_, variance=self.variance_,
                            shrinkage=np.array(self.shrinkage), fit_ids=np.array(self.fit_ids_))
