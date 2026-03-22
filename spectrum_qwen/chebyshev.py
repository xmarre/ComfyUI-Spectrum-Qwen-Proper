from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


def normalize_step_position(step_index: int, total_steps: int) -> float:
    if total_steps <= 1:
        return 0.0
    return float(step_index) / float(total_steps - 1) * 2.0 - 1.0


@dataclass(slots=True)
class _HistoryEntry:
    time_coord: float
    feature: torch.Tensor
    design_row: torch.Tensor

def chebyshev_basis(x: torch.Tensor, degree: int) -> torch.Tensor:
    x = x.to(dtype=torch.float32)
    basis = [torch.ones_like(x), x]
    if degree == 0:
        return basis[0].unsqueeze(-1)
    for _ in range(2, degree + 1):
        basis.append(2.0 * x * basis[-1] - basis[-2])
    return torch.stack(basis[: degree + 1], dim=-1)


@dataclass
class ChebyshevSpectrumForecaster:
    degree: int
    ridge_lambda: float
    max_history: int
    _history: list[_HistoryEntry] = field(default_factory=list, init=False, repr=False)
    _feature_shape: Optional[torch.Size] = field(default=None, init=False, repr=False)
    _feature_dtype: Optional[torch.dtype] = field(default=None, init=False, repr=False)
    _device: Optional[torch.device] = field(default=None, init=False, repr=False)
    _xtx: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _xth: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _coeff: Optional[torch.Tensor] = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        self._history.clear()
        self._feature_shape = None
        self._feature_dtype = None
        self._device = None
        self._xtx = None
        self._xth = None
        self._coeff = None

    def ready(self) -> bool:
        return self._feature_shape is not None and len(self._history) >= self.degree + 1

    def update(self, time_coord: float, feature: torch.Tensor) -> None:
        feat = feature.detach()
        if self._feature_shape is None:
            self._feature_shape = feat.shape
            self._feature_dtype = feat.dtype
            self._device = feat.device
        elif feat.shape != self._feature_shape:
            raise ValueError(
                f"Spectrum feature shape changed from {tuple(self._feature_shape)} to {tuple(feat.shape)}."
            )

        if self._device is None:
            raise RuntimeError("Spectrum forecaster has no target device.")

        design_row = chebyshev_basis(
            torch.tensor([float(time_coord)], device=self._device, dtype=torch.float32),
            self.degree,
        ).reshape(-1)
        flat_feat = feat.reshape(1, -1).to(device=self._device, dtype=torch.float32)

        p = design_row.numel()
        if self._xtx is None:
            self._xtx = torch.zeros((p, p), device=self._device, dtype=torch.float32)
        if self._xth is None:
            self._xth = torch.zeros((p, flat_feat.shape[1]), device=self._device, dtype=torch.float32)

        if len(self._history) >= self.max_history:
            oldest = self._history.pop(0)
            old_row = oldest.design_row.to(dtype=torch.float32)
            old_flat = oldest.feature.reshape(1, -1).to(device=self._device, dtype=torch.float32)
            self._xtx -= old_row[:, None] @ old_row[None, :]
            self._xth -= old_row[:, None] @ old_flat

        self._history.append(_HistoryEntry(float(time_coord), feat.to(device=self._device), design_row))
        self._xtx += design_row[:, None] @ design_row[None, :]
        self._xth += design_row[:, None] @ flat_feat

        self._coeff = None
        if self.ready():
            self._recompute_coefficients()

    def _recompute_coefficients(self) -> None:
        if self._xtx is None or self._xth is None:
            raise RuntimeError("Spectrum forecaster has no fit state to solve.")

        lhs = self._xtx
        if self.ridge_lambda > 0:
            lhs = lhs + torch.eye(lhs.shape[0], device=lhs.device, dtype=lhs.dtype) * float(self.ridge_lambda)
        try:
            chol = torch.linalg.cholesky(lhs)
        except RuntimeError:
            diag_mean = lhs.diag().mean() if lhs.numel() else torch.tensor(1.0, device=lhs.device)
            jitter = max(float(diag_mean.item()) * 1e-6, 1e-8)
            chol = torch.linalg.cholesky(lhs + jitter * torch.eye(lhs.shape[0], device=lhs.device, dtype=lhs.dtype))
        self._coeff = torch.cholesky_solve(self._xth, chol)

    def predict(self, time_coord: float) -> torch.Tensor:
        if not self.ready():
            raise RuntimeError("Spectrum forecaster is not ready yet.")
        if self._feature_shape is None or self._feature_dtype is None or self._device is None:
            raise RuntimeError("Spectrum forecaster has no cached feature history.")
        if self._coeff is None:
            self._recompute_coefficients()
        assert self._coeff is not None

        coord_star = torch.tensor([float(time_coord)], device=self._device, dtype=torch.float32)
        design_star = chebyshev_basis(coord_star, self.degree)
        spectral = (design_star @ self._coeff).reshape(self._feature_shape)
        return spectral.to(dtype=self._feature_dtype)
