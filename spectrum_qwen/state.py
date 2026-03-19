from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .config import QwenSpectrumConfig


@dataclass
class QwenSpectrumRuntime:
    config: QwenSpectrumConfig
    current_step_index: int
    total_steps: int
    current_sigma: float
    decision_actual: bool
    forecast_reason: str


@dataclass
class QwenSpectrumState:
    config: QwenSpectrumConfig
    history_sigmas: list[float] = field(default_factory=list)
    history_features: list[torch.Tensor] = field(default_factory=list)
    actual_count: int = 0
    forecast_count: int = 0
    consecutive_forecasts: int = 0
    output_factory: Callable[[Any, bool], Any] | None = None
    warned_control_once: bool = False
    warned_signature_once: bool = False
    warned_zero_cond_once: bool = False
    warned_capture_once: bool = False

    def reset(self) -> None:
        self.history_sigmas.clear()
        self.history_features.clear()
        self.actual_count = 0
        self.forecast_count = 0
        self.consecutive_forecasts = 0
        self.output_factory = None
        self.warned_control_once = False
        self.warned_signature_once = False
        self.warned_zero_cond_once = False
        self.warned_capture_once = False

    def record_actual(
        self,
        sigma: float,
        feature: torch.Tensor,
        output_factory: Callable[[Any, bool], Any] | None,
    ) -> None:
        self.actual_count += 1
        self.consecutive_forecasts = 0
        self.history_sigmas.append(float(sigma))
        self.history_features.append(feature)
        if len(self.history_features) > self.config.history_points:
            self.history_features = self.history_features[-self.config.history_points :]
            self.history_sigmas = self.history_sigmas[-self.config.history_points :]
        if output_factory is not None and self.output_factory is None:
            self.output_factory = output_factory

    def record_forecast(self) -> None:
        self.forecast_count += 1
        self.consecutive_forecasts += 1
