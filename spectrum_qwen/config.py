from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QwenSpectrumConfig:
    warmup_steps: int = 5
    tail_actual_steps: int = 2
    history_points: int = 5
    chebyshev_degree: int = 3
    max_consecutive_forecasts: int = 1
    ridge_lambda: float = 1e-4
    cache_device: str = "main_device"
    force_actual_on_control: bool = True
    debug: bool = False

    def validate(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.tail_actual_steps < 0:
            raise ValueError("tail_actual_steps must be >= 0")
        if self.history_points < 2:
            raise ValueError("history_points must be >= 2")
        if self.chebyshev_degree < 1:
            raise ValueError("chebyshev_degree must be >= 1")
        if self.chebyshev_degree >= self.history_points:
            raise ValueError("chebyshev_degree must be smaller than history_points")
        if self.max_consecutive_forecasts < 0:
            raise ValueError("max_consecutive_forecasts must be >= 0")
        if self.ridge_lambda < 0:
            raise ValueError("ridge_lambda must be >= 0")
        if self.cache_device not in {"main_device", "offload_device", "cpu"}:
            raise ValueError("cache_device must be one of: main_device, offload_device, cpu")
