from __future__ import annotations

from typing import Any

import torch

from .state import QwenSpectrumState



def find_step_index(sigmas: torch.Tensor, timestep: torch.Tensor) -> int:
    if sigmas.ndim == 0:
        return 0
    if timestep.numel() == 0:
        return 0

    target = timestep[0]
    matched = (sigmas == target).nonzero()
    if len(matched) > 0:
        return int(matched[0].item())

    for i in range(len(sigmas) - 1):
        if (sigmas[i] - target) * (sigmas[i + 1] - target) <= 0:
            return i
    return 0



def should_reset_for_step_zero(cond_or_uncond: list[int] | tuple[int, ...] | None) -> bool:
    if cond_or_uncond is None:
        return True
    if len(cond_or_uncond) == 2:
        return True
    return len(cond_or_uncond) == 1 and cond_or_uncond[0] == 1



def decide_actual_or_forecast(
    state: QwenSpectrumState,
    current_step_index: int,
    total_steps: int,
    control_present: bool,
    core: Any,
) -> tuple[bool, str]:
    cfg = state.config

    if current_step_index < cfg.warmup_steps:
        return True, "warmup"

    if current_step_index >= max(0, total_steps - cfg.tail_actual_steps):
        return True, "tail"

    if control_present and cfg.force_actual_on_control:
        return True, "control"

    if getattr(core, "gradient_checkpointing", False):
        return True, "gradient_checkpointing"

    if getattr(core, "zero_cond_t", False):
        return True, "zero_cond_t"

    if len(state.history_features) < cfg.history_points:
        return True, "insufficient_history"

    if state.consecutive_forecasts >= cfg.max_consecutive_forecasts:
        return True, "refresh"

    return False, "forecast"
