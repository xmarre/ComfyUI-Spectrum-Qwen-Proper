from __future__ import annotations

from typing import Any, Callable
from unittest.mock import patch

import torch

from .config import QwenSpectrumConfig
from .constants import RUNTIME_ATTR, STATE_ATTR
from .controller import decide_actual_or_forecast, find_step_index, should_reset_for_step_zero
from .forward_qwen import build_qwen_core_forward
from .state import QwenSpectrumRuntime, QwenSpectrumState
from .utils import log_debug



def create_qwen_spectrum_unet_wrapper(
    diffusion_model: Any,
    core: Any,
    config: QwenSpectrumConfig,
) -> Callable[[Callable[..., Any], dict[str, Any]], Any]:
    state = getattr(core, STATE_ATTR, None)
    if state is None:
        state = QwenSpectrumState(config=config)
        setattr(core, STATE_ATTR, state)

    def unet_wrapper_function(model_function: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
        input_tensor = kwargs["input"]
        timestep = kwargs["timestep"]
        c = kwargs["c"]
        transformer_options = c.setdefault("transformer_options", {})
        sigmas = transformer_options.get("sample_sigmas")
        cond_or_uncond = kwargs.get("cond_or_uncond")

        if sigmas is None or not isinstance(sigmas, torch.Tensor) or sigmas.numel() == 0:
            return model_function(input_tensor, timestep, **c)

        current_step_index = find_step_index(sigmas, timestep)
        total_steps = max(1, len(sigmas) - 1)

        if current_step_index == 0 and should_reset_for_step_zero(cond_or_uncond):
            state.reset()

        control_present = (
            kwargs.get("control") is not None
            or c.get("control") is not None
            or transformer_options.get("control") is not None
        )

        decision_actual, reason = decide_actual_or_forecast(
            state=state,
            current_step_index=current_step_index,
            total_steps=total_steps,
            control_present=control_present,
            core=core,
        )

        runtime = QwenSpectrumRuntime(
            config=config,
            current_step_index=current_step_index,
            total_steps=total_steps,
            current_sigma=float(sigmas[current_step_index].item()),
            decision_actual=decision_actual,
            forecast_reason=reason,
        )

        transformer_options["spectrum_actual_forward"] = decision_actual
        setattr(core, RUNTIME_ATTR, runtime)
        setattr(core, STATE_ATTR, state)

        original_forward = core.forward
        patched_forward = build_qwen_core_forward(core, original_forward).__get__(core, core.__class__)

        try:
            with patch.multiple(core, forward=patched_forward):
                out = model_function(input_tensor, timestep, **c)
        finally:
            if hasattr(core, RUNTIME_ATTR):
                delattr(core, RUNTIME_ATTR)

        if current_step_index + 1 == total_steps:
            log_debug(
                config.debug,
                (
                    "Spectrum Qwen summary: "
                    f"actual={state.actual_count} forecast={state.forecast_count} "
                    f"total={state.actual_count + state.forecast_count}"
                ),
            )

        return out

    return unet_wrapper_function
