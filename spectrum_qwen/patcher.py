from __future__ import annotations

from typing import Any, Callable
from unittest.mock import patch

import torch

from .config import QwenSpectrumConfig
from .constants import RUNTIME_ATTR, STATE_ATTR
from .chebyshev import normalize_step_position
from .controller import decide_actual_or_forecast, find_step_index
from .forward_qwen import build_qwen_core_forward
from .state import QwenSpectrumRootState, QwenSpectrumRuntime
from .utils import log_debug


def create_qwen_spectrum_unet_wrapper(
    diffusion_model: Any,
    core: Any,
    config: QwenSpectrumConfig,
) -> Callable[[Callable[..., Any], dict[str, Any]], Any]:
    root_state = getattr(core, "_spectrum_qwen_root_state", None)
    if root_state is None or root_state.config != config:
        root_state = QwenSpectrumRootState(config=config)
        setattr(core, "_spectrum_qwen_root_state", root_state)

    def unet_wrapper_function(model_function: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
        input_tensor = kwargs["input"]
        timestep = kwargs["timestep"]
        c = kwargs["c"]
        transformer_options = c.setdefault("transformer_options", {})
        sigmas = transformer_options.get("sample_sigmas")
        cond_or_uncond = kwargs.get("cond_or_uncond")
        branch_key = tuple(cond_or_uncond) if cond_or_uncond is not None else ()

        if sigmas is None or not isinstance(sigmas, torch.Tensor) or sigmas.numel() == 0:
            return model_function(input_tensor, timestep, **c)

        current_step_index = find_step_index(sigmas, timestep)
        sigmas_id = id(sigmas)
        total_steps = max(1, len(sigmas) - 1)

        if (
            root_state.last_sigmas_id is not None
            and root_state.last_sigmas_id != sigmas_id
        ) or (
            root_state.last_total_steps is not None
            and root_state.last_total_steps != total_steps
        ) or (
            root_state.last_global_step_index is not None
            and current_step_index < root_state.last_global_step_index
        ):
            root_state.reset_run()

        state = root_state.get_branch_state(branch_key)

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
            current_time_coord=normalize_step_position(current_step_index, total_steps),
            decision_actual=decision_actual,
            forecast_reason=reason,
            branch_key=branch_key,
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

        root_state.last_global_step_index = current_step_index
        root_state.last_total_steps = total_steps
        root_state.last_sigmas_id = sigmas_id

        if current_step_index + 1 == total_steps:
            root_state.final_step_seen_branches.add(branch_key)
            if len(root_state.final_step_seen_branches) == len(root_state.branch_states):
                actual_calls = sum(branch.actual_count for branch in root_state.branch_states.values())
                forecast_calls = sum(branch.forecast_count for branch in root_state.branch_states.values())
                log_debug(
                    config.debug,
                    (
                        "Spectrum Qwen summary: "
                        f"branches={len(root_state.branch_states)} "
                        f"actual_calls={actual_calls} forecast_calls={forecast_calls} "
                        f"total_calls={actual_calls + forecast_calls}"
                    ),
                )

        return out

    return unet_wrapper_function
