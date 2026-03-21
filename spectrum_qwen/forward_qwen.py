from __future__ import annotations

from typing import Any, Callable

import torch

from .chebyshev import forecast_feature
from .state import QwenSpectrumRuntime, QwenSpectrumState
from .utils import build_output_factory, log_debug, resolve_cache_target


def _call_time_text_embed(
    core: Any,
    timestep: torch.Tensor,
    guidance: torch.Tensor | None,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    timestep = timestep.to(hidden_states.dtype)
    if guidance is None:
        return core.time_text_embed(timestep, hidden_states)

    guidance = guidance.to(hidden_states.dtype) * 1000
    return core.time_text_embed(timestep, guidance, hidden_states)



def _run_actual_forward(
    core: Any,
    state: QwenSpectrumState,
    runtime: QwenSpectrumRuntime,
    original_forward: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    captured: dict[str, torch.Tensor] = {}

    def capture_pre_norm(_module: Any, hook_args: tuple[Any, ...]) -> None:
        if not hook_args:
            return
        hidden = hook_args[0]
        target_device, target_dtype = resolve_cache_target(hidden, state.config.cache_device)
        captured["feature"] = hidden.detach().to(device=target_device, dtype=target_dtype)

    handle = core.norm_out.register_forward_pre_hook(capture_pre_norm)
    try:
        out = original_forward(*args, **kwargs)
    finally:
        handle.remove()

    feature = captured.get("feature")
    if feature is None:
        if not state.warned_capture_once:
            from .utils import LOGGER

            LOGGER.warning("Spectrum Qwen: failed to capture pre-norm hidden state; using pure fallback behavior.")
            state.warned_capture_once = True
        return out

    state.record_actual(
        sigma=runtime.current_sigma,
        feature=feature,
        output_factory=build_output_factory(out),
    )
    log_debug(
        runtime.config.debug,
        (
            f"Spectrum Qwen step={runtime.current_step_index + 1}/{runtime.total_steps} "
            f"mode=actual reason={runtime.forecast_reason} "
            f"history={len(state.history_features)}"
        ),
    )
    return out



def _run_forecast_forward(
    core: Any,
    state: QwenSpectrumState,
    runtime: QwenSpectrumRuntime,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    guidance: torch.Tensor | None,
    return_dict: bool,
) -> Any:
    if state.output_factory is None:
        raise RuntimeError("output factory missing before forecast")

    pred = forecast_feature(
        history_sigmas=state.history_sigmas,
        history_features=state.history_features,
        target_sigma=runtime.current_sigma,
        degree=runtime.config.chebyshev_degree,
        ridge_lambda=runtime.config.ridge_lambda,
    )

    pred = pred.to(device=hidden_states.device, dtype=hidden_states.dtype)
    temb = _call_time_text_embed(core, timestep, guidance, pred)
    out_sample = core.proj_out(core.norm_out(pred, temb))
    state.record_forecast()
    log_debug(
        runtime.config.debug,
        (
            f"Spectrum Qwen step={runtime.current_step_index + 1}/{runtime.total_steps} "
            f"mode=forecast history={len(state.history_features)}"
        ),
    )
    return state.output_factory(out_sample, return_dict)



def build_qwen_core_forward(
    core: Any,
    original_forward: Callable[..., Any],
) -> Callable[..., Any]:
    def spectrum_qwen_forward(
        self: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        image_rotary_emb: Any = None,
        guidance: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        runtime: QwenSpectrumRuntime | None = getattr(self, "_spectrum_qwen_runtime", None)
        state: QwenSpectrumState | None = getattr(self, "_spectrum_qwen_state", None)
        if runtime is None or state is None:
            return original_forward(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                timestep=timestep,
                image_rotary_emb=image_rotary_emb,
                guidance=guidance,
                attention_kwargs=attention_kwargs,
                return_dict=return_dict,
                **kwargs,
            )

        if runtime.decision_actual:
            return _run_actual_forward(
                self,
                state,
                runtime,
                original_forward,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                timestep=timestep,
                image_rotary_emb=image_rotary_emb,
                guidance=guidance,
                attention_kwargs=attention_kwargs,
                return_dict=return_dict,
                **kwargs,
            )

        try:
            return _run_forecast_forward(
                self,
                state,
                runtime,
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                return_dict=return_dict,
            )
        except Exception:
            return _run_actual_forward(
                self,
                state,
                runtime,
                original_forward,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                timestep=timestep,
                image_rotary_emb=image_rotary_emb,
                guidance=guidance,
                attention_kwargs=attention_kwargs,
                return_dict=return_dict,
                **kwargs,
            )

    return spectrum_qwen_forward
