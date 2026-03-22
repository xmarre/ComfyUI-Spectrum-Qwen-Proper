from __future__ import annotations

from typing import Any, Callable

import torch

from .state import QwenSpectrumRuntime, QwenSpectrumState
from .utils import build_output_factory, log_debug, resolve_cache_target


def _call_time_text_embed(
    core: Any,
    timestep: torch.Tensor,
    hidden_states: torch.Tensor,
    additional_t_cond: Any,
) -> torch.Tensor:
    timestep = timestep.to(hidden_states.dtype)
    return core.time_text_embed(timestep, hidden_states, additional_t_cond)


def _reconstruct_qwen_output(
    core: Any,
    model_input: torch.Tensor,
    out_sample: torch.Tensor,
) -> torch.Tensor:
    try:
        tokenized_x, _, orig_shape = core.process_img(model_input)
    except Exception as exc:
        raise RuntimeError(
            "Spectrum Qwen forecast path could not reconstruct outer QwenImage output shape "
            "from process_img(model_input)."
        ) from exc

    num_embeds = tokenized_x.shape[1]
    patch_size = int(getattr(core, "patch_size", 2))

    out_sample = out_sample[:, :num_embeds].reshape(
        orig_shape[0],
        orig_shape[-2] // patch_size,
        orig_shape[-1] // patch_size,
        orig_shape[1],
        patch_size,
        patch_size,
    )
    out_sample = out_sample.permute(0, 3, 1, 4, 2, 5)
    return out_sample.reshape(orig_shape)[:, :, :, : model_input.shape[-2], : model_input.shape[-1]]



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
        time_coord=runtime.current_time_coord,
        feature=feature,
        output_factory=build_output_factory(out),
    )
    log_debug(
        runtime.config.debug,
        (
            f"Spectrum Qwen step={runtime.current_step_index + 1}/{runtime.total_steps} "
            f"branch={runtime.branch_key} "
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
    additional_t_cond: Any,
) -> Any:
    if state.output_factory is None:
        raise RuntimeError("output factory missing before forecast")
    pred = state.forecaster.predict(runtime.current_time_coord)

    pred = pred.to(device=hidden_states.device, dtype=hidden_states.dtype)
    temb = _call_time_text_embed(core, timestep, pred, additional_t_cond)
    out_sample = core.proj_out(core.norm_out(pred, temb))
    out_sample = _reconstruct_qwen_output(core, hidden_states, out_sample)
    state.record_forecast()
    log_debug(
        runtime.config.debug,
        (
            f"Spectrum Qwen step={runtime.current_step_index + 1}/{runtime.total_steps} "
            f"branch={runtime.branch_key} "
            f"mode=forecast history={len(state.history_features)}"
        ),
    )
    return state.output_factory(out_sample, True)



def build_qwen_core_forward(
    core: Any,
    original_forward: Callable[..., Any],
) -> Callable[..., Any]:
    def spectrum_qwen_forward(
        self: Any,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        ref_latents: Any = None,
        additional_t_cond: Any = None,
        transformer_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        runtime: QwenSpectrumRuntime | None = getattr(self, "_spectrum_qwen_runtime", None)
        state: QwenSpectrumState | None = getattr(self, "_spectrum_qwen_state", None)
        if runtime is None or state is None:
            return original_forward(
                x,
                timestep=timestep,
                context=context,
                attention_mask=attention_mask,
                ref_latents=ref_latents,
                additional_t_cond=additional_t_cond,
                transformer_options=transformer_options,
                **kwargs,
            )

        if runtime.decision_actual:
            return _run_actual_forward(
                self,
                state,
                runtime,
                original_forward,
                x,
                timestep=timestep,
                context=context,
                attention_mask=attention_mask,
                ref_latents=ref_latents,
                additional_t_cond=additional_t_cond,
                transformer_options=transformer_options,
                **kwargs,
            )

        try:
            return _run_forecast_forward(
                self,
                state,
                runtime,
                hidden_states=x,
                timestep=timestep,
                additional_t_cond=additional_t_cond,
            )
        except Exception:
            return _run_actual_forward(
                self,
                state,
                runtime,
                original_forward,
                x,
                timestep=timestep,
                context=context,
                attention_mask=attention_mask,
                ref_latents=ref_latents,
                additional_t_cond=additional_t_cond,
                transformer_options=transformer_options,
                **kwargs,
            )

    return spectrum_qwen_forward
