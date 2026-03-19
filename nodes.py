from __future__ import annotations

from .spectrum_qwen import QwenSpectrumConfig, create_qwen_spectrum_unet_wrapper
from .spectrum_qwen.model_introspection import resolve_qwen_core


class QwenSpectrumModelPatcher:
    CATEGORY = "model/optimization"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "warmup_steps": (
                    "INT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Number of initial solver steps forced to run as real Qwen forwards.",
                    },
                ),
                "tail_actual_steps": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Number of final solver steps forced to stay on the real path.",
                    },
                ),
                "history_points": (
                    "INT",
                    {
                        "default": 5,
                        "min": 2,
                        "max": 16,
                        "step": 1,
                        "tooltip": "Number of real hidden-state snapshots kept for the Chebyshev fit.",
                    },
                ),
                "chebyshev_degree": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Polynomial degree used by the ridge-regularized Chebyshev forecaster.",
                    },
                ),
                "max_consecutive_forecasts": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Maximum number of skipped Qwen transformer passes allowed in a row before a real refresh step is forced.",
                    },
                ),
                "ridge_lambda": (
                    "FLOAT",
                    {
                        "default": 0.0001,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.0001,
                        "round": 0.0001,
                        "tooltip": "Ridge regularization strength used during the Chebyshev coefficient solve.",
                    },
                ),
                "cache_device": (
                    ["main_device", "offload_device", "cpu"],
                    {
                        "default": "main_device",
                        "tooltip": "Where captured Qwen hidden-state history is stored for fitting and forecasting.",
                    },
                ),
                "force_actual_on_control": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Force real forwards when ControlNet/control residuals are present. Recommended.",
                    },
                ),
                "debug": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Log per-step mode decisions and a summary at the end of the run.",
                    },
                ),
            }
        }

    def patch(
        self,
        model,
        warmup_steps: int,
        tail_actual_steps: int,
        history_points: int,
        chebyshev_degree: int,
        max_consecutive_forecasts: int,
        ridge_lambda: float,
        cache_device: str,
        force_actual_on_control: bool,
        debug: bool,
    ):
        config = QwenSpectrumConfig(
            warmup_steps=warmup_steps,
            tail_actual_steps=tail_actual_steps,
            history_points=history_points,
            chebyshev_degree=chebyshev_degree,
            max_consecutive_forecasts=max_consecutive_forecasts,
            ridge_lambda=ridge_lambda,
            cache_device=cache_device,
            force_actual_on_control=force_actual_on_control,
            debug=debug,
        )
        config.validate()

        model_clone = model.clone()
        diffusion_model = model_clone.get_model_object("diffusion_model")
        core = resolve_qwen_core(diffusion_model)
        if core is None:
            raise ValueError(
                "Qwen Spectrum patcher could not find a native Qwen Image-family transformer core on this MODEL. "
                "Use it with ComfyUI's native Qwen Image / Qwen Image Edit loaders or a wrapper that still exposes the same transformer internals."
            )

        if "transformer_options" not in model_clone.model_options:
            model_clone.model_options["transformer_options"] = {}
        model_clone.model_options["transformer_options"]["qwen_spectrum_enabled"] = True
        model_clone.model_options["transformer_options"]["qwen_spectrum_history_points"] = history_points
        model_clone.model_options["transformer_options"]["qwen_spectrum_degree"] = chebyshev_degree

        wrapper = create_qwen_spectrum_unet_wrapper(
            diffusion_model=diffusion_model,
            core=core,
            config=config,
        )
        model_clone.set_model_unet_function_wrapper(wrapper)
        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    "QwenSpectrumModelPatcher": QwenSpectrumModelPatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenSpectrumModelPatcher": "Spectrum Qwen Model Patcher",
}
