from __future__ import annotations

import unittest

import torch

from spectrum_qwen.config import QwenSpectrumConfig
from spectrum_qwen.forward_qwen import (
    _reconstruct_qwen_output,
    _run_forecast_forward,
    _sanitize_forecast_feature,
)
from spectrum_qwen.state import QwenSpectrumRuntime, QwenSpectrumState


class _FakeCore:
    patch_size = 2

    def process_img(self, x: torch.Tensor):
        tokenized_x = torch.zeros((1, 4, 8), device=x.device, dtype=x.dtype)
        img_ids = torch.zeros((1, 4, 3), device=x.device, dtype=x.dtype)
        orig_shape = (1, 2, 1, 4, 4)
        return tokenized_x, img_ids, orig_shape


class _ForecastCore(_FakeCore):
    def __init__(self) -> None:
        self.seen_norm_dtype: torch.dtype | None = None

    def time_text_embed(self, timestep: torch.Tensor, hidden_states: torch.Tensor, additional_t_cond):
        return torch.zeros(
            (hidden_states.shape[0], hidden_states.shape[-1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    def norm_out(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        self.seen_norm_dtype = hidden_states.dtype
        return hidden_states

    def proj_out(self, hidden_states: torch.Tensor) -> torch.Tensor:
        extra = torch.zeros(
            (hidden_states.shape[0], 2, hidden_states.shape[-1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        return torch.cat([hidden_states, extra], dim=1)


class _FakeForecaster:
    def __init__(self, prediction: torch.Tensor) -> None:
        self.prediction = prediction

    def predict(self, time_coord: float) -> torch.Tensor:
        return self.prediction


class ReconstructQwenOutputTest(unittest.TestCase):
    def test_reconstructs_original_spatial_shape_with_crop_and_extra_tokens(self) -> None:
        core = _FakeCore()
        model_input = torch.zeros((1, 2, 1, 4, 2))
        out_sample = torch.arange(48, dtype=torch.float32).reshape(1, 6, 8)

        reconstructed = _reconstruct_qwen_output(core, model_input, out_sample)

        self.assertEqual(tuple(reconstructed.shape), tuple(model_input.shape))

    def test_forecast_uses_recorded_model_hidden_dtype_not_outer_input_dtype(self) -> None:
        core = _ForecastCore()
        state = QwenSpectrumState(config=QwenSpectrumConfig())
        state.output_factory = lambda sample, return_dict: sample
        state.model_feature_dtype = torch.bfloat16
        state.forecaster = _FakeForecaster(torch.ones((1, 4, 8), dtype=torch.float32))  # type: ignore[assignment]

        runtime = QwenSpectrumRuntime(
            config=state.config,
            current_step_index=3,
            total_steps=10,
            current_sigma=1.0,
            current_time_coord=0.0,
            decision_actual=False,
            forecast_reason="forecast",
            branch_key=(0,),
        )

        result = _run_forecast_forward(
            core,
            state,
            runtime,
            hidden_states=torch.zeros((1, 2, 1, 4, 2), dtype=torch.float16),
            timestep=torch.tensor([1.0], dtype=torch.float32),
            additional_t_cond=None,
        )

        self.assertEqual(core.seen_norm_dtype, torch.bfloat16)
        self.assertEqual(tuple(result.shape), (1, 2, 1, 4, 2))

    def test_forecast_sanitizer_clamps_nonfinite_values_for_target_dtype(self) -> None:
        feature = torch.tensor([0.0, float("nan"), float("inf"), -float("inf")], dtype=torch.float32)

        sanitized = _sanitize_forecast_feature(feature, torch.float16)

        self.assertEqual(sanitized.dtype, torch.float16)
        self.assertTrue(torch.isfinite(sanitized).all().item())

    def test_forecast_falls_back_to_prediction_dtype_when_model_dtype_is_missing(self) -> None:
        core = _ForecastCore()
        state = QwenSpectrumState(config=QwenSpectrumConfig())
        state.output_factory = lambda sample, return_dict: sample
        state.forecaster = _FakeForecaster(torch.ones((1, 4, 8), dtype=torch.float32))  # type: ignore[assignment]

        runtime = QwenSpectrumRuntime(
            config=state.config,
            current_step_index=3,
            total_steps=10,
            current_sigma=1.0,
            current_time_coord=0.0,
            decision_actual=False,
            forecast_reason="forecast",
            branch_key=(0,),
        )

        _run_forecast_forward(
            core,
            state,
            runtime,
            hidden_states=torch.zeros((1, 2, 1, 4, 2), dtype=torch.float16),
            timestep=torch.tensor([1.0], dtype=torch.float32),
            additional_t_cond=None,
        )

        self.assertEqual(core.seen_norm_dtype, torch.float32)

    def test_state_reset_clears_recorded_model_hidden_dtype(self) -> None:
        state = QwenSpectrumState(config=QwenSpectrumConfig())
        state.model_feature_dtype = torch.bfloat16

        state.reset()

        self.assertIsNone(state.model_feature_dtype)


if __name__ == "__main__":
    unittest.main()
