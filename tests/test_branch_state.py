from __future__ import annotations

import unittest

import torch
from torch import nn

from spectrum_qwen.chebyshev import ChebyshevSpectrumForecaster, normalize_step_position
from spectrum_qwen.config import QwenSpectrumConfig
from spectrum_qwen.patcher import create_qwen_spectrum_unet_wrapper


class _IdentityNorm(nn.Module):
    def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        return x


class _FakeCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm_out = _IdentityNorm()
        self.proj_out = nn.Identity()
        self.patch_size = 1
        self.gradient_checkpointing = False
        self.zero_cond_t = False

    def time_text_embed(
        self,
        timestep: torch.Tensor,
        hidden_states: torch.Tensor,
        additional_t_cond: object,
    ) -> torch.Tensor:
        return hidden_states

    def process_img(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
        tokenized_x = torch.zeros((1, x.shape[-2] * x.shape[-1], 1), device=x.device, dtype=x.dtype)
        img_ids = torch.zeros((1, tokenized_x.shape[1], 3), device=x.device, dtype=x.dtype)
        return tokenized_x, img_ids, tuple(x.shape)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        ref_latents: object = None,
        additional_t_cond: object = None,
        transformer_options: dict[str, object] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.norm_out(x)


class BranchStateWrapperTest(unittest.TestCase):
    def test_incremental_forecaster_matches_bruteforce_and_uses_cache(self) -> None:
        def brute_force_predict(
            history: list[tuple[float, torch.Tensor]],
            target: float,
            degree: int,
            ridge_lambda: float,
        ) -> torch.Tensor:
            coords = torch.tensor([coord for coord, _ in history], dtype=torch.float32)
            features = torch.stack([feat.reshape(-1).to(torch.float32) for _, feat in history], dim=0)

            def design(values: torch.Tensor) -> torch.Tensor:
                values = values.reshape(-1, 1)
                cols = [torch.ones((values.shape[0], 1), dtype=torch.float32)]
                if degree >= 1:
                    cols.append(values)
                    for _ in range(2, degree + 1):
                        cols.append(2.0 * values * cols[-1] - cols[-2])
                return torch.cat(cols[: degree + 1], dim=1)

            design_hist = design(coords)
            lhs = design_hist.transpose(0, 1) @ design_hist
            if ridge_lambda > 0:
                lhs = lhs + torch.eye(lhs.shape[0], dtype=torch.float32) * float(ridge_lambda)
            rhs = design_hist.transpose(0, 1) @ features
            coeff = torch.cholesky_solve(rhs, torch.linalg.cholesky(lhs))
            return (design(torch.tensor([target], dtype=torch.float32)) @ coeff).reshape(history[-1][1].shape)

        forecaster = ChebyshevSpectrumForecaster(degree=2, ridge_lambda=0.1, max_history=4)
        raw_history: list[tuple[float, torch.Tensor]] = []

        for step_index in range(5):
            coord = normalize_step_position(step_index, 5)
            feature = torch.tensor([coord, coord * coord], dtype=torch.float16)
            raw_history.append((coord, feature))
            forecaster.update(coord, feature)

        self.assertTrue(forecaster.ready())

        original_recompute = forecaster._recompute_coefficients

        def fail_recompute() -> None:
            raise AssertionError("predict() unexpectedly recomputed the fit")

        forecaster._recompute_coefficients = fail_recompute  # type: ignore[method-assign]
        try:
            pred = forecaster.predict(1.5)
        finally:
            forecaster._recompute_coefficients = original_recompute  # type: ignore[method-assign]

        want = brute_force_predict(raw_history[-4:], 1.5, degree=2, ridge_lambda=0.1).to(pred.dtype)
        self.assertTrue(torch.allclose(pred, want, atol=1e-3, rtol=1e-3))

    def test_separates_branches_and_resets_on_new_run(self) -> None:
        config = QwenSpectrumConfig(
            warmup_steps=0,
            tail_actual_steps=0,
            history_points=2,
            chebyshev_degree=1,
            max_consecutive_forecasts=1,
            debug=False,
        )
        core = _FakeCore()
        wrapper = create_qwen_spectrum_unet_wrapper(core, core, config)

        model_function = lambda input_tensor, timestep, **c: core.forward(input_tensor, timestep, **c)
        input_tensor = torch.zeros((1, 1, 2, 2, 2))
        sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)

        def invoke(step_sigma: float, branch: list[int]) -> None:
            wrapper(
                model_function,
                {
                    "input": input_tensor,
                    "timestep": torch.tensor([step_sigma], dtype=torch.float32),
                    "c": {"transformer_options": {"sample_sigmas": sigmas}},
                    "cond_or_uncond": branch,
                },
            )

        invoke(1.0, [0])
        invoke(1.0, [1])

        root_state = getattr(core, "_spectrum_qwen_root_state")
        branch_zero = root_state.branch_states[(0,)]
        branch_one = root_state.branch_states[(1,)]

        self.assertIsNot(branch_zero, branch_one)
        self.assertEqual(branch_zero.actual_count, 1)
        self.assertEqual(branch_one.actual_count, 1)
        self.assertEqual(len(branch_zero.history_features), 1)
        self.assertEqual(len(branch_one.history_features), 1)

        invoke(0.5, [0])
        invoke(0.5, [1])

        self.assertEqual(branch_zero.actual_count, 2)
        self.assertEqual(branch_one.actual_count, 2)
        self.assertEqual(len(branch_zero.history_features), 2)
        self.assertEqual(len(branch_one.history_features), 2)

        invoke(1.0, [0])

        self.assertEqual(set(root_state.branch_states.keys()), {(0,)})
        self.assertIsNot(root_state.branch_states[(0,)], branch_zero)
        self.assertEqual(root_state.branch_states[(0,)].actual_count, 1)
        self.assertEqual(len(root_state.branch_states[(0,)].history_features), 1)

        sigmas_same_shape_new_run = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
        wrapper(
            model_function,
            {
                "input": input_tensor,
                "timestep": torch.tensor([1.0], dtype=torch.float32),
                "c": {"transformer_options": {"sample_sigmas": sigmas_same_shape_new_run}},
                "cond_or_uncond": [0],
            },
        )

        self.assertEqual(set(root_state.branch_states.keys()), {(0,)})
        self.assertEqual(root_state.branch_states[(0,)].actual_count, 1)
        self.assertEqual(len(root_state.branch_states[(0,)].history_features), 1)

    def test_replaces_cached_root_state_when_config_changes(self) -> None:
        core = _FakeCore()

        first_config = QwenSpectrumConfig(
            warmup_steps=0,
            tail_actual_steps=0,
            history_points=2,
            chebyshev_degree=1,
            max_consecutive_forecasts=1,
            debug=False,
        )
        second_config = QwenSpectrumConfig(
            warmup_steps=1,
            tail_actual_steps=1,
            history_points=3,
            chebyshev_degree=1,
            max_consecutive_forecasts=2,
            debug=False,
        )

        create_qwen_spectrum_unet_wrapper(core, core, first_config)
        first_root_state = getattr(core, "_spectrum_qwen_root_state")
        first_root_state.get_branch_state((0,))

        create_qwen_spectrum_unet_wrapper(core, core, second_config)
        second_root_state = getattr(core, "_spectrum_qwen_root_state")

        self.assertIsNot(first_root_state, second_root_state)
        self.assertEqual(second_root_state.config, second_config)
        self.assertEqual(second_root_state.branch_states, {})


if __name__ == "__main__":
    unittest.main()
