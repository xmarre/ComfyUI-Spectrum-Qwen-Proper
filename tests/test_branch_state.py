from __future__ import annotations

import unittest

import torch
from torch import nn

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


if __name__ == "__main__":
    unittest.main()
