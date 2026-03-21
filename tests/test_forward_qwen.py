from __future__ import annotations

import unittest

import torch

from spectrum_qwen.forward_qwen import _reconstruct_qwen_output


class _FakeCore:
    patch_size = 2

    def process_img(self, x: torch.Tensor):
        tokenized_x = torch.zeros((1, 4, 8), device=x.device, dtype=x.dtype)
        img_ids = torch.zeros((1, 4, 3), device=x.device, dtype=x.dtype)
        orig_shape = (1, 2, 1, 4, 4)
        return tokenized_x, img_ids, orig_shape


class ReconstructQwenOutputTest(unittest.TestCase):
    def test_reconstructs_original_spatial_shape_with_crop_and_extra_tokens(self) -> None:
        core = _FakeCore()
        model_input = torch.zeros((1, 2, 1, 4, 2))
        out_sample = torch.arange(48, dtype=torch.float32).reshape(1, 6, 8)

        reconstructed = _reconstruct_qwen_output(core, model_input, out_sample)

        self.assertEqual(tuple(reconstructed.shape), tuple(model_input.shape))


if __name__ == "__main__":
    unittest.main()
