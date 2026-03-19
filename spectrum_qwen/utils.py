from __future__ import annotations

import copy
import logging
from types import SimpleNamespace
from typing import Any, Callable

import torch

LOGGER = logging.getLogger("ComfyUI-Spectrum-Qwen")


def resolve_cache_target(hidden: torch.Tensor, cache_device: str) -> tuple[torch.device, torch.dtype]:
    if cache_device == "cpu":
        return torch.device("cpu"), torch.float32
    if cache_device == "offload_device":
        try:
            import comfy.model_management as model_management

            return torch.device(model_management.unet_offload_device()), hidden.dtype
        except Exception:
            return hidden.device, hidden.dtype
    return hidden.device, hidden.dtype


def build_output_factory(example_output: Any) -> Callable[[Any, bool], Any]:
    if isinstance(example_output, tuple):
        def tuple_factory(sample: Any, return_dict: bool) -> Any:
            return (sample,)
        return tuple_factory

    if hasattr(example_output, "sample"):
        output_cls = example_output.__class__

        def dataclass_factory(sample: Any, return_dict: bool) -> Any:
            if not return_dict:
                return (sample,)
            try:
                return output_cls(sample=sample)
            except Exception:
                try:
                    cloned = copy.copy(example_output)
                    cloned.sample = sample
                    return cloned
                except Exception:
                    return SimpleNamespace(sample=sample)

        return dataclass_factory

    def raw_factory(sample: Any, return_dict: bool) -> Any:
        if not return_dict:
            return (sample,)
        return sample

    return raw_factory


def log_debug(enabled: bool, message: str) -> None:
    if enabled:
        LOGGER.info(message)
