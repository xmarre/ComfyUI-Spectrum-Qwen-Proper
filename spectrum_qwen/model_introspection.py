from __future__ import annotations

from collections import deque
from typing import Any

import torch.nn as nn

from .constants import SUPPORTED_FORWARD_FIELDS


KNOWN_INNER_ATTRS = (
    "model",
    "diffusion_model",
    "module",
    "inner_model",
    "unet",
    "transformer",
)


def is_qwen_like_core(obj: Any) -> bool:
    if obj is None:
        return False
    if not all(hasattr(obj, field) for field in SUPPORTED_FORWARD_FIELDS):
        return False
    type_name = type(obj).__name__.lower()
    module_name = getattr(type(obj), "__module__", "").lower()
    if "qwen" in type_name or "qwen" in module_name:
        return True
    return hasattr(obj, "transformer_blocks") and hasattr(obj, "txt_in") and hasattr(obj, "img_in")



def iter_candidate_children(obj: Any):
    for attr_name in KNOWN_INNER_ATTRS:
        if hasattr(obj, attr_name):
            yield getattr(obj, attr_name)

    if isinstance(obj, nn.Module):
        for child in obj.children():
            yield child


def resolve_qwen_core(diffusion_model: Any) -> Any | None:
    queue = deque([diffusion_model])
    seen: set[int] = set()

    while queue:
        current = queue.popleft()
        if current is None:
            continue
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)

        if is_qwen_like_core(current):
            return current

        for child in iter_candidate_children(current):
            queue.append(child)

    return None
