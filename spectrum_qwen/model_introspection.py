from __future__ import annotations

from collections import deque
from typing import Any

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

        for attr_name in KNOWN_INNER_ATTRS:
            if hasattr(current, attr_name):
                queue.append(getattr(current, attr_name))

    return None
