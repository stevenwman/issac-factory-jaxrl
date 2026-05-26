"""Source-tensor-type-agnostic DLPack bridge between Isaac (torch) and JAX.

When IsaacLab migrates to Newton/Warp (future spec D5), add a `warp.array`
branch to `to_jax` and a `target="warp"` branch to `from_jax`. The rest of the
bridge (env wrapper, training scripts) does not need to change.

Community-cited gotcha (IsaacLab GH discussion #24): torch tensors must be
`.contiguous()` before DLPack export. Non-contiguous strides cause errors or
silent corruption. We enforce it.
"""
from __future__ import annotations

from typing import Any, Literal

import jax
import torch
from torch.utils.dlpack import from_dlpack as torch_from_dlpack


def to_jax(x: Any) -> jax.Array:
    """Convert a torch.Tensor (or future warp.array) to jax.Array via DLPack.

    Zero-copy when source and JAX both live on the same CUDA device.
    Forces contiguity on torch source (DLPack contract).
    """
    if isinstance(x, torch.Tensor):
        if not x.is_contiguous():
            x = x.contiguous()
        # Modern jax API: pass array directly (consumes via __dlpack__ protocol).
        # The legacy torch_to_dlpack capsule path is deprecated in jax >= 0.4.
        return jax.dlpack.from_dlpack(x)
    raise TypeError(f"to_jax: unsupported source type {type(x).__name__}")


def from_jax(a: jax.Array, target: Literal["torch"] = "torch") -> Any:
    """Convert a jax.Array to the target tensor type via DLPack."""
    if target == "torch":
        # Modern: pass jax.Array directly; torch consumes via __dlpack__.
        return torch_from_dlpack(a)
    raise ValueError(f"from_jax: unsupported target {target!r}")
