"""gym.Wrapper exposing a jax.Array I/O over a torch-tensor Isaac env.

Preserves Isaac's `(num_envs, dim)` vectorized layout and dict-style obs
({"policy", "critic"} for asymmetric actor-critic envs like Factory).

Per spec §6: one `torch.cuda.synchronize()` per step as cheap stream-ordering
insurance. Profile + drop in M5b if not the bottleneck.
"""
from __future__ import annotations

from typing import Any

import jax
import torch

from .tensor_convert import from_jax, to_jax


def _torch_obs_to_jax(obs: Any):
    if isinstance(obs, dict):
        return {k: to_jax(v) for k, v in obs.items()}
    return to_jax(obs)


class JaxEnvWrapper:
    """Wrap a vectorized Isaac env so it speaks jax.Array on the outside.

    Duck-typed: requires the wrapped env to expose ``reset``, ``step``, ``close``
    matching gym's vectorized contract. Does NOT subclass ``gym.Wrapper`` because
    Isaac's ``DirectRLEnv`` is wrapped further by gym.make and we want to keep
    the wrapper independent of gym's type checks.
    """

    def __init__(self, env: Any):
        self.env = env

    @property
    def unwrapped(self):
        return getattr(self.env, "unwrapped", self.env)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return _torch_obs_to_jax(obs), info

    def step(self, action: jax.Array):
        action_torch = from_jax(action, target="torch")
        torch.cuda.synchronize()
        obs, reward, terminated, truncated, info = self.env.step(action_torch)
        torch.cuda.synchronize()
        return (
            _torch_obs_to_jax(obs),
            to_jax(reward),
            to_jax(terminated),
            to_jax(truncated),
            info,
        )

    def close(self):
        self.env.close()
