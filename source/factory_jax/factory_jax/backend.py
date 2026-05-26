"""IsaacLab backend for jax-learning's EnvBundle protocol.

Per spec D2: lives in our project; future upstream merge to jax-learning when stable.

Key design decisions (from Phase 1 research journal 2026-05-26):
- IsaacLabState mirrors GymState pattern from gym_backend.py (NamedTuple).
- obs keys renamed from Isaac's "policy"/"critic" -> "state"/"privileged_state" (spec D8).
- env_step does NOT clip actions - onpolicy_collect already clips at line 104.
- done = terminated | truncated (float, Brax convention); info["truncation"] = truncated.
- Factory's _get_dones returns (time_out, time_out) so terminated == truncated always;
  handle_truncation=True in TrainConfig is essential for correct GAE bootstrap.
- No manual env.reset() between training iterations - auto-reset handled inside step().
"""
from __future__ import annotations

import os
import sys
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

# jax-learning path-import (D10). Env-var overridable per plan reviewer rec.
sys.path.insert(0, os.environ.get(
    "JAX_LEARNING_PATH",
    "/home/stevenman/Desktop/Work/Research/jax-learning",
))

from jax_rl.training.env_bundle import EnvBundle
from jax_rl.training.env_backends import register_backend


class IsaacLabState(NamedTuple):
    """Mirrors GymState pattern from gym_backend.py. Carried through lax.scan."""
    obs: Any           # dict {"state": ..., "privileged_state": ...} of jax.Array
    reward: Any        # (num_envs,) jax.Array float
    done: Any          # (num_envs,) jax.Array float, terminated | truncated
    info: Any          # dict, must contain "truncation" key (per onpolicy_collect)


def _make_isaac_env(task_id: str, num_envs: int, device: str = "cuda:0"):
    """Build the wrapped Isaac env. Separated so tests can monkeypatch it.

    Must be called AFTER AppLauncher has booted (transitively imports pxr).
    """
    import gymnasium as gym
    import factory_jax.tasks  # noqa: F401  registers FactoryJax gym IDs
    from factory_jax.bridge.jax_env_wrapper import JaxEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(task_id, device=device, num_envs=num_envs)
    raw_env = gym.make(task_id, cfg=env_cfg)
    return JaxEnvWrapper(raw_env)


def _wrap_obs(obs_dict):
    """Rename Isaac's policy/critic keys to jax-learning's state/privileged_state."""
    return {"state": obs_dict["policy"], "privileged_state": obs_dict["critic"]}


def make_isaaclab_bundle(cfg, seed: int) -> EnvBundle:
    """Build EnvBundle for cfg.env_name='IsaacLab/<task>'."""
    task_id = cfg.env_name.removeprefix("IsaacLab/")
    num_envs = cfg.num_envs

    env = _make_isaac_env(task_id, num_envs)

    obs0, _ = env.reset(seed=seed)
    initial_state = IsaacLabState(
        obs=_wrap_obs(obs0),
        reward=jnp.zeros((num_envs,), dtype=jnp.float32),
        done=jnp.zeros((num_envs,), dtype=jnp.float32),
        info={"truncation": jnp.zeros((num_envs,), dtype=jnp.float32)},
    )

    def env_step(state: IsaacLabState, action: jax.Array) -> IsaacLabState:
        """jax-learning's onpolicy_collect calls this per scan step.

        Action is already clipped to [-1, 1] by onpolicy_collect.py:104.
        """
        obs, reward, terminated, truncated, info = env.step(action)
        done = jnp.asarray(terminated | truncated, dtype=jnp.float32)
        truncation = jnp.asarray(truncated, dtype=jnp.float32)
        return IsaacLabState(
            obs=_wrap_obs(obs),
            reward=jnp.asarray(reward, dtype=jnp.float32),
            done=done,
            info={"truncation": truncation},
        )

    return EnvBundle(
        env=env,
        env_step=env_step,
        env_state=initial_state,
        eval_env=env,
        obs_dim=19,
        action_dim=6,
        critic_obs_dim=43,
        has_privileged=True,
        dict_obs=True,
        key=jax.random.PRNGKey(seed),
        backend_kind="isaaclab",
        num_envs=num_envs,
        render_fn=None,
    )


# Register at module import time (lazy - function isn't called until detect_backend
# routes an "IsaacLab/..." env_name to us).
register_backend("isaaclab", make_isaaclab_bundle)
