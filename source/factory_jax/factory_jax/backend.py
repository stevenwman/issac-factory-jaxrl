"""IsaacLab backend for jax-learning's EnvBundle protocol.

Per spec D2: lives in our project; future upstream merge to jax-learning when stable.

Key design decisions (from Phase 1 research journal 2026-05-26):
- IsaacLabState mirrors GymState pattern from gym_backend.py (NamedTuple).
- obs keys renamed from Isaac's "policy"/"critic" → "state"/"privileged_state" (spec D8).
- env_step does NOT clip actions — onpolicy_collect already clips at line 104.
- done = terminated | truncated (float, Brax convention); info["truncation"] = truncated.
- Factory's _get_dones returns (time_out, time_out) so terminated == truncated always;
  handle_truncation=True in TrainConfig is essential for correct GAE bootstrap.
- No manual env.reset() between training iterations — auto-reset handled inside step().

Import isolation note:
  jax_rl.training.__init__ and jax_rl.training.env_backends.__init__ eagerly import
  mjx_backend (which requires ml_collections, mujoco_playground, mujoco — MJX deps not
  present in this venv). We bypass the package __init__ chains via importlib direct-load,
  pre-registering stub modules for mjx_backend and gym_backend so env_backends/__init__
  doesn't try to execute them. This is safe: we only need EnvBundle + register_backend,
  both of which load without MJX deps.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# jax-learning path setup + targeted imports (bypass MJX-heavy __init__ chains)
# ---------------------------------------------------------------------------

_JAX_LEARNING = os.environ.get(
    "JAX_LEARNING_PATH",
    "/home/stevenman/Desktop/Work/Research/jax-learning",
)


def _load_module_direct(module_name: str, file_path: str):
    """Load a module by file path and register it in sys.modules under module_name.

    Bypasses the package __init__ chains so we don't trigger MJX deps.
    Idempotent: returns existing module if already registered.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Ensure top-level package stubs exist so Python's import machinery is happy
# when submodule files do relative-style lookups. Intentionally NOT stubbing
# jax_rl.training.env_backends here — that one gets loaded directly below.
for _pkg in ("jax_rl", "jax_rl.training"):
    if _pkg not in sys.modules:
        sys.modules[_pkg] = types.ModuleType(_pkg)

# Stub out MJX-heavy backends BEFORE env_backends/__init__.py is loaded.
# env_backends/__init__.py has eager side-effect imports at the bottom:
#   from jax_rl.training.env_backends import mjx_backend
#   from jax_rl.training.env_backends import gym_backend
# Pre-registering them as empty stubs makes those lines no-ops.
for _stub in (
    "jax_rl.training.env_backends.mjx_backend",
    "jax_rl.training.env_backends.gym_backend",
):
    if _stub not in sys.modules:
        sys.modules[_stub] = types.ModuleType(_stub)

# Load the two modules we actually need, directly from their files.
# env_bundle.py has no heavy deps (only dataclass + numpy).
# env_backends/__init__.py defines register_backend / BACKEND_BUILDERS.
_eb_mod = _load_module_direct(
    "jax_rl.training.env_bundle",
    f"{_JAX_LEARNING}/jax_rl/training/env_bundle.py",
)
_ebi_mod = _load_module_direct(
    "jax_rl.training.env_backends",
    f"{_JAX_LEARNING}/jax_rl/training/env_backends/__init__.py",
)

EnvBundle = _eb_mod.EnvBundle
register_backend = _ebi_mod.register_backend


# ---------------------------------------------------------------------------
# IsaacLabState — the env-state struct carried through lax.scan
# ---------------------------------------------------------------------------

class IsaacLabState(NamedTuple):
    """Mirrors GymState pattern from gym_backend.py. Carried through lax.scan.

    Fields match the surface expected by onpolicy_collect.py:
    - obs: dict {"state": actor_obs, "privileged_state": critic_obs} (jax.Array values)
    - reward: (num_envs,) jax.Array
    - done: (num_envs,) jax.Array, float — terminated | truncated (Brax convention)
    - info: dict, must always contain "truncation" key per onpolicy_collect contract
    """
    obs: Any
    reward: Any
    done: Any
    info: Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_isaac_env(task_id: str, num_envs: int, device: str = "cuda:0"):
    """Build the wrapped Isaac env. Separated so tests can monkeypatch it.

    Must be called AFTER AppLauncher has booted (factory_jax.tasks + isaaclab_tasks
    transitively import pxr).
    """
    import gymnasium as gym
    import factory_jax.tasks  # noqa: F401  registers FactoryJax gym IDs
    from factory_jax.bridge.jax_env_wrapper import JaxEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(task_id, device=device, num_envs=num_envs)
    raw_env = gym.make(task_id, cfg=env_cfg)
    return JaxEnvWrapper(raw_env)


def _wrap_obs(obs_dict: dict) -> dict:
    """Rename Isaac's policy/critic keys to jax-learning's state/privileged_state.

    Isaac env returns {"policy": actor_obs, "critic": critic_obs}.
    jax-learning's obs_pipeline reads {"state": ..., "privileged_state": ...}.
    """
    return {
        "state": obs_dict["policy"],
        "privileged_state": obs_dict["critic"],
    }


# ---------------------------------------------------------------------------
# Bundle factory
# ---------------------------------------------------------------------------

def make_isaaclab_bundle(cfg, seed: int) -> EnvBundle:
    """Build EnvBundle for cfg.env_name='IsaacLab/<task>'.

    Per spec D8: obs keys renamed inside env_step (and in initial state construction).
    Per Phase 1 findings: Factory's terminated == truncated always; bootstrap via
    handle_truncation=True in TrainConfig is essential.

    Action clipping (spec D14) is handled upstream by onpolicy_collect.py:104 —
    env_step receives already-clipped actions and passes them through unchanged.
    """
    task_id = cfg.env_name.removeprefix("IsaacLab/")
    num_envs = cfg.num_envs

    env = _make_isaac_env(task_id, num_envs)

    # Initial state from env reset
    obs0, _ = env.reset(seed=seed)
    initial_state = IsaacLabState(
        obs=_wrap_obs(obs0),
        reward=jnp.zeros((num_envs,), dtype=jnp.float32),
        done=jnp.zeros((num_envs,), dtype=jnp.float32),
        info={"truncation": jnp.zeros((num_envs,), dtype=jnp.float32)},
    )

    def env_step(state: IsaacLabState, action: jax.Array) -> IsaacLabState:
        """Step the Isaac env once. Called per-timestep inside lax.scan.

        action: (num_envs, action_dim) already clipped to [-1, 1] by caller.
        Returns: new IsaacLabState with updated obs/reward/done/info.
        """
        obs, reward, terminated, truncated, _info = env.step(action)
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


# Register at module import. Lazy — function is only called when detect_backend
# routes an "IsaacLab/..." env_name here.
register_backend("isaaclab", make_isaaclab_bundle)
