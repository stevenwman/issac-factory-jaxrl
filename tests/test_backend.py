"""Unit tests for factory_jax.backend.make_isaaclab_bundle.

Uses a mocked Isaac env so tests run without booting AppLauncher.
"""
import sys
import os
import pytest

# jax-learning path-import for EnvBundle type
sys.path.insert(0, os.environ.get(
    "JAX_LEARNING_PATH",
    "/home/stevenman/Desktop/Work/Research/jax-learning",
))

import jax
import jax.numpy as jnp
import numpy as np
import torch


# ---------- Fake Isaac env ----------
class _FakeIsaacEnv:
    """Stand-in for the wrapped Isaac DirectRLEnv (post-JaxEnvWrapper).

    Returns torch tensors → JaxEnvWrapper would have converted to jax.Array,
    but here we return jax.Array directly to mimic the post-wrap interface.
    """
    num_envs = 4
    POLICY_DIM = 19
    CRITIC_DIM = 43
    ACTION_DIM = 6

    def __init__(self):
        self._t = 0

    @property
    def unwrapped(self):
        return self

    def reset(self, seed=None, options=None):
        self._t = 0
        return self._make_obs(), {}

    def step(self, action):
        # Mimic JaxEnvWrapper output: dict of jax.Array
        self._t += 1
        obs = self._make_obs()
        reward = jnp.full((self.num_envs,), float(self._t) * 0.01)
        terminated = jnp.zeros((self.num_envs,), dtype=jnp.bool_)
        truncated = jnp.zeros((self.num_envs,), dtype=jnp.bool_)
        return obs, reward, terminated, truncated, {}

    def close(self):
        pass

    def _make_obs(self):
        return {
            "policy": jnp.full((self.num_envs, self.POLICY_DIM), float(self._t)),
            "critic": jnp.full((self.num_envs, self.CRITIC_DIM), float(self._t) + 0.1),
        }


# ---------- Fake TrainConfig (minimal, just for the test) ----------
from dataclasses import dataclass
@dataclass
class _FakeCfg:
    env_name: str
    num_envs: int


@pytest.fixture
def fake_isaac_backend(monkeypatch):
    """Stub `_make_isaac_env` so the backend doesn't try to boot AppLauncher."""
    import factory_jax.backend as backend
    monkeypatch.setattr(backend, "_make_isaac_env", lambda *a, **kw: _FakeIsaacEnv())
    return backend


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for jax devices")
def test_bundle_protocol_fields(fake_isaac_backend):
    from jax_rl.training.env_bundle import EnvBundle
    cfg = _FakeCfg(env_name="IsaacLab/FactoryJax-NutThread-v0", num_envs=4)
    bundle = fake_isaac_backend.make_isaaclab_bundle(cfg, seed=0)
    assert isinstance(bundle, EnvBundle)
    assert bundle.backend_kind == "isaaclab"
    assert bundle.obs_dim == 19
    assert bundle.action_dim == 6
    assert bundle.critic_obs_dim == 43
    assert bundle.has_privileged is True
    assert bundle.dict_obs is True
    assert bundle.num_envs == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_env_state_has_renamed_keys(fake_isaac_backend):
    """Per spec D8: obs dict keys are 'state' + 'privileged_state', not 'policy'/'critic'."""
    cfg = _FakeCfg(env_name="IsaacLab/FactoryJax-NutThread-v0", num_envs=4)
    bundle = fake_isaac_backend.make_isaaclab_bundle(cfg, seed=0)
    state = bundle.env_state
    # state should have .obs attribute that's a dict with renamed keys
    assert hasattr(state, "obs"), f"env_state must have .obs attribute, got {type(state)}"
    assert isinstance(state.obs, dict)
    assert "state" in state.obs
    assert "privileged_state" in state.obs
    assert "policy" not in state.obs
    assert "critic" not in state.obs
    # also check .reward, .done, .info exist on the struct
    assert hasattr(state, "reward")
    assert hasattr(state, "done")
    assert hasattr(state, "info")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_env_step_struct_protocol(fake_isaac_backend):
    """env_step returns the same struct type, with updated obs/reward/done/info."""
    cfg = _FakeCfg(env_name="IsaacLab/FactoryJax-NutThread-v0", num_envs=4)
    bundle = fake_isaac_backend.make_isaaclab_bundle(cfg, seed=0)
    action = jnp.zeros((4, 6), dtype=jnp.float32)
    new_state = bundle.env_step(bundle.env_state, action)
    # Same struct type
    assert type(new_state) is type(bundle.env_state)
    # Obs is dict with renamed keys
    assert "state" in new_state.obs
    assert "privileged_state" in new_state.obs
    # info has truncation key (per Phase 1 finding)
    assert "truncation" in new_state.info


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_register_backend_called(fake_isaac_backend):
    """Importing factory_jax.backend should register 'isaaclab' in BACKEND_BUILDERS."""
    from jax_rl.training.env_backends import BACKEND_BUILDERS
    assert "isaaclab" in BACKEND_BUILDERS
