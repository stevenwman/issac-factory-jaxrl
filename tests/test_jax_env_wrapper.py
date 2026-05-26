"""Unit tests for JaxEnvWrapper using a mocked Isaac-style env (no Isaac needed)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from factory_jax.bridge.jax_env_wrapper import JaxEnvWrapper


class FakeIsaacEnv:
    """Minimal stand-in for an Isaac DirectRLEnv: torch tensors on cuda, dict obs."""
    num_envs = 4
    action_dim = 6
    policy_obs_dim = 19
    critic_obs_dim = 43

    def __init__(self):
        self._t = 0

    def reset(self, seed=None, options=None):
        self._t = 0
        return self._make_obs(), {}

    def step(self, action):
        assert isinstance(action, torch.Tensor) and action.is_cuda
        assert action.shape == (self.num_envs, self.action_dim)
        self._t += 1
        reward = torch.full((self.num_envs,), float(self._t) * 0.01, device="cuda")
        terminated = torch.zeros((self.num_envs,), dtype=torch.bool, device="cuda")
        truncated = torch.zeros((self.num_envs,), dtype=torch.bool, device="cuda")
        return self._make_obs(), reward, terminated, truncated, {}

    def close(self):
        pass

    def _make_obs(self):
        return {
            "policy": torch.full((self.num_envs, self.policy_obs_dim), float(self._t), device="cuda"),
            "critic": torch.full((self.num_envs, self.critic_obs_dim), float(self._t) + 0.1, device="cuda"),
        }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_wrapper_reset_returns_jax_dict():
    env = JaxEnvWrapper(FakeIsaacEnv())
    obs, info = env.reset()
    assert set(obs.keys()) == {"policy", "critic"}
    assert isinstance(obs["policy"], jax.Array)
    assert isinstance(obs["critic"], jax.Array)
    assert obs["policy"].shape == (4, 19)
    assert obs["critic"].shape == (4, 43)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_wrapper_step_jax_in_jax_out():
    env = JaxEnvWrapper(FakeIsaacEnv())
    env.reset()
    action = jnp.zeros((4, 6), dtype=jnp.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, jax.Array)
    assert isinstance(terminated, jax.Array)
    assert isinstance(truncated, jax.Array)
    np.testing.assert_allclose(np.asarray(reward), np.full(4, 0.01, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(obs["policy"]), np.full((4, 19), 1.0, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(obs["critic"]), np.full((4, 43), 1.1, dtype=np.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_wrapper_step_advances_state():
    env = JaxEnvWrapper(FakeIsaacEnv())
    env.reset()
    action = jnp.zeros((4, 6), dtype=jnp.float32)
    for expected_t in range(1, 6):
        obs, reward, _, _, _ = env.step(action)
        np.testing.assert_allclose(np.asarray(reward), np.full(4, 0.01 * expected_t, dtype=np.float32),
                                   rtol=1e-6, atol=1e-6)
