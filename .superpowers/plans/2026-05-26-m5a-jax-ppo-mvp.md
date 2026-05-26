# M5a JAX PPO MVP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make jax-learning's stock `train_ppo.py` train on `FactoryJax-NutThread-v0` by implementing the missing `isaaclab` backend adapter in our project. Reward at final iter ≥ 2× reward at iter 0.

**Architecture:** Implement `make_isaaclab_bundle(cfg, seed) -> EnvBundle` in `source/factory_jax/factory_jax/backend.py`, register it under `register_backend("isaaclab", ...)`. Write a 50-line wrapper script `scripts/train_jax_ppo.py` that boots AppLauncher, triggers the registration, constructs a `TrainConfig`, and delegates to jax-learning's `train_ppo.train()`. Zero new training logic.

**Tech Stack:** Isaac Sim 5.1 + IsaacLab 2.3.2 (torch+cu128, isaaclab editable) + jax 0.5.3 (cuda12) + jax-learning (path-import). Bridge (`tensor_convert` + `JaxEnvWrapper`) already implemented and tested.

**Spec:** `.superpowers/specs/2026-05-26-m5a-jax-ppo-mvp-design.md`

---

## File Structure

```
Research/isaaclab-factory-jax/
├── scripts/
│   └── train_jax_ppo.py            ← CREATE (~50 lines)
├── source/factory_jax/factory_jax/
│   ├── backend.py                  ← CREATE (~120 lines)
│   └── tasks.py                    (existing, unchanged)
├── tests/
│   └── test_backend.py             ← CREATE (~80 lines, mocked bundle)
└── docs/
    └── jax_learning_divergences.md ← CREATE only if D9/D16 branch-patch triggers
```

## Conventions

- **Working dir:** `/home/stevenman/Desktop/Work/Research/isaaclab-factory-jax/`
- **Branch:** continue on `main`. Push at end of milestone per parent spec.
- **Test isolation:** any pytest in `tests/` must NOT require Isaac bootstrap (no pxr). Mock or use fake objects.
- **Conventional commits.** Each task ends in a commit. Push at end of plan.
- **Subagent model tiering:** Haiku for mechanical edits, Sonnet for integration/judgment, Opus reserved.

---

## Phase 1 — Research / resolve open questions

### Task 1.1: Read jax-learning's onpolicy_collect + env_step protocol

**Goal:** Resolve spec Q1 — what does `EnvBundle.env_step` need to return for `onpolicy_collect`?

**Files:** none yet (read-only research)

- [ ] **Step 1: Read `onpolicy_collect.py`**

```bash
wc -l /home/stevenman/Desktop/Work/Research/jax-learning/jax_rl/training/onpolicy_collect.py
grep -nE "env_step\|bundle\.env_step\|env_state" /home/stevenman/Desktop/Work/Research/jax-learning/jax_rl/training/onpolicy_collect.py | head -20
```

- [ ] **Step 2: Read the gym backend's env_step closure as reference**

```bash
sed -n '230,300p' /home/stevenman/Desktop/Work/Research/jax-learning/jax_rl/training/env_backends/gym_backend.py
```

- [ ] **Step 3: Record findings**

Write a short note to `.context/journal/2026-05-26.md` (append):
- env_step signature: `(state, action) -> ?` (what is state? what does it return?)
- whether GAE is computed inside or outside ppo.update
- whether RolloutBuffer handles asymmetric critic_obs

These notes inform backend.py implementation in Task 2.x. No code change in this task.

### Task 1.2: Read jax-learning's `train_ppo.train()` signature + TrainConfig schema

**Goal:** Resolve spec Q2, Q4, Q5 — what does our wrapper need to pass to `train()`, and what's the minimal TrainConfig we can construct?

**Files:** none yet (read-only research)

- [ ] **Step 1: Read `train_ppo.train()` signature**

```bash
grep -n "^def train" /home/stevenman/Desktop/Work/Research/jax-learning/scripts/train_ppo.py
sed -n '67,95p' /home/stevenman/Desktop/Work/Research/jax-learning/scripts/train_ppo.py
```

- [ ] **Step 2: Read `TrainConfig` schema + presets**

```bash
ls /home/stevenman/Desktop/Work/Research/jax-learning/jax_rl/configs/
cat /home/stevenman/Desktop/Work/Research/jax-learning/jax_rl/configs/train_config.py
grep -nE "eval_interval\|eval_every\|eval_frequency" /home/stevenman/Desktop/Work/Research/jax-learning/jax_rl/configs/train_config.py
```

- [ ] **Step 3: Find the closest preset for an MLP-policy manipulation env**

```bash
cat /home/stevenman/Desktop/Work/Research/jax-learning/jax_rl/configs/env_presets.py | head -80
grep -nE "def get_preset" /home/stevenman/Desktop/Work/Research/jax-learning/jax_rl/configs/env_presets.py
```

- [ ] **Step 4: Record findings**

Append to `.context/journal/2026-05-26.md`:
- `train()` signature + which kwargs we need
- TrainConfig fields we MUST set vs ones that default
- Whether there's an eval-interval knob (per spec D16 Q4)
- Which preset is closest match (per spec D15 Q5) — likely a Playground locomotion preset; pick one with reasonable MLP + lr defaults

### Task 1.3: Check Isaac env reset semantics quickly

**Goal:** Resolve spec Q3 (auto-reset behavior) and D5 (truncation handling, retained from earlier draft).

- [ ] **Step 1: Read DirectRLEnv.reset + step**

```bash
sed -n '290,360p' /home/stevenman/Desktop/Work/IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py
```

- [ ] **Step 2: Confirm reset semantics + truncation flag**

Note in journal:
- Whether dones (`terminated | truncated`) auto-reset the env
- What obs is returned at the done step (pre-reset terminal state, or post-reset initial state)

This informs the bundle's env_step closure: if auto-reset returns post-reset obs, the standard `value(next_obs) * (1 - done)` bootstrap is correct (spec D5 trivially valid).

### Task 1.4: Commit research notes

- [ ] **Step 1: Commit journal entry**

```bash
git add .context/journal/2026-05-26.md
git commit -m "docs(M5a): research notes - env_step protocol, TrainConfig schema, reset semantics"
```

Do NOT push yet — wait until end of plan.

---

## Phase 2 — backend.py (the adapter)

### Task 2.1: Write unit tests for backend (mocked, no Isaac)

**Goal:** TDD — define the protocol we expect `make_isaaclab_bundle` to satisfy, with a mocked-AppLauncher fixture.

**Files:**
- Create: `tests/test_backend.py`

- [ ] **Step 1: Write tests for the bundle protocol**

Tests should verify (using monkeypatch to stub Isaac imports — Isaac isn't required for these tests):

```python
# tests/test_backend.py
import pytest
import jax
import torch
import gymnasium as gym  # available without Isaac

class _FakeIsaacEnv:
    """Stand-in for gym.make(IsaacLab/...) result."""
    num_envs = 4
    action_space = gym.spaces.Box(-float("inf"), float("inf"), (4, 6))
    observation_space = gym.spaces.Dict({...})
    @property
    def unwrapped(self): return self
    @property
    def device(self): return "cuda:0"
    def reset(self, seed=None):
        return ({"policy": torch.zeros(4, 19, device="cuda"),
                 "critic": torch.zeros(4, 43, device="cuda")}, {})
    def step(self, action):
        return (
            {"policy": torch.zeros(4, 19, device="cuda"),
             "critic": torch.zeros(4, 43, device="cuda")},
            torch.zeros(4, device="cuda"),
            torch.zeros(4, dtype=torch.bool, device="cuda"),
            torch.zeros(4, dtype=torch.bool, device="cuda"),
            {},
        )
    def close(self): pass

@pytest.fixture
def fake_isaac(monkeypatch):
    """Stub the Isaac-touching imports so backend.py builds without booting Isaac."""
    import factory_jax.backend as backend
    monkeypatch.setattr(backend, "_make_isaac_env", lambda *a, **kw: _FakeIsaacEnv())
    return backend

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA needed")
def test_make_isaaclab_bundle_protocol(fake_isaac):
    from jax_rl.training.env_bundle import EnvBundle  # via path-import; safe — no Isaac
    cfg = ...  # build minimal cfg with env_name="IsaacLab/FactoryJax-NutThread-v0", num_envs=4
    bundle = fake_isaac.make_isaaclab_bundle(cfg, seed=0)
    assert isinstance(bundle, EnvBundle)
    assert bundle.backend_kind == "isaaclab"
    assert bundle.obs_dim == 19
    assert bundle.action_dim == 6
    assert bundle.critic_obs_dim == 43
    assert bundle.has_privileged is True
    assert bundle.dict_obs is True
    assert bundle.num_envs == 4
    # env_state should be a dict with renamed keys (per spec D8)
    assert isinstance(bundle.env_state, dict)
    assert "state" in bundle.env_state
    assert "privileged_state" in bundle.env_state

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA needed")
def test_env_step_clips_actions(fake_isaac):
    """Per spec D14: actions must be clipped to [-1, 1] before passed to Isaac."""
    cfg = ...
    bundle = fake_isaac.make_isaaclab_bundle(cfg, seed=0)
    # Pass an action with values outside [-1, 1]; the closure should clip
    import jax.numpy as jnp
    action = jnp.array([[5.0, -3.0, 0.5, 2.0, -1.5, 0.0]] * 4)
    obs, reward, term, trunc, info = bundle.env_step(bundle.env_state, action)
    # If we monkeypatched _FakeIsaacEnv to record actions, we'd assert here.
    # Simpler: just verify the call doesn't raise. Action clip is tested in
    # the integration smoke (Task 4.1).

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA needed")
def test_env_step_returns_renamed_dict_obs(fake_isaac):
    """Per spec D8: obs keys renamed policy->state, critic->privileged_state."""
    cfg = ...
    bundle = fake_isaac.make_isaaclab_bundle(cfg, seed=0)
    import jax.numpy as jnp
    action = jnp.zeros((4, 6))
    obs, _, _, _, _ = bundle.env_step(bundle.env_state, action)
    assert isinstance(obs, dict)
    assert "state" in obs
    assert "privileged_state" in obs
    # Should NOT have the original keys
    assert "policy" not in obs
    assert "critic" not in obs
```

NB: cfg construction depends on Phase 1 findings. If TrainConfig is too complex to construct here, build a `SimpleNamespace`-style stub that has `.env_name`, `.num_envs` attributes.

- [ ] **Step 2: Run tests, verify they FAIL with ImportError** (backend.py doesn't exist yet)

```bash
source .venv/bin/activate
python -m pytest tests/test_backend.py -v
```
Expected: collection error or ImportError on `factory_jax.backend`.

### Task 2.2: Implement `backend.py`

**Files:**
- Create: `source/factory_jax/factory_jax/backend.py`

**Preflight:** confirm `EnvBundle.env_step`'s signature against the journal entry from Task 1.1. If the signature differs from the skeleton below `(state, action) -> 5-tuple`, adjust both `env_step` AND `test_backend.py` (Task 2.1) together.

- [ ] **Step 1: Skeleton with imports + register call**

```python
# source/factory_jax/factory_jax/backend.py
"""IsaacLab backend for jax-learning's EnvBundle protocol.

Registers under the 'isaaclab' key. Triggered by env_name='IsaacLab/<task>'.
Per spec D2: lives in our project for now; future upstream to jax-learning.
"""
from __future__ import annotations

import os
import sys
import jax
import jax.numpy as jnp

# jax-learning path-import (per spec D10). Env-var overridable for portability.
sys.path.insert(0, os.environ.get(
    "JAX_LEARNING_PATH",
    "/home/stevenman/Desktop/Work/Research/jax-learning",
))

from jax_rl.training.env_bundle import EnvBundle
from jax_rl.training.env_backends import register_backend


def _make_isaac_env(task_id: str, num_envs: int, device: str = "cuda:0"):
    """Build the wrapped Isaac env. Separated so tests can monkeypatch it."""
    import gymnasium as gym
    import factory_jax.tasks  # noqa: F401  registers FactoryJax gym IDs
    from factory_jax.bridge.jax_env_wrapper import JaxEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(task_id, device=device, num_envs=num_envs)
    raw_env = gym.make(task_id, cfg=env_cfg)
    return JaxEnvWrapper(raw_env)


def make_isaaclab_bundle(cfg, seed: int) -> EnvBundle:
    """Build EnvBundle for cfg.env_name='IsaacLab/<task>'."""
    task_id = cfg.env_name.removeprefix("IsaacLab/")
    num_envs = cfg.num_envs

    env = _make_isaac_env(task_id, num_envs)

    # Initial obs (reset). Rename keys per spec D8.
    obs0, _ = env.reset(seed=seed)
    obs0 = {"state": obs0["policy"], "privileged_state": obs0["critic"]}

    def env_step(state, action):
        """jax-learning's onpolicy_collect calls this each step."""
        # D14: clip actions to [-1, 1] (Factory env doesn't clip; algo must)
        action = jnp.clip(action, -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        # D8: rename obs keys to what jax-learning's _extract_obs expects
        obs = {"state": obs["policy"], "privileged_state": obs["critic"]}
        return obs, reward, terminated, truncated, info

    return EnvBundle(
        env=env,
        env_step=env_step,
        env_state=obs0,
        eval_env=env,        # D16: eval disabled; eval_env value is unused
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


# Register at import time (lazy — function isn't called until detect_backend
# routes an "IsaacLab/..." env_name to us).
register_backend("isaaclab", make_isaaclab_bundle)
```

NB: Sign of `env_step` (5-tuple) MAY differ from jax-learning's expectation — Phase 1 research dictates the actual signature. Adjust if needed.

- [ ] **Step 2: Run tests, verify they PASS**

```bash
python -m pytest tests/test_backend.py -v
```
Expected: 3 passed (or however many we wrote in 2.1).

If a test fails because the env_step signature differs from what jax-learning expects: update both the test and `env_step` together based on Phase 1 findings.

- [ ] **Step 3: Commit**

```bash
git add source/factory_jax/factory_jax/backend.py tests/test_backend.py
git commit -m "feat(M5a): isaaclab backend adapter for jax-learning EnvBundle protocol"
```

---

## Phase 3 — Wrapper script

### Task 3.1: Write `scripts/train_jax_ppo.py`

**Files:**
- Create: `scripts/train_jax_ppo.py`

**Preflight:** from journal Task 1.2, confirm:
- The exact import path for `train()` (e.g. `from jax_rl.scripts.train_ppo import train` vs `from jax_rl.training.train_onpolicy import train`)
- The exact kwargs `train()` accepts (`seed`, `use_wandb`, `wandb_project`, ...) — adjust the call site accordingly
- The exact preset name to use in `get_preset(...)` (replace `"CheetahRun"` placeholder)
- The exact field name for eval-disable (skeleton uses `eval_every_n_episodes` per `train_config.py:48`; if Task 1.2 found a different name, use that)

- [ ] **Step 1: Write the wrapper**

```python
# scripts/train_jax_ppo.py
"""MVP wrapper: jax-learning PPO on FactoryJax-NutThread-v0 via our isaaclab backend.

Per spec M5a:
- D13: XLA_PYTHON_CLIENT_PREALLOCATE=false before any jax import
- D15: Hardcoded minimal TrainConfig (stop-gap; promote to preset later)
- D16: eval_interval set to disable in-process eval
- D17: try/finally to close simulation_app
"""
import argparse
import os
import sys

# === D13: GPU memory ===
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

# === Path-import jax-learning (D10) — BEFORE AppLauncher because pure-python ===
sys.path.insert(0, "/home/stevenman/Desktop/Work/Research/jax-learning")

# === AppLauncher (must come BEFORE any isaaclab imports) ===
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="JAX PPO on FactoryJax-NutThread-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--total_env_steps", type=int, default=500_000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# === Now safe to import isaaclab + register backend ===
import jax
import factory_jax.tasks   # noqa: F401  gym.register FactoryJax-NutThread-v0
import factory_jax.backend # noqa: F401  register_backend("isaaclab", ...)

# === D15: TrainConfig stop-gap ===
# Adapt based on Phase 1 findings on TrainConfig schema. Below is a sketch;
# tune to the actual field names in jax_rl/configs/train_config.py.
from jax_rl.configs import TrainConfig, get_preset

# Start from closest preset (Phase 1 picks the exact one). Override env-specifics.
try:
    cfg = get_preset("CheetahRun")  # placeholder — Phase 1 picks the right name
except Exception:
    # If get_preset balks at unknown names, fall back to TrainConfig() defaults
    cfg = TrainConfig()

import dataclasses
cfg = dataclasses.replace(
    cfg,
    env_name="IsaacLab/FactoryJax-NutThread-v0",
    num_envs=args_cli.num_envs,
    total_timesteps=args_cli.total_env_steps,
    # D16: effectively disable in-process eval. Field name confirmed in
    # jax_rl/configs/train_config.py is `eval_every_n_episodes` (verify in Task 1.2).
    eval_every_n_episodes=10**12,
)

# === Train ===
from jax_rl.scripts.train_ppo import train

try:
    train(cfg, seed=args_cli.seed, use_wandb=args_cli.wandb,
          wandb_project="isaaclab-factory-jax")
finally:
    # D17: clean up Isaac
    simulation_app.close()
```

Several `__placeholder__` names (preset name, exact cfg fields, train() kwargs) need to be replaced based on Phase 1 research. Adjust freely.

- [ ] **Step 2: Commit (script alone, before smoke test)**

```bash
git add scripts/train_jax_ppo.py
git commit -m "feat(M5a): train_jax_ppo.py wrapper - delegates to jax-learning's train_ppo.train via isaaclab backend"
```

---

## Phase 4 — Smoke test + iterate

### Task 4.1: Run smoke training

**Goal:** Verify the whole stack lights up. Expected: imports succeed, AppLauncher boots, env loads, registration fires, jax-learning's train_ppo runs at least 5 iterations, wandb dashboard shows live reward data.

- [ ] **Step 1: Smoke run with short budget**

```bash
cd /home/stevenman/Desktop/Work/Research/isaaclab-factory-jax
source .venv/bin/activate
NO_COLOR=1 python scripts/train_jax_ppo.py \
    --num_envs 64 --total_env_steps 50000 --wandb \
    2>&1 | sed -u 's/\x1b\[[0-9;]*[mGKHfABCDsuJ]//g' > /tmp/m5a_smoke.log &
echo "PID=$!"
```

Run in foreground (not Bash background tool) so we can debug interactively. Alternatively background + tail.

- [ ] **Step 2: Watch live log**

```bash
tail -f /tmp/m5a_smoke.log
```

Look for:
- AppLauncher boot messages (no segfault, no Fabric clone error)
- `factory_jax.backend` import succeeds (no `register_backend` error)
- jax-learning's train_ppo prints config + first epoch
- Wandb URL prints

- [ ] **Step 3: If smoke fails, diagnose**

| Symptom | Likely cause | Fix |
|---|---|---|
| ImportError on `jax_rl.scripts.train_ppo.train` | Function not exported at that path | Find the actual path; jax-learning may use `jax_rl.training.train_onpolicy.train` or similar |
| `TypeError: train() got unexpected keyword` | Wrong kwarg names | Check Phase 1.2 signature, adjust |
| `KeyError: 'IsaacLab/...'` from `get_preset` | Preset name doesn't exist | Pick another preset name from `env_presets.py` or skip presets |
| Action dimension mismatch in PPO | Bundle obs/action dims wrong | Verify FactoryJax env shapes via M3.9 numbers (19, 6, 43) |
| Reward stays at random level after 20 iters | Possibly action not flowing through (clip too aggressive? wrong key in obs dict?) | Print `action.min/max` in env_step closure |
| OOM | jax preallocated despite env var | Verify env var set BEFORE jax import; lower num_envs to 32 |

For each fix, commit incrementally:
```bash
git add ...
git commit -m "fix(M5a): <what>"
```

### Task 4.2: Full smoke run + verify success criterion

- [ ] **Step 1: Once smoke passes 5+ iterations cleanly, run full 500k env-step budget**

```bash
NO_COLOR=1 python scripts/train_jax_ppo.py \
    --num_envs 64 --total_env_steps 500000 --wandb \
    --wandb-name m5a-jax-ppo-mvp-full \
    > /tmp/m5a_full.log 2>&1 &
```

Wall-clock estimate: 30-60 min at num_envs=64 + Isaac speed.

- [ ] **Step 2: After completion, check final reward vs iter-0 reward**

Pull from wandb run summary (or from `runs/m5a_jax_ppo_smoke/summary.json` if our wrapper writes it).

Pass: `final_reward / iter0_reward >= 2.0`.

Fail: investigate per spec R5 chain (inspect rollouts, lower lr, check action distribution).

- [ ] **Step 3: Write `docs/M5a_mvp_result.md`**

```markdown
# M5a — JAX PPO MVP result

**Date:** <YYYY-MM-DD>
**Run:** <wandb URL>
**Backend:** isaaclab (our adapter, factory_jax/backend.py)
**Algo:** jax-learning PPO (jax_rl.algos.ppo.PPO)

## Setup
- num_envs: 64
- total env_steps: 500,000
- seed: 0
- TrainConfig: copied from preset `<which>`, overrides env_name + num_envs + eval_interval

## Result

| | Value |
|---|---|
| Iter-0 mean reward | |
| Final mean reward (last 10 iters avg) | |
| Ratio (final / iter-0) | |
| Pass (≥ 2.0)? | yes / no |

## Verdict

<short paragraph: did MVP pass; any divergences logged; next step (M5b FlashSAC or revisit)>

## Caveats
- Hyperparams not matched to rl_games M2 (deferred per spec scope)
- jax 0.5.3 not 0.9+ (parent spec D10 downgrade)
- (any divergences from `docs/jax_learning_divergences.md`)
```

- [ ] **Step 4: Commit + push**

```bash
git add docs/M5a_mvp_result.md
git commit -m "docs(M5a): MVP result - <pass/fail> jax PPO trains on Factory NutThread"
git push origin main
```

---

## Final verification checklist

- [ ] `pytest tests/ -v` — all 11 tests pass (5 tensor_convert + 3 jax_env_wrapper + 3 backend)
- [ ] Wandb dashboard shows reward curve rising over 500k env-steps
- [ ] `docs/M5a_mvp_result.md` exists with concrete pass/fail
- [ ] Git log shows commits per phase (Phase 1 research notes, Phase 2 backend, Phase 3 wrapper, Phase 4 result)
- [ ] No jax-learning files modified except on a topic branch (per D11) — `cd jax-learning && git log main..HEAD --oneline` empty unless D9/D16 branches needed

When all boxed: M5a MVP done. Either promote to full M5a (matched-config comparison) or move directly to M5b FlashSAC.
