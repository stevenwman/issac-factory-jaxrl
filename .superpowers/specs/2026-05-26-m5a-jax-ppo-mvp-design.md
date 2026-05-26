# M5a — JAX PPO MVP on FactoryJax-NutThread-v0 (revised)

**Status:** Draft (revised after scope reality-check)
**Date:** 2026-05-26
**Owner:** robomechanicslab@gmail.com
**Parent spec:** `.superpowers/specs/2026-05-26-isaaclab-factory-jax-baseline-design.md`
**Sub-spec scope:** narrows M5a (originally "matched JAX PPO vs rl_games comparison") into the **smallest deliverable that proves jax-learning's existing `train_ppo.py` can train on our wrapped Isaac env via a single new adapter file.**

**Revision note:** initial draft mis-scoped this as "write our own training loop." After reading `jax_rl/training/env_bundle.py` and `env_backends/__init__.py`, jax-learning already has an `"isaaclab"` backend slot reserved — just unimplemented. The right move is to fill that slot, not to write a parallel trainer.

---

## 1. Goal

Implement the missing `isaaclab` backend for jax-learning's `EnvBundle` protocol, so that **jax-learning's stock `scripts/train_ppo.py` works against `FactoryJax-NutThread-v0` with only a CLI flag** (`--env IsaacLab/FactoryJax-NutThread-v0` or similar).

Success = reward rises during a short training run, observed live in wandb.

## 2. Success criterion

**Mean episode reward at the final iteration is at least 2× the mean episode reward at iteration 0**, measured by the training script itself and logged to wandb. Iteration 0 = the first PPO rollout (random policy from fresh init).

Self-contained pass/fail. No external baseline lookup. No matched config. No comparison number against rl_games.

## 3. Non-goals

- Writing a new training loop. **Use jax-learning's `scripts/train_ppo.py` unchanged.**
- Matched hyperparameter set with rl_games. Use jax-learning defaults.
- Eval cadence, best-ckpt tracking, video recording during training. (jax-learning's loop handles checkpointing; we accept whatever it does by default.)
- Obs normalization toggles, LSTM/recurrent policy.
- Multi-seed sweep.

## 4. Locked decisions

| # | Decision | Source / grounding |
|---|---|---|
| D1 | MVP scope: reward rises, not matched comparison | User decision |
| D2 | **Implement `isaaclab_backend` in OUR project** (`source/factory_jax/factory_jax/backend.py`), not in jax-learning. Register via `jax_rl.training.env_backends.register_backend("isaaclab", make_isaaclab_bundle)` at import time. | Lightweight-scope preference (memory `feedback-lightweight-scope`): avoids a long-lived jax-learning branch; our extension stays self-contained. Future merge upstream tracked as follow-up. |
| D3 | Env name convention: `IsaacLab/FactoryJax-NutThread-v0` (matches `detect_backend` prefix at `env_backends/__init__.py:43`) | jax-learning's existing dispatch logic |
| D4 | Use jax-learning's `scripts/train_ppo.py` **unmodified**. Invoke via our wrapper script `scripts/train_jax_ppo.py` which sys.path-inserts jax-learning, imports `factory_jax.backend` (triggers registration), then defers to jax-learning's `train()`. | Per spec D11: don't touch jax-learning's `main`. Our wrapper provides the registration hook + AppLauncher boot. |
| D5 | AppLauncher boot order: AppLauncher → `import factory_jax.tasks` (register gym ID) → `import factory_jax.backend` (register backend) → call `jax_rl.scripts.train_ppo.train(cfg, seed)`. | pxr import constraint: isaaclab modules can only be imported AFTER AppLauncher init. |
| D6 | num_envs: 64. Total env-step budget: 500k (~100-200 PPO iter at num_envs=64, num_steps=128). | MVP discipline + GPU memory headroom |
| D7 | jax-learning PPO config: `get_preset(...)` or `PPOConfig()` default; override only `num_envs`. **No further tuning.** | Avoid hyperparam debate in MVP |
| D8 | Asymmetric actor-critic: bundle declares `has_privileged=True`, `dict_obs=True`, `critic_obs_dim=43`. jax-learning's PPO already plumbs this via `train_ppo.py`'s `_extract_obs` helper (handles dict obs → policy + privileged_state split). **Our env returns dict with keys `policy` and `critic`; jax-learning's `_extract_obs` expects keys `state` and `privileged_state`.** Two options: (a) rename keys in bundle's `env_step` wrapper, (b) add a thin obs-key remap in our backend. We pick (a) — rename in env_step. | Read `train_ppo.py:43-49` for the expected keys. |
| D9 | jax 0.5.3 compatibility risk (jax-learning was written against 0.9+). Patch jax-learning ONLY on branch `isaaclab-factory-jax/m5a-jax05-compat` if hit at runtime. Record in `docs/jax_learning_divergences.md`. | Per parent spec D11 |
| D10 | Seed: 0. Single training run. | MVP discipline |
| D11 | Wandb on. Project `isaaclab-factory-jax`, entity `sman2`, name `m5a-jax-ppo-mvp-<timestamp>`. jax-learning's `train_ppo.py` already does wandb plumbing — pass `use_wandb=True`. | Parent spec wandb conventions |
| D12 | Future upstream: when jax-learning maintainer adds an official isaaclab backend, deprecate our `backend.py`. Until then, ours is the only one. | Long-term hygiene |
| D13 | **GPU memory contention:** set `XLA_PYTHON_CLIENT_PREALLOCATE=false` (or `XLA_PYTHON_CLIENT_MEM_FRACTION=0.3`) in our wrapper script env, BEFORE any jax import. jax otherwise grabs 75% of GPU memory at first use, leaving Isaac OOMing during sim build. Runtime config knobs to dial down memory if pressure surfaces: drop `num_envs`, drop `minibatch_size`, drop `num_steps` (rollout horizon). | RTX 5080 has 16 GB; Isaac wants ~6 GB; preallocated jax would grab ~12 GB → OOM. |
| D14 | **Action clipping**: set `clip_actions=1.0` in the bundle (rl_games convention; matches Isaac Factory's expectation). Factory's `action_space` is `Box(-inf, inf, (N, 6))` so the env itself doesn't clip; the algo must. Applies to **PPO AND any future algo** (FlashSAC, etc.) — bundle is algo-agnostic. Implementation: in the bundle's `env_step` closure, `action = jnp.clip(action, -1.0, 1.0)` before `from_jax`. | Per Factory paper's controller assumption + rl_games config (`clip_actions: 1.0`). |
| D15 | **TrainConfig construction stop-gap**: copy the closest jax-learning preset (e.g. `get_preset("CheetahRun")` or `get_preset("default_manipulation")` if it exists) into our wrapper, override `env_name=`, `num_envs=`, and any IsaacLab-specific fields. We do NOT add a new `IsaacLab/...` preset to jax-learning's `env_presets.py` for MVP — that's a branch-and-merge dance. Hardcode in our wrapper for now; promote to a real preset later. | Lightweight-scope memory + spec D11. |
| D16 | **Eval disabled for MVP**: Isaac convention is to NOT run in-process eval (SimulationApp singleton blocks two kits). jax-learning's `train_ppo.train()` calls `evaluate_gym` periodically. The cfg field name is **`eval_every_n_episodes`** (per `jax_rl/configs/train_config.py:48`); set to a very large number (e.g. `10**12`) to effectively disable eval. Fallback if field name is wrong or doesn't actually gate eval: patch `train_ppo.train` on branch `isaaclab-factory-jax/m5a-skip-eval` to skip eval when `backend_kind == "isaaclab"`. After training, run `scripts/play_jax_ppo.py` (separate Isaac process) for eval. | Spec D11 + Isaac kit-process constraint (kvdb lock observed during M2). |
| D17 | **simulation_app cleanup**: wrapper script uses `try / finally: simulation_app.close()` so Isaac kit shuts down cleanly even if training crashes mid-iter. | Avoids zombie procs holding GPU memory. |
| D18 | **Obs normalization handling**: jax-learning's `train_ppo.train()` does obs normalization inline (`norm_init` / `norm_normalize` / `norm_update`, train_ppo.py:153-207). It's hardcoded — not a cfg toggle. **We accept this as a feature** — Factory obs have varying dynamic ranges, normalization should help. We don't have to plumb anything. Note: norm_state is saved on checkpoint via `load_checkpoint` (line 160). | Verified by reading train_ppo.py. |

## 5. Architecture

### Files touched

```
Research/isaaclab-factory-jax/
├── scripts/
│   └── train_jax_ppo.py            ← CREATE (~40 lines: AppLauncher + registration hooks + delegate to jax-learning's train_ppo.train)
├── source/factory_jax/factory_jax/
│   └── backend.py                  ← CREATE (~120 lines: make_isaaclab_bundle, register at import)
└── docs/
    └── jax_learning_divergences.md ← CREATE only if D9 triggers
```

**No new training loop. No new RolloutBuffer. No new GAE.** All that lives in jax-learning, unchanged.

### `backend.py` (the meat — one new file)

Conforms to the builder protocol per `env_backends/__init__.py`:

```python
def make_isaaclab_bundle(cfg: TrainConfig, seed: int) -> EnvBundle:
    """Build EnvBundle for an IsaacLab env. cfg.env_name must be 'IsaacLab/<task-id>'."""
    # 1. Resolve task ID
    task_id = cfg.env_name.removeprefix("IsaacLab/")  # e.g. "FactoryJax-NutThread-v0"

    # 2. AppLauncher must already be running (handled by caller, scripts/train_jax_ppo.py)
    import gymnasium as gym
    import factory_jax.tasks  # noqa: F401  (registers FactoryJax-* gym IDs)
    from factory_jax.bridge.jax_env_wrapper import JaxEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    # 3. Make wrapped env
    env_cfg = parse_env_cfg(task_id, device="cuda:0", num_envs=cfg.num_envs)
    raw_env = gym.make(task_id, cfg=env_cfg)
    env = JaxEnvWrapper(raw_env)

    # 4. env_step closure: jax-learning's onpolicy_collect expects a step fn
    def env_step(state, action):
        obs, reward, terminated, truncated, info = env.step(action)
        # Rename obs keys to what jax-learning's _extract_obs expects
        obs = {"state": obs["policy"], "privileged_state": obs["critic"]}
        return obs, reward, terminated, truncated, info

    # 5. Initial obs
    obs0, _ = env.reset(seed=seed)
    obs0 = {"state": obs0["policy"], "privileged_state": obs0["critic"]}

    # 6. eval_env: same env for MVP (no separate eval). Defer eval rigor.
    return EnvBundle(
        env=env, env_step=env_step, env_state=obs0, eval_env=env,
        obs_dim=19, action_dim=6, critic_obs_dim=43,
        has_privileged=True, dict_obs=True,
        key=jax.random.PRNGKey(seed),
        backend_kind="isaaclab", num_envs=cfg.num_envs,
    )

register_backend("isaaclab", make_isaaclab_bundle)
```

### `scripts/train_jax_ppo.py` (the wrapper)

```python
# 0. ENV: set XLA_PYTHON_CLIENT_PREALLOCATE=false (D13) — BEFORE importing jax
# 1. argparse + AppLauncher boot
# 2. sys.path.insert for jax-learning
# 3. import factory_jax.tasks    # gym.register
# 4. import factory_jax.backend  # register_backend("isaaclab", ...)
# 5. Build TrainConfig (D15 stop-gap: copy a preset, override env_name + num_envs + eval_interval=inf per D16)
# 6. try: jax_rl.scripts.train_ppo.train(cfg, seed, use_wandb=True)
#    finally: simulation_app.close()  (D17)
```

Total: ~50 lines, no training logic of our own.

## 6. Data flow

Identical to jax-learning's existing PPO flow. The bundle's `env_step` is the only Isaac-specific call site. Everything downstream (rollout, GAE, PPO update, wandb logging, checkpointing) is jax-learning's stock code.

## 7. Risks + mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | jax 0.5 API gap inside jax-learning's PPO breaks at runtime | Patch on branch `isaaclab-factory-jax/m5a-jax05-compat` in jax-learning. Log in `docs/jax_learning_divergences.md`. |
| R2 | `EnvBundle.env_step` signature jax-learning expects doesn't match what I sketched in §5 | Read `onpolicy_collect.py` first thing in plan phase before writing backend.py |
| R3 | jax-learning's `train_ppo.train()` may not accept all args we need (e.g. `use_wandb`) — its signature is fixed | Read `scripts/train_ppo.py:67` (`def train(...)`) signature first. Add a thin shim in our wrapper if needed. |
| R4 | jax-learning's `_extract_obs` expects `obs["state"]` + `obs["privileged_state"]`, but we have `policy`+`critic`. Bundle's env_step closure remaps these. | D8 — already designed |
| R5 | Reward doesn't rise | Investigate: (a) inspect rollouts (b) try obs norm in jax-learning's cfg (c) lower lr (d) check that action is being fed correctly through bridge |
| R6 | Isaac OOM with num_envs=64 + jax memory on same GPU | Drop to 32, document |
| R7 | jax-learning may import a non-existent module at top of `train_ppo.py` (e.g. `mujoco_playground`) that we don't have | uv add the missing dep, or stub-out the import — minor friction |
| R8 | jax + Isaac OOM on shared GPU (jax preallocates 75%) | **D13 mitigates** — set `XLA_PYTHON_CLIENT_PREALLOCATE=false` env var in wrapper. If still tight: drop num_envs, drop minibatch_size, drop num_steps. |
| R9 | Policy emits unbounded actions, Isaac doesn't clip, exploration diverges | **D14 mitigates** — `jnp.clip(action, -1, 1)` in bundle's env_step closure. |
| R10 | TrainConfig has fields we can't fill from a preset (Playground-specific) | **D15 stop-gap** — hardcode minimal cfg in wrapper; promote to real preset later. |
| R11 | jax-learning's hardcoded periodic eval resets our env mid-training and corrupts rollout state | **D16 mitigates** — disable eval via cfg if possible, else branch-patch. |

## 8. Verification checkpoints

| When | Pass condition | If fail |
|---|---|---|
| `import factory_jax.backend` succeeds (post-AppLauncher) | `"isaaclab" in BACKEND_BUILDERS` | Check registration call |
| Bundle constructed | `EnvBundle.backend_kind == "isaaclab"`, obs/action dims match, env_state is dict with `state`+`privileged_state` keys | Trace `make_isaaclab_bundle` output |
| First rollout collected by jax-learning's onpolicy_collect | No exception, no NaN | This is most likely failure point — R2/R4 mitigation |
| First `ppo.update` returns | TrainingState updated, info dict non-empty | R1 mitigation |
| 10 iterations complete | wandb live, reward + losses streaming | Check wandb auth |
| Final iteration (~100 PPO iters at 500k env-steps budget) | Mean ep reward at last iter ≥ 2× mean ep reward at iter 0 | R5 mitigation |

## 9. Open questions — MUST be resolved as the first plan task before writing backend.py

- **Q1 (blocking):** What is `EnvBundle.env_step`'s exact signature expected by `onpolicy_collect`? (state, action) → ? Need to read `jax_rl/training/onpolicy_collect.py` for the iteration loop. Determines whether we return 5-tuple, dict, etc.
- **Q2 (blocking):** What's `jax_rl.scripts.train_ppo.train()`'s signature? Need to call it correctly from our wrapper. Read the signature + figure out which args we need to pass (cfg, seed, use_wandb=True/False, wandb_project=..., etc.).
- **Q3 (non-blocking):** Does jax-learning's `train_ppo.train()` import anything Playground-specific at module-load time (e.g. `mujoco_playground` registry queries)? If yes, may need a stub or uv-installed dep. Resolve empirically by running the wrapper script.
- **Q4 (blocking — D16):** Does `TrainConfig` expose an `eval_interval` (or `eval_every`, etc.) knob that can be set to `inf` / `0` to disable in-process eval? If not, escalate to D16 path (b) — patch `train_ppo.train` on a jax-learning branch. Read `jax_rl/configs/train_config.py`.
- **Q5 (blocking — D15):** Which preset in `jax_rl/configs/env_presets.py` is closest to a manipulation env with the right network sizes / lr / gamma? E.g. CheetahRun (locomotion, MLP), AnymalCFlat (locomotion), or there's a manipulation preset. Pick the most appropriate as MVP starting point; override env_name + num_envs.

## 10. Out-of-scope follow-ups

- **Full M5a per parent spec**: matched hyperparams + comparison doc. Open as separate sub-spec after MVP passes.
- **M5b FlashSAC**: parent spec's primary deliverable. Should work with the SAME `isaaclab_backend.py` since it's algo-agnostic — that's the win of this design. Sub-spec to be written after M5a MVP.
- **Upstream backend.py to jax-learning**: when MVP is stable, open a PR to jax-learning that moves `backend.py` from our project into `jax_rl/training/env_backends/isaaclab_backend.py`. Per D12.
- **Eval cadence + best-ckpt tracking + render_fn**: backend currently passes `eval_env=env` (same) and `render_fn=None`. Add proper eval env + render function in follow-up.

## 11. Glossary delta

- **EnvBundle**: jax-learning's backend-agnostic env handle, see `jax_rl/training/env_bundle.py`.
- **`IsaacLab/<task-id>`** env_name prefix: jax-learning's dispatch convention (`env_backends/__init__.py:43`).
