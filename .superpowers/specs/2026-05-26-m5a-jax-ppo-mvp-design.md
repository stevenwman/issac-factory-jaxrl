# M5a — JAX PPO MVP on FactoryJax-NutThread-v0

**Status:** Draft
**Date:** 2026-05-26
**Owner:** robomechanicslab@gmail.com
**Parent spec:** `.superpowers/specs/2026-05-26-isaaclab-factory-jax-baseline-design.md`
**Sub-spec scope:** narrows M5a (originally "matched JAX PPO vs rl_games comparison") into a smaller MVP that **proves the bridge + jax-learning's PPO can train on Isaac at all**. Full matched-config comparison is deferred.

---

## 1. Goal

Stand up an end-to-end training script that drives jax-learning's `PPO` on the `FactoryJax-NutThread-v0` env via our `JaxEnvWrapper`, and demonstrate that the resulting setup **learns** — i.e. reward rises above the random-policy baseline over a short training run.

This is the **first time** the full pipeline is exercised: bridge + jax-learning's algo + asymmetric actor-critic on a real Isaac env. The MVP confirms the plumbing works before we invest in matched-config rigor (the original M5a) or FlashSAC training (M5b).

## 2. Success criterion (single, narrow, self-contained)

**Mean episode reward at the final iteration is at least 2× the mean episode reward at iteration 0**, where both numbers are measured by the script itself during this run and logged to wandb. Iteration-0 reward is computed by rolling out the freshly-initialized (random) policy for one full rollout (`num_steps × num_envs` env-steps) before any PPO update fires; the final reward is the rolling mean over the last 10 iterations' episodes. Both numbers are written to `runs/m5a_jax_ppo_smoke/summary.json` at end of training.

No external baseline lookup. No comparison number against rl_games. Self-contained pass/fail.

## 3. Non-goals

- Matched hyperparameter set with rl_games' `rl_games_ppo_cfg.yaml`. Use jax-learning sensible defaults.
- Wall-clock comparison vs rl_games (parent spec's M5a Table). Defer.
- Reaching M2's final reward number (809 @ ep 50). Defer.
- Obs normalization (`RunningMeanStd`). Skip.
- Eval cadence + best-checkpoint tracking. Skip — save final checkpoint only.
- LSTM / recurrent policy. jax-learning PPO doesn't have it; use MLP-only.
- Video recording during training. Defer to a separate play step.
- Multi-seed / statistical comparison.

## 4. Locked decisions

| # | Decision | Source / grounding |
|---|---|---|
| D1 | MVP scope, not full matched-config comparison | User decision in M5a brainstorm |
| D2 | Network: MLP [256, 128] elu actor + critic | jax-learning defaults; matches Factory rl_games MLP shape modulo the LSTM layer they prepend (which jax-learning lacks) |
| D3 | Hyperparams: take from `jax_rl.configs.ppo_config.PPOConfig` defaults; override only `num_envs` and `num_steps` from CLI | Avoid debating each knob in MVP |
| D4 | Asymmetric actor-critic: actor on `obs["policy"]` (19-dim), critic on `obs["critic"]` (43-dim) | Factory env structure confirmed in M1 |
| D5 | Truncation handling: treat `terminated | truncated` as single `done` flag in GAE | Isaac doesn't separately signal truncation by default; differentiating would require env-cfg work out of scope for MVP |
| D6 | No obs normalization | Simplifies bringup. Acceptable for MVP demonstration. Add later if reward stagnates. |
| D7 | No periodic eval, no best-ckpt tracking | Wandb reward curve = the verification artifact. Final ckpt saved as `runs/m5a_jax_ppo_smoke/final.ckpt`. |
| D8 | Wandb on. Project `isaaclab-factory-jax`, entity `sman2`, run name `m5a-jax-ppo-smoke-N` where N is a counter. `wandb.define_metric("*", step_metric="env_step")` so env_step is the x-axis everywhere. | User preference from M2 wandb discussion |
| D9 | Inline mini-bundle (env, env_step closure, obs_dim, action_dim, critic_obs_dim, num_envs) — do NOT use `jax_rl.training.make_env_bundle` | `make_env_bundle` is Playground/MJX/gym-specific and would need adapter work. Defer. |
| D10 | jax-learning consumed via `sys.path.insert(...)` (path-import, per parent spec); NOT installed editable | Per parent spec's M3.5 SKIP decision |
| D11 | If jax-learning's PPO internals break under jax 0.5.3 (parent spec records the version downgrade from 0.9+), patch on a `jax-learning` topic branch `isaaclab-factory-jax/m5a-jax05-compat` and log in `docs/jax_learning_divergences.md` | Per parent spec D11 |
| D12 | Single training run, single seed (seed=0). No multi-seed sweep. | MVP discipline |
| D13 | num_envs: 64 (parent had 128; drop for MVP to leave GPU headroom for jax memory alongside Isaac's PhysX) | Parent spec Risk #4 + M2 actually-observed memory |
| D14 | Total env-step budget: 500k (≈ 100-200 PPO iterations at num_envs=64, num_steps=128) — small enough to fit in ~30 min, large enough to see reward rise on a learnable env | NutThread learns within ~50 rl_games epochs ≈ 800k env-steps; halve for MVP |

## 5. Architecture

### Files touched

```
Research/isaaclab-factory-jax/
├── scripts/
│   └── train_jax_ppo.py          ← CREATE (new)
├── source/factory_jax/factory_jax/
│   ├── tasks.py                   (existing, unchanged)
│   └── bridge/                    (existing, unchanged)
└── docs/
    └── jax_learning_divergences.md  ← CREATE on first divergence (lazy)
```

### Script structure

```python
# 1. AppLauncher boot (must come first)
# 2. sys.path insert for jax-learning + import factory_jax.tasks
# 3. gym.make + JaxEnvWrapper
# 4. Construct PPO(config, obs_dim=19, action_dim=6, ..., critic_obs_dim=43)
# 5. wandb.init + define_metric(env_step)
# 6. Training loop:
#    for iter in range(num_iterations):
#        rollout: collect num_steps × num_envs transitions into a RolloutBatch
#        ppo.update(state, batch, key, next_obs, critic_obs, critic_next_obs)
#        wandb.log({reward, losses, perf}, step=env_step)
# 7. Save final ckpt + close env
```

### Three "units" inside the script

- **mini-bundle** (lines ~50): inline closure that mimics what `make_env_bundle` would have returned. One function, ~30 lines.
- **rollout collector** (~40 lines): vectorized step loop that fills a `RolloutBatch` with policy obs, critic obs, actions, log-probs, rewards, dones. Asymmetric — stores both obs streams in parallel arrays.
- **train loop** (~50 lines): instantiate PPO, build optimizers, run iterations, log to wandb, save final ckpt.

If any of these grows past its budget, split into a helper file. For MVP, keep one file.

## 6. Data flow per iteration

```
                ┌──────────────────────┐
                │  jax PRNG key         │
                └───────────┬──────────┘
                            ▼
              ppo.select_action(state, policy_obs, key, critic_obs)
                            │  → (action, log_prob, value)
                            ▼
              JaxEnvWrapper.step(action)   ── DLPack ── Isaac.step(torch_action)
                            │  → ({"policy": obs_p, "critic": obs_c}, reward, term, trunc, info)
                            ▼
              accumulate into rollout buffer for num_steps
                            │
                            ▼
              compute (or let PPO compute internally) GAE advantages
                            ▼
              ppo.update(state, batch, key, next_obs, critic_obs, critic_next_obs)
                            │  → (new_state, train_info)
                            ▼
              wandb.log({reward/mean, losses/*, perf/*}, step=env_step)
```

env_step counter ticks `num_envs * num_steps` per iteration.

## 7. Risks + mitigations

| # | Risk | Mitigation / fallback |
|---|---|---|
| R1 | jax 0.5 vs 0.9 API gap inside jax-learning's PPO breaks at runtime | First fix attempt: patch the call site on a topic branch in jax-learning (per D11). If breakage is too pervasive, escalate: bump our `pyproject.toml` numpy override to `numpy>=2` and try `jax[cuda12]>=0.9` again. |
| R2 | Reward doesn't rise — flat training | Investigate in order: (a) inspect rollout obs/reward distributions, (b) enable obs normalization, (c) lower lr, (d) inspect action distribution, (e) sanity-check that PPO sees the right shapes. If still flat, M5a fails; escalate. |
| R3 | Isaac OOM on shared GPU when jax also runs there | num_envs=64 by default. Drop to 32 if needed. Document. |
| R4 | DLPack contiguous-guard catches an obs we didn't expect to be non-contiguous (Factory env may return strided slices) | `to_jax` already calls `.contiguous()`. If we see perf issues, profile + investigate. |
| R5 | Wall-clock per iter dominated by Isaac, not algo | Acceptable — MVP doesn't try to beat rl_games on speed. Just verify learning. |
| R6 | jax-learning's PPO expects a `next_obs` shape we don't readily have at iteration boundary | The rollout collector keeps the last obs from the iteration as `next_obs`. Standard PPO pattern. |
| R7 | Asymmetric obs shape mismatch — critic obs not propagated through wrapper | Bridge already preserves dict; wrapper test (`test_wrapper_reset_returns_jax_dict`) verifies. Re-verify by printing shapes in the smoke run. |

## 8. Verification checkpoints

| When | Pass condition | If fail |
|---|---|---|
| Script imports + AppLauncher boots + env loads | No exception, FactoryJax-NutThread-v0 registered | Investigate import order / EULA env var |
| First rollout collected | `RolloutBatch.obs.shape == (num_steps, num_envs, 19)`, no NaN, reward field populated | Print shapes, dtype check, ensure JAX arrays not Python floats |
| First `ppo.update` returns | TrainingState updated, info dict non-empty, no NaN losses | This is most likely R1 failure mode. Patch via D11. |
| 5 iterations complete | wandb dashboard live, reward + losses logged with env_step axis | Check wandb auth + `define_metric` calls |
| Final iteration (~100 PPO iters) | Mean episode reward > 2× random baseline | This is the deliverable. If fail, R2 mitigation. |

## 9. Open questions — **MUST be resolved as the FIRST plan task** before any rollout-collector code is written. These determine the rollout collector's shape and interface.

- **Q1 (blocking — shapes rollout collector):** Does `jax_rl.algos.ppo.PPO.update` compute GAE internally, or does the caller need to provide `advantages` in the batch? If internal: collector stores raw `rewards`, `dones`, `values`. If external: collector must also compute bootstrapped advantages and pass them in. Resolve by reading `jax_rl/algos/ppo.py` line 273 onward + `jax_rl/buffers/rollout_buffer.py`.
- **Q2 (blocking — shapes rollout collector):** Does jax-learning's `RolloutBuffer` support asymmetric `critic_obs` storage? If yes: use it. If no: hand-roll a parallel `critic_obs` array (shape `(num_steps, num_envs, 43)`) outside the buffer and pass to `ppo.update(critic_obs=...)`. Resolve by reading `jax_rl/buffers/rollout_buffer.py` + `RolloutBatch` dataclass.
- **Q3 (non-blocking — affects reset semantics):** Isaac may auto-reset dones via the `DirectRLEnv` mechanism (Factory probably does) or require an explicit reset call. Verify by reading `factory_env.py` or empirically. Most likely auto-reset is on and we never call `env.reset()` mid-training. Has no impact on code structure if assumed auto-reset.

## 10. Out-of-scope follow-ups

- **Full M5a per parent spec**: matched hyperparameters, equal env-step budget, wall-clock & sample-efficiency table comparing rl_games vs jax-learning PPO. Open as a separate sub-spec **after** MVP passes.
- **M5b FlashSAC**: parent spec's primary deliverable. Sub-spec to be written after M5a MVP.
- **Obs normalization** (`RunningMeanStd`): add if MVP reward plateaus too early.

## 11. Glossary delta from parent spec

- **MVP** here means "minimum proof that the pipeline trains end-to-end," NOT "minimum viable product to ship." Internal-only term for this sub-spec.
- **Mini-bundle** = inline-constructed dict/closure that takes the place of `jax_rl.training.make_env_bundle` for our Isaac env.
