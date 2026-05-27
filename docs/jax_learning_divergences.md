# jax-learning divergences (per parent spec D11)

All changes live on branch `isaaclab-factory-jax/lazy-mjx-import` in `/home/stevenman/Desktop/Work/Research/jax-learning/`. `main` untouched.

| Date | File | Change | Reason | Recommendation |
|---|---|---|---|---|
| 2026-05-26 | `jax_rl/training/env_backends/__init__.py` | Wrap `mjx_backend` + `gym_backend` side-effect imports in try/except + `warnings.warn` on failure | warp-lang and mujoco_warp version conflict (warp 1.13 vs `wp.array2d[int]` syntax in mujoco 3.8). IsaacLab pins warp-lang<1.14, mujoco>=3.5 → no resolution path keeps both happy + lets mjx_backend import. | Upstream PR after M5a passes; pattern is reusable for any backend with optional heavy deps. |
| 2026-05-26 | `jax_rl/training/env_setup.py` | Wrap re-export `from .mjx_backend import make_envs, _make_nan_safe_step` in try/except. Set `make_envs=None` and `_make_nan_safe_step = lambda fn: fn` (passthrough) on failure. | Same root cause. Passthrough preserves the call-site `_make_nan_safe_step(env.step)` in `jax_rl/utils/eval.py:66` for non-MJX backends. | Upstream alongside `__init__.py` patch. |
| 2026-05-26 | `scripts/train_ppo.py` | Skip end-of-training `_eval_fn(...)` call when `bundle.backend_kind == "isaaclab"`. Set `eval_metrics = {"eval_mean": nan}` instead. | jax-learning's `_eval_fn` falls through to Brax `evaluate` for non-`gym` backends. Brax eval JITs `env.step` and calls `env.reset(jax.random.split(...))` (batched keys) — incompatible with Isaac's gym-style env. Spec D16. | Upstream after we add a proper isaaclab-aware eval path. |

## Branch status

Branch: `isaaclab-factory-jax/lazy-mjx-import` in jax-learning repo.
Commits ahead of `main`:
- `584017a` lazy mjx/gym backend imports (try/except)
- `16c4dad` also lazy mjx import in env_setup re-export (same branch)
- `a35b1b2` passthrough `_make_nan_safe_step` stub + skip final-eval for isaaclab backend

## Related unfixed jax-learning bugs

Surfaced during M5a but NOT patched (handled differently):

- **`metrics_logger.py` references `wandb.run` / `wandb.init` without guarding `import wandb`** — if wandb isn't installed in the consuming env, attribute lookups fail at attempted-call sites rather than as a clean `ImportError`. Workaround: `wandb` is now a hard dep in our project's `pyproject.toml` even when training without cloud sync (use `WANDB_MODE=offline`).
