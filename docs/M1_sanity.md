# M1 — Sanity check (NutThread + random actions)

**Date:** 2026-05-26
**IsaacLab version:** 2.3.2 (from `IsaacLab/VERSION`)
**Isaac Sim version:** 5.1 (kit log path: `~/isaacsim/kit/data/Kit/Isaac-Sim/5.1/`)
**Command:**
```bash
./isaaclab.sh -p /tmp/m1_sanity.py
# (one-off script wrapping AppLauncher + gym.make + 200 random-action steps; not committed)
```

## Observed (physics-only run, no camera)

- **Exit status:** 0 (clean shutdown after 200 steps, ~37 s wall)
- **Task:** `Isaac-Factory-NutThread-Direct-v0`
- **num_envs used:** 4
- **observation_space:** `Box(-inf, inf, (4, 19), float32)` — vectorized
- **action_space:** `Box(-inf, inf, (4, 6), float32)` — vectorized
- **obs dict keys at first reset:** `["policy", "critic"]` (asymmetric actor-critic, matches spec §6 + Risk #3)
- **obs key shapes:**
  - `policy`: `(4, 19)`, `torch.float32`, `cuda:0` — per-env dim **19**
  - `critic`: `(4, 43)`, `torch.float32`, `cuda:0` — per-env dim **43**
- **Device:** `cuda:0`
- **GPU:** NVIDIA GeForce RTX 5080 (Blackwell)
- **Driver:** 595.71.05 (nvidia-smi reports CUDA 13.2)
- **Kernel:** `6.17.0-29-generic`

## Notable spec correction

Spec (§ multiple) and plan referenced `single_observation_space=(21,)` based on an older grep of `factory_env_cfg.py`. Actual current value is **19** (policy obs), **43** (critic obs). Update spec / plan when next touched. Functionally harmless — wrapper is shape-agnostic.

## Video / rendering — BLOCKED with RTX 5080 + Isaac Sim 5.1

Attempting to enable cameras (`AppLauncher.enable_cameras=True` + `gym.make(..., render_mode="rgb_array")` + `gym.wrappers.RecordVideo`) crashes the kit subprocess with a SIGSEGV in `librtx.scenedb.plugin.so` during `omni::usd::UsdManager::createHydraEngine`. Stack trace logged to `~/isaacsim/kit/data/Kit/Isaac-Sim/5.1/<uuid>.dmp` and to `/tmp/m1_run.log`.

Likely root cause: Isaac Sim 5.1's bundled RTX raytracer plugin not yet validated on Blackwell-generation GPUs (RTX 5080 architecture is newer than Isaac Sim 5.1's release). Driver 595.71.05 supports CUDA 13.2, but the renderer crashes regardless of CUDA path.

**Workaround paths (decide before M5b at the latest):**

| Path | Notes |
|---|---|
| Downgrade driver to a known-good Isaac Sim 5.1 version (e.g. 555.x) | Risk: may break other GPU-using code on this machine |
| Upgrade Isaac Sim to 5.2 / latest (if a Blackwell fix has shipped) | Requires verifying IsaacLab 2.3.2 compatibility |
| Use a non-RTX renderer (Storm / OpenGL hydra) for video frames | Worse quality, but enough for a sanity video |
| Run video capture on a different machine (different GPU) | Most flexible, slowest workflow |
| Skip video entirely for M1/M5b and just rely on reward curves + scalar metrics | Loses visual proof; not ideal for "see the policy threading" eval |

**Impact on milestones:**
- M1.1 video: deferred
- M2: rl_games training itself does NOT need cameras (training runs headless without RTX renderer); only the `--video` flag for periodic clip dumps would be affected. Safe to proceed.
- M5a: same as M2.
- M5b eval video: blocked until renderer is fixed.

## Notes / warnings during run

- `[gpu.foundation.plugin] PCIe link width current (8) and maximum (16) for device 0 don't match.` — GPU running at PCIe 4.0 x8, not x16. Hardware/board config, harmless for training.
- `[gpu.foundation.plugin] IOMMU is enabled.` — may slightly increase memory copy overhead; not blocking.
- `[isaaclab.envs.direct_rl_env] Seed not set` — expected (script didn't pass a seed).
- `[direct_rl_env.py] WARNING: The render interval (1) is smaller than the decimation (8). Multiple render calls will happen for each environment step.` — only matters when rendering is active; benign in physics-only mode.
- `[omni.usd] Unexpected reference count of 2 for UsdStage ... while being closed` — known IsaacLab/USD shutdown warning, not a leak that affects training.
