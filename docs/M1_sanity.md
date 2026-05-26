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

## Video / rendering — RESOLVED via driver downgrade

**Initial state (driver 595.71.05):** RTX renderer SIGSEGV in `librtx.scenedb.plugin.so` during `omni::usd::UsdManager::createHydraEngine`. Root cause = NVIDIA driver 595.x regression on Blackwell GPUs (RTX 5080). Isaac Sim 5.1's spec minimum driver is 580.65.06.

**Fix:** downgraded driver to `nvidia-driver-580-open` (apt-pulled `580.159.03`). Reboot. Verified via `nvidia-smi`.

**Post-fix verification:**
- Re-ran sanity script with `--record_video`. No crash.
- Video written: `videos/m1_nutthread_random-step-0.mp4` (528 KB).
- `ffprobe`: `Duration 00:00:13.27, h264 yuv420p 1280x720, 15 fps`. 200 steps / 15 fps ≈ 13.27 s — matches.
- 200 sim steps completed OK.

**Citations (investigation):**
- IsaacSim issue #537 — "fails with 595.79, works with 580"
- Isaac Sim 5.1 docs — minimum driver 580.65.06
- NVIDIA forum: RTX 5080 + Ubuntu 24.04 same `librtx.scenedb` segfault
- Multiple Blackwell GPUs (5060 Ti / 5070 Ti / 5080 / 5090) report identical crash on driver 595.x

**Residual risks (monitor):**
- Warp `cuDeviceGetUuid` bug on driver 580 — workaround `WARP_DISABLE_CUDA=0` if it appears (IsaacLab #3477).
- TiledCamera hang on Blackwell (IsaacLab #4951) — switch to `CameraCfg` instead of `TiledCameraCfg` if Factory tasks ever use tiled cams.

## Notes / warnings during run

- `[gpu.foundation.plugin] PCIe link width current (8) and maximum (16) for device 0 don't match.` — GPU running at PCIe 4.0 x8, not x16. Hardware/board config, harmless for training.
- `[gpu.foundation.plugin] IOMMU is enabled.` — may slightly increase memory copy overhead; not blocking.
- `[isaaclab.envs.direct_rl_env] Seed not set` — expected (script didn't pass a seed).
- `[direct_rl_env.py] WARNING: The render interval (1) is smaller than the decimation (8). Multiple render calls will happen for each environment step.` — only matters when rendering is active; benign in physics-only mode.
- `[omni.usd] Unexpected reference count of 2 for UsdStage ... while being closed` — known IsaacLab/USD shutdown warning, not a leak that affects training.
