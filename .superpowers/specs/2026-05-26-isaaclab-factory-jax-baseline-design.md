# IsaacLab Factory + JAX: Bolt-Tightening RL Baseline

**Status:** Draft
**Date:** 2026-05-26
**Owner:** robomechanicslab@gmail.com
**Project location:** `Research/isaaclab-factory-jax/` (new external project, new git repo)

---

## 1. Goal

Stand up a trained RL policy for **bolt tightening** in IsaacLab's Factory simulation, executed by a JAX algorithm (FlashSAC from `jax-learning`) reading observations and emitting actions through a thin GPU bridge to Isaac's torch-based environment. Establish a workflow where the JAX algorithms in `jax-learning` are reusable across future Isaac tasks with minimal plumbing.

"Bolt tightening" is interpreted as Isaac's stock `Isaac-Factory-NutThread-Direct-v0` task — a Franka threading a nut onto a fixed bolt, which is the closest match in the Factory paper / IsaacLab task suite. Custom bolt-tighten task design (torque thresholds, multi-turn success, etc.) is **out of scope** for this spec; revisit after the baseline trains.

## 2. Success criteria

This spec is done when all of the following hold:

1. `Research/isaaclab-factory-jax/` exists as a self-contained external IsaacLab project, scaffolded via Isaac's `./isaaclab.sh --new` generator.
2. NutThread loads, accepts random actions, and a recorded video shows the Franka moving inside the scene. (M1)
3. `rl_games` PPO trains NutThread to a reward curve that visibly rises above the random baseline, with the checkpoint and final reward saved as a **reference number**. (M2)
4. A JAX↔Isaac GPU bridge exists and passes a roundtrip + parity test against an Isaac-only rollout under the same seed. (M4)
5. `jax-learning`'s **PPO** trains NutThread through the bridge with hyperparameters configured to match the rl_games config from M2 wherever the two algos share a knob. The M5a doc reports a complete side-by-side comparison: final reward at equal env-step budget, final reward at equal wall-clock budget, env-steps-to-reward curve, wall-clock-to-reward curve, plus per-phase throughput (rollout steps/sec, update steps/sec). (M5a)
6. `jax-learning`'s **FlashSAC** trains NutThread through the same bridge; checkpoint saved, eval video rendered. This is the deliverable "RL policy trained baseline of the bolt-tightening task." (M5b)
7. `docs/robot_swap.md` documents the procedure for swapping Franka → UR5 in this project's task config, without yet executing it. (M6)

## 3. Non-goals

- A custom `BoltTighten` task with bolt USD, torque sensing, multi-turn success criterion. (Future spec.)
- Implementing the UR5 swap in code. (Documented only.)
- Migration to IsaacLab 3.0 / Newton physics. (Future spec; see §10.)
- Touching `jax-learning`'s **`main` branch** — it is consumed as an editable dependency. Per D11, project-specific algo changes go on a `jax-learning` topic branch (or are ported into this project case-by-case). `main` is off-limits from this project.
- Custom reward shaping. Use Isaac's stock NutThread reward for the baseline; revisit later.

## 4. Locked decisions and grounding

| # | Decision | Source / grounding |
|---|---|---|
| D1 | External project scaffolded by Isaac's `./isaaclab.sh --new` generator | IsaacLab docs `overview/own-project/template.rst` (lines 13–22): "External project (recommended): An isolated project that is not part of the Isaac Lab repository… ensuring that your development efforts remain self-contained." |
| D2 | Task target = `Isaac-Factory-NutThread-Direct-v0` (stock) | Factory paper; IsaacLab `source/isaaclab_tasks/.../direct/factory/__init__.py` registers exactly three Factory IDs and NutThread is the screwing task. |
| D3 | Sequence: rl_games PPO baseline → JAX bridge → JAX PPO → FlashSAC | User's pick + Factory's `agents/` directory ships **only** `rl_games_ppo_cfg.yaml` — no sb3, rsl_rl, or skrl configs exist for Factory. All three Factory tasks register **only** `rl_games_cfg_entry_point`. rl_games is the de-facto reference because it is the only one NVIDIA published. |
| D4 | UR5 swap = docs only this spec | User's pick. |
| D5 | Stay on **IsaacLab 2.3.2** (PhysX + torch), not 3.0 beta-2 | Newton beta-2 docs (`experimental-features/newton-physics-integration/index.html`): "only a limited set of classic RL and flat terrain locomotion reinforcement learning examples are included." Factory tasks not yet ported. Choosing the beta would mean porting a contact-rich manipulation env onto a beta engine while simultaneously building the JAX bridge — two unknowns at once. |
| D6 | Bridge = DLPack zero-copy GPU, **source-tensor-type abstracted behind a single `to_jax(x)` function** | Both `torch.Tensor` and `warp.array` expose DLPack / `__cuda_array_interface__`. Migrating to Newton/Warp later = swap one function's implementation. |
| D7 | Primary JAX algorithm = **FlashSAC** from `jax-learning/jax_rl/algos/flash_sac.py` | User's pick. SAC variant is more sample-efficient than PPO for contact-rich manipulation. |
| D8 | Also train **JAX PPO** for direct comparison with rl_games PPO, **hyperparameter-matched** | User's pick. Lets us measure whether differences are algorithmic (PPO ≠ SAC) or implementation (rl_games PPO vs jax-learning PPO). |
| D9 | Spec path = `Research/isaaclab-factory-jax/.superpowers/specs/` | Matches `.superpowers/` convention from `jax-learning/CLAUDE.md`. |
| D10 | **Dedicated uv venv** at `Research/isaaclab-factory-jax/.venv/` with Python **3.11**. Use `uv add` (records deps in `pyproject.toml` + lockfile) with named indices declared in `[tool.uv.index]`. Install order from NVIDIA's quickstart: `torch==2.7.0` + `torchvision==0.22.0` (cu128 wheels) **first**, then `isaacsim[all,extscache]==5.1.0` from `pypi.nvidia.com`, then `isaaclab*` editable from the existing `IsaacLab/source/`, then `jax-learning` editable, then our project editable. Pin **`jax[cuda12]`** via `[tool.uv] override-dependencies` to replace jax-learning's `jax[cuda13]`. | Mirrors NVIDIA's quickstart (`docs/source/setup/quickstart.html`). `uv add` over `uv pip install` because adds are recorded in `pyproject.toml` and lockfile — reproducible without separate `requirements.txt`. Order matters: torch first prevents isaacsim's resolver from pulling a different torch wheel. cu128 ≡ CUDA 12.8 = the Isaac Sim 5.1.0 target. jax[cuda12] is wheel-compatible with cu128 drivers. Project-local venv leaves your existing `IsaacLab/env_isaaclab` and jax-learning venvs untouched. |
| D11 | **jax-learning algo modification policy: branch by default, port case-by-case, decide migration later.** If a jax-learning algo needs changes to work with the Isaac env (e.g. obs-dict handling, vectorization quirks, FlashSAC tuning), default action is to **create a new branch in the `jax-learning` repo** (e.g. `isaaclab-factory-jax/<topic>`) and make the change there. Editable install picks it up live — no reinstall needed. **Do not modify `main` or any shared branch in jax-learning.** Alternative: **port the file** into `source/factory_jax/factory_jax/algos/<copied_algo>.py` and modify in-place — choose this only when the change is so divergent it would clutter jax-learning's history (e.g. a Factory-specific actor architecture). Migration question (merge branch back into jax-learning vs keep ported copy) is **deferred until after M5b baseline trains**. | Protects the user's jax-learning `main` branch from project-specific changes. Editable install means a branched jax-learning is consumed transparently. Per-change decision (branch vs port) prevents premature commitment to either. |

## 5. Architecture

```
Research/
├── jax-learning/                        (existing, editable dep, untouched by this project)
└── isaaclab-factory-jax/                (NEW)
    ├── .git/                            (own repo)
    ├── pyproject.toml                   (project metadata + deps)
    ├── README.md
    ├── docs/
    │   └── robot_swap.md                (M6)
    ├── scripts/
    │   ├── sanity_check.py              (M1, optional — Isaac's stock random_agent works too)
    │   ├── train_rl_games.py            (M2, thin wrapper around Isaac's train.py)
    │   ├── train_jax_ppo.py             (M5a)
    │   ├── train_jax_flashsac.py        (M5b)
    │   └── play.py                      (eval + record)
    ├── source/factory_jax/
    │   ├── pyproject.toml               (extension package)
    │   ├── setup.py
    │   ├── config/extension.toml        (Isaac Sim extension metadata)
    │   └── factory_jax/
    │       ├── __init__.py
    │       ├── bridge/
    │       │   ├── __init__.py
    │       │   ├── tensor_convert.py    (to_jax / from_jax — DLPack, source-agnostic)
    │       │   └── jax_env_wrapper.py   (gym.Wrapper exposing jax.Array I/O)
    │       └── tasks/
    │           └── direct/factory_nut_thread/
    │               ├── __init__.py      (gym.register: "FactoryJax-NutThread-v0")
    │               └── env_cfg.py       (subclass of Isaac's FactoryTaskNutThreadCfg)
    └── .superpowers/
        ├── specs/2026-05-26-isaaclab-factory-jax-baseline-design.md   (this file)
        └── plans/                       (filled by writing-plans skill)
```

### Three units, each one responsibility

- **`bridge/`** — pure plumbing. Knows nothing about Factory. Converts between any GPU array source (torch today, warp tomorrow) and `jax.Array` via DLPack. Owns one gym wrapper that swaps obs/action dtypes at the env interface.
- **`tasks/`** — registration + task-config customization point. A thin subclass of Isaac's NutThread config so we own the gym ID (`FactoryJax-NutThread-v0`) and have a single place to override (e.g. robot articulation, action gains) without forking Isaac's `factory_env.py`.
- **`scripts/`** — entrypoints. Each script is small: parse args, build env, build algo, run training/eval. Sharing is by extracting helpers into `factory_jax/training/` only when a real duplication appears, not preemptively.

### Why a subclass rather than a copy of `factory_env.py`

Forking Isaac's env means we drift from upstream every time NVIDIA fixes a Factory bug. Subclassing `FactoryTaskNutThreadCfg` and re-registering the gym ID gives us a stable seam to customize (robot, gains, reward terms) while still inheriting upstream fixes for free. If we later need physics-level changes, we can override specific methods on `FactoryEnv` itself.

## 6. Bridge data flow (the hot path)

Per env step:

```
                       ┌─────────────────────────┐
                       │   JAX policy / buffer   │
                       └────────────┬────────────┘
                                    │ jax.Array action (num_envs, 6)
                                    ▼
                  JaxEnvWrapper.step(jax_action)
                                    │ from_jax(jax_action) → torch.Tensor (DLPack, zero-copy)
                                    ▼
              Isaac FactoryEnv.step(torch_action)
                                    │ returns (obs_dict, reward, done, trunc, info), all torch on cuda:0
                                    │ obs_dict = {"policy": tensor, "critic": tensor}
                                    ▼
                  JaxEnvWrapper post-step
                                    │ to_jax(obs_dict["policy"]) / to_jax(obs_dict["critic"])
                                    │ to_jax(reward) / to_jax(done) (DLPack)
                                    ▼
                  back to JAX policy / buffer
```

`tensor_convert.py` exposes exactly two functions:

```python
def to_jax(x) -> jax.Array:        # accepts torch.Tensor or warp.array; dispatches on type
def from_jax(x: jax.Array, target: Literal["torch"] | type = "torch"):
```

Stream-ordering insurance: one `torch.cuda.synchronize()` per step in M4. Profile in M5b and drop if the kernel timeline shows the bridge isn't the bottleneck.

## 7. Milestones

Each milestone has: one concrete goal, named new code (or "none"), and **explicit verification commands**.

### M1 — Sanity: NutThread loads, random actions, video saved

- **Goal:** prove install, GPU, Factory assets all work end-to-end.
- **New code:** none.
- **Command:**
  ```bash
  cd Research/jax-learning  # or wherever IsaacLab is callable from
  ./IsaacLab/isaaclab.sh -p IsaacLab/scripts/environments/random_agent.py \
      --task Isaac-Factory-NutThread-Direct-v0 \
      --num_envs 4 --video --video_length 200 --enable_cameras --headless
  ```
- **Verification:**
  - Command exits 0.
  - `videos/*.mp4` exists, plays, shows Franka thrashing in scene.
  - Console reports `single_observation_space=(21,)` and `single_action_space=(6,)`.
- **Failure modes to watch:** CUDA OOM (drop `--num_envs`), missing assets (re-run Isaac asset download), GPU driver mismatch.

### M2 — rl_games PPO reference baseline

- **Goal:** establish a known-good reward curve and final number on stock NutThread.
- **New code:** none (this milestone uses Isaac's stock training script).
- **Command:**
  ```bash
  ./IsaacLab/isaaclab.sh -p IsaacLab/scripts/reinforcement_learning/rl_games/train.py \
      --task Isaac-Factory-NutThread-Direct-v0 \
      --num_envs 128 --headless --max_iterations <N>
  ```
- **Verification:**
  - Training reward (rl_games tensorboard) monotonically trends up over the first ~1M env steps.
  - **`docs/M2_baseline.md` records, at minimum:**
    - Final mean episode reward.
    - Success rate over 20 eval episodes from the saved checkpoint.
    - **Wall-clock metrics** (these feed M5a's comparison table):
      - Env steps/sec measured during the rollout phase only.
      - Update steps/sec (or seconds per minibatch update) measured during the gradient phase only.
      - Total wall-clock per training iteration (rollout + update).
      - Total wall-clock to reach 80% of final reward (this becomes R* in M5a).
    - The full `rl_games_ppo_cfg.yaml` hyperparameters verbatim — these become the **target config** for M5a.
- **Decision gate before M3:** if M2 fails (reward flat, training crashes), debug here before any custom code is written. JAX bridge cannot fix a broken env.

### M3 — External project scaffold

- **Goal:** generate the new project, install it, re-register NutThread under our gym ID, reproduce M1 video through it.
- **New code:**
  - Generated project skeleton via `./isaaclab.sh --new` → choose external, direct workflow, RL libraries: `rl_games` + (none for JAX yet, we'll add manually).
  - `source/factory_jax/factory_jax/tasks/direct/factory_nut_thread/__init__.py` — `gym.register` with `id="FactoryJax-NutThread-v0"`, pointing at `factory_env_cfg:FactoryJaxNutThreadCfg`.
  - `env_cfg.py` — `class FactoryJaxNutThreadCfg(FactoryTaskNutThreadCfg): pass` to start. Customization seam, not yet customized.
- **Verification:**
  - `python scripts/list_envs.py` (Isaac scaffolds this) lists `FactoryJax-NutThread-v0`.
  - `random_agent.py --task FactoryJax-NutThread-v0 --video` produces a video identical-in-behavior to M1.
- **Environment & install (uv-native — `uv add` records each dep in `pyproject.toml` + lockfile automatically; order from NVIDIA quickstart):**
  1. **Create dedicated venv** (Python 3.11 matches `env_isaaclab`):
     ```bash
     cd Research/isaaclab-factory-jax
     uv init --no-package --python 3.11      # creates pyproject.toml if --new didn't already
     uv venv --python 3.11
     source .venv/bin/activate
     ```
  2. **Declare named indices once** in `pyproject.toml` so subsequent `uv add` calls know where to look:
     ```toml
     [[tool.uv.index]]
     name = "pytorch-cu128"
     url = "https://download.pytorch.org/whl/cu128"
     explicit = true

     [[tool.uv.index]]
     name = "nvidia"
     url = "https://pypi.nvidia.com"
     explicit = true

     [tool.uv.sources]
     torch        = { index = "pytorch-cu128" }
     torchvision  = { index = "pytorch-cu128" }
     isaacsim     = { index = "nvidia" }
     ```
     (`explicit = true` keeps these indices opt-in per package, so they don't pollute resolution for unrelated deps.)
  3. **Add PyTorch (cu128) first** — NVIDIA's quickstart order; ensures isaacsim's resolver doesn't pull a different torch wheel:
     ```bash
     uv add "torch==2.7.0" "torchvision==0.22.0"
     ```
  4. **Add Isaac Sim 5.1.0** (the version paired with IsaacLab 2.3.2 in NVIDIA's quickstart):
     ```bash
     uv add "isaacsim[all,extscache]==5.1.0"
     ```
  5. **Add IsaacLab packages editable from the existing clone:**
     ```bash
     for pkg in isaaclab isaaclab_assets isaaclab_rl isaaclab_tasks isaaclab_mimic; do
       uv add --editable /home/stevenman/Desktop/Work/IsaacLab/source/$pkg
     done
     ```
     (editable so upstream factory fixes flow through when IsaacLab is updated)
  6. **Add jax-learning editable:**
     ```bash
     uv add --editable /home/stevenman/Desktop/Work/Research/jax-learning
     ```
  7. **Add our project's extension editable** (after `./isaaclab.sh --new` has scaffolded `source/factory_jax/`):
     ```bash
     uv add --editable source/factory_jax
     ```
  8. **Pin `jax[cuda12]`** to override jax-learning's `jax[cuda13]` — this is a `pyproject.toml`-level setting, not a CLI flag:
     ```toml
     [tool.uv]
     override-dependencies = ["jax[cuda12]==<pinned-version>"]
     ```
     Then re-sync: `uv sync`. cu128 (CUDA 12.8) drivers and `jax[cuda12]` wheels are compatible — CUDA 12.x is forward-compatible across patch versions within the 12 series.
- **Verification of the env (precondition for any other M3 verification):**
  - `python -c "import isaacsim; import isaaclab; import jax_rl; import jax; print(jax.devices())"` succeeds and reports a CUDA device.
  - `python -c "import torch, jax; print(torch.version.cuda, jax.devices()[0].device_kind)"` — both report CUDA 12.x.
  - `python -c "import jax, torch; from torch.utils.dlpack import to_dlpack; x = torch.randn(4, device='cuda'); a = jax.dlpack.from_dlpack(to_dlpack(x)); print(a.devices())"` — DLPack bridge works end-to-end (this also de-risks M4 early).
- **Scaffold steps after env works:**
  - Run `./IsaacLab/isaaclab.sh --new` from a separate scratch dir, then merge the generated files into `Research/isaaclab-factory-jax/` so this spec's `.superpowers/` directory survives.
  - Add `source/factory_jax/factory_jax/tasks/direct/factory_nut_thread/__init__.py` calling `gym.register(id="FactoryJax-NutThread-v0", ...)` pointing at `FactoryJaxNutThreadCfg`.
  - Add `env_cfg.py` — `class FactoryJaxNutThreadCfg(FactoryTaskNutThreadCfg): pass` (customization seam, not yet customized).

### M4 — JAX↔Isaac bridge

- **Goal:** working zero-copy GPU bridge, parity-tested against Isaac-only rollouts.
- **New code:**
  - `bridge/tensor_convert.py` — `to_jax(x)` dispatches on type: `torch.Tensor` → `jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))`. Same shape for `warp.array` (single-line addition, future). `from_jax(arr, target="torch")` is the inverse.
  - `bridge/jax_env_wrapper.py` — `gym.Wrapper` that overrides `reset()` and `step()` to call `to_jax` on outputs and `from_jax` on inputs. Preserves vectorized layout `(num_envs, dim)`.
- **Verification:**
  - **Unit test:** for representative dtypes/shapes, `to_jax(from_jax(x)) == x` and the underlying CUDA pointer is identical (DLPack should be zero-copy).
  - **Parity test:** load Isaac NutThread directly, fix seed, run a deterministic action sequence for 200 steps, log `(obs, reward, done)`. Repeat through `JaxEnvWrapper` with the same seed and action sequence, log the same. Assert `max(|obs_a - obs_b|) < 1e-5`, same `done` flags, same rewards.
  - **Bridge overhead measurement (data, not a threshold):** at `num_envs=128`, record (a) raw Isaac `env.step(action)` ms/step and (b) `JaxEnvWrapper.step(jax_action)` ms/step. Compute `overhead_ms = wrapper_ms - raw_ms` and `overhead_pct = overhead_ms / raw_ms`. Save to `docs/M4_bridge_overhead.md`. This becomes the **denominator** for M5a's wall-clock comparison: if JAX PPO is faster than rl_games PPO end-to-end *despite* paying this per-step cost, that's the JIT win quantified.

### M5a — JAX PPO, config-matched to rl_games PPO

- **Goal:** measure jax-learning's PPO against rl_games' PPO with hyperparameters held equal. This isolates the algo implementation as a variable, separately from algo choice (PPO vs SAC).
- **New code:**
  - `scripts/train_jax_ppo.py` — builds `FactoryJax-NutThread-v0` env, wraps it in `JaxEnvWrapper`, instantiates `jax_rl.algos.ppo.PPO`, runs training loop.
  - A `configs/jax_ppo_matched.yaml` (or python file) holding the hyperparameter set extracted from the M2 rl_games config: learning rate, batch size, minibatch size, rollout length, num env, num epochs, GAE λ, γ, PPO clip ε, value clip, entropy coef, value coef, network MLP shape, activation.
- **Verification:**
  - Configs match: a one-line diff between the matched yaml and `rl_games_ppo_cfg.yaml` documents any unavoidable differences (e.g. param names that don't translate; explicit note for each).
  - Training reward curve rises; final number logged.
  - **`docs/M5a_jax_vs_rlgames_ppo.md`** records all of the following side-by-side for the two implementations:

    | Metric | rl_games PPO (M2) | jax-learning PPO (M5a) |
    |---|---|---|
    | Env steps/sec (rollout phase only) | | |
    | Update steps/sec (gradient phase only) | | |
    | Total wall-clock per training step (rollout + update) | | |
    | Wall-clock to reach reward R* (define R* = some fraction of M2 final reward, e.g. 80%) | | |
    | Env steps to reach R* | | |
    | Final mean episode reward after equal env-step budget | | |
    | Final mean episode reward after equal wall-clock budget | | |
    | (For JAX) bridge overhead ms/step from M4 | n/a | |

    Plot both **wall-clock-to-reward** and **env-steps-to-reward** curves overlaid. Two stories must be tellable from this doc: (a) which is more sample-efficient (env-step axis), (b) which is faster to a target reward in real time (wall-clock axis). The bridge cost from M4 is the explanatory variable for any gap on the wall-clock axis.
- **Expected outcome:** the hypothesis worth testing is that JIT-compiled JAX PPO has high enough rollout/update throughput to claw back the bridge overhead and still beat or match rl_games on wall-clock. Numbers may say otherwise — that's the point. We measure, we don't assume.

### M5b — JAX FlashSAC (the deliverable baseline)

- **Goal:** train the actual bolt-tightening policy with the sample-efficient algorithm.
- **New code:**
  - `scripts/train_jax_flashsac.py` — analogous to M5a but with `jax_rl.algos.flash_sac.FlashSAC`. Uses `JaxReplayBuffer` from jax-learning. FlashSAC tuning takes the per-task preset path: start from `jax_rl.configs.env_presets.get_flash_sac_preset(...)` and override only what's NutThread-specific.
- **Verification:**
  - Reward rises beyond random and beyond the M2 rl_games PPO number in fewer environment steps (sample efficiency claim).
  - Saved checkpoint loadable; `scripts/play.py --algo flashsac --task FactoryJax-NutThread-v0` renders an eval video showing the nut being threaded (or visibly attempting to).
  - **Success metric for "trained baseline":** mean episode reward over 20 eval episodes ≥ M2's rl_games PPO final number, measured after an equal environment-step budget. (The Factory paper's number is on a different codebase / hyperparameter set and is not apples-to-apples; use M2 as the only reference.)

### M6 — UR5 swap procedure (docs only)

- **Goal:** a documented, code-grounded recipe for swapping the robot.
- **New code:** `docs/robot_swap.md` only.
- **Contents:**
  - Where Franka enters the env (Isaac's `factory_env_cfg.py` robot articulation cfg, USD path, default joint pos, EE link name, gripper cfg).
  - Which cfg fields would change for UR5 (articulation USD path, joint names list, EE link name, joint default angles, gripper attachment if used).
  - UR5 USD asset source: `isaaclab_assets/UR/UR5e.usd` (verify presence; otherwise Isaac Sim Assets path).
  - Control implications: Factory env uses a task-space (operational-space) controller. UR5 has 6 joints (vs Franka 7) and no built-in redundancy resolution, so DLS may need retuning. Reference Isaac's `factory_control.py` Jacobian/damping params.
  - **Known unknowns** explicitly listed: gripper choice for UR5 (Robotiq vs custom), workspace dimensions, joint limits' interaction with Factory's reset distribution.
- **Verification:** reviewer can follow the doc on paper and identify exact file paths and field names to edit. No environment change required to pass.

## 8. Verification checkpoints summary

| After | Pass condition | If fail |
|---|---|---|
| M1 | Video recorded, env loads, obs/action shapes confirmed | Stop, debug install/GPU before any custom code |
| M2 | rl_games reward curve rises, final number recorded | Stop, debug Isaac config/assets; do not start M3 |
| M3 | `FactoryJax-NutThread-v0` listed and runnable | Inspect extension.toml + `__init__.py` gym.register path |
| M4 | DLPack roundtrip + parity test pass | Bridge is wrong — fix before M5 |
| M5a | JAX PPO curve recorded vs M2 | If wildly different despite matched configs, dig into implementation differences |
| M5b | FlashSAC checkpoint trains above M2 number | This is the bolt-tightening baseline; ship the eval video |
| M6 | Doc reviewable; no code change | N/A |

## 9. Risks and mitigations

1. **CUDA version mismatch.** `jax-learning` pins `jax[cuda13]`. Isaac Sim ships with CUDA 12. Mitigation = D10: dedicated uv venv at `.venv/`, declare `jax[cuda12]` in this project's `pyproject.toml` under `[tool.uv] override-dependencies`, which replaces jax-learning's dep without touching jax-learning's own pin. **Residual risk:** if `jaxlib` on CUDA 12 lacks a feature jax-learning's code path needs (e.g. a specific op only built into the CUDA 13 wheel), surface it during M5a/b and decide whether to (a) bump Isaac's CUDA toolkit, (b) downgrade the jax-learning code path, or (c) re-pin jax differently. Likelihood low — both CUDA 12 and 13 wheels are first-class for current JAX.
2. **DLPack stream ordering.** torch and jax may run on different CUDA streams. Insurance: `torch.cuda.synchronize()` per step in M4. Profile in M5b. Remove if not bottleneck.
3. **Factory env returns dict observations.** Isaac's NutThread returns `obs_dict = {"policy": tensor, "critic": tensor}` (asymmetric actor-critic, verify exact keys in M3). The wrapper must convert both keys to `jax.Array` and pass the dict through. jax-learning's algos consume `(num_envs, obs_dim)` ndarrays, so M5a/b decide per-algo which key to feed the policy and which to feed the critic (FlashSAC's critic can take the richer `"critic"` obs while the actor uses `"policy"`). Codify this in `JaxEnvWrapper`'s contract.
4. **num_envs scaling.** Factory is contact-heavy. Start at 4 (M1), 128 (M2/M5), only tune up to 1024+ after M5b is alive. Bigger num_envs first time is a noise generator.
5. **Newton/Warp migration.** Listed as Future Work (§10). Bridge designed so migration is a one-function swap in `tensor_convert.py`.
6. **rl_games vs jax PPO config translation.** Some hyperparameters do not have 1:1 names. The M5a config diff must explicitly document each unmapped knob with the chosen default, so the comparison stays interpretable.
7. **FlashSAC has non-trivial machinery** (BatchNorm, weight norm, reward scaling, Zeta noise). The wrapper must not block any of these; treat the env as a black box. If FlashSAC's reward scaler needs CPU updates per step, batch them.
8. **`./isaaclab.sh --new` may clobber existing files in the target dir.** Run it into a temp dir and merge into `Research/isaaclab-factory-jax/` so this spec's `.superpowers/` survives.

## 10. Future work (explicitly deferred)

- **IsaacLab 3.0 / Newton physics migration.** When NVIDIA ports Factory tasks to Newton, swap `to_jax`'s torch branch to a warp branch. Quaternion convention change (wxyz→xyzw) needs an audit of any custom math we wrote (probably zero for this spec). Own spec.
- **Custom BoltTighten task.** With M5b's pipeline live, design a real bolt-tighten env (USD asset, threaded-hole contact model, torque-based success criterion, multi-turn reward). Own spec.
- **UR5 implementation.** From the M6 doc.
- **Other Factory tasks (PegInsert, GearMesh) and a multi-task baseline.**
- **Distributed training / async data collection.** Single-GPU until M5b stable.
- **Custom controller / impedance & admittance control.** The M5b baseline uses Isaac's stock Factory task-space PD controller — same as the rl_games M2 baseline, so the comparison stays apples-to-apples. After M5b, an own spec opens to evaluate (a) tuning Isaac's actuator stiffness/damping for compliance, (b) swapping in a frax-style JIT IK so the policy can emit Cartesian impedance targets, (c) full external solver in JAX. Survey lives at `docs/controllers_survey.md` (produced in plan task M0.2).

## 11. Open questions

- **CUDA major version Isaac Sim 2025.1 ships against** — likely 12, must confirm in M3 before pinning JAX. If 11, JAX wheel selection changes.
- **`rl_games_ppo_cfg.yaml`'s exact tuning vs Factory paper's** — diff at M2 time; if Isaac's stock cfg differs from the paper, decide which to call "the baseline" for the M5a comparison.
- **Whether to record video during JAX training (M5a/b) or only at eval** — eval-only by default; revisit if debugging.

## 12. Debugging resources (search-first for downstream agents)

When something jax-learning-related misbehaves during M4/M5a/M5b — slow training, divergence, BatchNorm weirdness, replay buffer issues, deterministic / GPU memory problems, manipulation-specific reward weirdness — **search `Research/jax-learning/.context/` before diagnosing from scratch.** That directory carries prior incidents and fixes from the locomotion / manipulation work that built these algos.

Likely-relevant files (grep first, read what hits):

| Symptom | File |
|---|---|
| Algo seems slower than expected on GPU | `lessons/jax_performance.md`, `lessons/gpu_management.md` |
| FlashSAC / SAC instability, replay weirdness | `lessons/offpolicy.md`, `lessons/learner.md` |
| Bringing up a new env / wrapping issues | `lessons/algo_port_protocol.md`, `lessons/env_backends.md` |
| Manipulation reward shaping intuition | `lessons/manipulation.md` |
| BatchNorm / distributional critic (C51) | `lessons/distributional.md`, `lessons/learner.md` |
| Run determinism, seeding | `lessons/determinism.md` |
| What the algos were last validated on | `LESSONS.md`, `AGENT_HANDOFF.md` |

Rules:

- **Do not pull from `.context/` for pure Isaac-Sim issues** (USD, articulation cfg, contact tuning, etc.). Those have nothing to do with jax-learning's history.
- **Do not modify** files in `jax-learning/.context/` from this project. Read-only.
- Treat it as a search corpus, not gospel: a prior lesson may be outdated. Verify against the current `jax-learning/` code before acting on it.

### Modifying jax-learning's algo code (per D11)

If diagnosis points to a jax-learning algo change being needed:

1. **Default — branch jax-learning, do not touch `main`:**
   ```bash
   cd /home/stevenman/Desktop/Work/Research/jax-learning
   git checkout -b isaaclab-factory-jax/<topic>      # e.g. isaaclab-factory-jax/m5b-obs-dict
   # edit, commit on the branch
   ```
   Our project's editable install picks the change up immediately — no reinstall. Record the branch name in the PR / journal so we can review later for merge-back.

2. **Alternative — port the algo into this project**, only when the change would clutter jax-learning's history (e.g. Factory-specific actor architecture):
   ```
   source/factory_jax/factory_jax/algos/<algo>_factory.py
   ```
   Update `scripts/train_jax_*.py` to import the ported version. Add a one-line provenance comment at the top noting the source commit hash from jax-learning.

3. **Decision is per-change.** If unsure, default to branching. Don't pre-commit to porting — migration is decided **after M5b's baseline trains**, not before.

4. **Don't fork-and-forget.** Each branch / port that survives M5b must be listed in `docs/jax_learning_divergences.md` with: file, change summary, reason, and recommendation (merge back / keep ported / revert).

## 13. Glossary

- **External project:** an IsaacLab project living outside the `IsaacLab/` clone, scaffolded by Isaac's `./isaaclab.sh --new`. Officially recommended.
- **Direct workflow:** the IsaacLab task style where the env class subclasses `DirectRLEnv` (vs the manager-based workflow). Factory uses direct.
- **DLPack:** a vendor-neutral tensor exchange format that lets torch / jax / warp / cupy share GPU buffers without copies.
- **NutThread:** Isaac Factory task where a Franka screws a nut onto a fixed bolt. Closest stock task to "bolt tightening."
- **FlashSAC:** SAC variant in `jax-learning` (Kim et al. 2026) with inverted residual blocks, BatchNorm, weight norm, and a C51 critic. Significantly more sample-efficient than vanilla SAC/PPO on contact tasks.
