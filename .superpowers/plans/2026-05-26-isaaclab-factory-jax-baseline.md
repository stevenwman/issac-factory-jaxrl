# IsaacLab Factory + JAX: Bolt-Tightening RL Baseline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a trained JAX/FlashSAC policy on Isaac Factory NutThread (bolt-tightening primitive) in an external IsaacLab project, with a config-matched JAX PPO vs rl_games PPO comparison along the way.

**Architecture:** External IsaacLab project at `Research/isaaclab-factory-jax/` with its own uv venv. A DLPack zero-copy bridge converts torch (Isaac) ↔ jax.Array. Algorithms come from `jax-learning` consumed as an editable dep, modified only via topic branches per spec D11.

**Tech Stack:** IsaacLab 2.3.2 (PhysX + torch), Isaac Sim 5.1.0 (pip wheel), Python 3.11, uv, torch 2.7.0+cu128, `jax[cuda12]`, jax-learning (FlashSAC, PPO).

**Spec:** `Research/isaaclab-factory-jax/.superpowers/specs/2026-05-26-isaaclab-factory-jax-baseline-design.md`

---

## File Structure (created/modified)

```
Research/isaaclab-factory-jax/
├── .venv/                                          (uv venv, gitignored)
├── pyproject.toml                                  ← CREATE (M3)
├── uv.lock                                         ← AUTO (M3)
├── .gitignore                                      ← CREATE (M3)
├── README.md                                       ← CREATE (M3)
├── docs/
│   ├── M1_sanity.md                                ← CREATE (M1)
│   ├── M2_baseline.md                              ← CREATE (M2)
│   ├── M4_bridge_overhead.md                       ← CREATE (M4)
│   ├── M5a_jax_vs_rlgames_ppo.md                   ← CREATE (M5a)
│   ├── M5b_flashsac.md                             ← CREATE (M5b)
│   ├── robot_swap.md                               ← CREATE (M6)
│   └── jax_learning_divergences.md                 ← CREATE (M3, append-only)
├── scripts/
│   ├── train_rl_games.py                           ← CREATE (M2, thin wrapper)
│   ├── train_jax_ppo.py                            ← CREATE (M5a)
│   ├── train_jax_flashsac.py                       ← CREATE (M5b)
│   ├── play.py                                     ← CREATE (M5b)
│   ├── profile_bridge_overhead.py                  ← CREATE (M4)
│   └── list_envs.py                                ← FROM ./isaaclab.sh --new (M3)
├── source/factory_jax/                             ← FROM ./isaaclab.sh --new (M3)
│   ├── pyproject.toml                              (generator output)
│   ├── setup.py                                    (generator output)
│   ├── config/extension.toml                       (generator output)
│   └── factory_jax/
│       ├── __init__.py
│       ├── bridge/
│       │   ├── __init__.py                         ← CREATE (M4)
│       │   ├── tensor_convert.py                   ← CREATE (M4)
│       │   └── jax_env_wrapper.py                  ← CREATE (M4)
│       ├── tasks/direct/factory_nut_thread/
│       │   ├── __init__.py                         ← CREATE (M3)
│       │   └── env_cfg.py                          ← CREATE (M3)
│       └── configs/
│           └── matched_ppo_config.yaml             ← CREATE (M5a)
├── tests/
│   ├── test_tensor_convert.py                      ← CREATE (M4)
│   ├── test_jax_env_wrapper.py                     ← CREATE (M4)
│   └── test_isaac_parity.py                        ← CREATE (M4)
└── .superpowers/                                   (already exists from spec)
    ├── specs/2026-05-26-isaaclab-factory-jax-baseline-design.md
    └── plans/2026-05-26-isaaclab-factory-jax-baseline.md   (this file)
```

**Untouched (read-only references):**
- `/home/stevenman/Desktop/Work/IsaacLab/` — IsaacLab clone
- `/home/stevenman/Desktop/Work/Research/jax-learning/` — `main` branch off-limits per D11

**One-responsibility-per-file principle:** `tensor_convert.py` is pure plumbing (no env knowledge). `jax_env_wrapper.py` is pure gym-wrapper (no algo knowledge). Training scripts are pure entrypoints (no shared logic). If shared logic appears later, extract into `factory_jax/training/`.

---

## Conventions for this plan

- **Working directory.** All `bash` blocks assume `cd /home/stevenman/Desktop/Work/Research/isaaclab-factory-jax/` unless explicitly noted otherwise (e.g. M1 uses Isaac's clone).
- **Isaac Sim entrypoint.** From inside the new project's venv, use `python` (not `./IsaacLab/isaaclab.sh -p`). The shell wrapper is only needed when running against IsaacLab's own env.
- **Long-running tasks.** Training milestones (M2, M5a, M5b) are time-bounded by env-step budget, not wall-clock. Monitor reward curves, stop on plateau or budget exhaustion (whichever first).
- **Commits.** Every task ends with a commit. Use Conventional Commits (`feat:`, `chore:`, `docs:`, `test:`). Commit messages include the milestone tag, e.g. `feat(M4): add tensor_convert.to_jax`.
- **TDD.** Code tasks: red → green → commit. Setup tasks: install → smoke test → commit. Training tasks: launch → record → commit results.

---

## M0 — Git setup (do this before any other commit)

**Goal:** initialize the project as its own git repo with the GitHub remote configured. Every milestone's commits go to this repo from M0 onward.

### Task M0.1: git init + remote add + first commit

**Files:**
- Already exists: `.superpowers/specs/2026-05-26-isaaclab-factory-jax-baseline-design.md`, `.superpowers/plans/2026-05-26-isaaclab-factory-jax-baseline.md`

- [ ] **Step 1: Confirm we are not inside an existing git repo**

```bash
cd /home/stevenman/Desktop/Work/Research/isaaclab-factory-jax
git rev-parse --is-inside-work-tree 2>/dev/null && echo "ALREADY A REPO — investigate" || echo "OK, not a repo yet"
```
If "ALREADY A REPO": stop and inspect — something earlier created a repo unexpectedly. Reconcile before continuing.

- [ ] **Step 2: Initialize repo**

```bash
git init -b main
```

- [ ] **Step 3: Add `.gitignore` (minimal — full version comes in M3.1 once we have a venv)**

```bash
cat > .gitignore <<'EOF'
.venv/
__pycache__/
*.pyc
EOF
```

- [ ] **Step 4: First commit (spec + plan)**

```bash
git add .superpowers/specs/2026-05-26-isaaclab-factory-jax-baseline-design.md \
        .superpowers/plans/2026-05-26-isaaclab-factory-jax-baseline.md \
        .gitignore
git commit -m "chore(M0): init repo with spec + plan"
```

- [ ] **Step 5: Add GitHub remote**

```bash
git remote add origin git@github.com:stevenwman/issac-factory-jaxrl.git
git remote -v   # verify origin appears
```

- [ ] **Step 6: Push first commit (creates the remote branch)**

```bash
git push -u origin main
```
If push fails with "Repository not found": create the empty repo on GitHub first (no README, no .gitignore — we already have ours). Then retry.

- [ ] **Step 7: Verify**

```bash
git log --oneline -1
git branch -vv   # should show: main ... [origin/main]
```

**Push cadence going forward:**
- After the **last task of every milestone**, run `git push origin main`. Do not let an entire milestone's commits stay local. The GitHub mirror is the off-machine backup.
- More frequent pushes per task are fine but not required.

### Task M0.2: Controller-and-IK survey (parallel research task — write findings, defer decisions)

**Why:** The user is interested in impedance / admittance control and fast IK (mink-style, frax-style) for downstream work beyond the NutThread baseline. Before M5b's bolt-tightening policy is trained, we want a written survey of (a) what controllers Isaac actually ships and uses in Factory today, (b) what's plausible to swap in or layer on, and (c) a deferred decision on whether to touch the controller in this spec or not.

**Files:**
- Create: `docs/controllers_survey.md`

- [ ] **Step 1: Catalog Isaac's built-in controller surface**

Read and summarize:
- `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_control.py` — Factory's task-space controller (Jacobian, damping, gains)
- `IsaacLab/source/isaaclab/isaaclab/controllers/` — Isaac's controller library (likely: joint PD, differential IK, operational-space / OSC, RMP-like)
- `IsaacLab/source/isaaclab/isaaclab/actuators/` — actuator models (PD, implicit PD, etc.) and where stiffness/damping live
- `IsaacLab/docs/source/features/`  (or `how-to/`) for any controller docs

Output for each: what it does, where Factory uses it (if anywhere), what its action interface is.

- [ ] **Step 2: Cross-reference external solvers**

Compare against:
- **mink** (https://github.com/kevinzakka/mink) — Kevin Zakka's QP-based differential IK for MuJoCo. Solves task-space targets via quadratic program; supports inequality constraints, posture tasks, frame tasks. Tied to MuJoCo's `mjData`.
- **frax** (https://github.com/StanfordASL/frax) — StanfordASL fast IK in JAX. JIT-compiled, batched. Designed for sim throughput.
- (Optional) **placo**, **pink** (pinocchio-based) — listed for context; pure CPU, slower.

Summarize for each:
- Backend / dependency footprint
- Differentiable or not (matters for end-to-end RL)
- Batched / JIT-friendly
- Realistic to call per env step at `num_envs=128`?

- [ ] **Step 3: Write `docs/controllers_survey.md`** with the catalog + a "controller decision matrix"

```markdown
# Controller and IK survey (M0.2 — research, not implementation)

## Isaac's built-in controllers (as of IsaacLab 2.3.2)

| Location | Controller | Action interface | Used by Factory? |
|---|---|---|---|
| `isaaclab/controllers/differential_ik.py` | Differential IK (Jacobian inverse, dls) | task-space target | TODO confirm |
| `isaaclab/controllers/operational_space.py` | OSC (task-space inertia-weighted) | wrench / accel | TODO confirm |
| `factory/factory_control.py` | Factory-specific task-space PD with DLS Jacobian | 6-DoF task-space delta + gripper | yes (NutThread/PegInsert/GearMesh) |
| `isaaclab/actuators/implicit_pd.py` | Joint-space PD | torque or position | underlying actuator |

(Fill in exact paths and signatures after reading the source.)

## External solvers surveyed

| Tool | Backend | Diff'ble | Batched / JIT | Per-step at num_envs=128? | Notes |
|---|---|---|---|---|---|
| mink | MuJoCo + QP | no | no (one env at a time) | no | QP solver per step; would be the bottleneck |
| frax | JAX | yes | yes | yes (designed for it) | Closest to a drop-in for our JAX setup; not MuJoCo-bound |
| Isaac OSC | torch/PhysX | n/a (eager) | yes (Isaac's batched envs) | yes (free, already running) | Default |
| Isaac dls-IK | torch/PhysX | n/a | yes | yes | Default |

## Impedance / admittance control — feasibility

- **Stock Factory:** task-space PD ≈ implicit Cartesian impedance; no explicit contact-force feedback in the action interface (just delta-pose). For true impedance, we'd need force-torque sensing in the obs and a controller that maps target compliance to torques.
- **Path A — stay inside Isaac:** use Isaac's actuator stiffness/damping fields + add F/T sensor in `env_cfg`; treat policy output as compliant target. Cheapest; works inside the current bridge.
- **Path B — swap in frax-style JIT IK:** policy emits Cartesian impedance target → frax solves joint commands → Isaac applies. Requires the bridge to additionally transport joint commands torch-side; frax must JIT against the kinematics.
- **Path C — fully external solver in JAX (frax + custom impedance law in JAX):** maximum flexibility, biggest implementation cost.

## Decision (this spec)

**For the NutThread M5b baseline: use Isaac's stock Factory controller unchanged.** Reasons:
1. We want the baseline number to be apples-to-apples vs rl_games PPO M2 (same controller).
2. Custom controller introduces another variable that would confound the M5a/M5b comparisons.
3. Impedance control is more useful for contact-rich tasks where we have F/T sensing, which would warrant its own spec.

**For follow-on work:** open a separate spec ("controllers-and-ik") that picks one of Paths A/B/C based on what we learn from the M5b policy's failure modes. Frax-style JIT IK is the strongest fit if we go beyond stock.

## Open questions
- Does Factory's `factory_control.py` expose stiffness/damping as cfg fields, or are they hard-coded? (Read carefully.)
- Does Isaac's OSC controller support a contact-aware variant?
- What's frax's coverage of robots — is UR5e supported, or just Franka-class?
```

- [ ] **Step 4: Commit + push**

```bash
git add docs/controllers_survey.md
git commit -m "docs(M0): controller + IK survey (decision deferred; baseline uses Isaac stock)"
git push origin main
```

---

## M1 — Sanity: NutThread Loads + Random Actions + Video

**Goal:** Confirm install, GPU, Factory assets all work via Isaac's bundled tooling. Zero new code.

### Task M1.1: Verify IsaacLab can launch NutThread with random actions and record a video

**Files:**
- Create: `docs/M1_sanity.md` (after this task — directory may not yet exist; create on first commit)

**Working directory for this task:** `/home/stevenman/Desktop/Work/IsaacLab/`

- [ ] **Step 1: Confirm IsaacLab env_isaaclab activates and Isaac Sim imports**

```bash
cd /home/stevenman/Desktop/Work/IsaacLab
source env_isaaclab/bin/activate
python -c "import isaaclab; import isaaclab_tasks; print('isaaclab', isaaclab.__file__)"
```
Expected: prints path to isaaclab `__init__.py`. If it errors, the existing env is broken — stop and fix before proceeding.

- [ ] **Step 2: Run the random agent against NutThread with video**

```bash
./isaaclab.sh -p scripts/environments/random_agent.py \
    --task Isaac-Factory-NutThread-Direct-v0 \
    --num_envs 4 \
    --video --video_length 200 --enable_cameras \
    --headless
```
Expected:
- Exit code 0
- A new `videos/` directory (Isaac creates one under the IsaacLab repo or current dir — find it via `find . -name '*.mp4' -newer /tmp -type f` if unsure)
- Console prints `single_observation_space=Box(...,(21,))` and `single_action_space=Box(...,(6,))` early in startup

If OOM: drop to `--num_envs 2`. If asset-download errors: re-run `./isaaclab.sh --install` per IsaacLab docs.

- [ ] **Step 3: Record findings in `docs/M1_sanity.md`** (back in the new project)

Create the file at `/home/stevenman/Desktop/Work/Research/isaaclab-factory-jax/docs/M1_sanity.md` with:
```markdown
# M1 — Sanity check (NutThread + random actions)

**Date:** <YYYY-MM-DD>
**IsaacLab version:** 2.3.2 (from IsaacLab/VERSION)
**Command:** (paste the exact command run)

## Observed
- Exit status: 0
- Video saved to: `<path/to/video.mp4>`
- Video length: <N> frames
- `single_observation_space`: Box(-inf, inf, (21,))
- `single_action_space`: Box(-1, 1, (6,))  ← confirm exact bounds from the run
- obs dict keys returned by env at first reset: ["policy", "critic"]  ← confirm exact keys
- num_envs used: 4
- GPU device: <CUDA device name from torch.cuda.get_device_name(0)>

## Notes
- Any warnings / oddities to watch for in later milestones
```

To confirm obs-dict keys, add a one-off inline check (don't commit this script):
```bash
./isaaclab.sh -p -c "
import gymnasium as gym
import isaaclab_tasks  # noqa
env = gym.make('Isaac-Factory-NutThread-Direct-v0', num_envs=2)
obs, _ = env.reset()
print('OBS KEYS:', list(obs.keys()) if hasattr(obs, 'keys') else type(obs))
for k, v in (obs.items() if hasattr(obs, 'items') else []):
    print(' ', k, v.shape, v.dtype, v.device)
env.close()
"
```

- [ ] **Step 4: Commit (repo already exists from M0)**

```bash
cd /home/stevenman/Desktop/Work/Research/isaaclab-factory-jax
mkdir -p docs
# write M1_sanity.md, then:
git add docs/M1_sanity.md
git commit -m "docs(M1): sanity check video + obs/action shapes recorded"
git push origin main
```

**Failure modes:**
- `single_observation_space` not (21,) → Factory has been updated upstream; cross-check against `factory_env_cfg.py` and update the spec.
- Obs is a tensor (not a dict) → wrapper assumptions change; update spec §6 + Risk #3 and re-run M1 doc.

---

## M2 — rl_games PPO reference baseline

**Goal:** Establish a known-good reward curve and wall-clock numbers for NutThread under Isaac's stock training pipeline. These become the comparison target for M5a.

### Task M2.1: Launch rl_games PPO training

**Files:**
- None (Isaac's stock script).

**Working directory:** `/home/stevenman/Desktop/Work/IsaacLab/`

- [ ] **Step 1: Identify the rl_games config to study**

```bash
cat source/isaaclab_tasks/isaaclab_tasks/direct/factory/agents/rl_games_ppo_cfg.yaml
```
Save a verbatim copy of this file's content for later — it becomes the **target config** for M5a:
```bash
mkdir -p /home/stevenman/Desktop/Work/Research/isaaclab-factory-jax/source/factory_jax/factory_jax/configs
cp source/isaaclab_tasks/isaaclab_tasks/direct/factory/agents/rl_games_ppo_cfg.yaml \
   /home/stevenman/Desktop/Work/Research/isaaclab-factory-jax/source/factory_jax/factory_jax/configs/rl_games_ppo_cfg_reference.yaml
```

- [ ] **Step 2: Launch training**

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Factory-NutThread-Direct-v0 \
    --num_envs 128 \
    --headless \
    --max_iterations 1000 \
    --experiment factory_nut_thread_m2
```

`--max_iterations 1000` is a **starting budget**; raise to 2000 / 4000 if curve has not plateaued. rl_games iterations are not env steps — read the script's `horizon_length` × `num_envs` to get env steps per iteration.

Monitor via tensorboard (rl_games writes logs under `runs/` or `logs/rl_games/`):
```bash
tensorboard --logdir runs --port 6006 --bind_all
```

Stop when: reward curve has visibly plateaued OR max budget exhausted.

- [ ] **Step 3: Time per-phase wall-clock**

rl_games prints per-iteration time in its log. Capture into a file:
```bash
# from the IsaacLab dir during training, in a separate terminal:
tail -f runs/factory_nut_thread_m2/summaries/*.tfevents.*  # check timestamps
# OR add a hook: track 'steps_per_sec' from rl_games's stdout
```
Or rerun a short pinned profile after the main run (e.g. 10 iterations) to compute env-steps/sec and update-steps/sec cleanly.

- [ ] **Step 4: Save the final checkpoint path**

rl_games saves to `runs/factory_nut_thread_m2/nn/*.pth`. Note the best checkpoint path in `docs/M2_baseline.md` (created next task).

- [ ] **Step 5: Commit reference config**

```bash
cd /home/stevenman/Desktop/Work/Research/isaaclab-factory-jax
git add source/factory_jax/factory_jax/configs/rl_games_ppo_cfg_reference.yaml
git commit -m "chore(M2): vendor rl_games PPO reference config from IsaacLab"
```
**Temporal-ordering note:** `source/factory_jax/` tree does not exist until M3.6 runs `./isaaclab.sh --new`. If this commit fails because the path doesn't exist:
1. Stash the copied yaml to `/tmp/rl_games_ppo_cfg_reference.yaml`.
2. Add a TODO line at the top of `docs/M2_baseline.md`: `TODO: commit /tmp/rl_games_ppo_cfg_reference.yaml after M3.6 creates source/factory_jax/factory_jax/configs/`.
3. In **M3.6 Step 3** (the merge commit), explicitly include this deferred file in the commit — see the back-reference there.

### Task M2.2: Eval rl_games checkpoint + record M2_baseline.md

**Files:**
- Create: `docs/M2_baseline.md`

- [ ] **Step 1: Run the play script against the best checkpoint**

```bash
cd /home/stevenman/Desktop/Work/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
    --task Isaac-Factory-NutThread-Direct-v0 \
    --checkpoint runs/factory_nut_thread_m2/nn/<best>.pth \
    --num_envs 20 \
    --headless --video --video_length 400 --enable_cameras
```
Record:
- Mean episode reward over 20 envs
- Success rate (Factory env's "success" key in info dict if present; else fraction of episodes ending with positive terminal reward)

- [ ] **Step 2: Write `docs/M2_baseline.md`**

```markdown
# M2 — rl_games PPO baseline (NutThread)

**Date:** <YYYY-MM-DD>
**Task:** `Isaac-Factory-NutThread-Direct-v0`
**Config:** `source/factory_jax/factory_jax/configs/rl_games_ppo_cfg_reference.yaml` (verbatim copy of IsaacLab's)
**num_envs:** 128
**Total env steps:** <N>
**Iterations:** <I>

## Final metrics

| Metric | Value |
|---|---|
| Mean episode reward (train) | |
| Mean episode reward (eval, 20 envs) | |
| Success rate (eval) | |
| Final checkpoint path | `runs/factory_nut_thread_m2/nn/<best>.pth` |

## Wall-clock (feeds M5a comparison table)

| Phase | Value |
|---|---|
| Env steps/sec (rollout only) | |
| Update steps/sec (gradient only) | |
| Total seconds per training iteration (rollout + update) | |
| Wall-clock to reach R* = 80% of final reward | |

## R* target (used in M5a)

R* = `<0.8 * final_reward>` mean episode reward.

## Full hyperparameter snapshot

See `source/factory_jax/factory_jax/configs/rl_games_ppo_cfg_reference.yaml`. Key knobs we will need to translate for M5a:
- `horizon_length`, `num_envs`, `mini_epochs`, `minibatch_size`, `learning_rate`, `gamma`, `tau` (GAE λ), `e_clip` (PPO ε), `clip_value`, `entropy_coef`, `critic_coef`, `network.mlp.units`, `network.mlp.activation`, `network.rnn` (LSTM!)

## Network architecture note

rl_games config uses **LSTM** (`rnn.name: lstm, units: 1024, layers: 2`). jax-learning's PPO does not (as of writing) ship an LSTM variant. M5a will document this as an **unavoidable difference**: jax-learning PPO will run with MLP-only [512, 128, 64] elu — same MLP head as the rl_games config, no recurrence. This means the comparison is "rl_games (LSTM + MLP) vs jax-learning (MLP only)."

## Notes / anomalies
```

- [ ] **Step 3: Commit M2 baseline doc + push (end of milestone)**

```bash
git add docs/M2_baseline.md
git commit -m "docs(M2): rl_games PPO baseline metrics + R* target + LSTM mismatch note"
git push origin main
```

**Decision gate before M3:**
- If reward curve flat or training crashed, do NOT proceed. Debug Factory env / config first.

---

## M3 — External project scaffold (uv venv + isaaclab.sh --new + register task)

**Goal:** Working external project with its own venv, NutThread re-registered under our gym ID, M1 video reproducible against the new ID.

### Task M3.1: Create dedicated venv + project init

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`

- [ ] **Step 1: Init project**

```bash
cd /home/stevenman/Desktop/Work/Research/isaaclab-factory-jax
uv init --no-package --python 3.11
uv venv --python 3.11
source .venv/bin/activate
python -V        # should print 3.11.x
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
*.egg-info/
dist/
build/
videos/
runs/
logs/
wandb/
checkpoints/
*.mp4
*.pth
*.ckpt
.ipynb_checkpoints/
```

- [ ] **Step 3: Configure `pyproject.toml`** (indices + sources + uv config)

Edit the auto-generated `pyproject.toml` to include:
```toml
[project]
name = "isaaclab-factory-jax"
version = "0.1.0"
description = "IsaacLab Factory NutThread RL baseline with jax-learning algos"
requires-python = ">=3.11,<3.12"
dependencies = []

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[[tool.uv.index]]
name = "nvidia"
url = "https://pypi.nvidia.com"
explicit = true

[tool.uv.sources]
torch       = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
isaacsim    = { index = "nvidia" }

[tool.uv]
# Pinned later in M3.7 once jax-learning is installed and we know the actual jax version it pulls
# override-dependencies = ["jax[cuda12]==<pinned>"]
```

- [ ] **Step 4: Stub `README.md`**

```markdown
# isaaclab-factory-jax

External IsaacLab project for training JAX-based RL policies on the Factory NutThread (bolt-tightening) task.

See `.superpowers/specs/2026-05-26-isaaclab-factory-jax-baseline-design.md` for the design.
See `.superpowers/plans/2026-05-26-isaaclab-factory-jax-baseline.md` for the implementation plan.

## Quickstart
TODO — fill in after M3 completes.
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore README.md
git commit -m "chore(M3): init project + uv pyproject (no deps yet)"
```

### Task M3.2: Install torch (cu128) first

- [ ] **Step 1: Add torch + torchvision**

```bash
uv add "torch==2.7.0" "torchvision==0.22.0"
```

- [ ] **Step 2: Verify torch sees CUDA 12.8**

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```
Expected: `2.7.0+cu128 12.8 True <GPU NAME>`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(M3): add torch 2.7.0 + torchvision 0.22.0 (cu128)"
```

### Task M3.3: Install Isaac Sim from pip

- [ ] **Step 1: Add isaacsim**

```bash
uv add "isaacsim[all,extscache]==5.1.0"
```
This will pull ~2 GB. Allow 5–15 min for download + install. Watch for "Successfully installed isaacsim-5.1.0".

- [ ] **Step 2: Smoke test isaacsim import**

```bash
python -c "import isaacsim; print('isaacsim OK')"
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(M3): add isaacsim 5.1.0 (pypi.nvidia.com)"
```

### Task M3.4: Install IsaacLab packages editable

- [ ] **Step 1: Add each isaaclab package as editable**

```bash
for pkg in isaaclab isaaclab_assets isaaclab_rl isaaclab_tasks isaaclab_mimic; do
  uv add --editable /home/stevenman/Desktop/Work/IsaacLab/source/$pkg
done
```
(Some packages depend on others; uv resolves the graph. If any fail, fix the failing package first; do not move on with a half-installed set.)

- [ ] **Step 2: Smoke test isaaclab import**

```bash
python -c "import isaaclab, isaaclab_tasks; print('isaaclab', isaaclab.__file__); print('tasks', isaaclab_tasks.__file__)"
```
Expected: both paths point inside `/home/stevenman/Desktop/Work/IsaacLab/source/`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(M3): add isaaclab* editable from local clone"
```

### Task M3.5: Install jax-learning editable

- [ ] **Step 1: Add jax-learning**

```bash
uv add --editable /home/stevenman/Desktop/Work/Research/jax-learning
```

- [ ] **Step 2: Smoke test**

```bash
python -c "import jax_rl; print(jax_rl.__file__)"
python -c "from jax_rl.algos.flash_sac import FlashSAC; print('FlashSAC OK')"
python -c "from jax_rl.algos.ppo import PPO; print('PPO OK')"
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(M3): add jax-learning editable from local clone"
```

### Task M3.6: Run isaaclab.sh --new + merge scaffold into project

- [ ] **Step 1: Run the generator into a scratch dir**

```bash
mkdir -p /tmp/factory_jax_scratch
cd /home/stevenman/Desktop/Work/IsaacLab
./isaaclab.sh --new
# Prompts:
#   - External or Internal: External
#   - Path: /tmp/factory_jax_scratch/factory_jax
#   - Workflow: Direct workflow
#   - RL libraries: rl_games (we'll wire JAX manually)
#   - Algorithm: PPO (placeholder, we ignore the generated train.py)
```

- [ ] **Step 2: Merge scaffold into project, preserving `.superpowers/`**

```bash
cd /home/stevenman/Desktop/Work/Research/isaaclab-factory-jax
# Copy generated files but DO NOT clobber .superpowers/, .git/, .gitignore, README.md, pyproject.toml
rsync -av \
    --exclude='.superpowers/' \
    --exclude='.git/' \
    --exclude='.gitignore' \
    --exclude='README.md' \
    --exclude='pyproject.toml' \
    /tmp/factory_jax_scratch/factory_jax/ ./
```
Inspect the diff carefully:
```bash
git status
git diff --stat
```
Cherry-pick: keep the scaffold's `source/factory_jax/`, `scripts/`, and `tests/` skeleton; reject any generator-added `.gitignore` or top-level pyproject changes.

If the scaffold's top-level `pyproject.toml` conflicts with ours, merge by hand: keep our `[tool.uv]` blocks; pull in any extension-related entries from the scaffold.

- [ ] **Step 3: Commit scaffold + (if applicable) the deferred M2 reference config**

```bash
# pick up the M2-deferred config if it exists in /tmp
if [ -f /tmp/rl_games_ppo_cfg_reference.yaml ]; then
  mkdir -p source/factory_jax/factory_jax/configs
  mv /tmp/rl_games_ppo_cfg_reference.yaml \
     source/factory_jax/factory_jax/configs/rl_games_ppo_cfg_reference.yaml
fi

git add source/ scripts/ tests/   # or whatever the scaffold added
git add source/factory_jax/factory_jax/configs/rl_games_ppo_cfg_reference.yaml 2>/dev/null || true
git commit -m "chore(M3): merge isaaclab.sh --new scaffold + close M2 deferred config commit"
```
If the M2 file was committed, remove the TODO line from `docs/M2_baseline.md`.

### Task M3.7: Install our project's extension editable + pin jax[cuda12]

- [ ] **Step 1: Add our extension as editable**

```bash
uv add --editable source/factory_jax
```

- [ ] **Step 2: Identify jax version jax-learning pulled**

```bash
uv pip show jax | grep -E "Version|Requires"
```
Note the version pulled (likely `0.9.x` cuda13).

- [ ] **Step 3: Pin `jax[cuda12]` via override**

Edit `pyproject.toml`'s `[tool.uv]` section:
```toml
[tool.uv]
override-dependencies = ["jax[cuda12]==<exact-version-from-step-2>"]
```

- [ ] **Step 4: Resync**

```bash
uv sync
```

- [ ] **Step 5: Verify CUDA alignment**

```bash
python -c "import torch, jax; print('torch CUDA:', torch.version.cuda); print('jax devices:', jax.devices())"
```
Expected: `torch CUDA: 12.8` and a `cuda` device in `jax.devices()`. If `jax.devices()` is `[CpuDevice(id=0)]`, the cuda12 plugin didn't install correctly — debug before continuing (re-pin to the exact `jax[cuda12]` matching jax-learning's jax version).

- [ ] **Step 6: DLPack smoke test (de-risks M4 early)**

```bash
python -c "
import jax, torch
from torch.utils.dlpack import to_dlpack
x = torch.randn(4, 8, device='cuda')
a = jax.dlpack.from_dlpack(to_dlpack(x))
print('jax array:', a.shape, a.dtype, a.devices())
print('match:', (a.shape == x.shape) and str(a.devices()[0]).startswith('cuda'))
"
```
Expected: `match: True`.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(M3): add factory_jax extension editable + pin jax[cuda12] override"
```

### Task M3.8: Register `FactoryJax-NutThread-v0`

**Files:**
- Create: `source/factory_jax/factory_jax/tasks/direct/factory_nut_thread/__init__.py`
- Create: `source/factory_jax/factory_jax/tasks/direct/factory_nut_thread/env_cfg.py`
- Modify: `source/factory_jax/factory_jax/__init__.py` (to import the task module so gym.register runs)

- [ ] **Step 1: Create the task package**

```bash
mkdir -p source/factory_jax/factory_jax/tasks/direct/factory_nut_thread
touch source/factory_jax/factory_jax/tasks/__init__.py
touch source/factory_jax/factory_jax/tasks/direct/__init__.py
```

- [ ] **Step 2: Write `env_cfg.py`**

```python
# source/factory_jax/factory_jax/tasks/direct/factory_nut_thread/env_cfg.py

from __future__ import annotations

from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskNutThreadCfg


class FactoryJaxNutThreadCfg(FactoryTaskNutThreadCfg):
    """Customization seam for our NutThread variant.

    Currently no overrides — pure subclass. Add robot, gains, or reward
    overrides here as the project evolves (e.g. UR5 swap per docs/robot_swap.md).
    """
    pass
```

- [ ] **Step 3: Write task package `__init__.py`** (registers the gym ID)

```python
# source/factory_jax/factory_jax/tasks/direct/factory_nut_thread/__init__.py

import gymnasium as gym

from .env_cfg import FactoryJaxNutThreadCfg

gym.register(
    id="FactoryJax-NutThread-v0",
    entry_point="isaaclab_tasks.direct.factory.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryJaxNutThreadCfg,
    },
)
```

- [ ] **Step 4: Ensure extension `__init__.py` imports the task** (so `gym.register` runs on package import)

Edit `source/factory_jax/factory_jax/__init__.py`:
```python
# source/factory_jax/factory_jax/__init__.py
# (whatever the scaffold put here — keep it; add this line:)
from factory_jax.tasks.direct import factory_nut_thread  # noqa: F401
```
(If the scaffold's `__init__.py` already auto-discovers tasks, this line is unnecessary — verify by listing envs in step 5.)

- [ ] **Step 5: Confirm gym ID is registered**

```bash
python -c "
import factory_jax  # noqa: F401
import gymnasium as gym
ids = [s.id for s in gym.envs.registry.values() if 'FactoryJax' in s.id]
print('registered:', ids)
"
```
Expected: `registered: ['FactoryJax-NutThread-v0']`

- [ ] **Step 6: Commit**

```bash
git add source/factory_jax/factory_jax/tasks
git add source/factory_jax/factory_jax/__init__.py
git commit -m "feat(M3): register FactoryJax-NutThread-v0 (thin subclass of Isaac's NutThread)"
```

### Task M3.9: Reproduce M1 video against `FactoryJax-NutThread-v0`

**Files:**
- Use: scaffold's `scripts/random_agent.py` (or equivalent)

- [ ] **Step 1: Identify the scaffold's random agent script**

```bash
find scripts -name '*.py' -exec grep -l 'random' {} \;
```
If none exists, copy from IsaacLab:
```bash
cp /home/stevenman/Desktop/Work/IsaacLab/scripts/environments/random_agent.py scripts/random_agent.py
# IMPORTANT: edit the new file to `import factory_jax  # noqa: F401` before `gym.make`
```

- [ ] **Step 2: Run random agent against our task ID**

```bash
python scripts/random_agent.py \
    --task FactoryJax-NutThread-v0 \
    --num_envs 4 \
    --video --video_length 200 --enable_cameras --headless
```
Expected: same console output as M1.2 (obs/action shapes match), new video in `videos/`.

- [ ] **Step 3: Confirm video matches M1 visually**

```bash
ls -la videos/
# play the latest video manually or via ffprobe to confirm it's non-empty
ffprobe videos/<latest>.mp4 2>&1 | grep "Duration"
```

- [ ] **Step 4: Update `docs/M1_sanity.md`** with a section "Reproduced via FactoryJax-NutThread-v0 (M3)" linking the new video path.

- [ ] **Step 5: Commit**

```bash
git add docs/M1_sanity.md
# also commit scripts/random_agent.py if you copied it in
git add scripts/random_agent.py 2>/dev/null || true
git commit -m "feat(M3): reproduce M1 video against FactoryJax-NutThread-v0"
```

### Task M3.10: Create `docs/jax_learning_divergences.md` (append-only ledger)

**Files:**
- Create: `docs/jax_learning_divergences.md`

- [ ] **Step 1: Write the ledger header**

```markdown
# jax-learning divergences (per spec D11)

Any time we modify jax-learning on a branch OR port a file into this project, append a row here.

| Date | File (in jax-learning) | Branch / Ported-to | Change summary | Reason | Recommendation (merge / keep / revert) |
|---|---|---|---|---|---|
| | | | | | |
```

- [ ] **Step 2: Commit**

```bash
git add docs/jax_learning_divergences.md
git commit -m "docs(M3): add jax-learning divergence ledger (per spec D11)"
git push origin main   # end of M3
```

---

## M4 — JAX↔Isaac DLPack bridge

**Goal:** working zero-copy GPU bridge between Isaac (torch) and jax-learning (jax.Array), with parity test and bridge overhead measurement.

### Task M4.1: TDD — `to_jax` / `from_jax` (torch.Tensor source)

**Files:**
- Create: `tests/test_tensor_convert.py`
- Create: `source/factory_jax/factory_jax/bridge/__init__.py`
- Create: `source/factory_jax/factory_jax/bridge/tensor_convert.py`

- [ ] **Step 1: Write the failing test**

`tests/test_tensor_convert.py`:
```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from factory_jax.bridge.tensor_convert import to_jax, from_jax


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_to_jax_torch_cuda_roundtrip_values():
    x_torch = torch.randn(8, 16, device="cuda", dtype=torch.float32)
    a = to_jax(x_torch)
    assert isinstance(a, jax.Array)
    assert a.shape == (8, 16)
    assert str(a.devices()[0]).startswith("cuda"), f"expected cuda device, got {a.devices()}"
    np.testing.assert_allclose(np.asarray(a), x_torch.cpu().numpy(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_from_jax_torch_roundtrip_values():
    a = jnp.arange(32, dtype=jnp.float32).reshape(4, 8)
    a = jax.device_put(a, jax.devices()[0])
    x = from_jax(a, target="torch")
    assert isinstance(x, torch.Tensor)
    assert x.is_cuda
    assert x.shape == (4, 8)
    np.testing.assert_allclose(x.cpu().numpy(), np.asarray(a), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_roundtrip_torch_jax_torch_value_exact():
    x_in = torch.randn(64, device="cuda", dtype=torch.float32)
    x_out = from_jax(to_jax(x_in), target="torch")
    np.testing.assert_array_equal(x_in.cpu().numpy(), x_out.cpu().numpy())
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/test_tensor_convert.py -v
```
Expected: `ImportError: cannot import name 'to_jax' from 'factory_jax.bridge.tensor_convert'` (file doesn't exist).

- [ ] **Step 3: Implement bridge**

`source/factory_jax/factory_jax/bridge/__init__.py`:
```python
from .tensor_convert import to_jax, from_jax  # noqa: F401
```

`source/factory_jax/factory_jax/bridge/tensor_convert.py`:
```python
"""Source-tensor-type-agnostic DLPack bridge between Isaac (torch) and JAX.

When IsaacLab migrates to Newton/Warp (future spec), add a `warp.array` branch
to `to_jax` and a `target="warp"` branch to `from_jax`. The rest of the bridge
(env wrapper, training scripts) does not need to change.
"""
from __future__ import annotations

from typing import Any, Literal

import jax
import torch
from torch.utils.dlpack import from_dlpack as torch_from_dlpack
from torch.utils.dlpack import to_dlpack as torch_to_dlpack


def to_jax(x: Any) -> jax.Array:
    """Convert a torch.Tensor (or future warp.array) to jax.Array via DLPack.

    Zero-copy when source and JAX both live on the same CUDA device.
    """
    if isinstance(x, torch.Tensor):
        return jax.dlpack.from_dlpack(torch_to_dlpack(x))
    raise TypeError(f"to_jax: unsupported source type {type(x).__name__}")


def from_jax(a: jax.Array, target: Literal["torch"] = "torch") -> Any:
    """Convert a jax.Array to the target tensor type via DLPack."""
    if target == "torch":
        return torch_from_dlpack(jax.dlpack.to_dlpack(a))
    raise ValueError(f"from_jax: unsupported target {target!r}")
```

- [ ] **Step 4: Run test, verify pass**

```bash
pytest tests/test_tensor_convert.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_tensor_convert.py source/factory_jax/factory_jax/bridge
git commit -m "feat(M4): add tensor_convert to_jax/from_jax (torch via DLPack)"
```

### Task M4.2: TDD — `JaxEnvWrapper` (mocked env)

**Files:**
- Create: `tests/test_jax_env_wrapper.py`
- Create: `source/factory_jax/factory_jax/bridge/jax_env_wrapper.py`

- [ ] **Step 1: Write the failing test (mocked env, no Isaac required)**

`tests/test_jax_env_wrapper.py`:
```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from factory_jax.bridge.jax_env_wrapper import JaxEnvWrapper


class FakeIsaacEnv:
    """Minimal stand-in for an Isaac DirectRLEnv.

    Returns torch tensors on cuda; obs is a dict with 'policy' and 'critic' keys.
    """
    num_envs = 4
    action_dim = 6
    policy_obs_dim = 21
    critic_obs_dim = 27

    def __init__(self):
        self._t = 0

    def reset(self, seed=None, options=None):
        self._t = 0
        obs = {
            "policy": torch.full((self.num_envs, self.policy_obs_dim), 0.5, device="cuda"),
            "critic": torch.full((self.num_envs, self.critic_obs_dim), 0.7, device="cuda"),
        }
        return obs, {}

    def step(self, action):
        assert isinstance(action, torch.Tensor) and action.is_cuda
        assert action.shape == (self.num_envs, self.action_dim)
        self._t += 1
        obs = {
            "policy": torch.full((self.num_envs, self.policy_obs_dim), float(self._t), device="cuda"),
            "critic": torch.full((self.num_envs, self.critic_obs_dim), float(self._t) + 0.1, device="cuda"),
        }
        reward = torch.full((self.num_envs,), float(self._t) * 0.01, device="cuda")
        terminated = torch.zeros((self.num_envs,), dtype=torch.bool, device="cuda")
        truncated = torch.zeros((self.num_envs,), dtype=torch.bool, device="cuda")
        return obs, reward, terminated, truncated, {}

    def close(self):
        pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_wrapper_reset_returns_jax_dict():
    env = JaxEnvWrapper(FakeIsaacEnv())
    obs, info = env.reset()
    assert set(obs.keys()) == {"policy", "critic"}
    assert isinstance(obs["policy"], jax.Array)
    assert isinstance(obs["critic"], jax.Array)
    assert obs["policy"].shape == (4, 21)
    assert obs["critic"].shape == (4, 27)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_wrapper_step_accepts_jax_action_returns_jax_outputs():
    env = JaxEnvWrapper(FakeIsaacEnv())
    env.reset()
    action = jnp.zeros((4, 6), dtype=jnp.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, jax.Array)
    assert isinstance(terminated, jax.Array)
    assert isinstance(truncated, jax.Array)
    np.testing.assert_allclose(np.asarray(reward), np.full(4, 0.01, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(obs["policy"]), np.full((4, 21), 1.0, dtype=np.float32))
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/test_jax_env_wrapper.py -v
```
Expected: `ImportError`.

- [ ] **Step 3: Implement `JaxEnvWrapper`**

`source/factory_jax/factory_jax/bridge/jax_env_wrapper.py`:
```python
"""gym.Wrapper that swaps torch tensors at the env interface for jax.Array.

Preserves Isaac's `(num_envs, dim)` batched layout and dict-style obs
({"policy", "critic"} for asymmetric actor-critic envs like Factory).

Per spec §6: one `torch.cuda.synchronize()` per step as cheap stream-ordering
insurance. Profile and drop in M5b if not the bottleneck.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import jax
import torch

from .tensor_convert import from_jax, to_jax


class JaxEnvWrapper(gym.Wrapper):
    def __init__(self, env: Any):
        super().__init__(env)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _torch_obs_to_jax(obs):
        if isinstance(obs, dict):
            return {k: to_jax(v) for k, v in obs.items()}
        return to_jax(obs)

    # ------------------------------------------------------------------ API
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._torch_obs_to_jax(obs), info

    def step(self, action: jax.Array):
        action_torch = from_jax(action, target="torch")
        torch.cuda.synchronize()
        obs, reward, terminated, truncated, info = self.env.step(action_torch)
        torch.cuda.synchronize()
        return (
            self._torch_obs_to_jax(obs),
            to_jax(reward),
            to_jax(terminated),
            to_jax(truncated),
            info,
        )

    def close(self):
        self.env.close()
```

- [ ] **Step 4: Run test, verify pass**

```bash
pytest tests/test_jax_env_wrapper.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_jax_env_wrapper.py source/factory_jax/factory_jax/bridge/jax_env_wrapper.py
git commit -m "feat(M4): add JaxEnvWrapper (gym.Wrapper, torch<->jax via tensor_convert)"
```

### Task M4.3: Integration parity test against real Isaac NutThread

**Why two scripts + a diff test, not one in-process test:** Isaac Sim's `SimulationApp` is a process-global singleton. Instantiating `gym.make("FactoryJax-NutThread-v0", ...)` twice in the same Python process is unreliable — the second instance may fail to start, or carry state from the first. The robust pattern: each rollout runs in its own Python process (its own SimulationApp lifecycle), dumps its trajectory to disk, and the pytest test only reads the two NPZ files and diffs them.

**Files:**
- Create: `scripts/dump_rollout.py` (one binary that does raw OR wrapped rollout based on a flag)
- Create: `tests/test_isaac_parity.py` (offline diff of two NPZ dumps)

- [ ] **Step 1: Write `scripts/dump_rollout.py`**

```python
"""Run a deterministic rollout against FactoryJax-NutThread-v0 and dump the
trajectory to NPZ. One rollout per process (Isaac's SimulationApp is a
process-global singleton).

Usage:
    python scripts/dump_rollout.py --mode raw     --out /tmp/parity_raw.npz
    python scripts/dump_rollout.py --mode wrapped --out /tmp/parity_wrapped.npz
"""
from __future__ import annotations

import argparse
import numpy as np

import gymnasium as gym
import torch
import jax.numpy as jnp

import factory_jax  # noqa: F401


def run_raw(num_envs: int, n_steps: int, seed: int):
    env = gym.make("FactoryJax-NutThread-v0", num_envs=num_envs)
    obs, _ = env.reset(seed=seed)
    policy = [obs["policy"].cpu().numpy().copy()]
    rewards, dones = [], []
    a = torch.zeros((num_envs, 6), device="cuda")
    for _ in range(n_steps):
        obs, r, term, trunc, _ = env.step(a)
        policy.append(obs["policy"].cpu().numpy().copy())
        rewards.append(r.cpu().numpy().copy())
        dones.append((term | trunc).cpu().numpy().copy())
    env.close()
    return np.stack(policy), np.stack(rewards), np.stack(dones)


def run_wrapped(num_envs: int, n_steps: int, seed: int):
    from factory_jax.bridge.jax_env_wrapper import JaxEnvWrapper
    env = JaxEnvWrapper(gym.make("FactoryJax-NutThread-v0", num_envs=num_envs))
    obs, _ = env.reset(seed=seed)
    policy = [np.asarray(obs["policy"])]
    rewards, dones = [], []
    a = jnp.zeros((num_envs, 6), dtype=jnp.float32)
    for _ in range(n_steps):
        obs, r, term, trunc, _ = env.step(a)
        policy.append(np.asarray(obs["policy"]))
        rewards.append(np.asarray(r))
        dones.append(np.asarray(term | trunc))
    env.close()
    return np.stack(policy), np.stack(rewards), np.stack(dones)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("raw", "wrapped"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    runner = run_raw if args.mode == "raw" else run_wrapped
    policy, rewards, dones = runner(args.num_envs, args.n_steps, args.seed)
    np.savez(args.out, policy=policy, rewards=rewards, dones=dones)
    print(f"wrote {args.out}: policy={policy.shape} rewards={rewards.shape} dones={dones.shape}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the failing pytest test**

`tests/test_isaac_parity.py`:
```python
"""Offline parity test: assumes scripts/dump_rollout.py has been run twice
(once with --mode raw, once with --mode wrapped) under the same seed.

The test only loads the dumps and diffs — it does not start Isaac.
"""
from pathlib import Path

import numpy as np
import pytest


RAW = Path("/tmp/parity_raw.npz")
WRAPPED = Path("/tmp/parity_wrapped.npz")


@pytest.mark.skipif(not (RAW.exists() and WRAPPED.exists()),
                    reason="run scripts/dump_rollout.py first (raw + wrapped)")
def test_jax_wrapper_parity_with_raw_isaac():
    r = np.load(RAW)
    w = np.load(WRAPPED)

    diff_policy = np.max(np.abs(r["policy"] - w["policy"]))
    diff_rewards = np.max(np.abs(r["rewards"] - w["rewards"]))

    assert diff_policy < 1e-5, f"policy obs max|diff|={diff_policy}"
    assert diff_rewards < 1e-5, f"reward max|diff|={diff_rewards}"
    assert np.array_equal(r["dones"], w["dones"]), "done flags mismatch"
```

- [ ] **Step 3: Run test, verify it skips (no dumps yet)**

```bash
pytest tests/test_isaac_parity.py -v
```
Expected: 1 skipped (reason: dumps don't exist).

- [ ] **Step 4: Produce the two dumps (each in its own process)**

```bash
python scripts/dump_rollout.py --mode raw     --out /tmp/parity_raw.npz
python scripts/dump_rollout.py --mode wrapped --out /tmp/parity_wrapped.npz
```
Each invocation boots Isaac fresh, runs the rollout, exits. Look for "wrote /tmp/parity_*.npz" lines.

- [ ] **Step 5: Re-run test, verify it passes**

```bash
pytest tests/test_isaac_parity.py -v
```
Expected: 1 passed.

If diff > 1e-5: the bridge is wrong — inspect `policy[0]` (reset obs) first; if reset already differs, seeding plumbed differently in the two paths.

- [ ] **Step 6: Commit**

```bash
git add scripts/dump_rollout.py tests/test_isaac_parity.py
git commit -m "test(M4): parity test (two-process dump + offline diff)"
```

### Task M4.4: Measure bridge overhead

**Files:**
- Create: `scripts/profile_bridge_overhead.py`
- Create: `docs/M4_bridge_overhead.md`

- [ ] **Step 1: Write the profiler script**

`scripts/profile_bridge_overhead.py`:
```python
"""Measure per-step latency: raw Isaac env.step vs JaxEnvWrapper.step.

Reports ms/step at the configured num_envs. Writes results to
docs/M4_bridge_overhead.md.
"""
from __future__ import annotations

import argparse
import statistics
import time

import gymnasium as gym
import jax.numpy as jnp
import torch

import factory_jax  # noqa: F401
from factory_jax.bridge.jax_env_wrapper import JaxEnvWrapper


def time_loop(env, action, n_warmup=20, n_measure=200, is_jax=False):
    for _ in range(n_warmup):
        env.step(action)
    torch.cuda.synchronize()
    samples = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        env.step(action)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--n-measure", type=int, default=200)
    args = parser.parse_args()

    # Raw
    env_raw = gym.make("FactoryJax-NutThread-v0", num_envs=args.num_envs)
    env_raw.reset(seed=0)
    a_raw = torch.zeros((args.num_envs, 6), device="cuda")
    raw_ms = time_loop(env_raw, a_raw, args.n_warmup, args.n_measure)
    env_raw.close()

    # Wrapped
    env_wrapped = JaxEnvWrapper(gym.make("FactoryJax-NutThread-v0", num_envs=args.num_envs))
    env_wrapped.reset(seed=0)
    a_jax = jnp.zeros((args.num_envs, 6), dtype=jnp.float32)
    wrapped_ms = time_loop(env_wrapped, a_jax, args.n_warmup, args.n_measure, is_jax=True)
    env_wrapped.close()

    raw_med = statistics.median(raw_ms)
    wrapped_med = statistics.median(wrapped_ms)
    overhead_ms = wrapped_med - raw_med
    overhead_pct = overhead_ms / raw_med * 100

    print(f"raw     : median {raw_med:.3f} ms/step (min {min(raw_ms):.3f}, max {max(raw_ms):.3f})")
    print(f"wrapped : median {wrapped_med:.3f} ms/step (min {min(wrapped_ms):.3f}, max {max(wrapped_ms):.3f})")
    print(f"overhead: {overhead_ms:.3f} ms/step ({overhead_pct:+.1f}%)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run profiler**

```bash
python scripts/profile_bridge_overhead.py --num-envs 128 2>&1 | tee /tmp/bridge_profile.txt
```

- [ ] **Step 3: Write `docs/M4_bridge_overhead.md`**

```markdown
# M4 — Bridge overhead

**Date:** <YYYY-MM-DD>
**Command:** `python scripts/profile_bridge_overhead.py --num-envs 128`
**GPU:** <torch.cuda.get_device_name(0)>

## Numbers

| Path | Median ms/step | Min | Max |
|---|---|---|---|
| Raw Isaac (`env.step`) | | | |
| Wrapped (`JaxEnvWrapper.step`) | | | |
| **Overhead** | **(wrapped - raw)** | | |
| **Overhead %** | **((wrapped - raw) / raw) × 100** | | |

## Interpretation

The overhead is the per-step cost of: DLPack action conversion + `torch.cuda.synchronize()` (×2) + DLPack obs/reward/done conversion. This number is the **denominator** for M5a's wall-clock comparison: any wall-clock gap between JAX PPO and rl_games PPO that's smaller than `overhead_ms × env_steps_per_iter` is "JAX won despite the bridge tax."

## Follow-ups
- If overhead is dominated by `torch.cuda.synchronize()`, profile dropping one of the two syncs in M5b once we trust the stream ordering.
```

- [ ] **Step 4: Commit**

```bash
git add scripts/profile_bridge_overhead.py docs/M4_bridge_overhead.md
git commit -m "feat(M4): bridge overhead profiler + measurement doc"
git push origin main   # end of M4
```

---

## M5a — JAX PPO config-matched to rl_games PPO

**Goal:** Train jax-learning's PPO on NutThread with hyperparameters matched to M2's rl_games config (modulo unmappable knobs like LSTM, documented). Produce the side-by-side comparison.

### Task M5a.1: Extract matched PPO config

**Files:**
- Create: `source/factory_jax/factory_jax/configs/matched_ppo_config.yaml`

**Pre-req:** consult `Research/jax-learning/.context/lessons/algo_port_protocol.md` and `lessons/learner.md` per spec §12.

- [ ] **Step 1: Read both configs side by side**

```bash
diff -y -W 200 \
  source/factory_jax/factory_jax/configs/rl_games_ppo_cfg_reference.yaml \
  /home/stevenman/Desktop/Work/Research/jax-learning/jax_rl/configs/ppo_config.py || true
```
(The latter is python, so diff is imperfect; use it as a side-by-side inspection.)

- [ ] **Step 2: Write the matched config**

`source/factory_jax/factory_jax/configs/matched_ppo_config.yaml`:
```yaml
# Matched PPO config for FactoryJax-NutThread-v0.
# Source of truth: ../configs/rl_games_ppo_cfg_reference.yaml (M2 baseline).
# Every key here either MATCHES rl_games verbatim or is flagged as a documented
# unavoidable difference. See `docs/M5a_jax_vs_rlgames_ppo.md` for the diff.

# === HYPERPARAMETERS — direct mappings from rl_games_ppo_cfg.yaml ===
num_envs: 128
horizon_length: <COPY from rl_games>
mini_epochs: <COPY>
minibatch_size: <COPY>
learning_rate: <COPY>
gamma: <COPY>
gae_lambda: <COPY (rl_games key: tau)>
clip_epsilon: <COPY (rl_games key: e_clip)>
value_clip: <COPY (rl_games key: clip_value)>
entropy_coef: <COPY>
value_coef: <COPY (rl_games key: critic_coef)>
max_grad_norm: <COPY (rl_games key: grad_norm)>

# === NETWORK — partial match (no LSTM in jax-learning PPO) ===
network:
  mlp_units: [512, 128, 64]   # MATCHES rl_games
  activation: elu             # MATCHES rl_games
  recurrent: false            # DIVERGES: rl_games uses LSTM (units=1024, layers=2)

# === DIVERGENCES (recorded; explain in docs/M5a_jax_vs_rlgames_ppo.md) ===
divergences:
  - key: network.rnn
    rl_games: "LSTM units=1024 layers=2 before_mlp=True"
    jax_learning: "none (MLP only)"
    reason: "jax-learning PPO does not implement a recurrent variant"
  # Add more rows here for each rl_games key without a jax-learning equivalent.
```

(Replace `<COPY>` placeholders with actual values from `rl_games_ppo_cfg_reference.yaml`.)

- [ ] **Step 3: Commit**

```bash
git add source/factory_jax/factory_jax/configs/matched_ppo_config.yaml
git commit -m "feat(M5a): matched PPO config (with documented divergences from rl_games)"
```

### Task M5a.2: Write `train_jax_ppo.py`

**Files:**
- Create: `scripts/train_jax_ppo.py`

- [ ] **Step 1: Cross-reference jax-learning's existing PPO trainer**

```bash
head -200 /home/stevenman/Desktop/Work/Research/jax-learning/scripts/train_ppo.py
```
Identify: how it builds the env bundle, instantiates `PPO`, runs the loop, logs metrics. Mirror that pattern but swap in our wrapped Isaac env.

- [ ] **Step 2: Write the training script**

`scripts/train_jax_ppo.py` (skeleton — fill in based on jax-learning's `train_ppo.py` patterns):
```python
"""Train jax-learning PPO on FactoryJax-NutThread-v0 through the JAX bridge.

Config: source/factory_jax/factory_jax/configs/matched_ppo_config.yaml
(matched to M2's rl_games PPO config; divergences documented).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import yaml

import factory_jax  # noqa: F401  registers FactoryJax-NutThread-v0
from factory_jax.bridge.jax_env_wrapper import JaxEnvWrapper
from jax_rl.algos.ppo import PPO
# Additional imports based on jax-learning patterns: configs, training utils, etc.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="source/factory_jax/factory_jax/configs/matched_ppo_config.yaml")
    parser.add_argument("--total-env-steps", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", default="runs/jax_ppo_m5a")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 1) Build env
    env = JaxEnvWrapper(gym.make("FactoryJax-NutThread-v0", num_envs=cfg["num_envs"]))

    # 2) Instantiate PPO (mirror jax-learning's pattern — see jax_rl.algos.ppo.PPO.__init__ signature)
    # ... build optimizer, build PPO algo, initial training state ...

    # 3) Training loop: rollout -> update -> log
    # Track per-phase wall-clock so M5a doc gets filled in
    # ... rollout_seconds_total, update_seconds_total, env_steps_total ...

    # 4) Periodic eval + checkpoint
    # ... use jax-learning's checkpoint manager ...

    # 5) Final eval, save metrics


if __name__ == "__main__":
    main()
```
(Implementation details flow from jax-learning's `train_ppo.py`. Treat that as the template and adjust env construction + obs-dict handling.)

- [ ] **Step 3: Smoke run (1k env steps, sanity only — not a real training run)**

```bash
python scripts/train_jax_ppo.py --total-env-steps 1000 --log-dir /tmp/jax_ppo_smoke
```
Expected: no crashes, reward number printed, log dir populated.

- [ ] **Step 4: Commit (training script alone, no training results yet)**

```bash
git add scripts/train_jax_ppo.py
git commit -m "feat(M5a): train_jax_ppo.py (smoke-tested, full run pending)"
```

### Task M5a.3: Full training run + comparison doc

- [ ] **Step 1: Launch full run**

```bash
python scripts/train_jax_ppo.py --total-env-steps 5_000_000 --log-dir runs/jax_ppo_m5a
```
Monitor reward + wall-clock. Stop on plateau or budget.

- [ ] **Step 2: Run eval against best checkpoint**

```bash
python scripts/play.py --algo jax_ppo --task FactoryJax-NutThread-v0 \
    --checkpoint runs/jax_ppo_m5a/checkpoints/best.ckpt \
    --num-envs 20 --record-video --video-length 400
```
(See M5b.2 for `scripts/play.py`; create it earlier if needed.)

- [ ] **Step 3: Write `docs/M5a_jax_vs_rlgames_ppo.md`**

```markdown
# M5a — jax-learning PPO vs rl_games PPO (config-matched)

## Side-by-side

| Metric | rl_games PPO (M2) | jax-learning PPO (M5a) |
|---|---|---|
| Env steps/sec (rollout only) | | |
| Update steps/sec (gradient only) | | |
| Total seconds per training iteration | | |
| Wall-clock to reach R* (= 80% of M2 final reward) | | |
| Env steps to reach R* | | |
| Final mean ep reward — equal env-step budget | | |
| Final mean ep reward — equal wall-clock budget | | |
| Bridge overhead from M4 (ms/step) | n/a | |

## Plots

- `plots/wallclock_vs_reward.png` — both curves on (wall-clock seconds, reward)
- `plots/envsteps_vs_reward.png` — both curves on (env steps, reward)

## Divergences from rl_games config

| Key | rl_games | jax-learning | Reason |
|---|---|---|---|
| network.rnn | LSTM 1024×2 before MLP | none (MLP only) | jax-learning PPO does not implement a recurrent variant |
| ... | | | |

## Interpretation

(Fill in: which is more sample-efficient? Faster in wall-clock to R*? Was the bridge tax visible? What does this say about JIT vs torch eager / recurrence vs MLP?)
```

- [ ] **Step 4: Commit**

```bash
git add docs/M5a_jax_vs_rlgames_ppo.md docs/plots/* 2>/dev/null
git commit -m "docs(M5a): jax_ppo vs rl_games comparison + plots"
git push origin main   # end of M5a
```

---

## M5b — JAX FlashSAC (the deliverable baseline)

**Goal:** Train FlashSAC on NutThread through the bridge. Produce the checkpoint + eval video that is the bolt-tightening baseline.

### Task M5b.1: Write `train_jax_flashsac.py`

**Files:**
- Create: `scripts/train_jax_flashsac.py`

**Pre-req:** consult `Research/jax-learning/.context/lessons/offpolicy.md`, `lessons/learner.md`, `lessons/manipulation.md` per spec §12.

- [ ] **Step 1: Cross-reference jax-learning's FlashSAC trainer**

```bash
head -300 /home/stevenman/Desktop/Work/Research/jax-learning/scripts/train_flashsac.py
```

- [ ] **Step 2: Write training script**

`scripts/train_jax_flashsac.py` mirrors `train_flashsac.py` from jax-learning. Key differences from M5a's PPO script:
- Off-policy: needs `JaxReplayBuffer`
- Use `jax_rl.configs.env_presets.get_flash_sac_preset(...)` as the starting config; override NutThread-specific knobs only
- FlashSAC's reward scaler updates per env step on CPU side — make sure the wrapper passes scalar reward through unchanged

Skeleton:
```python
"""Train jax-learning FlashSAC on FactoryJax-NutThread-v0 through the JAX bridge.
The DELIVERABLE baseline policy for bolt tightening.
"""
from __future__ import annotations

import argparse
import gymnasium as gym
import jax
import jax.numpy as jnp

import factory_jax  # noqa: F401
from factory_jax.bridge.jax_env_wrapper import JaxEnvWrapper
from jax_rl.algos.flash_sac import FlashSAC, NoiseState
from jax_rl.buffers.jax_replay_buffer import JaxReplayBuffer
from jax_rl.configs.flash_sac_config import FlashSACConfig
from jax_rl.configs.env_presets import get_flash_sac_preset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--total-env-steps", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", default="runs/flashsac_m5b")
    args = parser.parse_args()

    # Use a NutThread-ish preset if available; else start from a generic manipulation preset
    # and override num_envs / obs_dim / action_dim.
    cfg = get_flash_sac_preset("FactoryNutThread") or get_flash_sac_preset("default_manipulation")
    cfg = dataclasses.replace(cfg, num_envs=args.num_envs)

    env = JaxEnvWrapper(gym.make("FactoryJax-NutThread-v0", num_envs=args.num_envs))
    # ... rest: build FlashSAC, replay buffer, run train loop, periodic eval+checkpoint ...


if __name__ == "__main__":
    main()
```

If `get_flash_sac_preset("FactoryNutThread")` returns `None`, we may need to add a preset to `jax_rl/configs/env_presets.py`.

- [ ] **Step 2b: jax-learning divergence procedure (only if Step 2 needed a preset addition)**

Per spec D11:
```bash
cd /home/stevenman/Desktop/Work/Research/jax-learning
git checkout -b isaaclab-factory-jax/flashsac-nutthread-preset
# edit jax_rl/configs/env_presets.py to add a get_flash_sac_preset("FactoryNutThread") entry
git add jax_rl/configs/env_presets.py
git commit -m "feat: add FactoryNutThread FlashSAC preset (isaaclab-factory-jax)"
# do not merge to main
cd /home/stevenman/Desktop/Work/Research/isaaclab-factory-jax
```

Then append a row to `docs/jax_learning_divergences.md`:
```markdown
| <YYYY-MM-DD> | jax_rl/configs/env_presets.py | branch: isaaclab-factory-jax/flashsac-nutthread-preset | added FactoryNutThread preset (lr, batch, network sizes) | M5b needs Factory-specific FlashSAC tuning | TBD after M5b — likely merge back to main if numbers are good |
```

- [ ] **Step 3: Smoke run**

```bash
python scripts/train_jax_flashsac.py --total-env-steps 1000 --log-dir /tmp/flashsac_smoke
```

- [ ] **Step 4: Commit**

```bash
git add scripts/train_jax_flashsac.py
git commit -m "feat(M5b): train_jax_flashsac.py (smoke-tested)"
```

### Task M5b.2: Write `scripts/play.py`

**Files:**
- Create: `scripts/play.py`

- [ ] **Step 1: Write generic player**

`scripts/play.py`:
```python
"""Load a trained policy + record an eval video.

Usage:
    python scripts/play.py --algo flashsac --task FactoryJax-NutThread-v0 \
        --checkpoint runs/flashsac_m5b/checkpoints/best.ckpt \
        --num-envs 20 --record-video --video-length 400
"""
from __future__ import annotations

import argparse
import os

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

import factory_jax  # noqa: F401  registers FactoryJax-NutThread-v0
from factory_jax.bridge.jax_env_wrapper import JaxEnvWrapper


def load_policy(algo: str, checkpoint: str):
    """Return a callable: jax.Array obs -> jax.Array action (deterministic)."""
    if algo == "flashsac":
        from jax_rl.algos.flash_sac import FlashSAC  # noqa
        # Use jax-learning's checkpoint loader; the exact API lives in
        # jax_rl.training.checkpointing.CheckpointManager (see train_flashsac.py).
        # ... load state, return a lambda obs: actor_apply(state.actor_params, obs)
        raise NotImplementedError("fill in based on jax-learning's FlashSAC eval pattern")
    if algo == "jax_ppo":
        from jax_rl.algos.ppo import PPO  # noqa
        raise NotImplementedError("fill in based on jax-learning's PPO eval pattern")
    raise ValueError(f"unknown algo: {algo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=("flashsac", "jax_ppo"), required=True)
    parser.add_argument("--task", default="FactoryJax-NutThread-v0")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-length", type=int, default=400)
    parser.add_argument("--video-dir", default="videos")
    args = parser.parse_args()

    # Make env (with video wrapper if requested)
    env_kwargs = {"num_envs": args.num_envs}
    if args.record_video:
        env_kwargs["render_mode"] = "rgb_array"
    env = gym.make(args.task, **env_kwargs)
    if args.record_video:
        os.makedirs(args.video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(env, video_folder=args.video_dir,
                                      video_length=args.video_length,
                                      step_trigger=lambda s: s == 0)
    env = JaxEnvWrapper(env)

    policy = load_policy(args.algo, args.checkpoint)

    # Roll out
    ep_rewards = np.zeros((args.num_envs,))
    ep_counts = np.zeros((args.num_envs,), dtype=np.int32)
    obs, _ = env.reset(seed=0)
    while ep_counts.min() < args.n_episodes:
        act = policy(obs["policy"])
        obs, r, term, trunc, _ = env.step(act)
        ep_rewards += np.asarray(r)
        done = np.asarray(term | trunc)
        ep_counts += done.astype(np.int32)

    print(f"mean episode reward across {args.num_envs} envs: {(ep_rewards / ep_counts).mean():.3f}")
    env.close()


if __name__ == "__main__":
    main()
```
Fill in `load_policy`'s two branches by referencing the eval/play snippets in `jax-learning/scripts/train_flashsac.py` and `jax-learning/scripts/train_ppo.py` respectively (look for `actor_apply` / `policy_apply` usage and `CheckpointManager.restore_latest()`).

- [ ] **Step 2: Commit**

```bash
git add scripts/play.py
git commit -m "feat(M5b): play.py (checkpoint eval + video recorder)"
```

### Task M5b.3: Full FlashSAC training + eval

- [ ] **Step 1: Launch full run**

```bash
python scripts/train_jax_flashsac.py --total-env-steps 5_000_000 --log-dir runs/flashsac_m5b
```

- [ ] **Step 2: Eval best checkpoint, record video**

```bash
python scripts/play.py --algo flashsac --task FactoryJax-NutThread-v0 \
    --checkpoint runs/flashsac_m5b/checkpoints/best.ckpt \
    --num-envs 20 --record-video --video-length 400
```

- [ ] **Step 3: Write `docs/M5b_flashsac.md`**

```markdown
# M5b — FlashSAC bolt-tightening baseline

## Setup
- Task: `FactoryJax-NutThread-v0`
- num_envs: 128
- Total env steps: <N>
- Config: jax-learning preset `<preset_name>` with overrides `<list>`
- Seed: 0

## Final metrics
- Mean episode reward (eval, 20 envs): <X>
- Success rate (eval): <Y>
- Best checkpoint: `runs/flashsac_m5b/checkpoints/best.ckpt`
- Eval video: `videos/flashsac_eval.mp4`

## Comparison to M2 (rl_games PPO)
- M2 final reward (equal env-step budget): <M2_X>
- M5b final reward (same budget): <X>
- Verdict: **<met / not met>** spec §2 success criterion #6 ("≥ M2's rl_games PPO final number at equal env-step budget").

## Notes / surprises
```

- [ ] **Step 4: Commit**

```bash
git add docs/M5b_flashsac.md videos/flashsac_eval.mp4 2>/dev/null
git commit -m "docs(M5b): FlashSAC bolt-tightening baseline metrics + eval video"
git push origin main   # end of M5b
```

---

## M6 — UR5 swap procedure (docs only)

**Goal:** A documented, code-grounded recipe for swapping Franka → UR5 in this project's task config. No code change.

### Task M6.1: Write `docs/robot_swap.md`

**Files:**
- Create: `docs/robot_swap.md`

- [ ] **Step 1: Cross-reference Isaac's Factory robot config**

```bash
grep -E "robot|articulation|franka|panda" /home/stevenman/Desktop/Work/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py | head -20
grep -rE "franka_panda|FRANKA" /home/stevenman/Desktop/Work/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/ | head -10
find /home/stevenman/Desktop/Work/IsaacLab/source/isaaclab_assets -name "*ur*" -o -name "*UR*" | head -10
```

- [ ] **Step 2: Write the doc**

`docs/robot_swap.md`:
```markdown
# Swapping Franka → UR5 in `FactoryJax-NutThread-v0`

This doc describes the swap procedure without executing it. The goal is to make the swap actionable when we're ready.

## Where Franka enters the env (today)

Isaac's stock Factory env hard-codes the Franka articulation in:
- `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py` — robot articulation cfg block (asset USD path, default joint positions, EE link name)
- `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_control.py` — task-space (operational-space) controller; Franka-tuned Jacobian damping and gain defaults
- `IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/franka.py` — Franka USD + joint name list

The customization seam in our project is `source/factory_jax/factory_jax/tasks/direct/factory_nut_thread/env_cfg.py`. Override the robot cfg on `FactoryJaxNutThreadCfg` to swap robots **without forking** Isaac's env code.

## Files to change for UR5

| File (in this project) | Change |
|---|---|
| `source/factory_jax/factory_jax/tasks/direct/factory_nut_thread/env_cfg.py` | Override `robot` field with a UR5e articulation cfg (USD path + joint defaults + EE link name) |
| (new) `source/factory_jax/factory_jax/assets/ur5e.py` | Define the UR5e articulation cfg modeled on `IsaacLab/source/isaaclab_assets/.../franka.py` |

## UR5 USD asset source

Check `IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/` for existing UR support. If absent, Isaac Sim Assets ships UR5e at `Isaac/Robots/UniversalRobots/ur5e/ur5e.usd` (verify via the Isaac Sim asset browser; path may have changed). Source from there.

## Joint and EE differences

- Franka: 7 DoF + Panda gripper, EE link `panda_hand` (or `panda_grip_site` depending on cfg)
- UR5e: 6 DoF, no built-in gripper, EE link `tool0` (or `wrist_3_link` depending on attachment)
- Joint name lists differ: replace Franka's `[panda_joint1, ..., panda_joint7]` with UR5's `[shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint]`
- Default joint positions: pick a home-pose that places the EE above the bolt initial position (see Factory env's reset distribution)

## Control implications

Factory uses a task-space controller (`factory_control.py`). UR5 has no redundancy (6 DoF, 6-DoF task), so:
- DLS (Damped Least Squares) damping coefficient likely needs to be smaller than Franka's default — too much damping causes near-singular motions to stall
- Workspace is smaller and reach is different; the reset distribution for the nut/bolt may need to be re-centered
- No null-space task to exploit

## Gripper

Factory's NutThread reward depends on the gripper closing on the nut. UR5 has no default gripper. Options (pick one when implementing):
- Add a Robotiq 2F-85 USD (Isaac Sim Assets includes it)
- Replace the gripper-grasp success condition with a virtual constraint (suction or attach-on-contact)

## Known unknowns
- Whether IsaacLab 2.3.2 ships a UR5e USD in `isaaclab_assets/robots/`. If not, sourcing the USD is step zero.
- Factory's contact-rich tuning is Franka-specific; expect to retune contact stiffness/damping for UR5 even if everything else is correct.

## How to verify a swap (when implementing — out of scope for this spec)
1. Random agent run with UR5 cfg, video — confirms USD loads and robot moves.
2. Re-run M2 rl_games training with UR5 to see if the env is still trainable.
3. Re-run M5b FlashSAC; expect a fresh tuning cycle.
```

- [ ] **Step 3: Verify file paths and field names in the doc by cross-ref**

```bash
# spot-check every IsaacLab path mentioned in the doc actually exists
for p in \
  source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py \
  source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_control.py \
  source/isaaclab_assets/isaaclab_assets/robots/franka.py; do
  test -e /home/stevenman/Desktop/Work/IsaacLab/$p && echo "OK: $p" || echo "MISSING: $p"
done
```
Fix any "MISSING" entries by adjusting paths in the doc.

- [ ] **Step 4: Commit**

```bash
git add docs/robot_swap.md
git commit -m "docs(M6): UR5 swap procedure (paths verified, implementation out of scope)"
git push origin main   # end of M6 — plan complete
```

---

## Final verification checklist

Run before declaring the plan complete:

- [ ] `pytest tests/ -v` — all unit + parity tests pass
- [ ] `docs/M1_sanity.md`, `docs/M2_baseline.md`, `docs/M4_bridge_overhead.md`, `docs/M5a_jax_vs_rlgames_ppo.md`, `docs/M5b_flashsac.md`, `docs/robot_swap.md`, `docs/jax_learning_divergences.md` all exist and are filled in
- [ ] M5b reports a final reward ≥ M2's rl_games PPO final reward at equal env-step budget (spec §2 success criterion #6)
- [ ] Eval video at `videos/flashsac_eval.mp4` shows the policy attempting (or succeeding at) the thread-on-bolt motion
- [ ] `docs/jax_learning_divergences.md` lists every jax-learning change made on a topic branch (with branch name) or ported into this project
- [ ] Git log shows commits per milestone tag (`(M1)`, `(M2)`, `(M3)`, `(M4)`, `(M5a)`, `(M5b)`, `(M6)`)

When all boxes ticked: spec §2 success criteria 1–7 are all satisfied. The bolt-tightening RL baseline exists.
