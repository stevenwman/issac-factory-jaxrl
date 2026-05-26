# Controller and IK survey (M0.2 — research, not implementation)

> **Status:** research / docs only — no code changes.
> **Decision (TL;DR):** use Isaac's stock Factory controller unchanged for the M5b baseline. Controller swap deferred to a follow-on spec.

---

## Isaac's built-in controllers (as of IsaacLab 2.3.2)

### Controllers (`isaaclab/source/isaaclab/isaaclab/controllers/`)

| File | Controller | Action interface | Used by Factory? | Notes |
|---|---|---|---|---|
| `differential_ik.py` | `DifferentialIKController` | Position (N,3), pose-abs (N,7), or pose-rel delta (N,6) | No (not used by Factory tasks directly) | Four IK backends: `pinv`, `svd`, `trans`, `dls`. Outputs target joint positions (N, num_joints). Configured via `DifferentialIKControllerCfg`. |
| `operational_space.py` | `OperationalSpaceController` | Pose-abs (N,7), pose-rel (N,6), wrench-abs (N,6), or combinations; optional +6 stiffness / +6 damping-ratio appended when `impedance_mode="variable_kp"/"variable"` | No (separate from Factory's custom controller) | Full OSC: Jacobian transpose, optional inertial decoupling (`arm_mass_matrix`), gravity compensation, null-space position control. Outputs joint torques (N, num_DoF). Stiffness/damping are cfg fields and can be policy-variable. |
| `joint_impedance.py` | `JointImpedanceController` | Joint positions (N, num_dof); optional +stiffness / +damping appended in `variable_kp` / `variable` mode | No | Joint-space spring-damper. Optionally inertial compensation (`mass_matrix`) + gravity correction. Outputs desired torques (N, num_dof). |
| `rmp_flow.py` | `RmpFlowController` | EE pose (N, 7): position + quat (w,x,y,z) | No | Wraps NVIDIA LULA / RMPFlow. Requires `isaacsim.robot_motion.lula` extension. **Not batched** — iterates one robot at a time in a Python loop via `ArticulationMotionPolicy`. Action dim = 7. Outputs (dof_pos_target, dof_vel_target). |
| `pink_ik/pink_ik.py` | `PinkIKController` | Per-task targets (EE pose, joint posture, etc.) configured via `PinkIKControllerCfg.variable_input_tasks` | No | Wraps [pink](https://github.com/stephane-caron/pink) QP-based differential IK (pinocchio backend). **Not batched** — single-env numpy, calls `solve_ik(..., solver="daqp")` per step. Useful for teleoperation / scripted demos, not for RL training at scale. |

### Factory task-space controller (`isaaclab_tasks/.../factory/factory_control.py`)

| File | Controller | Action interface | Used by Factory? | Notes |
|---|---|---|---|---|
| `factory_control.py` | `compute_dof_torque` (free function) | Absolute EE pose target: `ctrl_target_fingertip_midpoint_pos` (N,3) + `ctrl_target_fingertip_midpoint_quat` (N,4); gains `task_prop_gains` (N,6) + `task_deriv_gains` (N,6) passed in per-call | Yes — all three tasks (NutThread, PegInsert, GearMesh) | Geometric Jacobian task-space PD + operational-space mass-matrix inertia decoupling + 7-DoF null-space control. Torques clamped ±100 Nm. See detailed breakdown below. |

**factory_control.py — detailed breakdown**

Control law (`compute_dof_torque`):

1. **Task-space PD wrench:**
   ```python
   task_wrench[:, 0:3] = task_prop_gains[:, 0:3] * pos_error
                         + task_deriv_gains[:, 0:3] * (0.0 - fingertip_linvel)
   task_wrench[:, 3:6] = task_prop_gains[:, 3:6] * axis_angle_error
                         + task_deriv_gains[:, 3:6] * (0.0 - fingertip_angvel)
   ```
   Target velocity is zero (no velocity feedforward). `task_prop_gains` and `task_deriv_gains` are passed in per call, so they can vary per environment / step — the env sets them from `cfg.ctrl`.

2. **Map to joint torques via J^T:**
   ```python
   dof_torque[:, 0:7] = (jacobian_T @ task_wrench.unsqueeze(-1)).squeeze(-1)
   ```

3. **Null-space stabilization** (7-DoF redundancy):
   ```python
   u_null = cfg.ctrl.kd_null * -dof_vel[:, :7] + cfg.ctrl.kp_null * distance_to_default_dof_pos
   torque_null = (I - J^T @ j_eef_inv) @ (arm_mass_matrix @ u_null)
   ```
   Uses `kp_null=10.0`, `kd_null=6.3246` (from `CtrlCfg`). Default posture defined in `cfg.ctrl.default_dof_pos_tensor`.

4. **Torque clamp:** `torch.clamp(dof_torque, min=-100.0, max=100.0)`

**Key cfg fields from `CtrlCfg` (`factory_env_cfg.py`):**

| Field | Value | Role |
|---|---|---|
| `default_task_prop_gains` | `[100,100,100,30,30,30]` | Default Kp (lin, rot) |
| `reset_task_prop_gains` | `[300,300,300,20,20,20]` | Kp during resets |
| `reset_rot_deriv_scale` | `10.0` | Derivative scale at reset |
| `ema_factor` | `0.2` | EMA smoothing on action |
| `kp_null` | `10.0` | Null-space position gain |
| `kd_null` | `6.3246` | Null-space damping gain |
| `pos_action_bounds` | `[0.05, 0.05, 0.05]` m | Max delta-pos per step |
| `rot_action_bounds` | `[1.0, 1.0, 1.0]` rad | Max delta-rot per step |

**Action space (from `factory_env_cfg.py`):**
- `action_space = 6` — the policy emits a 6-DoF delta pose (3 pos + 3 rot in axis-angle).
- The env applies EMA smoothing (`ema_factor=0.2`), then converts delta → absolute target pose before calling `compute_dof_torque`.

**No F/T sensing in default obs:** `OBS_DIM_CFG` and `STATE_DIM_CFG` in `factory_env_cfg.py` do not include any contact force or wrench observation. There is no wrist F/T sensor wired into the default observation.

**Actuators (arm):** Factory configures arm joints with `ImplicitActuatorCfg(stiffness=0.0, damping=0.0)` — pure torque control mode. PhysX receives raw joint torques from `compute_dof_torque`. Gripper uses `ImplicitActuatorCfg(stiffness=7500.0, damping=173.0)` — position-controlled.

---

### Actuators (`isaaclab/source/isaaclab/isaaclab/actuators/`)

| File | Actuator | Interface | Used by Factory? | Notes |
|---|---|---|---|---|
| `actuator_pd.py` | `ImplicitActuator` | Joint positions / velocities / efforts → passthrough | Yes (arm + gripper) | PD integrated by PhysX (continuous-time). For Factory arm: stiffness=0, damping=0 (torque passthrough). For gripper: stiffness=7500, damping=173. |
| `actuator_pd.py` | `IdealPDActuator` | Joint positions → joint torques | No | Explicit PD: τ = kp·(q_des−q) + kd·(dq_des−dq) + τ_ff. Effort-limited. |
| `actuator_pd.py` | `DCMotor` | Joint positions → joint torques | No | Extends IdealPD with velocity-based torque-speed saturation curve. |
| `actuator_pd.py` | `DelayedPDActuator` | Joint positions → joint torques (delayed) | No | Extends IdealPD with configurable command delay buffer. |
| `actuator_pd.py` | `RemotizedPDActuator` | Joint positions → joint torques (angle-dependent limits) | No | Extends DelayedPD with lookup-table torque limits (e.g., tendon-driven joints). |
| `actuator_net.py` | `ActuatorNetLSTM` | Desired joint positions → joint torques (LSTM) | No | Learned actuator model (recurrent). TorchScript network. Useful for legged robots. |
| `actuator_net.py` | `ActuatorNetMLP` | Desired joint positions + history → joint torques (MLP) | No | Learned actuator model (feedforward + history buffer). TorchScript network. |

---

## External solvers surveyed

| Tool | Backend | Diff'ble | Batched / JIT | Per-step at num_envs=128? | Notes |
|---|---|---|---|---|---|
| **mink** (kevinzakka/mink) | MuJoCo + QP (OSQP/clarabel) | No (QP interior-point) | No — single env, numpy/scipy, Python loop | No — QP per env × 128 envs would dominate step time; not designed for batched RL training | Kevin Zakka's differential IK for MuJoCo. Elegant API, model-based (MJCF), not MuJoCo-bound in principle but tightly coupled. No JAX path. |
| **frax** (StanfordASL/frax) | JAX | Yes (through jax.grad / jax.jit) | Yes — designed for batched, JIT-compiled IK | Yes — vmapped over environments; < 1 ms/batch at 128 envs on GPU expected | JAX-native forward/inverse kinematics. Robot defined via URDF→JAX. Supports Franka and custom robots. Closest fit to our JAX RL setup. Does not require MuJoCo. |
| **Isaac OSC** (`operational_space.py`) | PyTorch / PhysX | No (eager) | Yes — batched over num_envs natively | Yes — already running inside simulation | Mature, well-tested. Supports inertial decoupling, gravity compensation, null-space control, variable impedance. No additional deps. |
| **Isaac diff-IK** (`differential_ik.py`) | PyTorch / PhysX | No (eager) | Yes — batched | Yes — already running | Four IK backends (pinv, svd, trans, dls). Outputs joint positions; actuator applies PD internally. |
| **Isaac Factory controller** (`factory_control.py`) | PyTorch / PhysX | No (eager) | Yes — batched | Yes — already running (decimation=8) | Task-space PD + inertia decoupling + null-space. Action = 6-DoF delta pose. The baseline we are preserving. |
| **pink** (stephane-caron/pink, via `pink_ik/`) | pinocchio + QP (daqp) | No | No — single env, numpy | No — not suitable for RL training | Already integrated in IsaacLab as `PinkIKController`. Useful for scripted demos / teleoperation. |

---

## Impedance / admittance control — feasibility

### What Factory currently exposes

Factory's `factory_control.py` implements task-space PD, which is structurally impedance control (spring-damper in task space). However:

- **Stiffness/damping are passed as arguments per call**, not baked into a cfg-locked constant. `task_prop_gains` and `task_deriv_gains` are (N,6) tensors. The policy could in principle modulate these — but the current training setup passes them from `cfg.ctrl.default_task_prop_gains`, not from the policy output.
- **No F/T sensing in observations.** `OBS_DIM_CFG` contains fingertip pose, linear velocity, and angular velocity. There is no contact wrench or force sensor in the default obs vector. Without F/T sensing, closed-loop force control (true admittance control) is not possible without first adding a contact sensor to the env config.
- **Dead-zone simulation is supported** (`dead_zone_thresholds` argument to `compute_dof_torque`), which is used to model actuator unreliability near zero force.

### Path A — Stay inside Isaac with variable stiffness/damping + F/T sensing

**What it takes:**
1. Add a contact/F/T sensor to `factory_env_cfg.py` (Isaac's `ContactSensorCfg` or a wrist frame force sensor).
2. Add the wrench observation to `OBS_DIM_CFG` / `STATE_DIM_CFG`.
3. Change the policy output from 6-DoF delta pose to 6-DoF delta pose + 6 stiffness (or delta pose + 6 stiffness + 6 damping-ratio).
4. Pass policy-output gains directly to `compute_dof_torque` as `task_prop_gains` / `task_deriv_gains`.

**Fit:** Stays entirely inside PyTorch/PhysX. No new deps. Conceptually the lowest-risk path. The `OperationalSpaceController` (`impedance_mode="variable_kp"` or `"variable"`) already implements this pattern and could replace Factory's custom controller with minor wiring.

**Limitation:** No feedforward dynamics compensation beyond what Factory already does (no feedforward wrench term for contact forces). True closed-loop force control requires the F/T feedback signal.

### Path B — Swap in frax-style JIT IK

**What it takes:**
1. Integrate frax (JAX) to compute joint targets from Cartesian impedance targets.
2. Policy emits a Cartesian impedance target (pose + stiffness); frax converts to joint angles; Isaac applies via `ImplicitActuatorCfg` in position mode.
3. Requires bridging JAX ↔ PyTorch (DLPack or `.numpy()` round-trip) at each step.

**Fit:** Maximum differentiability — gradient can flow through the IK step if desired. frax is batched and JIT-compiled, so it should be fast enough at num_envs=128. The JAX/PyTorch boundary is manageable but adds engineering overhead. This path is best suited if the RL training loop itself moves to JAX (e.g., JAXRL / Brax-style training).

**Limitation:** frax robot coverage should be verified for the Franka Panda specifically before committing. Franka is a standard robot in robotics research but coverage of exact joint limits / inertia model needs confirmation.

### Path C — Fully external solver in JAX

**What it takes:**
1. Move the entire control stack (IK + impedance + contact model) to JAX.
2. Policy, controller, and IK all live in JAX; only physics integration remains in PhysX (via Isaac).
3. Maximum flexibility: end-to-end differentiability, custom contact models, arbitrary impedance parameterization.

**Fit:** Highest capability ceiling, highest engineering cost. The JAX↔PhysX boundary means control signals still need to cross to/from PyTorch at every physics step. Most justified if we are building a custom physics simulator or need to differentiate through contact dynamics — neither of which is in scope for this project.

**Cost:** Significant upfront investment. Not recommended until Path A or B prove insufficient.

---

## Decision (this spec)

**For the NutThread M5b baseline: use Isaac's stock Factory controller unchanged.**

Specifically: `factory_control.py`'s `compute_dof_torque` is called as-is, with `task_prop_gains` and `task_deriv_gains` set from `cfg.ctrl.default_task_prop_gains` (not from the policy). Action space remains 6-DoF delta pose. No F/T sensor, no variable impedance.

**Reasons:**

1. **M5b vs M2 comparison stays apples-to-apples.** The M2 baseline used Factory's stock controller. Any controller change would confound the comparison between the Isaac-based M2 baseline and the JAX-trained M5b agent.
2. **Custom controller adds a variable.** The research question in M5a/M5b is about the training loop (JAX RL vs Isaac RL), not the controller. Adding a controller change mixes two variables simultaneously.
3. **Impedance control is more useful when we have F/T sensing.** Without a wrist force sensor in the obs, a variable-stiffness policy has no feedback signal to exploit. Adding F/T sensing + variable impedance is a self-contained change that warrants its own spec and experiment.
4. **Factory's controller is already inertia-decoupled and null-space stabilized.** It is not a naive PD controller — it already handles the key stability concerns for precision assembly tasks.

**For follow-on work:** open a separate spec ("controllers-and-ik") after M5b results are in. Recommended priority order:

- **Path A first:** add F/T sensing + variable stiffness output, staying entirely in Isaac. Low risk, validates whether compliance helps.
- **Path B if going JAX:** frax-style JIT IK makes sense if the training loop moves fully to JAX. Evaluate frax's Franka coverage before committing.
- **Path C deferred:** only if contact-differentiability becomes a hard requirement.

---

## Open questions

1. **Does frax cover the Franka Panda specifically?** The StanfordASL/frax repo documents general URDF loading — Franka Panda is a common target but the exact joint limit / inertia fidelity vs the `franka_mimic.usd` used by Factory should be verified before adopting frax in Path B.

2. **Does Isaac's `OperationalSpaceController` support contact-aware compliance?** The implementation (`operational_space.py`) accepts `current_ee_force_b` for closed-loop force control (`contact_wrench_stiffness_task`). However it only uses the linear force component (not the full 6-DoF wrench), and the sensor would need to be wired into the env. This is a Path A enabler — confirm whether the existing `ContactSensorCfg` in IsaacLab provides the required signal format.

3. **What's frax's differentiability boundary in practice?** frax is described as differentiable through IK, but the boundary with PhysX (which is not differentiable) means end-to-end gradients stop at the actuation layer. Confirm whether this matters for the intended training algorithm (it does not matter for PPO/SAC; it matters for model-based / trajectory optimization methods).

4. **Is `dead_zone_thresholds` used in production Factory training?** The `compute_dof_torque` signature accepts it but the M2 published results may not have used it. Clarify before M5b to ensure identical conditions.

---

## Files cited (all verified to exist)

- `/home/stevenman/Desktop/Work/IsaacLab/source/isaaclab/isaaclab/controllers/differential_ik.py`
- `/home/stevenman/Desktop/Work/IsaacLab/source/isaaclab/isaaclab/controllers/differential_ik_cfg.py`
- `/home/stevenman/Desktop/Work/IsaacLab/source/isaaclab/isaaclab/controllers/operational_space.py`
- `/home/stevenman/Desktop/Work/IsaacLab/source/isaaclab/isaaclab/controllers/operational_space_cfg.py`
- `/home/stevenman/Desktop/Work/IsaacLab/source/isaaclab/isaaclab/controllers/joint_impedance.py`
- `/home/stevenman/Desktop/Work/IsaacLab/source/isaaclab/isaaclab/controllers/rmp_flow.py`
- `/home/stevenman/Desktop/Work/IsaacLab/source/isaaclab/isaaclab/controllers/pink_ik/pink_ik.py`
- `/home/stevenman/Desktop/Work/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py`
- `/home/stevenman/Desktop/Work/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd_cfg.py`
- `/home/stevenman/Desktop/Work/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_net.py`
- `/home/stevenman/Desktop/Work/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_control.py`
- `/home/stevenman/Desktop/Work/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py`
