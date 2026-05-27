"""MVP wrapper: jax-learning PPO on FactoryJax-NutThread-v0 via our isaaclab backend.

Usage:
    NO_COLOR=1 python scripts/train_jax_ppo.py --num_envs 64 --total_env_steps 50000 --wandb

Phase 1 findings applied:
- train() is at scripts/train_ppo.py (not a package); import via sys.path.insert into scripts/
- Preset "CartpoleBalance" is the minimal stop-gap (confirmed in Task 1.2 journal)
- episode_length=450 (Factory NutThread: 30s @ decimation=8, dt=1/120s; Task 1.3 verified)
- eval_every_n_episodes=999_999_999 disables scheduled eval (D16); backend_kind="isaaclab"
  falls through to the Brax evaluate path which won't work — must not trigger
- handle_truncation already True in CartpoleBalance preset; Factory term==trunc always (Task 1.3)
- onpolicy_collect clips actions at line 104; no duplicate clip needed here
"""
import argparse
import dataclasses
import os
import sys

# === D13: GPU memory — must set BEFORE any jax import ===
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

# === jax-learning path-imports (BEFORE AppLauncher: pure-python, no pxr) ===
# scripts/ has no __init__.py so it is not a package; path-insert is the only safe import.
_JAX_LEARNING = os.environ.get(
    "JAX_LEARNING_PATH",
    "/home/stevenman/Desktop/Work/Research/jax-learning",
)
sys.path.insert(0, _JAX_LEARNING)
sys.path.insert(0, os.path.join(_JAX_LEARNING, "scripts"))  # exposes train_ppo.py

# === AppLauncher (must precede any isaaclab/factory_jax import) ===
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="JAX PPO MVP on FactoryJax-NutThread-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--total_env_steps", type=int, default=500_000)
parser.add_argument("--wandb_name", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force headless — this script is always run without a display
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# === Now safe to import isaaclab + register backend ===
import factory_jax.tasks   # noqa: F401  registers FactoryJax-NutThread-v0 gym ID
import factory_jax.backend  # noqa: F401  register_backend("isaaclab", make_isaaclab_bundle)

# === Build TrainConfig from CartpoleBalance preset + mandatory overrides ===
from jax_rl.configs import get_preset
from train_ppo import train  # imported via sys.path insert into jax-learning/scripts/

cfg = get_preset("CartpoleBalance")
cfg = dataclasses.replace(
    cfg,
    env_name="IsaacLab/FactoryJax-NutThread-v0",  # routes to our isaaclab backend
    num_envs=args_cli.num_envs,
    total_timesteps=args_cli.total_env_steps,
    episode_length=450,               # Factory NutThread: 30s @ decimation=8, dt=1/120s
    eval_every_n_episodes=999_999_999,  # D16: disable in-process eval; Brax eval path won't work
)

# handle_truncation must be True — Factory _get_dones returns (time_out, time_out)
# so term==trunc always; GAE bootstrap requires truncation masking (Task 1.3).
# CartpoleBalance preset defaults to True, but be explicit in case preset changes.
if hasattr(cfg, "handle_truncation") and not cfg.handle_truncation:
    cfg = dataclasses.replace(cfg, handle_truncation=True)

# Silence jax DEBUG spam (not useful, clutters log)
import logging
logging.getLogger("jax").setLevel(logging.WARNING)

# === Run training ===
# IMPORTANT: print exception trace BEFORE simulation_app.close(), because Isaac's
# headless shutdown can hang on get_timeline_interface (observed in M5a smoke).
# A hung close() with no traceback printed first hides the real failure.
import traceback
try:
    train(
        cfg,
        seed=args_cli.seed,
        use_wandb=args_cli.wandb,
        wandb_project="isaaclab-factory-jax",
    )
except Exception:
    print("=" * 80, flush=True)
    print("TRAINING RAISED:", flush=True)
    traceback.print_exc()
    print("=" * 80, flush=True)
finally:
    # D17: clean Isaac shutdown (may hang on shutdown — known kit bug, log captures
    # the train traceback above regardless).
    simulation_app.close()
