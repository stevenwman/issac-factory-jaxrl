"""factory_jax: thin extension that re-registers Factory NutThread under our own gym ID.

Importing this module triggers `gym.register("FactoryJax-NutThread-v0", ...)`.
The task config is a pure subclass of Isaac's `FactoryTaskNutThreadCfg` so we own
the gym ID + a customization seam (robot swap, controller, etc.) without forking
Isaac's env code.
"""
from __future__ import annotations

import gymnasium as gym

from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskNutThreadCfg


class FactoryJaxNutThreadCfg(FactoryTaskNutThreadCfg):
    """Customization seam. No overrides yet — pure subclass."""
    pass


gym.register(
    id="FactoryJax-NutThread-v0",
    entry_point="isaaclab_tasks.direct.factory.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryJaxNutThreadCfg,
        # Reuse Isaac's stock rl_games config for our task ID
        "rl_games_cfg_entry_point": "isaaclab_tasks.direct.factory.agents:rl_games_ppo_cfg.yaml",
    },
)
