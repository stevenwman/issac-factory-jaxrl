"""Gym task registrations. Import after AppLauncher has bootstrapped Isaac.

Defines `FactoryJaxNutThreadCfg` (subclass of Isaac's stock NutThread cfg)
and registers `FactoryJax-NutThread-v0` so we own the gym ID + customization
seam without forking Isaac's env code.
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
        "rl_games_cfg_entry_point": "isaaclab_tasks.direct.factory.agents:rl_games_ppo_cfg.yaml",
    },
)
