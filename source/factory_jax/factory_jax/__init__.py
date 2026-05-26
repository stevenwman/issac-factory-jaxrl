"""factory_jax: Factory NutThread task + JAX bridge.

`factory_jax.bridge.*` is Isaac-free (just torch + jax). Import freely.

To register the gym task, explicitly `import factory_jax.tasks` AFTER you have
launched Isaac via `AppLauncher` (per Isaac's bootstrap order).
"""
