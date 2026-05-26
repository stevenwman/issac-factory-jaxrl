# .context/

Mirror of `jax-learning/.context/` for this project. Both files here and in `jax-learning/.context/` are project-scoped knowledge stores.

## Layout

- `lessons/` — reusable findings from this project (e.g. `isaac_install.md`, `dlpack_bridge.md`, `flashsac_isaac.md`). One file per discrete topic.
- `journal/` — daily/per-session notes. Filename convention: `YYYY-MM-DD.md`. Append-only.

## When to write here vs in `docs/`

- `docs/M<N>_*.md` — milestone deliverables (metrics, baselines, comparisons). Stable artifacts referenced by the spec.
- `.context/lessons/*.md` — "next time we hit X, do Y" insights. Not tied to a milestone.
- `.context/journal/YYYY-MM-DD.md` — what we did + decisions made today. Not curated.

## Cross-references

`jax-learning/.context/` is the read-only debugging corpus per spec §12. Do not duplicate jax-learning lessons here; link to them when relevant.
