# CLAUDE.md

This repository is an Agent Skill bundle, not an application. Read
`AGENTS.md` first for routing rules.

## Layout

- `skills/kernel/`: CUDA, PTX, Triton, CUTLASS, DeepGEMM, kernel profiling,
  persistent-kernel tuning, performance modeling, and kernel references.
- `skills/framework/`: SGLang and framework/runtime-facing skills.
- `skills/other/`: communication, AMD/ROCm, local knowledge bases, and
  non-GPU specialties.
- `skills/routers/`: catch-all and domain routers.
- `repos/`: external skill catalogs as git submodules.
- `repos/tilelang-cuda-skills/`: TileLang kernel authoring, debugging,
  optimization, profiling, and design skills.
- `kernel-design-agents/`: KDA workflow plus the nested `KernelWiki` skill.

The installed skill name is independent of its repository path. For example,
`skills/kernel/cuda-skill` installs as `cuda-skill`.

## Common Commands

```bash
# Initialize and update submodules and upstream source checkouts.
bash bootstrap.sh --agent codex

# Update one source or submodule.
bash update-repos.sh triton
bash update-repos.sh kernel-design-agents
bash update-repos.sh tilelang-cuda-skills
bash update-repos.sh nvidia-skills

# Install into an isolated directory for validation.
bash install.sh --dest /tmp/agent-gpu-skills-test --no-veloq

# Refresh CUDA documentation.
python3 scrape_docs.py all --force

# Refresh AMD documentation cache (local and git-ignored).
python3 skills/other/amd-gpu-docs/scripts/sync_amd_docs.py --profile core
```

## Maintenance Rules

- Keep local skills in one of the four `skills/` categories.
- Add or change a router whenever a new domain skill is introduced.
- Keep the parallel `SKILL_NAMES` and `SKILL_DIRS` arrays in `install.sh`
  aligned; the script must remain compatible with Bash 3.2.
- Prefer profile evidence over performance speculation.
- Do not commit AMD PDFs, documentation caches, sparse checkouts, or generated
  profile output.
- Local skills win over same-named skills from external catalogs.
- The TileLang suite's upstream `cuda_skill` is intentionally not installed;
  use the local `cuda-skill` as the canonical CUDA entry.
- When an upstream repository moves a referenced path, update
  `update-repos.sh`, the skill documentation, and install verification checks
  together.
