# CLAUDE.md

This repository is an Agent Skill bundle, not an application. Read
`AGENTS.md` first for routing rules.

## Layout

- `skills/kernel/`: CUDA, PTX, Triton, CUTLASS, DeepGEMM, kernel profiling,
  persistent-kernel tuning, performance modeling, and kernel references.
- `skills/framework/`: SGLang, verl (RL post-training), train-inference
  consistency (from vLLM), and framework/runtime-facing skills.
- `skills/other/`: communication, AMD/ROCm, local knowledge bases, and
  non-GPU specialties.
- `skills/routers/`: catch-all and domain routers, plus `agent-skill-router`
  (broad router, generated inventory, and `scripts/route_skill.py`).
- `repos/`: external skill catalogs as git submodules, plus git-ignored
  checkouts of the external sources listed in `sources/external-sources.conf`.
- `sources/external-sources.conf`: tracked manifest of non-submodule external
  sources (ref, mode, runtime, provenance). Consumed by `install-external.sh`
  and `install.sh`.
- `tests/`: unittest suites for the installers and the router.
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
bash update-repos.sh tech-doc-style-chinese   # upstream clone, for diffing only
bash update-repos.sh vllm                     # pinned sparse vLLM checkout for train-inference-consistency

# Test the customized Chinese technical-writing skill.
for t in skills/other/tech-doc-style-chinese/tests/test_*.py; do python3 "$t"; done

# Install into an isolated directory for validation (skills only; no VeloQ, no runtimes, no real HOME writes).
bash install.sh --dest /tmp/agent-gpu-skills-test --no-veloq

# Fetch external sources (sources/external-sources.conf) and install their runtime CLIs.
bash install-external.sh            # clone missing checkouts, install runtimes
bash update-repos.sh external       # update to pinned refs
bash install-rea-mcp.sh --agent codex   # REA MCP for codex: rea setup --client codex --skill=false --yes

# Repository self-checks (no network, temp dirs only).
bash -n bootstrap.sh install.sh update-repos.sh install-veloq.sh install-external.sh install-rea-mcp.sh scripts/node-preflight.sh
python3 -m unittest discover -s tests -p 'test_install_integrations.py' -v
python3 -m unittest discover -s tests -p 'test_skills_router.py' -v

# Regenerate and verify the router's skill inventory after changing checkouts.
python3 skills/routers/agent-skill-router/scripts/build_inventory.py
python3 skills/routers/agent-skill-router/scripts/build_inventory.py --check

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
- `skills/other/tech-doc-style-chinese` overrides the upstream clone in
  `repos/tech-doc-style-chinese`; its `README.md` lists which files are
  verbatim upstream copies and which are customized. Keep that table accurate.
- `skills/framework/train-inference-consistency` is a pinned copy of vLLM's
  `.agents/skills/train-inference-consistency`; its sparse vLLM checkout lives
  in the skill's git-ignored `repos/vllm`. Its `README.md` lists the local
  modifications and the sync procedure.
- The TileLang suite's upstream `cuda_skill` is intentionally not installed;
  use the local `cuda-skill` as the canonical CUDA entry.
- When an upstream repository moves a referenced path, update
  `update-repos.sh`, the skill documentation, and install verification checks
  together.
- A source in `sources/external-sources.conf` needs a 40-character `ref`, a
  `mode`, and a `runtime` chosen after reading its README, manifests, and
  license. Record the findings in `provenance`. `pending` is only for a source
  that has not been inspected; a real install must fail on it, never report it
  as installed. Use a version-pinned npm spec (`npm:pkg@version:bin`) for CLIs.
- `install.sh` never lets a catalog overwrite a local skill, a user-level
  symlink, or a name claimed by an earlier catalog. Keep the protected names
  (`claude-delegate`, `codex-background`) in `PROTECTED_SKILL_NAMES` when adding
  new delegate-style skills.
- After changing `install.sh` arrays, the external source manifest, or the
  checked-out catalogs, regenerate `skills/routers/agent-skill-router/references/inventory.tsv`
  and make sure `build_inventory.py --check` passes. A real `bootstrap.sh` run
  does this automatically as its last preparation step; `--dest` never does.
- REA MCP registration is scoped: Codex only, `rea setup --client codex
  --skill=false --yes --json`, REA's own backup logic. Never add
  `--all-detected` or `--install-hopper`, and never parse or edit a user's
  Codex config from our scripts.
- Node engine rules for npm runtimes live in `scripts/node-preflight.sh`. They
  follow the engines of the published npm package that the installer installs,
  not the source checkout's `package.json`, which can carry newer engines under
  the same version number. Do not change global Node versions from installer code.
- Bash constructs: write `${VAR}` before any non-ASCII punctuation (for example
  `${url}（`). Bash on macOS can read the first byte of a full-width character as
  part of the variable name.
