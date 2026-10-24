---
name: agent-skill-router
description: >
  Broad router for the whole agent-gpu-skills bundle outside the narrow GPU routers: GPU
  kernels, profiling, frameworks and serving (via the GPU routers), scientific papers
  (paper reading, literature search, paper review, rebuttal, submission checks, Zotero
  libraries), Chinese technical writing, general software design and code review (TDD,
  architecture, diagnosis, issues, plans), and the four integrations added in
  sources/external-sources.conf (WeChat wx-cli, mattpocock skills, sepia, rea). Use when the
  user does not name a skill and the domain is unclear or mixed, or when the task spans
  several domains. Also use when the user names a skill, to check that it is installed.
  中文触发词：不知道用哪个 skill、论文/文献/Zotero/审稿/rebuttal、技术文章润色、代码审查、
  TDD、架构、微信聊天记录、skill 路由。
---

# Agent Skill Router

Use this skill to choose one primary skill (plus at most two companions) from the
bundle before answering. It does not answer the domain question itself.

## Procedure

1. **Explicit request first.** If the user names a skill (`$name`, `` `name` ``,
   `name skill`, `skill: name`), use that skill. If it is not installed, say so and
   show its `requires`; do not substitute another skill silently.
2. **Route with evidence.** Run the bundled router from this skill's directory:

   ```bash
   python3 scripts/route_skill.py "<the user's request, verbatim>" --artifact <path> ...
   python3 scripts/route_skill.py "<request>" --json   # when parsing the result
   ```

   Pass real artifacts (`.ncu-rep`, `.nsys-rep`, `.pdf`, `.docx`, `.bib`, `.cu`) with
   `--artifact`; file types outrank keywords.
3. **Read only what was chosen.** Open the primary `SKILL.md` (and at most two companion
   `SKILL.md` files). Do not load `references/inventory.tsv` or the rules file into the
   context; grep them when you need a specific name.
4. **Ask at most one question.** Use the `澄清` line only when the output contains one.
   Otherwise proceed with the primary skill and state the assumption.
5. **Narrow routers still win for GPU-only work.** For pure GPU questions, start from
   `gpu-development-catchall` or the narrow router named in `AGENTS.md`. Use this router
   for mixed or non-GPU domains, and whenever the bundle has no clear narrow route.

## Availability labels

The output reports three separate facts. Do not merge them:

- `source`: `installable` means the inventory lists the skill from a checked-out source;
  `absent` means it is not in the inventory; `user-level` means a machine-global skill.
- `skill_copy`: `found` or `not-found` in the skill directories passed with `--skills-root`;
  `not-checked` when none were passed. An installable source is not proof of a copy in the host.
- `runtime`: `on-path` or `missing` for the declared command (`runtime_bin`); `n/a` if none.

An explicitly named skill that is not present is reported as `absent`; the router never
substitutes another skill for it. Catalog-only matches (no rule) are labelled low confidence.

## Instruction-only skills vs runtime tools

Check the `kind` field before promising an action:

- `instruction`: read `SKILL.md` and follow it; no host install needed.
- `runtime-cli`: needs a command on PATH and may need local configuration or account
  authorization (for example `zotero-cli config`, the `veloq` binary, `wx-cli`). Report
  the prerequisite; never fake a result when the command is missing.
- `user-level`: a machine-global skill outside this repository (`claude-delegate`,
  `codex-background`). Do not install, overwrite, or re-link it from this bundle.

## Gates

- **WeChat (`wx-cli`)**: a direct request from the user to query their local WeChat data is
  the authorization for that ordinary query; do not ask for a second confirmation. Sending,
  editing, or deleting messages, or reading beyond what the request names, needs an explicit
  instruction in the current request. Never run `sudo wx init`, codesign, tccutil, or restart
  WeChat as part of routing; those are separate user-authorized setup steps.
- **Zotero and paper libraries**: these are the user's own references. Read and search
  them as needed; do not write, delete, or re-import items unless asked.
- **Writing and review outputs**: show the draft; do not submit, publish, or send.
- **Credentials and accounts**: never ask for or store secrets in a skill. Tell the user the
  exact command to run themselves.

## Delegated or relayed work

When the work is handed to a relay agent that may not have skills or MCP tools enabled,
copy the selected skill's required rules and the exact commands into the brief. Do not
rely on the relay loading this bundle.

## Maintenance

- Rules: `references/routing-rules.json` (signals, weights, gates, requires).
- Catalog: `references/inventory.tsv`, generated from the checkout by
  `python3 scripts/build_inventory.py`; verify with
  `python3 scripts/build_inventory.py --check` after changing submodules or external sources.
- Tests: `python3 -m unittest discover -s tests -p 'test_skills_router.py' -v` from the repo root.
