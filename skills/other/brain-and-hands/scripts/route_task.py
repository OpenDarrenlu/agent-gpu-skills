#!/usr/bin/env python3
"""Resolve a brain-and-hands role against an explicit model catalog; no network."""

import argparse
import json
import os
from pathlib import Path
import re
import sys


FAMILIES = {"debug": ("astra", "opus"), "test": ("sol", "sonnet"),
            "worker": ("sol", "sonnet"), "researcher": ("sol", "sonnet"),
            "operator": ("sol", "sonnet"), "scout": ("terra", "haiku")}
CLAUDE_KINDS = ("opus", "sonnet", "haiku", "fable")
BRAIN_KINDS = ("opus", "fable")
CURSOR_HAND_PREFIXES = ("deepseek-", "grok-", "gemini-", "muse-")


def version(model, family):
    match = re.fullmatch(r"gpt-(\d+(?:\.\d+)*)-" + re.escape(family), model)
    return tuple(map(int, match[1].split("."))) if match else None


def claude_kind(model):
    """Return (kind, numeric version). A bare alias has an empty version."""
    if model in CLAUDE_KINDS:
        return model, ()
    match = re.fullmatch(
        r"claude-(opus|sonnet|haiku|fable)-(\d+(?:[.-]\d+)*)(?:-[A-Za-z0-9_.-]+)?(?:\[[A-Za-z0-9_.-]+\])?",
        model)
    if not match:
        return None
    return match[1], tuple(int(part) for part in re.split(r"[.-]", match[2]))


def is_brain_slug(host, model):
    if host == "codex":
        return model == "astra" or version(model, "astra") is not None
    parsed = claude_kind(model)
    return parsed is not None and parsed[0] in BRAIN_KINDS


def is_cursor_hand_slug(model):
    parsed = claude_kind(model)
    if parsed and parsed[0] in ("sonnet", "haiku"):
        return True
    if version(model, "sol") or version(model, "terra"):
        return True
    return model.startswith(CURSOR_HAND_PREFIXES)


def is_noir(model):
    """Accept exact a7n Noir IDs, with an optional provider and context suffix."""
    return isinstance(model, str) and re.fullmatch(
        r"(?:[a-z0-9_.-]+/)?a7n-noir(?:-\d+(?:[.-]\d+)*)?(?:\[[a-z0-9_.-]+\])?", model, re.I) is not None


def load_catalog(path):
    try:
        catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("cannot read model catalog; pass --models-file or a family model pin: " + str(exc)) from exc
    if not isinstance(catalog, dict) or not isinstance(catalog.get("models"), list):
        raise ValueError("model catalog must contain a models list")
    slugs = []
    for entry in catalog["models"]:
        if not isinstance(entry, dict) or entry.get("visibility") in ("hide", "hidden") or entry.get("supported_in_api") is False:
            continue
        slug = entry.get("slug", "")
        if isinstance(slug, str) and slug:
            slugs.append(slug)
    return slugs


def latest_claude(slugs, kind):
    found = [(parsed[1], slug) for slug in slugs if (parsed := claude_kind(slug)) and parsed[0] == kind and parsed[1]]
    return max(found)[1] if found else None


def resolve_cursor(task, models_file, model, env):
    path = models_file or env.get("BAH_CURSOR_MODELS_FILE")
    slugs = load_catalog(path) if path else None
    if task == "debug":
        selected = model or env.get("BAH_CURSOR_BRAIN_MODEL")
        if not selected:
            return {"host": "cursor", "task": task, "family": "fable-or-opus", "model": "",
                    "session": "main", "reason": "retain the current Fable or Opus main session"}
        parsed = claude_kind(selected)
        if not parsed or parsed[0] not in BRAIN_KINDS:
            raise ValueError("cursor debug pin must be a Fable or Opus model")
        if slugs is not None and parsed[1] and selected not in slugs:
            raise ValueError("cursor model pin is not in the available model list")
        return {"host": "cursor", "task": task, "family": parsed[0], "model": selected,
                "session": "brain-subagent" if parsed[1] else "main", "reason": "explicit pin"}
    family = "haiku" if task == "scout" else "sonnet"
    selected = model or env.get("BAH_CURSOR_" + family.upper() + "_MODEL")
    if selected:
        if is_brain_slug("cursor", selected):
            raise ValueError("do not use a Fable or Opus model as a hand")
        if slugs is None and not is_cursor_hand_slug(selected):
            raise ValueError("cursor hand pin is not a recognized hand model; pass --models-file")
        if slugs is not None and selected not in slugs:
            raise ValueError("cursor model pin is not in the available model list")
        parsed = claude_kind(selected)
        return {"host": "cursor", "task": task, "family": parsed[0] if parsed else "cursor-hand",
                "model": selected, "session": "native-hand-or-fresh-cli", "reason": "explicit pin"}
    if slugs is None:
        raise ValueError("cursor hand selection needs --models-file listing this session's available models")
    chosen = latest_claude(slugs, family)
    if chosen:
        return {"host": "cursor", "task": task, "family": family, "model": chosen,
                "session": "native-hand-or-fresh-cli",
                "reason": "latest available family version in the cursor catalog"}
    others = [slug for slug in slugs if not is_brain_slug("cursor", slug)]
    raise ValueError("no available " + family + " model in the cursor catalog; pass --model from: " + ", ".join(others))


def resolve(host, task, models_file=None, model=None, environ=None):
    env = os.environ if environ is None else environ
    if task == "simple-test":
        selected = model or env.get("BAH_NOIR_MODEL") or "anthropic/a7n-noir[1m]"
        if not is_noir(selected):
            raise ValueError("simple-test requires a Noir model identifier")
        return {"host": "claude", "task": task, "family": "noir", "model": selected,
                "session": "fresh-cli", "reason": "explicit Noir test lane from either host"}
    if host == "cursor":
        return resolve_cursor(task, models_file, model, env)
    if host == "claude" and task == "debug":
        selected = model or env.get("BAH_CLAUDE_OPUS_MODEL") or env.get("BAH_CLAUDE_FABLE_MODEL")
        if selected:
            parsed = claude_kind(selected)
            if not parsed or parsed[0] not in BRAIN_KINDS:
                raise ValueError("model pin must belong to the opus or fable family")
            return {"host": host, "task": task, "family": parsed[0], "model": selected,
                    "session": "main", "reason": "explicit pin"}
        return {"host": host, "task": task, "family": "opus", "model": "opus", "session": "main",
                "reason": "provider latest-family alias (verify actual session model)"}
    codex_family, claude_family = FAMILIES[task]
    family = codex_family if host == "codex" else claude_family
    selected = model or env.get("BAH_" + host.upper() + "_" + family.upper() + "_MODEL")
    reason = "explicit pin" if selected else "latest available family version"
    if host == "claude":
        parsed = claude_kind(selected) if selected else None
        if selected and (not parsed or parsed[0] != family):
            raise ValueError("model pin must belong to the " + family + " family")
        selected = selected or family
        if reason != "explicit pin":
            reason = "provider latest-family alias (verify actual session model)"
    elif selected:
        if version(selected, family) is None:
            raise ValueError("model pin must belong to the " + family + " family")
    else:
        path = Path(models_file) if models_file else Path(env.get("CODEX_HOME", str(Path.home() / ".codex"))) / "models_cache.json"
        try:
            catalog = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("cannot read model catalog; pass --models-file or a family model pin: " + str(exc)) from exc
        candidates = []
        if not isinstance(catalog, dict) or not isinstance(catalog.get("models"), list):
            raise ValueError("model catalog must contain a models list")
        for entry in catalog["models"]:
            if not isinstance(entry, dict) or entry.get("visibility") in ("hide", "hidden") or entry.get("supported_in_api") is False:
                continue
            slug = entry.get("slug", "")
            if not isinstance(slug, str):
                continue
            ver = version(slug, family)
            if ver is not None:
                candidates.append((ver, slug))
        if not candidates:
            raise ValueError("no available " + family + " model; do not substitute a different family")
        selected = max(candidates)[1]
    return {"host": host, "task": task, "family": family, "model": selected,
            "session": "main" if task == "debug" else "native-hand-or-fresh-cli", "reason": reason}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("codex", "claude", "cursor"), required=True)
    parser.add_argument("--task", choices=("simple-test", *FAMILIES), required=True)
    parser.add_argument("--models-file")
    parser.add_argument("--model", help="explicit family pin; overrides environment and catalog")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(resolve(args.host, args.task, args.models_file, args.model), ensure_ascii=False))
    except (ValueError, TypeError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
