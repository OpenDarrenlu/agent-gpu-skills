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


def version(model, family):
    match = re.fullmatch(r"gpt-(\d+(?:\.\d+)*)-" + re.escape(family), model)
    return tuple(map(int, match[1].split("."))) if match else None


def is_noir(model):
    """Accept exact a7n Noir IDs, with an optional provider and context suffix."""
    return isinstance(model, str) and re.fullmatch(
        r"(?:[a-z0-9_.-]+/)?a7n-noir(?:-\d+(?:[.-]\d+)*)?(?:\[[a-z0-9_.-]+\])?", model, re.I) is not None


def resolve(host, task, models_file=None, model=None, environ=None):
    env = os.environ if environ is None else environ
    if task == "simple-test":
        selected = model or env.get("BAH_NOIR_MODEL") or "anthropic/a7n-noir[1m]"
        if not is_noir(selected):
            raise ValueError("simple-test requires a Noir model identifier")
        return {"host": "claude", "task": task, "family": "noir", "model": selected,
                "session": "fresh-cli", "reason": "explicit Noir test lane from either host"}
    codex_family, claude_family = FAMILIES[task]
    family = codex_family if host == "codex" else claude_family
    selected = model or env.get("BAH_" + host.upper() + "_" + family.upper() + "_MODEL")
    reason = "explicit pin" if selected else "latest available family version"
    if host == "claude":
        if selected and not (selected == family or re.fullmatch(r"claude-" + family + r"-\d+(?:[-.]\d+)*(?:\[[A-Za-z0-9_.-]+\])?", selected)):
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
    parser.add_argument("--host", choices=("codex", "claude"), required=True)
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
