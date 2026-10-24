#!/usr/bin/env python3
"""按任务证据选择一个主 skill 与至多两个伴随 skill。

只读 references/routing-rules.json、references/inventory.tsv，以及可选的 skill 目录（只检查名称是否存在，
不读取 skill 正文）。标准库，无依赖。

用法:
  python3 route_skill.py "帮我看下这个 kernel 为什么慢" --artifact prof/run.ncu-rep
  python3 route_skill.py "用 grill-me 帮我拷问这个方案" --json
  python3 route_skill.py "..." --skills-root ~/.codex/skills     # 额外检查这些目录里是否已有副本

三个层次的可用性（不混用）:
  source         installable  清单中该名称来自已检出的来源（可以安装）；absent = 清单中没有
  skill_copy     found / not-found / not-checked  是否已在检查过的 skill 目录中存在
  runtime        on-path / missing / n/a  规则声明的可执行名是否在 PATH 中

规则:
  - 显式点名（$name、`name`、name skill、skill: name 等）优先，覆盖清单中的全部 skill，不只限于规则表。
    未安装时返回 absent，不替换为其他 skill。
  - 制品后缀（--artifact）比关键词更强；性能问题无制品时标注 needs_evidence：先采集 profiling。
  - 规则没有命中时，在清单的名称与描述中做保守的词面匹配（catalog 回退），并标注为低置信度。
  - 至多两个伴随 skill，且需要至少两条独立证据；同分或近分且无显式点名/制品时只给一个澄清问题。
"""

import argparse
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
REFERENCES = os.path.join(os.path.dirname(HERE), "references")
RULES_PATH = os.path.join(REFERENCES, "routing-rules.json")
INVENTORY_PATH = os.path.join(REFERENCES, "inventory.tsv")

EXPLICIT_SCORE = 100
ARTIFACT_SCORE = 50
DEFAULT_WEIGHT = 12
EXCLUDE_PENALTY = 20
MAX_PHRASE_HITS = 3
MAX_COMPANIONS = 2
ASK_MARGIN = 10
CATALOG_MIN_HITS = 2
# 清单回退时忽略的通用词：它们出现在几乎所有工程任务中，单独命中不能说明任何领域。
STOPWORDS = {"the", "and", "for", "with", "this", "that", "use", "skill", "skills", "please", "help", "me",
             "pipeline", "pipelines", "baseline", "layout", "design", "data", "service", "services", "document",
             "documents", "report", "analysis", "compare", "comparison", "workflow", "task", "tasks", "project",
             "帮我", "一下", "这个", "一个", "可以", "使用"}


def load_rules(path=RULES_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["rules"]


def load_inventory(path=INVENTORY_PATH):
    """返回 {skill 名: {status, catalog, source, description}}；同名取第一条（与安装顺序一致）。"""
    skills = {}
    with open(path, encoding="utf-8") as f:
        columns = None
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if columns is None:
                columns = fields
                continue
            row = dict(zip(columns, fields))
            if row.get("kind") != "skill":
                continue
            skills.setdefault(row["name"], {
                "status": row["status"],
                "catalog": row["catalog"],
                "source": row["source"],
                "description": row["description"],
            })
    return skills


def load_installed(path=INVENTORY_PATH):
    """兼容旧接口：返回 {名称: (catalog, source)}，只包含清单中可安装（installed）的名称。"""
    return {name: (info["catalog"], info["source"])
            for name, info in load_inventory(path).items() if info["status"] == "installed"}


def _ascii_term(term):
    return re.fullmatch(r"[a-z0-9 .+_-]+", term) is not None


def phrase_hit(phrase, text):
    """ASCII 短语按词边界匹配；中文短语按子串匹配。text 已转小写。"""
    term = phrase.lower().strip()
    if not term:
        return False
    if _ascii_term(term):
        pattern = r"(?<![a-z0-9_])" + re.escape(term) + r"(?![a-z0-9_])"
        return re.search(pattern, text) is not None
    return term in text


def explicit_position(skill, text):
    """只识别明确的 skill 点名语法，返回首次出现的位置（没有则 None）。

    支持: $name、`name`、name skill、name 技能、skill name、skill: name、用/使用/调用/use name skill。
    普通英文词（如 review、kernel）不会因为出现在句子中而被视为点名。
    """
    name = re.escape(skill.lower())
    patterns = [
        r"\$" + name + r"(?![a-z0-9_-])",
        r"`" + name + r"`",
        r"(?<![a-z0-9_-])" + name + r"\s*(skill|技能)",
        r"(?<![a-z0-9_-])(skill|技能)\s*[:：]?\s*" + name + r"(?![a-z0-9_-])",
        r"(用|使用|调用|use|invoke)\s+(the\s+)?" + name + r"\s*(skill|技能)",
    ]
    positions = []
    for p in patterns:
        match = re.search(p, text)
        if match:
            positions.append(match.start())
    return min(positions) if positions else None


def explicit_hit(skill, text):
    return explicit_position(skill, text) is not None


def score_rule(rule, text, artifacts):
    """返回 (score, reasons, explicit_pos)。score <= 0 表示无证据。

    requires_any: 领域门控。规则只有在文本含其中至少一个领域词时才计分，
    从而让 pipeline、debug、layout 等通用动作词不能单独选中某个领域 skill。
    显式点名与强制制品（ARTIFACT_SCORE）可以绕过门控。
    """
    weight = rule.get("weight", DEFAULT_WEIGHT)
    score = 0
    reasons = []

    pos = explicit_position(rule["skill"], text)
    if pos is not None:
        score += EXPLICIT_SCORE
        reasons.append("explicit:" + rule["skill"])

    artifact_hit = False
    for ext in rule.get("artifacts", []):
        if any(a.lower().endswith(ext) for a in artifacts):
            score += ARTIFACT_SCORE
            reasons.append("artifact:" + ext)
            artifact_hit = True

    gate = rule.get("requires_any")
    if gate and pos is None and not artifact_hit:
        if not any(phrase_hit(term, text) for term in gate):
            return 0, [], pos

    hits = [p for p in rule.get("phrases", []) if phrase_hit(p, text)][:MAX_PHRASE_HITS]
    for p in hits:
        score += weight
        reasons.append("phrase:" + p)

    if any(phrase_hit(p, text) for p in rule.get("not", [])):
        score -= EXCLUDE_PENALTY
        reasons.append("exclude")

    return score, reasons, pos


def _tokens(text):
    """返回 (ASCII 词集合, 中文二元组集合)。"""
    text = text.lower()
    words = set(w for w in re.findall(r"[a-z][a-z0-9-]{2,}", text) if w not in STOPWORDS)
    bigrams = set()
    for run in re.findall(r"[一-鿿]+", text):
        bigrams.update(run[i:i + 2] for i in range(len(run) - 1))
    return words, bigrams


def _catalog_match(query_tokens, target_tokens):
    """ASCII 词至少命中 1 个且总命中 >= 2，或中文二元组至少命中 3 个。单个通用中文字对不构成匹配。"""
    q_words, q_bigrams = query_tokens
    t_words, t_bigrams = target_tokens
    word_hits = q_words & t_words
    bigram_hits = q_bigrams & t_bigrams
    if word_hits and len(word_hits) + len(bigram_hits) >= CATALOG_MIN_HITS:
        return sorted(word_hits) + sorted(bigram_hits)[:4]
    if len(bigram_hits) >= 3:
        return sorted(bigram_hits)[:4]
    return None


def catalog_candidates(query, skills, limit=3):
    """清单词面回退：只在规则表未命中时使用，结果一律标注为低置信度。"""
    query_tokens = _tokens(query)
    found = []
    for name, info in skills.items():
        if info["status"] not in ("installed", "shadowed"):
            continue
        name_tokens = _tokens(name.replace("-", " "))
        desc_tokens = _tokens(info["description"])
        target = (name_tokens[0] | desc_tokens[0], name_tokens[1] | desc_tokens[1])
        shared = _catalog_match(query_tokens, target)
        if shared:
            found.append((len(shared), name, shared, info))
    found.sort(key=lambda item: (-item[0], item[1]))
    return found[:limit]


def _find_skill_copy(name, roots):
    if roots is None:
        return "not-checked", None
    for root in roots:
        candidate = os.path.join(os.path.expanduser(root), name, "SKILL.md")
        if os.path.isfile(candidate):
            return "found", candidate
    return "not-found", None


def _availability(rule, info, skill_roots):
    copy_state, copy_path = _find_skill_copy(rule["skill"], skill_roots)
    if rule.get("kind") == "user-level" or (info is not None and info["status"] == "protected"):
        source = "user-level"
    elif info is None:
        source = "absent"
    else:
        source = "installable"
    runtime_bin = rule.get("runtime_bin")
    if runtime_bin:
        runtime = "on-path" if shutil.which(runtime_bin) else "missing"
    else:
        runtime = "n/a"
    return {"source": source, "skill_copy": copy_state, "skill_copy_path": copy_path, "runtime": runtime,
            "runtime_bin": runtime_bin or ""}


def _candidate(rule, score, reasons, pos, info, skill_roots):
    return {
        "skill": rule["skill"],
        "domain": rule.get("domain", info["catalog"] if info else ""),
        "kind": rule.get("kind", "instruction"),
        "score": score,
        "reasons": reasons,
        "explicit": pos is not None,
        "explicit_pos": pos,
        "available": info is not None and info["status"] in ("installed", "shadowed"),
        "info": info,
        "availability": _availability(rule, info, skill_roots),
        "requires": rule.get("requires", ""),
        "gate": rule.get("gate", ""),
        "needs_evidence": bool(rule.get("needs_evidence")),
    }


def route(query, artifacts=(), rules=None, inventory=None, skill_roots=None):
    rules = load_rules() if rules is None else rules
    inventory = load_inventory() if inventory is None else inventory
    text = query.lower()
    artifacts = list(artifacts)

    # 规则覆盖的 skill 按规则评分；清单中其余 skill 只参与显式点名（不需要规则也能被点名）。
    ruled = {}
    candidates = []
    for index, rule in enumerate(rules):
        ruled[rule["skill"]] = True
        score, reasons, pos = score_rule(rule, text, artifacts)
        if score <= 0:
            continue
        info = inventory.get(rule["skill"])
        cand = _candidate(rule, score, reasons, pos, info, skill_roots)
        cand["_index"] = index
        if cand["needs_evidence"] and artifacts:
            cand["needs_evidence"] = False
        candidates.append(cand)
    for name, info in inventory.items():
        if name in ruled:
            continue
        pos = explicit_position(name, text)
        if pos is None:
            continue
        rule = {"skill": name, "domain": info["catalog"], "kind": "user-level" if info["status"] == "protected"
                else "instruction"}
        cand = _candidate(rule, EXPLICIT_SCORE, ["explicit:" + name], pos, info, skill_roots)
        cand["_index"] = len(rules) + len(candidates)
        candidates.append(cand)

    candidates.sort(key=lambda c: (-c["score"], c["_index"]))

    result = {"primary": None, "companions": [], "ask": None, "unavailable": [], "notes": [],
              "catalog_fallback": False}
    if not candidates:
        fallback = catalog_candidates(query, inventory)
        if fallback:
            result["catalog_fallback"] = True
            hits, name, shared, info = fallback[0]
            rule = {"skill": name, "domain": info["catalog"], "kind": "instruction"}
            cand = _candidate(rule, hits, ["catalog:" + "/".join(shared)], None, info, skill_roots)
            result["primary"] = _public(cand)
            result["notes"].append("低置信度：规则表未命中，按清单名称与描述词面匹配；请确认后再使用")
            others = [c for c in fallback[1:] if c[3]["status"] == "installed"]
            if others:
                result["ask"] = "你是要 %s 还是 %s？" % (name, others[0][1])
            return result
        result["notes"].append("无可用证据：直接回答或只问一个澄清问题；GPU 问题可从 gpu-development-catchall 开始")
        return result

    explicit_candidates = [c for c in candidates if c["explicit"]]
    if explicit_candidates:
        # 用户先提到的 skill 优先。
        primary = min(explicit_candidates, key=lambda c: (c["explicit_pos"], -c["score"]))
        result["primary"] = _public(primary)
        if not primary["available"]:
            result["notes"].append("用户点名的 %s 不在清单中或未安装；不会替换为其他 skill（见 requires）"
                                   % primary["skill"])
        if primary["gate"]:
            result["notes"].append("授权边界: " + primary["gate"])
        return result

    available = [c for c in candidates if c["available"]]
    if not available:
        primary = candidates[0]
        result["primary"] = _public(primary)
        result["notes"].append("最可能的 skill %s 未安装；按 requires 处理，不替换为其他 skill" % primary["skill"])
        if primary["gate"]:
            result["notes"].append("授权边界: " + primary["gate"])
        return result

    primary = available[0]
    result["unavailable"] = [_public(c) for c in candidates if not c["available"]][:MAX_COMPANIONS + 1]
    result["primary"] = _public(primary)
    if primary["needs_evidence"]:
        result["notes"].append("性能问题缺少 profiling 制品：先采集 ncu/nsys 报告，再给结论")
    if primary["gate"]:
        result["notes"].append("授权边界: " + primary["gate"])

    # 伴随 skill 需要至少两条独立证据（约两个短语，或一个制品），单条短语不足以加入
    threshold = max(2 * DEFAULT_WEIGHT, primary["score"] // 2)
    for c in candidates:
        if c is primary or not c["available"] or c["score"] < threshold:
            continue
        result["companions"].append(_public(c))
        if len(result["companions"]) == MAX_COMPANIONS:
            break

    runner_up = next((c for c in candidates if c is not primary and c["available"]), None)
    # 只有两个候选得分接近（差距不大且 runner-up 至少为主候选的 3/4）时才提问；明显更弱的候选不构成歧义。
    if (runner_up is not None and not artifacts
            and primary["score"] - runner_up["score"] <= ASK_MARGIN
            and runner_up["score"] * 4 >= primary["score"] * 3
            and runner_up["score"] >= DEFAULT_WEIGHT and primary["score"] >= DEFAULT_WEIGHT
            and not any(r.startswith("explicit") for r in primary["reasons"] + runner_up["reasons"])):
        result["ask"] = "你是要 %s（%s）还是 %s（%s）？" % (
            primary["skill"], primary["domain"], runner_up["skill"], runner_up["domain"])
    return result


def _public(c):
    info = c["info"] or {}
    return {
        "skill": c["skill"],
        "status": "installable" if c["available"] else (
            "user-level" if c["availability"]["source"] == "user-level" else "absent"),
        "kind": c["kind"],
        "domain": c["domain"],
        "score": c["score"],
        "why": c["reasons"],
        "catalog": info.get("catalog", ""),
        "source": info.get("source", ""),
        "description": info.get("description", ""),
        "availability": c["availability"],
        "requires": c["requires"],
        "gate": c["gate"],
    }


def format_text(result):
    lines = []
    primary = result["primary"]
    if primary is None:
        lines.append("主 skill: 无")
    else:
        avail = primary["availability"]
        lines.append("主 skill: %s [%s, %s] score=%d" % (
            primary["skill"], primary["status"], primary["kind"], primary["score"]))
        lines.append("  why: %s" % ", ".join(primary["why"]))
        lines.append("  availability: source=%s, skill_copy=%s, runtime=%s%s" % (
            avail["source"], avail["skill_copy"],
            avail["runtime"], (" (%s)" % avail["runtime_bin"]) if avail["runtime_bin"] else ""))
        if primary["requires"]:
            lines.append("  requires: %s" % primary["requires"])
    for c in result["companions"]:
        lines.append("伴随 skill: %s [%s] score=%d" % (c["skill"], c["kind"], c["score"]))
    for c in result["unavailable"]:
        lines.append("未就绪候选: %s（%s）" % (c["skill"], c["requires"] or c["status"]))
    for note in result["notes"]:
        lines.append("注意: " + note)
    if result["ask"]:
        lines.append("澄清: " + result["ask"])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", help="用户任务的原文（中英文均可）")
    parser.add_argument("--artifact", action="append", default=[], help="相关制品路径，可重复（按后缀识别）")
    parser.add_argument("--skills-root", action="append", default=None,
                        help="检查 skill 副本是否存在的目录，可重复。默认不检查（显示 not-checked）")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)

    result = route(args.query, args.artifact, skill_roots=args.skills_root)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_text(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
