#!/usr/bin/env python3
"""生成 SOURCES.md：本仓库 install.sh 提供的全部 skill 的来源登记。

一个 skill 的来源分四类：
  local     本仓库原创（无上游），来源为本仓库自身
  vendored  从上游仓库逐字复制的快照（允许本地修改，同步时 merge 而非覆盖）
  catalog   submodule 引入的 skill 目录树（安装器只读取，不复制进本仓库）
  external  sources/external-sources.conf 中的外部来源（git-ignored checkout + 运行时 CLI）

catalog 与 external 的 skill 列表从实际 checkout 枚举（与 install.sh 的发现规则一致）；
local 与 vendored 从 install.sh 的 SKILL_NAMES/SKILL_DIRS 读取（与安装名一致）。

用法:
  python3 scripts/build_sources.py              # 重新生成 SOURCES.md
  python3 scripts/build_sources.py --check      # 与现有文件比较，不一致返回 1

上游更新检查用 `git ls-remote <repo> main` 或 GitHub API 手工执行，结论记录在
sources/update-log.md（本文件的"更新记录"一节）。

"更新记录"一节来自 sources/update-log.md（手工维护，不自动生成）。
"""

import argparse
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "SOURCES.md")
LOG = os.path.join(ROOT, "sources", "update-log.md")
GITMODULES = os.path.join(ROOT, ".gitmodules")
REPO_REMOTE = "https://github.com/OpenDarrenlu/agent-gpu-skills"

# vendored 快照：从上游仓库复制到本仓库的 skill。local_mods 说明相对上游的本地修改
# （同步上游时，有本地修改的文件必须 merge，其余可直接覆盖）。
VENDORED = [
    {
        "root": "skills/framework/sglang-repo-skills",
        "skills_dir": "sglang-repo-skills",
        "repo": "https://github.com/sgl-project/sglang.git",
        "path": ".agents/skills",
        "ref": "3bcde0c80eeb07b40d3334c4b3d7341602efd089",
        "local_mods": "无（全部逐字复制）",
        "note": "上游 .claude/skills 是 .agents/skills 的 symlink；未收录的上游 skill 见其 README",
    },
    {
        "root": "skills/kernel/tilelang-repo-skills",
        "skills_dir": "tilelang-repo-skills",
        "repo": "https://github.com/tile-ai/tilelang.git",
        "path": ".agents/skills",
        "ref": "a8f3d3bcfe7fdd1a3887b05ff8f054e7d08980de",
        "local_mods": "无（全部逐字复制）",
        "note": "未收录上游 tilelang-build / tilelang-dev-guide / tilelang-simplification / tilelang-submit-pr / tilelang-tvm-ir",
    },
    {
        "root": "skills/framework/zero-mismatch-rl",
        "skills_dir": "zero-mismatch-rl",
        "repo": "https://github.com/Inferact/zero-mismatch-rl.git",
        "path": "skill/（另有 cases/ 模型案例）",
        "ref": "136aca87d8158adae94b5a272fa4a8a6d5cd6939",
        "local_mods": "SKILL.md 与 README.md 为本地新增入口与说明；skill/、cases/、LICENSE、README.upstream.md 逐字复制",
        "note": "skill/ 与 cases/ 必须保持同级（skill/guides 中以 ../../../cases/... 引用案例）",
    },
    {
        "root": "skills/framework/train-inference-consistency",
        "skills_dir": "train-inference-consistency",
        "repo": "https://github.com/vllm-project/vllm.git",
        "path": ".agents/skills/train-inference-consistency",
        "ref": "2d8efee9a73366d6b4e753780af32443d2fef91a",
        "ref_follow": "main（update-repos.sh 的 VLLM_SKILLS_REF）",
        "local_mods": "SKILL.md 增加中文触发词、vLLM 文档链接改为本目录稀疏 clone、增加 zero-mismatch-rl 交叉引用与来源注释；新增 README.md 与 LICENSE；其余为上游逐字复制",
        "note": "2026-11-08 检查：上游 vLLM main 已删除 .agents/skills/train-inference-consistency，本 vendored 副本成为唯一来源，不再有上游更新可同步",
    },
    {
        "root": "skills/other/tech-doc-style-chinese",
        "skills_dir": "tech-doc-style-chinese",
        "repo": "https://github.com/Fenng/Tech-Doc-Style-Chinese.git",
        "path": ".（仓库根）",
        "ref": "b119d01c9bb7cd132b3c62fafdff34227c4410d1",
        "local_mods": "本地定制覆盖版（修改对照见其 README.md）；上游 clone 在 repos/tech-doc-style-chinese（git-ignored）",
        "note": "同步上游必须按其 README.md 的步骤 merge，不能直接覆盖本地修改的文件",
    },
]

# catalog 列表：submodule（pin 来自 git）与 clone（pin 为本地 HEAD）。
# skills 取"安装名 -> 上游相对路径"；枚举规则与 install.sh 一致。
CATALOGS = [
    {
        "label": "nvidia-skills",
        "kind": "submodule",
        "path": "repos/nvidia-skills",
        "repo": "https://github.com/NVIDIA/skills.git",
        "enumerate": [("repos/nvidia-skills/skills/*", "dirname"),
                      ("repos/nvidia-skills/plugins/nvidia-skills/skills/*", "dirname")],
        "note": "install.sh 遍历 skills/ 与 plugins/nvidia-skills/skills/；同名先到先得（本地 skill 优先）",
    },
    {
        "label": "amd-skills",
        "kind": "submodule",
        "path": "repos/amd-skills",
        "repo": "https://github.com/amd/skills.git",
        "enumerate": [("repos/amd-skills/skills/*", "dirname")],
        "note": "安装前检查其 skills/ 目录；本仓库只安装 skills/，不安装其他目录",
    },
    {
        "label": "ccfa-skills",
        "kind": "submodule",
        "path": "repos/ccfa-skills",
        "repo": "https://github.com/mikubaka88/CCFA-Skills.git",
        "enumerate": [("repos/ccfa-skills/ccf-*", "dirname")],
        "note": "ccf-* 含 SKILL.md 者为 runtime skill（安装名 = 目录名）；不含者为支撑目录（直接链接）",
    },
    {
        "label": "cursor-skills",
        "kind": "submodule",
        "path": "repos/cursor-skills",
        "repo": "https://github.com/Saddss/cursor-skills.git",
        "enumerate": [("repos/cursor-skills/skills/*", "dirname")],
        "note": "同名时本地 catalog 优先（install.sh 先到先得）",
    },
    {
        "label": "ponytail",
        "kind": "submodule",
        "path": "repos/ponytail",
        "repo": "https://github.com/DietrichGebert/ponytail.git",
        "enumerate": [("repos/ponytail/skills/*", "dirname")],
        "note": "安装名与本地数组一致",
    },
    {
        "label": "tilelang-cuda-skills",
        "kind": "submodule",
        "path": "repos/tilelang-cuda-skills",
        "repo": "https://github.com/sablin39/tilelang-cuda-skills.git",
        "enumerate": [
            ("repos/tilelang-cuda-skills/skills/tilelang-optimization-loop", "fixed:tilelang-optimization-loop"),
            ("repos/tilelang-cuda-skills/skills/designing-with-tilefoundry", "fixed:designing-with-tilefoundry"),
            ("repos/tilelang-cuda-skills/skills/tilelang/debugging-tilelang-programs", "fixed:debugging-tilelang-programs"),
            ("repos/tilelang-cuda-skills/skills/tilelang/optimizing-tilelang-programs", "fixed:optimizing-tilelang-programs"),
            ("repos/tilelang-cuda-skills/skills/tilelang/profiling-tilelang-programs", "fixed:profiling-tilelang-programs"),
            ("repos/tilelang-cuda-skills/skills/tilelang/testing-fwd-bwd-kernels", "fixed:testing-fwd-bwd-kernels"),
            ("repos/tilelang-cuda-skills/skills/tilelang/torch-profiling-tilelang-programs", "fixed:torch-profiling-tilelang-programs"),
            ("repos/tilelang-cuda-skills/skills/tilelang/writing-tilelang-design-docs", "fixed:writing-tilelang-design-docs"),
            ("repos/tilelang-cuda-skills/skills/tilelang/writing-tilelang-kernels", "fixed:writing-tilelang-kernels"),
        ],
        "note": "嵌套 submodule external/TileFoundry 由 update-repos.sh --recursive 维护；其余上游 skill 未安装（见 skills/other 与 skills/kernel 的本地/vendored 对应项）",
    },
    {
        "label": "kernel-design-agents",
        "kind": "submodule",
        "path": "kernel-design-agents",
        "repo": "https://github.com/OpenDarrenlu/kernel-design-agents.git",
        "enumerate": [("kernel-design-agents/skills/KernelWiki", "fixed:KernelWiki")],
        "note": "只安装 KernelWiki；其嵌套 skills（ncu-report-skill）已在上游合并为目录内引用",
    },
    {
        "label": "tech-doc-style-chinese (clone)",
        "kind": "clone",
        "path": "repos/tech-doc-style-chinese",
        "repo": "https://github.com/Fenng/Tech-Doc-Style-Chinese.git",
        "enumerate": [],
        "note": "git-ignored 完整 clone，仅作 diff 参考；安装的是本地定制版 skills/other/tech-doc-style-chinese",
    },
]


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def sh(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, cwd=ROOT, timeout=120).stdout.strip()
    except Exception:
        return ""


def parse_conf():
    path = os.path.join(ROOT, "sources", "external-sources.conf")
    rows = []
    if not os.path.isfile(path):
        return rows
    for line in read(path).splitlines():
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        f = line.split("|")
        if len(f) < 6:
            continue
        rows.append({"name": f[0], "url": f[1], "ref": f[2], "mode": f[3],
                     "skills_dir": f[4], "runtime": f[5],
                     "provenance": f[6] if len(f) > 6 else ""})
    return rows


def parse_install_arrays():
    text = read(os.path.join(ROOT, "install.sh"))
    names = re.search(r"^SKILL_NAMES=\((.*?)^\)", text, re.S | re.M)
    dirs = re.search(r"^SKILL_DIRS=\((.*?)^\)", text, re.S | re.M)
    if not names or not dirs:
        raise SystemExit("无法从 install.sh 读取 SKILL_NAMES / SKILL_DIRS")
    n, d = names.group(1).split(), dirs.group(1).split()
    if len(n) != len(d):
        raise SystemExit("install.sh 中 SKILL_NAMES 与 SKILL_DIRS 数量不一致")
    return list(zip(n, d))


def frontmatter_name(skill_dir):
    """与 install.sh 的 frontmatter_skill_name 一致：name 合法则用它，否则用目录名。"""
    md = os.path.join(skill_dir, "SKILL.md")
    if not os.path.isfile(md):
        return ""
    m = re.search(r"^name:\s*(.+)$", read(md), re.M)
    if not m:
        return ""
    name = re.sub(r"[\"' \r]", "", m.group(1).rstrip("\n"))
    if re.fullmatch(r"[A-Za-z0-9._-]+", name):
        return name
    return os.path.basename(os.path.abspath(skill_dir))


def enumerate_catalog(cat):
    """返回 [(安装名, 上游相对路径, kind)]；kind: skill | support（支撑目录不安装）。"""
    out = []
    for pattern, how in cat["enumerate"]:
        for d in sorted(glob.glob(os.path.join(ROOT, pattern))):
            if not os.path.isdir(d):
                continue
            rel = os.path.relpath(d, os.path.join(ROOT, cat["path"])).replace(os.sep, "/")
            name = frontmatter_name(d) if how == "frontmatter" else os.path.basename(d)
            if how.startswith("fixed:"):
                name = how.split(":", 1)[1]
            if not os.path.isfile(os.path.join(d, "SKILL.md")):
                out.append((name, rel, "support"))
            else:
                out.append((name, rel, "skill"))
    return out


def enumerate_external(conf):
    """与 install.sh 的 install_conf_catalogs 相同：递归查找 skills_dir 下的 SKILL.md，
    同名先到先得（后到的重复目录不安装）。"""
    installed, seen = [], set()
    for sd in conf["skills_dir"].split(","):
        root = os.path.join(ROOT, "repos", conf["name"], sd.strip())
        for md in sorted(glob.glob(os.path.join(root, "**", "SKILL.md"), recursive=True)):
            d = os.path.dirname(md)
            if "node_modules" in d.split(os.sep):
                continue
            rel = os.path.relpath(d, os.path.join(ROOT, "repos", conf["name"])).replace(os.sep, "/")
            name = frontmatter_name(d)
            if name in seen:
                continue
            seen.add(name)
            installed.append((name, rel))
    return installed


def submodule_pin(rel):
    return sh(["git", "submodule", "status", rel])


def checkout_head(rel):
    return sh(["git", "-C", rel, "rev-parse", "HEAD"])




def pin_of(kind, rel):
    if kind == "submodule":
        out = submodule_pin(rel)
        m = re.match(r"[-+U ]*([0-9a-f]{7,40})", out)
        return m.group(1) if m else ""
    return checkout_head(rel)


def ref_of_vendored(root):
    """vendored 的 ref 以本地 README 的基线 commit 为准（脚本常量仅作 fallback）。"""
    readme = os.path.join(ROOT, root, "README.md")
    if os.path.isfile(readme):
        m = re.search(r"基线[^`]*`([0-9a-f]{40})`", read(readme))
        if m:
            return m.group(1)
    return ""




def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    pairs = parse_install_arrays()
    conf = {c["name"]: c for c in parse_conf()}
    inventory = os.path.join(ROOT, "skills", "routers", "agent-skill-router", "references", "inventory.tsv")
    inv_catalog = {}
    if os.path.isfile(inventory):
        for line in read(inventory).splitlines():
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) > 3 and f[0] == "skill":
                inv_catalog[f[2]] = f[1]

    # SKILL_DIRS 中的每一项按 vendored root 或 catalog path 归类
    vend_by_root = {v["root"]: [] for v in VENDORED}
    local_pairs = []
    cat_by_path = {}
    for cat in CATALOGS:
        cat_by_path[cat["path"]] = cat
    for name, d in pairs:
        placed = False
        for v in VENDORED:
            if d == v["root"] or d.startswith(v["root"] + "/"):
                vend_by_root[v["root"]].append((name, d))
                placed = True
                break
        if placed:
            continue
        for path, cat in sorted(cat_by_path.items(), key=lambda kv: -len(kv[0])):
            if d == path or d.startswith(path + "/"):
                cat.setdefault("_pairs", []).append((name, d))
                placed = True
                break
        if not placed:
            local_pairs.append((name, d))


    L = []
    L.append("# Skill 来源登记（SOURCES.md）")
    L.append("")
    L.append("本文件由 `python3 scripts/build_sources.py` 生成，请勿手工编辑正文。")
    L.append("它列出 `install.sh` 安装的全部 skill 的来源：本仓库路径、上游仓库与引用、本地修改情况。")
    L.append("")
    L.append("## 来源分类")
    L.append("")
    L.append("| 类别 | 含义 | 同步方式 |")
    L.append("| --- | --- | --- |")
    L.append("| local | 本仓库原创，无上游 | 直接修改本仓库 |")
    L.append("| vendored | 从上游逐字复制的快照，可能有本地修改 | 有本地修改的文件 merge，其余直接覆盖 |")
    L.append("| catalog | submodule 引入的 skill 目录树 | `bash update-repos.sh <name>` 更新 pin |")
    L.append("| external | `sources/external-sources.conf` 的外部来源 | 更新 conf 中的 ref 后 `bash update-repos.sh external` |")
    L.append("")
    L.append("上游 URL 模板：`<repo>/tree/<pin>/<上游路径>`（vendored 与 catalog 逐 skill 给出，external 见 conf）。")
    L.append("")

    # 1. local
    L.append("## 1. 本仓库原创 skill（local，%d 个）" % len(local_pairs))
    L.append("")
    L.append("来源：本仓库 " + REPO_REMOTE + "。")
    L.append("")
    L.append("| skill | 本地路径 |")
    L.append("| --- | --- |")
    for n, d in sorted(local_pairs):
        L.append("| `%s` | `%s` |" % (n, d))
    L.append("")

    # 2. vendored
    L.append("## 2. vendored 固定上游快照（%d 个目录）" % len(VENDORED))
    L.append("")
    L.append("| skill | 上游仓库 | 上游路径 | 基线 commit | 本地修改 |")
    L.append("| --- | --- | --- | --- | --- |")
    for v in VENDORED:
        entries = vend_by_root[v["root"]]
        names = ", ".join(sorted("`%s`" % n for n, _ in entries)) or "-"
        ref = ref_of_vendored(v["root"]) or v["ref"]
        L.append("| %s | %s | `%s` | `%s` | %s |"
                 % (names, v["repo"], v["path"], ref, v["local_mods"]))
        if not entries:
            L.append("| - | %s | 本地目录 `%s` 未在 install.sh 数组中 | | | |" % (v["repo"], v["root"]))
    L.append("")
    L.append("各 vendored 目录的说明：")
    L.append("")
    for v in VENDORED:
        L.append("- **`%s`**（%s）：%s" % (v["root"], v["skills_dir"], v["note"]))
    L.append("")

    # 3. catalogs
    total_cat = sum(len(enumerate_catalog(c)) for c in CATALOGS)
    L.append("## 3. catalog（submodule / clone，%d 个来源）" % len(CATALOGS))
    L.append("")
    L.append("| catalog | 上游仓库 | 本地 pin | 安装 skill 数 |")
    L.append("| --- | --- | --- | --- |")
    for cat in CATALOGS:
        entries = enumerate_catalog(cat)
        pin = pin_of(cat["kind"], cat["path"])
        L.append("| `%s` | %s | `%s` | %d |"
                 % (cat["label"], cat["repo"], pin, len([e for e in entries if e[2] == "skill"])))
    L.append("")
    L.append("install.sh 的 SKILL_DIRS 直接安装的 tilelang-cuda-skills skill：%s。"
             % ", ".join("`%s`" % n for n, _ in sorted(cat_by_path["repos/tilelang-cuda-skills"].get("_pairs", []))))
    L.append("")
    for cat in CATALOGS:
        entries = [e for e in enumerate_catalog(cat) if e[2] == "skill"]
        if not entries:
            continue
        L.append("### %s" % cat["label"])
        L.append("")
        L.append("- 上游：%s" % cat["repo"])
        L.append("- 本地 pin：`%s`（%s）" % (pin_of(cat["kind"], cat["path"]), cat["kind"]))
        L.append("- 上游 URL：`%s/tree/%s/<路径>`" % (cat["repo"], pin_of(cat["kind"], cat["path"])))
        L.append("- %s" % cat["note"])
        L.append("")
        L.append("| 安装名 | 上游相对路径 |")
        L.append("| --- | --- |")
        for name, rel, kind in entries:
            L.append("| `%s` | `%s` |" % (name, rel))
        L.append("")

    # 4. external
    L.append("## 4. 外部来源（sources/external-sources.conf，%d 个）" % len(conf))
    L.append("")
    L.append("| name | 上游仓库 | ref | runtime | 安装 skill |")
    L.append("| --- | --- | --- | --- | --- |")
    ext_entries = {}
    for name, c in sorted(conf.items()):
        entries = enumerate_external(c) if c["mode"] != "none" else []
        ext_entries[name] = entries
        L.append("| `%s` | %s | `%s` | `%s` | %s |"
                 % (name, c["url"], c["ref"][:12], c["runtime"],
                    ", ".join("`%s`" % n for n, _ in entries)[:200] or "-"))
    L.append("")
    for name, c in sorted(conf.items()):
        entries = ext_entries[name]
        L.append("### %s" % name)
        L.append("")
        L.append("- 上游：%s" % c["url"])
        L.append("- ref：`%s`" % c["ref"])
        L.append("- mode：`%s`；skills_dir：`%s`" % (c["mode"], c["skills_dir"]))
        L.append("- runtime：`%s`" % c["runtime"])
        L.append("- 安装 skill：")
        L.append("")
        L.append("| 安装名 | 上游相对路径 |")
        L.append("| --- | --- |")
        for n, rel in entries:
            L.append("| `%s` | `%s` |" % (n, rel))
        L.append("")
        L.append("provenance：%s" % c["provenance"])
        L.append("")

    total_installed = len(local_pairs) + sum(len(v) for v in vend_by_root.values()) + total_cat + \
        sum(len(e) for e in ext_entries.values())
    L.append("安装 skill 总数（含各 catalog 枚举与 external，可能有同名先到先得）：%d。" % total_installed)
    L.append("")
    L.append("## 检查与更新流程")
    L.append("")
    L.append("```bash")
    L.append("python3 scripts/build_sources.py --fetch   # 查询上游 tip 并写入本文件的更新状态列")
    L.append("python3 scripts/build_sources.py --check   # 校验本文件与 checkout 一致")
    L.append("bash update-repos.sh all                   # 更新全部 submodule / 外部 checkout")
    L.append("```")
    L.append("")
    L.append("逐来源的检查结论记录在 `sources/update-log.md`（手工维护）。")
    L.append("")

    if os.path.isfile(LOG):
        L.append("<!-- update-log -->")
        L.append(read(LOG).rstrip("\n"))
        L.append("")

    text = "\n".join(L) + "\n"
    if args.check:
        try:
            current = read(OUT)
        except OSError:
            current = ""
        if current != text:
            print("SOURCES.md 已过期，运行 python3 scripts/build_sources.py 重新生成")
            return 1
        print("SOURCES.md 与当前 checkout 一致")
        return 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print("已写入 %s（%d 行；local %d、vendored %d、catalog skill %d、external skill %d）"
          % (os.path.relpath(OUT, ROOT), text.count("\n"), len(local_pairs),
             sum(len(v) for v in vend_by_root.values()), total_cat, sum(len(e) for e in ext_entries.values())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
