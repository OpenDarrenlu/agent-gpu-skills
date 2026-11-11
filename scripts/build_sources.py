#!/usr/bin/env python3
"""生成 SOURCES.md：本仓库全部 skill 的来源登记（逐 skill 一行，附上游链接）。

来源类别（class）:
  vendored          从上游逐字复制的快照（VENDORED），有本地修改的文件需 merge，不能整目录覆盖
  catalog           submodule / clone 中的 skill（CATALOGS），安装器只读取，不复制进本仓库
  external          sources/external-sources.conf 的外部来源（git-ignored checkout，runtime 另行安装）
  nested            catalog 的嵌套 submodule（NESTED），以嵌套仓库自己的 origin 与 gitlink 为准
  external-declared 本地目录内有上游声明（UPSTREAM.md / README），仅记录来源，本仓库不含上游快照
  protected         用户级保留名（claude-delegate、codex-background），安装器不覆盖，不计为可安装
  unknown           本地目录，未找到上游声明；来源未知，不默认为原创，需人工确认
  optional-runtime  install-veloq.sh 安装的 VeloQ skill（上游 installer 取 main，未固定 ref）

skill 的发现、先到先得与状态（installed / shadowed / protected / missing-source）由
skills/routers/agent-skill-router/scripts/build_inventory.py 的 build_rows() 统一给出；
本脚本只附加来源，不再自行枚举目录。

用法:
  python3 scripts/build_sources.py              # 重新生成 SOURCES.md
  python3 scripts/build_sources.py --check      # 与现有文件比较，不一致返回 1

上游更新检查需要网络，用 git ls-remote <repo> main 或 GitHub API 手工执行，结论记录在
sources/update-log.md（手工维护，本文件的"更新记录"一节原样嵌入）。无网络时不得记为"无更新"。
"""

import argparse
import importlib.util
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "SOURCES.md")
LOG = os.path.join(ROOT, "sources", "update-log.md")
INVENTORY_PY = os.path.join(ROOT, "skills", "routers", "agent-skill-router", "scripts", "build_inventory.py")
INSTALL_SH = os.path.join(ROOT, "install.sh")
VELOQ_SH = os.path.join(ROOT, "install-veloq.sh")
REPO_REMOTE = "https://github.com/OpenDarrenlu/agent-gpu-skills"

# vendored 快照：从上游仓库复制到本仓库的 skill。link_path 为上游中对应的目录（链接用），
# local_mods 说明相对上游的本地修改（同步上游时，有本地修改的文件必须 merge，其余可直接覆盖）。
VENDORED = [
    {
        "root": "skills/framework/sglang-repo-skills",
        "skills_dir": "sglang-repo-skills",
        "repo": "https://github.com/sgl-project/sglang.git",
        "path": ".agents/skills",
        "link_path": ".agents/skills",
        "ref": "3bcde0c80eeb07b40d3334c4b3d7341602efd089",
        "local_mods": "无（全部逐字复制）",
        "note": "上游 .claude/skills 是 .agents/skills 的 symlink；未收录的上游 skill 见其 README",
    },
    {
        "root": "skills/kernel/tilelang-repo-skills",
        "skills_dir": "tilelang-repo-skills",
        "repo": "https://github.com/tile-ai/tilelang.git",
        "path": ".agents/skills",
        "link_path": ".agents/skills",
        "ref": "a8f3d3bcfe7fdd1a3887b05ff8f054e7d08980de",
        "local_mods": "无（全部逐字复制）",
        "note": "未收录上游 tilelang-build / tilelang-dev-guide / tilelang-simplification / tilelang-submit-pr / tilelang-tvm-ir",
    },
    {
        "root": "skills/framework/zero-mismatch-rl",
        "skills_dir": "zero-mismatch-rl",
        "repo": "https://github.com/Inferact/zero-mismatch-rl.git",
        "path": "skill/（另有 cases/ 模型案例）",
        "link_path": "skill",
        "ref": "136aca87d8158adae94b5a272fa4a8a6d5cd6939",
        "local_mods": "SKILL.md 与 README.md 为本地新增入口与说明；skill/、cases/、LICENSE、README.upstream.md 逐字复制",
        "note": "skill/ 与 cases/ 必须保持同级（skill/guides 中以 ../../../cases/... 引用案例）",
    },
    {
        "root": "skills/framework/train-inference-consistency",
        "skills_dir": "train-inference-consistency",
        "repo": "https://github.com/vllm-project/vllm.git",
        "path": ".agents/skills/train-inference-consistency",
        "link_path": ".agents/skills/train-inference-consistency",
        "ref": "2d8efee9a73366d6b4e753780af32443d2fef91a",
        "ref_follow": "main（update-repos.sh 的 VLLM_SKILLS_REF）",
        "local_mods": "SKILL.md 增加中文触发词、vLLM 文档链接改为本目录稀疏 clone、增加 zero-mismatch-rl 交叉引用与来源注释；新增 README.md 与 LICENSE；其余为上游逐字复制",
        "note": "2026-11-08 审计时的上游 tip 不含 .agents/skills/train-inference-consistency；保留本地固定副本（不覆盖）；未来需网络恢复后重新检查上游是否恢复该路径",
    },
    {
        "root": "skills/other/tech-doc-style-chinese",
        "skills_dir": "tech-doc-style-chinese",
        "repo": "https://github.com/Fenng/Tech-Doc-Style-Chinese.git",
        "path": ".（仓库根）",
        "link_path": "",
        "ref": "b119d01c9bb7cd132b3c62fafdff34227c4410d1",
        "local_mods": "本地定制覆盖版（修改对照见其 README.md）；上游 clone 在 repos/tech-doc-style-chinese（git-ignored）",
        "note": "同步上游必须按其 README.md 的步骤 merge，不能直接覆盖本地修改的文件",
    },
]

# catalog：submodule（pin 来自 git gitlink）与 clone（pin 为本地 HEAD）。安装器只读取这些目录。
CATALOGS = [
    {
        "label": "nvidia-skills",
        "kind": "submodule",
        "path": "repos/nvidia-skills",
        "repo": "https://github.com/NVIDIA/skills.git",
        "note": "install.sh 遍历 skills/ 与 plugins/nvidia-skills/skills/；同名先到先得（本地 skill 优先）",
    },
    {
        "label": "amd-skills",
        "kind": "submodule",
        "path": "repos/amd-skills",
        "repo": "https://github.com/amd/skills.git",
        "note": "安装前检查其 skills/ 目录；本仓库只安装 skills/，不安装其他目录",
    },
    {
        "label": "ccfa-skills",
        "kind": "submodule",
        "path": "repos/ccfa-skills",
        "repo": "https://github.com/mikubaka88/CCFA-Skills.git",
        "note": "ccf-* 含 SKILL.md 者为 runtime skill（安装名 = 目录名）；不含者为支撑目录（直接链接，不计为 skill）",
    },
    {
        "label": "cursor-skills",
        "kind": "submodule",
        "path": "repos/cursor-skills",
        "repo": "https://github.com/Saddss/cursor-skills.git",
        "note": "同名时本地 catalog 优先（install.sh 先到先得）",
    },
    {
        "label": "ponytail",
        "kind": "submodule",
        "path": "repos/ponytail",
        "repo": "https://github.com/DietrichGebert/ponytail.git",
        "note": "安装名与本地 SKILL_NAMES 数组一致（SKILL_DIRS 指向 repos/ponytail/skills/*）",
    },
    {
        "label": "tilelang-cuda-skills",
        "kind": "submodule",
        "path": "repos/tilelang-cuda-skills",
        "repo": "https://github.com/sablin39/tilelang-cuda-skills.git",
        "note": "install.sh SKILL_DIRS 直接安装其中 9 个 skill；嵌套 submodule external/TileFoundry 见第 4 节；其余上游 skill 未安装",
    },
    {
        "label": "kernel-design-agents",
        "kind": "submodule",
        "path": "kernel-design-agents",
        "repo": "https://github.com/OpenDarrenlu/kernel-design-agents.git",
        "note": "只安装 KernelWiki（install.sh 从 kernel-design-agents/skills/KernelWiki 复制）；其 .gitmodules 的嵌套 submodule 见第 4 节",
    },
    {
        "label": "tech-doc-style-chinese (clone)",
        "kind": "clone",
        "path": "repos/tech-doc-style-chinese",
        "repo": "https://github.com/Fenng/Tech-Doc-Style-Chinese.git",
        "note": "git-ignored 完整 clone，仅作 diff 参考；安装的是本地定制版 skills/other/tech-doc-style-chinese",
    },
]

# 嵌套 submodule：parent 为直接持有 gitlink 的 checkout（相对仓库根）。pin 取 parent HEAD 的 gitlink，
# origin 取嵌套 checkout 的 remote（实际来源），.gitmodules URL 作为对照记录。
NESTED = [
    {"parent": "kernel-design-agents", "path": "skills/KernelWiki", "installed_as": "KernelWiki",
     "note": "install.sh 安装的 KernelWiki 来自此嵌套目录"},
    {"parent": "kernel-design-agents", "path": "skills/ncu-report-skill", "installed_as": None,
     "note": "未安装；本仓库 skills/kernel/ncu-report-skill 为本地副本，来源见第 5 节"},
    {"parent": "repos/tilelang-cuda-skills", "path": "external/TileFoundry", "installed_as": None,
     "note": "未安装（tilelang-cuda-skills 的嵌套 submodule）"},
    {"parent": "repos/tilelang-cuda-skills/external/TileFoundry", "path": "third_party/cutlass", "installed_as": None,
     "note": "未安装（TileFoundry 的嵌套 submodule）"},
]

# 本地目录的来源证据（按安装名）。未列出者一律为 unknown，不默认原创。
# repo 为上游 web 地址（无 .git）；ref 只记录文档中的明确 pin，文档没有给出则为空，不得编造。
LOCAL_ORIGINS = {
    "iket-profiling": {
        "class": "external-declared",
        "repo": "https://github.com/humanfia/iket-profiling-skill",
        "ref": "",
        "evidence": "skills/kernel/iket-profiling/UPSTREAM.md",
        "note": "UPSTREAM.md 声明安装来源为 humanfia/iket-profiling-skill（无 pin）；内容是 NVIDIA CUTLASS 官方 IKET 文档的摘要，NVIDIA 文档是参考来源，不是本 skill 的 origin",
    },
    "salary-negotiation-skill": {
        "class": "external-declared",
        "repo": "https://github.com/Ssupercoder/Salary-Negotiation-Skill",
        "ref": "2688ef8e4931ab3c069593967ceeed1f79186b3e",
        "evidence": "skills/other/salary-negotiation-skill/UPSTREAM.md",
        "note": "MIT；UPSTREAM.md 记载源路径 谈薪skill_easy/ 与 ref 2688ef8e（文档文字，本次未在线核对）；本地 SKILL.md 保留出处",
    },
    "ncu-report-skill": {
        "class": "external-declared",
        "repo": "https://github.com/mit-han-lab/ncu-report-skill",
        "ref": "",
        "evidence": "skills/kernel/ncu-report-skill/README.md",
        "note": "README 称为 KDA 的独立 submodule；KDA .gitmodules 指向 mit-han-lab/ncu-report-skill；README 安装示例另指 DongyunZou/ncu-report-skill（两处来源待确认）；本地副本无 .git，ref 未知",
    },
    "claude-delegate": {
        "class": "protected",
        "repo": "https://github.com/amElnagdy/delegate-skills",
        "ref": "",
        "evidence": "sources/local-provenance.md（claude-delegate 节）",
        "note": "lockfile 声明来源；ref 未知；保留本地 macOS 补丁，用户级保留名，安装器不覆盖。LICENSE 为 MIT，版权人 Ahmed Mohammed (amElnagdy)",
    },
    "codex-background": {
        "class": "protected",
        "repo": "",
        "ref": "",
        "evidence": "sources/local-provenance.md（codex-background 节）",
        "note": "MIT，版权人 Ahmed Mohammed (amElnagdy)；本次审计未确立上游 URL（unknown）；保留本地补丁，用户级保留名，安装器不覆盖",
    },
    "zotero-cli": {
        "class": "external-declared",
        "repo": "https://github.com/54yyyu/zotero-mcp",
        "ref": "",
        "evidence": "sources/local-provenance.md（zotero-cli 节）",
        "note": "2026-11-12 编排器核验：安装包 zotero-mcp-server 0.15.0 声明 Homepage https://github.com/54yyyu/zotero-mcp；skill 由 `zotero-mcp install-skill` 生成（AGENTS.md 托管块）。本地 skill 与包内模板未逐字核对（baseline equality unknown）",
    },
}

# VeloQ：install-veloq.sh 安装的可选 skill（不在 install.sh 数组中）。
VELOQ_REPO = "https://github.com/lucifer1004/veloq"


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def sh(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, cwd=ROOT, timeout=120).stdout.strip()
    except Exception:
        return ""


def web(repo):
    """仓库地址去掉 .git 后缀，用于 /tree/ 链接。"""
    return repo[:-4] if repo.endswith(".git") else repo


def tree_link(repo, ref, path=""):
    """GitHub 链接：有 ref 时 <repo>/tree/<ref>[/<path>]；无 ref 只给仓库地址；无 repo 为空。"""
    if not repo:
        return ""
    if not ref:
        return repo
    return "%s/tree/%s%s" % (repo, ref, "/" + path if path else "")


def load_inventory():
    spec = importlib.util.spec_from_file_location("build_inventory", INVENTORY_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_conf():
    path = os.path.join(ROOT, "sources", "external-sources.conf")
    rows = []
    if not os.path.isfile(path):
        return rows
    for line in read(path).splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split("|")
        if len(f) < 6:
            continue
        rows.append({"name": f[0], "url": f[1], "ref": f[2], "mode": f[3],
                     "skills_dir": f[4], "runtime": f[5],
                     "provenance": f[6] if len(f) > 6 else ""})
    return rows


def parse_install_names():
    text = read(INSTALL_SH)
    names = re.search(r"^SKILL_NAMES=\((.*?)^\)", text, re.S | re.M)
    if not names:
        raise SystemExit("无法从 install.sh 读取 SKILL_NAMES")
    return names.group(1).split()


def parse_veloq_names():
    m = re.search(r"^VELOQ_SKILLS=\((.*?)\)", read(VELOQ_SH), re.M)
    return m.group(1).split() if m else []


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


def gitlink_of(parent_abs, path):
    """父 checkout HEAD 中 path 的 gitlink（不联网，只读本地对象库）。"""
    m = re.match(r"160000 commit ([0-9a-f]{40})", sh(["git", "-C", parent_abs, "ls-tree", "HEAD", path]))
    return m.group(1) if m else ""


def ref_of_vendored(root):
    """vendored 的 ref 以本地 README 的基线 commit 为准（脚本常量仅作 fallback）。"""
    readme = os.path.join(ROOT, root, "README.md")
    if os.path.isfile(readme):
        m = re.search(r"基线[^`]*`([0-9a-f]{40})`", read(readme))
        if m:
            return m.group(1)
    return ""


def checkout_roots():
    """仓库相对路径（checkout 根）-> 来源记录。用于把 inventory 的 source 映射到上游。"""
    roots = {}
    for cat in CATALOGS:
        if cat["kind"] == "clone":
            continue
        roots[cat["path"]] = {"class": "catalog", "repo": web(cat["repo"]),
                              "ref": pin_of(cat["kind"], cat["path"]), "label": cat["label"]}
    for row in parse_conf():
        roots["repos/%s" % row["name"]] = {"class": "external", "repo": web(row["url"]),
                                           "ref": row["ref"], "label": row["name"],
                                           "runtime": row["runtime"]}
    for nd in NESTED:
        parent_abs = os.path.join(ROOT, nd["parent"])
        base = os.path.join(nd["parent"], nd["path"]).replace(os.sep, "/")
        origin = sh(["git", "-C", os.path.join(ROOT, base), "remote", "get-url", "origin"])
        gitmodules = sh(["git", "-C", parent_abs, "config", "-f", ".gitmodules", "--get",
                         "submodule.%s.url" % nd["path"]])
        roots[base] = {"class": "nested", "repo": web(origin or gitmodules), "ref": gitlink_of(parent_abs, nd["path"]),
                       "label": base, "parent": nd["parent"], "origin": web(origin), "gitmodules": web(gitmodules),
                       "installed_as": nd["installed_as"], "note": nd["note"]}
    return roots


def match_root(source, keys):
    best = ""
    for key in keys:
        if (source == key or source.startswith(key + "/")) and len(key) > len(best):
            best = key
    return best


def origin_of(source, name, status, roots):
    """返回一个 skill 的来源记录：key（所属来源）、class、url、ref、note。"""
    for vend in VENDORED:
        if source == vend["root"] or source.startswith(vend["root"] + "/"):
            rel = source[len(vend["root"]):].lstrip("/")
            path = "/".join(p for p in (vend["link_path"], rel) if p)
            ref = ref_of_vendored(vend["root"]) or vend["ref"]
            return {"key": vend["root"], "class": "vendored", "url": tree_link(web(vend["repo"]), ref, path),
                    "ref": ref, "note": vend["local_mods"]}
    key = match_root(source, roots.keys())
    if key:
        info = roots[key]
        rel = source[len(key):].lstrip("/")
        path = rel
        if info["class"] == "external" and info["runtime"].startswith("npm:") and rel.startswith("runtime-skills"):
            path = ""  # runtime-skills 由 npm 包生成，不在 git tree 中
        note = info.get("note", "")
        if info["class"] == "nested":
            note = "%s；origin=%s；.gitmodules=%s" % (info["note"], info["origin"], info["gitmodules"])
        return {"key": key, "class": info["class"], "url": tree_link(info["repo"], info["ref"], path),
                "ref": info["ref"], "note": note}
    if name in LOCAL_ORIGINS:
        origin = LOCAL_ORIGINS[name]
        return {"key": "local", "class": origin["class"],
                "url": tree_link(origin["repo"], origin["ref"]) if origin["repo"] else "",
                "ref": origin["ref"], "note": origin["note"]}
    return {"key": "local", "class": "unknown", "url": "", "ref": "",
            "note": "本次审计未确立来源（未逐一读取该目录全部文件，不断言其无上游声明）"}


def collect_skills(inv, roots):
    """inventory 的 skill 行（状态由 build_inventory 决定）+ VeloQ 可选 skill。"""
    skills = []
    for row in inv.build_rows(ROOT):
        if row[0] != "skill":
            continue
        _, catalog, name, status, shadowed_by, source, _ = row
        origin = origin_of(source, name, status, roots)
        skills.append({"name": name, "catalog": catalog, "status": status, "shadowed_by": shadowed_by,
                       "source": source, **origin})
    for name in parse_veloq_names():
        skills.append({"name": name, "catalog": "veloq", "status": "optional-runtime", "shadowed_by": "-",
                       "source": "install-veloq.sh", "key": "veloq", "class": "optional-runtime",
                       "url": VELOQ_REPO, "ref": "",
                       "note": "install-veloq.sh 从 raw.githubusercontent.com/lucifer1004/veloq/main/scripts/install.sh 安装（main，无 pin）；--no-veloq 跳过"})
    return skills


def tracked_uncovered(skills):
    """仓库内 git 跟踪的 SKILL.md 目录中，未被任何已列 skill 覆盖的（即仓库有、未安装）。"""
    dirs = sorted({os.path.dirname(p) for p in sh(["git", "ls-files", "--", "*SKILL.md"]).splitlines()})
    covered = [s["source"] for s in skills]
    out = []
    for d in dirs:
        if any(d == c or d.startswith(c + "/") or c.startswith(d + "/") for c in covered):
            continue
        out.append(d)
    return out


def missing_install_names(skills):
    """install.sh SKILL_NAMES 中没有被 installed/protected 覆盖的名称（应为空）。"""
    have = {s["name"] for s in skills if s["status"] in ("installed", "protected")}
    return [n for n in parse_install_names() if n not in have]


def cell(text):
    return str(text).replace("|", "\\|") if text else "-"


def render(skills, roots):
    L = []
    L.append("# Skill 来源登记（SOURCES.md）")
    L.append("")
    L.append("本文件由 `python3 scripts/build_sources.py` 生成，请勿手工编辑正文。")
    L.append("skill 的发现、同名先到先得与状态来自 `skills/routers/agent-skill-router/scripts/build_inventory.py`；")
    L.append("本文件逐 skill 给出来源类别、上游链接、ref 与说明。")
    L.append("")
    L.append("## 类别说明")
    L.append("")
    L.append("| 类别 | 含义 | 同步方式 |")
    L.append("| --- | --- | --- |")
    L.append("| vendored | 从上游逐字复制的快照，可能有本地修改 | 有本地修改的文件 merge，其余直接覆盖 |")
    L.append("| catalog | submodule / clone 中的 skill 目录，只读取 | `bash update-repos.sh <name>` 更新 pin |")
    L.append("| external | `sources/external-sources.conf` 的外部来源 | 更新 conf 中的 ref 后 `bash update-repos.sh external` |")
    L.append("| nested | catalog 的嵌套 submodule | 以嵌套仓库的 origin 与 gitlink 为准 |")
    L.append("| external-declared | 第三方来源已在本地文件或包元数据中声明；基线与内容比对未验证 | 按声明的上游手工核对，比对后再决定是否 merge |")
    L.append("| protected | 用户级保留名，安装器不覆盖；来源可能已声明，保留本地补丁 | 不由本仓库同步或覆盖 |")
    L.append("| unknown | 本次审计未确立来源（不断言无声明） | 人工确认来源；不得默认为原创 |")
    L.append("| optional-runtime | 可选运行时 skill（VeloQ） | 由 install-veloq.sh 管理 |")
    L.append("")
    L.append("上游链接：`<repo>/tree/<ref>/<路径>`，仓库地址不带 `.git`。无 ref 时只给仓库地址。")
    L.append("")

    counts = {}
    for s in skills:
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    uncovered = tracked_uncovered(skills)
    L.append("## 1. 汇总")
    L.append("")
    L.append("| 状态 | 数量 | 含义 |")
    L.append("| --- | --- | --- |")
    L.append("| installed | %d | 来源可安装（install.sh 选中并可装入，不表示已安装到宿主） |" % counts.get("installed", 0))
    L.append("| shadowed | %d | 同名已被先到来源占用，不安装 |" % counts.get("shadowed", 0))
    L.append("| protected | %d | 用户级保留名，不计为可安装 |" % counts.get("protected", 0))
    L.append("| missing-source | %d | 本地缺文件 |" % counts.get("missing-source", 0))
    L.append("| optional-runtime | %d | VeloQ，可选，不在 install.sh 数组中 |" % counts.get("optional-runtime", 0))
    L.append("| tracked-not-installed | %d | 仓库内有 SKILL.md，未安装 |" % len(uncovered))
    L.append("")
    L.append("注：installed 数量与 `agent-skill-router/references/inventory.tsv` 的 installed 行一致，只表示安装器会选中该来源，不验证宿主上的安装结果。")
    L.append("")

    L.append("## 2. 逐来源概览")
    L.append("")
    L.append("| 来源 | 类别 | 上游仓库 | ref | 安装 skill 数 |")
    L.append("| --- | --- | --- | --- | --- |")
    for v in VENDORED:
        n = len([s for s in skills if s["key"] == v["root"] and s["status"] == "installed"])
        ref = ref_of_vendored(v["root"]) or v["ref"]
        L.append("| `%s` | vendored | %s | `%s` | %d |" % (v["root"], web(v["repo"]), ref, n))
    for cat in CATALOGS:
        key = cat["path"]
        n = len([s for s in skills if (s["key"] == key or s["key"].startswith(key + "/")) and s["status"] == "installed"])
        pin = pin_of(cat["kind"], cat["path"])
        L.append("| `%s` | catalog | %s | `%s` | %d |" % (cat["label"], web(cat["repo"]), pin, n))
    for name in sorted(k for k in roots if roots[k]["class"] == "external"):
        info = roots[name]
        n = len([s for s in skills if s["key"] == name and s["status"] == "installed"])
        L.append("| `%s` | external | %s | `%s` | %d |" % (info["label"], info["repo"], info["ref"], n))
    L.append("| `veloq` | optional-runtime | %s | 未固定（main） | %d |"
             % (VELOQ_REPO, len([s for s in skills if s["catalog"] == "veloq"])))
    L.append("")

    L.append("## 3. 逐 skill 来源")
    L.append("")
    L.append("| 安装名 | 状态 | 占用者 | 类别 | 上游链接 | ref | 本地路径 | 说明 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in sorted(skills, key=lambda x: (x["name"], x["status"], x["source"])):
        L.append("| `%s` | %s | %s | %s | %s | %s | `%s` | %s |"
                 % (s["name"], s["status"], cell(s["shadowed_by"]), s["class"],
                    cell(s["url"]), cell(s["ref"][:12]) if s["ref"] else "-",
                    s["source"], cell(s["note"])))
    for path in uncovered:
        L.append("| `%s` | tracked-not-installed | - | - | - | - | `%s` | 仓库内有 SKILL.md，install.sh 未安装 |"
                 % (os.path.basename(path), path))
    L.append("")

    L.append("## 4. 嵌套 checkout（nested）")
    L.append("")
    L.append("| 嵌套路径 | 父 checkout | 实际 origin / 来源 | gitlink ref | 安装 | 说明 |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for key, info in sorted(roots.items()):
        if info["class"] != "nested":
            continue
        L.append("| `%s` | `%s` | %s | `%s` | %s | %s |"
                 % (key, info["parent"], info["origin"] or "-",
                    info["ref"] or "-", info["installed_as"] or "否", info["note"]))
    L.append("")

    L.append("## 5. 本地 skill 来源证据")
    L.append("")
    L.append("本表只列有明确证据的条目。其余本地 skill 为 `unknown`，不默认原创。")
    L.append("")
    L.append("| 安装名 | 类别 | 上游仓库 | ref | 证据文件 | 说明 |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for name in sorted(LOCAL_ORIGINS):
        o = LOCAL_ORIGINS[name]
        L.append("| `%s` | %s | %s | %s | `%s` | %s |"
                 % (name, o["class"], o["repo"] or "-", "`%s`" % o["ref"] if o["ref"] else "-",
                    o["evidence"], o["note"]))
    L.append("")
    unknown = sorted(s["name"] for s in skills if s["class"] == "unknown")
    L.append("未验证来源（unknown，%d 个）：%s。" % (len(unknown), ", ".join("`%s`" % n for n in unknown) or "无"))
    L.append("")

    L.append("## 6. vendored 本地修改与 merge 要求")
    L.append("")
    for v in VENDORED:
        L.append("- **`%s`**（%s）：%s。%s" % (v["root"], v["skills_dir"], v["local_mods"], v["note"]))
    L.append("")
    L.append("2026-11-08 审计时上游 tip 不含路径的 vendored 副本（如 train-inference-consistency）保留本地固定副本，不得以上游缺失为由删除本地 skill。")
    L.append("")

    L.append("## 7. 覆盖检查")
    L.append("")
    missing = missing_install_names(skills)
    L.append("- install.sh SKILL_NAMES 未被 installed/protected 覆盖：%s" % (", ".join(missing) if missing else "无"))
    L.append("- 仓库内 SKILL.md 未安装（tracked-not-installed）：%d 个（见第 3 节）" % len(uncovered))
    L.append("")

    L.append("## 检查与更新流程")
    L.append("")
    L.append("```bash")
    L.append("python3 scripts/build_sources.py            # 重新生成 SOURCES.md")
    L.append("python3 scripts/build_sources.py --check    # 校验 SOURCES.md 与当前 checkout 一致")
    L.append("bash update-repos.sh all                    # 更新全部 submodule / 外部 checkout")
    L.append("```")
    L.append("")
    L.append("上游 tip 的检查结论记录在 `sources/update-log.md`（手工维护）。无网络时记为失败，不得记为无更新。")
    L.append("")

    if os.path.isfile(LOG):
        L.append("<!-- update-log -->")
        L.append(read(LOG).rstrip("\n"))
        L.append("")

    return "\n".join(L) + "\n"


def build_text():
    inv = load_inventory()
    roots = checkout_roots()
    skills = collect_skills(inv, roots)
    return render(skills, roots)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    text = build_text()
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
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print("已写入 %s（%d 行）" % (os.path.relpath(OUT, ROOT), text.count("\n")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
