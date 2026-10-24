#!/usr/bin/env python3
"""生成 agent-skill-router 的安装清单 references/inventory.tsv。

清单按 install.sh 的安装顺序模拟同名占用（先到先得）：
本地 SKILL_NAMES -> KernelWiki -> NVIDIA -> Cursor -> CCFA -> AMD -> sources/external-sources.conf 中 mode=skills 的来源。
保留名（claude-delegate、codex-background）永远不计为可安装 skill。

只读取各 SKILL.md 的 frontmatter（name/description），不索引 skill 正文或 references。

用法:
  python3 build_inventory.py                 # 重新生成 references/inventory.tsv
  python3 build_inventory.py --check         # 与现有文件比较；不一致返回 1，不写文件
  python3 build_inventory.py --root DIR      # 指定仓库根目录（测试或其他工作树）

清单依赖当前 checkout：submodule/外部来源缺失时对应 catalog 会标为 absent。
更换 checkout 或更新 submodule 后，请先运行 bash update-repos.sh 或 bash bootstrap.sh --no-update，
再运行本脚本生成清单。
"""

import argparse
import glob
import os
import re
import sys

PROTECTED_NAMES = ("claude-delegate", "codex-background")
DESC_LIMIT = 140
HEADER = (
    "# 由 skills/routers/agent-skill-router/scripts/build_inventory.py 生成；请勿手工编辑。\n"
    "# 列: kind  catalog  name  status  shadowed_by  source  description\n"
    "# status: installed | shadowed（同名先到者占用） | protected（保留名） | missing-source（本地缺文件）\n"
    "#         present / absent 仅用于 kind=catalog 的 catalog 行\n"
)
COLUMNS = ("kind", "catalog", "name", "status", "shadowed_by", "source", "description")


def default_root():
    here = os.path.realpath(__file__)
    # scripts -> agent-skill-router -> routers -> skills -> repo
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here)))))


def read_install_arrays(root):
    with open(os.path.join(root, "install.sh"), encoding="utf-8") as f:
        text = f.read()
    names = re.search(r"^SKILL_NAMES=\((.*?)^\)", text, re.S | re.M)
    dirs = re.search(r"^SKILL_DIRS=\((.*?)^\)", text, re.S | re.M)
    if not names or not dirs:
        raise SystemExit("无法从 install.sh 读取 SKILL_NAMES / SKILL_DIRS")
    name_list = names.group(1).split()
    dir_list = dirs.group(1).split()
    if len(name_list) != len(dir_list):
        raise SystemExit("install.sh 中 SKILL_NAMES 与 SKILL_DIRS 数量不一致")
    return list(zip(name_list, dir_list))


def read_frontmatter(path):
    """返回 (name, description)；只解析 frontmatter 的 name 与 description 字段。"""
    name = ""
    parts = []
    in_description = False
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return name, ""
    if not lines or lines[0].strip() != "---":
        return name, ""
    for line in lines[1:]:
        if line.strip() == "---":
            break
        match = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if match:
            key, value = match.group(1), match.group(2).strip()
            in_description = key == "description"
            if key == "name":
                name = value.strip("\"'")
            elif in_description:
                parts = [value] if value not in ("", ">", "|", ">-", "|-") else []
        elif in_description and line[:1] in (" ", "\t"):
            parts.append(line.strip())
        else:
            in_description = False
    description = " ".join(p.strip("\"'") for p in parts if p)
    description = re.sub(r"\s+", " ", description).strip()
    return name, description[:DESC_LIMIT].rstrip()


def skill_dirs_under(base, pattern="*"):
    """直接子目录中含 SKILL.md 的 skill 目录，按名称排序。"""
    found = []
    for path in sorted(glob.glob(os.path.join(base, pattern))):
        if os.path.isfile(os.path.join(path, "SKILL.md")):
            found.append(path)
    return found


def recursive_skill_dirs(base):
    """递归查找 SKILL.md（与 install.sh 的 install_conf_catalogs 同一规则）。"""
    found = []
    if not os.path.isdir(base):
        return found
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d != "node_modules")
        if "SKILL.md" in filenames:
            found.append(dirpath)
    return sorted(found)


def read_conf_rows(root):
    conf = os.path.join(root, "sources", "external-sources.conf")
    rows = []
    if not os.path.isfile(conf):
        return rows
    with open(conf, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("|")
            if len(fields) < 7:
                raise SystemExit("external-sources.conf 字段不足: " + line)
            rows.append(fields[:7])
    return rows


def build_catalogs(root):
    """返回 [(catalog_id, catalog_root_rel, [(skill_dir_abs, install_name_or_None)] , present)]，顺序即安装优先级。"""
    catalogs = []

    local = []
    for name, rel in read_install_arrays(root):
        local.append((os.path.join(root, rel), name))
    catalogs.append(("local", "install.sh:SKILL_DIRS", local))

    kernelwiki = os.path.join(root, "kernel-design-agents", "skills", "KernelWiki")
    catalogs.append(("kernel-design-agents", "kernel-design-agents/skills/KernelWiki",
                     [(kernelwiki, "KernelWiki")] if os.path.isfile(os.path.join(kernelwiki, "SKILL.md")) else []))

    nvidia = []
    for base in ("repos/nvidia-skills/skills", "repos/nvidia-skills/plugins/nvidia-skills/skills"):
        nvidia += [(d, None) for d in skill_dirs_under(os.path.join(root, base))]
    catalogs.append(("nvidia-skills", "repos/nvidia-skills", nvidia))

    cursor = [(d, None) for d in skill_dirs_under(os.path.join(root, "repos/cursor-skills/skills"))]
    catalogs.append(("cursor-skills", "repos/cursor-skills/skills", cursor))

    ccfa = [(d, None) for d in skill_dirs_under(os.path.join(root, "repos/ccfa-skills"), "ccf-*")]
    catalogs.append(("ccfa-skills", "repos/ccfa-skills", ccfa))

    amd = [(d, None) for d in skill_dirs_under(os.path.join(root, "repos/amd-skills/skills"))]
    catalogs.append(("amd-skills", "repos/amd-skills/skills", amd))

    # pending 来源没有可安装的 skill（尚未检视），仍列为 absent，便于路由器说明缺口。
    # skills_dir 可用逗号列出多个目录；skill 名取 frontmatter name（与 install.sh 的 frontmatter_skill_name 一致）。
    for name, url, ref, mode, skills_dir, runtime, provenance in read_conf_rows(root):
        if mode == "none":
            continue
        if mode == "pending":
            catalogs.append((name, "repos/%s" % name, []))
            continue
        dirs = []
        rels = []
        for sd in skills_dir.split(","):
            rel = os.path.normpath("repos/%s/%s" % (name, sd))
            rels.append(rel)
            for d in recursive_skill_dirs(os.path.join(root, rel)):
                dirs.append((d, frontmatter_name(d)))
        catalogs.append((name, ", ".join(rels), dirs))
    return catalogs


def frontmatter_name(skill_dir):
    """与 install.sh 的 frontmatter_skill_name 相同：name 合法则用它，否则用目录的真实名称。"""
    name = ""
    try:
        with open(os.path.join(skill_dir, "SKILL.md"), encoding="utf-8", errors="replace") as f:
            for line in f:
                match = re.match(r"^name:\s*(.*)$", line)
                if match:
                    # 与 install.sh 的 tr -d "\"' \r" 相同
                    name = re.sub(r"[\"' \r]", "", match.group(1).rstrip("\n"))
                    break
    except OSError:
        pass
    if not name or not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        return os.path.basename(os.path.abspath(skill_dir))
    return name


def describe(skill_dir, install_name):
    # 安装名取目录名（与 install.sh 一致）；frontmatter name 只用于描述。
    _, description = read_frontmatter(os.path.join(skill_dir, "SKILL.md"))
    return (install_name or os.path.basename(skill_dir)), description


def build_rows(root):
    rows = []
    claimed = {}
    for catalog_id, catalog_rel, candidates in build_catalogs(root):
        present = bool(candidates)
        rows.append(("catalog", catalog_id, "-", "present" if present else "absent", "-", catalog_rel, "-"))
        for skill_dir, install_name in candidates:
            if not os.path.isfile(os.path.join(skill_dir, "SKILL.md")):
                # 本地 SKILL_DIRS 指向的目录缺文件：不占用名称。
                name = install_name or os.path.basename(skill_dir)
                rows.append(("skill", catalog_id, name, "missing-source", "-", _rel(root, skill_dir), "-"))
                continue
            name, description = describe(skill_dir, install_name)
            source = _rel(root, skill_dir)
            if name in PROTECTED_NAMES:
                rows.append(("skill", catalog_id, name, "protected", "-", source, description))
            elif name in claimed:
                rows.append(("skill", catalog_id, name, "shadowed", claimed[name], source, description))
            else:
                claimed[name] = catalog_id
                rows.append(("skill", catalog_id, name, "installed", "-", source, description))
    return rows


def _rel(root, path):
    return os.path.relpath(path, root).replace(os.sep, "/")


def render(rows):
    lines = [HEADER.rstrip("\n"), "\t".join(COLUMNS)]
    for row in rows:
        lines.append("\t".join(_clean(col) for col in row))
    return "\n".join(lines) + "\n"


def _clean(value):
    return re.sub(r"[\t\r\n]+", " ", value)


def inventory_path(root):
    return os.path.join(root, "skills", "routers", "agent-skill-router", "references", "inventory.tsv")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=default_root(), help="仓库根目录")
    parser.add_argument("--check", action="store_true", help="只检查现有 inventory.tsv 是否与当前 checkout 一致")
    args = parser.parse_args(argv)

    root = os.path.abspath(args.root)
    text = render(build_rows(root))
    path = inventory_path(root)

    if args.check:
        current = ""
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                current = f.read()
        if current == text:
            print("inventory 与当前 checkout 一致（%d 行）" % text.count("\n"))
            return 0
        old = set(current.splitlines())
        new = set(text.splitlines())
        print("inventory 已过期: 新增 %d 行，删除 %d 行。运行 python3 %s 重新生成"
              % (len(new - old), len(old - new), os.path.relpath(os.path.realpath(__file__), root)))
        return 1

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("已写入 %s（%d 行）" % (os.path.relpath(path, root), text.count("\n")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
