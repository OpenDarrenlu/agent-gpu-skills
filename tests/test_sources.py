"""SOURCES.md 生成器测试（scripts/build_sources.py）。

运行（仓库根目录）:
  python3 -m unittest discover -s tests -p 'test_sources*.py' -v

测试分五类:
  1. 链接规范：仓库地址去掉 .git；/tree/ 链接不含 .git/tree；无 ref 时只给仓库地址。
  2. 状态：installed 与 inventory 一致；shadowed / protected 不计为安装；不接受 --fetch。
  3. 来源分类：有证据的本地 skill 不标为原创；无声明者为 unknown；保留名为 protected；证据文件存在且 ref 可核对。
  4. 嵌套 checkout：KernelWiki 的 ref 为 kernel-design-agents HEAD 的 gitlink，origin 为嵌套 checkout 的 remote。
  5. 覆盖与稳定性：install.sh 的 SKILL_NAMES 全部被覆盖；VeloQ 可选 skill 被列出；重复生成结果一致且与磁盘一致。

真实仓库数据只读取一次（setUpModule），不联网。
"""

import importlib.util
import os
import re
import subprocess
import sys
import unittest
from collections import Counter

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(REPO, "scripts", "build_sources.py")
HEX40 = re.compile(r"^[0-9a-f]{40}$")

M = None
SKILLS = None
ROOTS = None
TEXT = None


def load_module():
    spec = importlib.util.spec_from_file_location("build_sources", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setUpModule():
    global M, SKILLS, ROOTS, TEXT
    M = load_module()
    ROOTS = M.checkout_roots()
    SKILLS = M.collect_skills(M.load_inventory(), ROOTS)
    TEXT = M.build_text()


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def by_name(name, status="installed"):
    matches = [s for s in SKILLS if s["name"] == name and s["status"] == status]
    if len(matches) != 1:
        raise AssertionError("%s/%s: expected 1 row, got %d" % (name, status, len(matches)))
    return matches[0]


class LinkTests(unittest.TestCase):
    def test_web_strips_git_suffix(self):
        self.assertEqual(M.web("https://github.com/a/b.git"), "https://github.com/a/b")
        self.assertEqual(M.web("https://github.com/a/b"), "https://github.com/a/b")

    def test_tree_link_shapes(self):
        self.assertEqual(M.tree_link("https://x/y", ""), "https://x/y")
        self.assertEqual(M.tree_link("https://x/y", "abc"), "https://x/y/tree/abc")
        self.assertEqual(M.tree_link("https://x/y", "abc", "p/q"), "https://x/y/tree/abc/p/q")
        self.assertEqual(M.tree_link("", "abc", "p"), "")

    def test_registry_has_no_git_tree_links(self):
        self.assertIsNone(re.search(r"\.git/tree/", TEXT))

    def test_catalog_and_vendored_links_pin_to_ref(self):
        for s in SKILLS:
            if s["class"] in ("catalog", "external", "vendored", "nested") and s["status"] == "installed":
                self.assertTrue(HEX40.match(s["ref"]), (s["name"], s["ref"]))
                self.assertIn("/tree/" + s["ref"], s["url"], s["name"])
                self.assertNotIn(".git/tree", s["url"], s["name"])


class StatusTests(unittest.TestCase):
    def test_installed_count_matches_inventory(self):
        inv_rows = [r for r in M.load_inventory().build_rows(M.ROOT) if r[0] == "skill"]
        expected = sum(1 for r in inv_rows if r[3] == "installed")
        self.assertEqual(sum(1 for s in SKILLS if s["status"] == "installed"), expected)

    def test_installed_names_are_unique(self):
        counts = Counter(s["name"] for s in SKILLS if s["status"] == "installed")
        self.assertEqual([n for n, c in counts.items() if c > 1], [])

    def test_shadowed_rows_name_their_owner_and_are_not_installed(self):
        shadowed = [s for s in SKILLS if s["status"] == "shadowed"]
        self.assertTrue(shadowed)
        for s in shadowed:
            self.assertNotEqual(s["shadowed_by"], "-", s["name"])
        self.assertEqual(by_name("tdd", "shadowed")["shadowed_by"], "cursor-skills")
        self.assertEqual(by_name("tdd")["catalog"], "cursor-skills")

    def test_protected_rows_are_protected_regardless_of_origin(self):
        # 保留名的来源可以已声明或未确立；两者都不能改变"保留、不覆盖"的分类
        for name in ("claude-delegate", "codex-background"):
            s = by_name(name, "protected")
            self.assertEqual(s["class"], "protected")
            self.assertEqual(s["status"], "protected")

    def test_summary_counts_exclude_shadowed_and_protected(self):
        fixture = [
            {"name": "a", "status": "installed", "shadowed_by": "-", "class": "local", "url": "", "ref": "",
             "source": "x/a", "note": "", "key": "local", "catalog": "local"},
            {"name": "b", "status": "shadowed", "shadowed_by": "local", "class": "catalog", "url": "", "ref": "",
             "source": "x/b", "note": "", "key": "repos/c", "catalog": "c"},
            {"name": "c", "status": "protected", "shadowed_by": "-", "class": "protected", "url": "", "ref": "",
             "source": "x/c", "note": "", "key": "local", "catalog": "local"},
        ]
        text = M.render(fixture, {})
        self.assertIn("| installed | 1 |", text)
        self.assertIn("| shadowed | 1 |", text)
        self.assertIn("| protected | 1 |", text)

    def test_fetch_flag_is_rejected_and_not_documented(self):
        # update-log 中的历史说明可以提到 --fetch；只检查生成部分
        generated = TEXT.split("<!-- update-log -->")[0]
        self.assertNotIn("--fetch", generated)
        proc = subprocess.run([sys.executable, SCRIPT, "--fetch"], capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("unrecognized", proc.stderr)


class ProvenanceTests(unittest.TestCase):
    def test_declared_locals_are_external_not_original(self):
        for name in ("iket-profiling", "salary-negotiation-skill", "ncu-report-skill"):
            s = by_name(name)
            self.assertEqual(s["class"], "external-declared", name)
            self.assertTrue(s["url"].startswith("https://github.com/"), name)

    def test_zotero_origin_from_installed_package_metadata(self):
        s = by_name("zotero-cli")
        self.assertEqual(s["class"], "external-declared")
        self.assertEqual(s["url"], "https://github.com/54yyyu/zotero-mcp")
        self.assertIn("baseline equality unknown", s["note"])

    def test_claude_delegate_origin_from_tracked_note_and_stays_protected(self):
        # 来源事实记录在 tracked 的 sources/local-provenance.md，而非 git-ignored 的 skills-lock.json
        note = read(os.path.join(REPO, PROVENANCE))
        self.assertIn("amElnagdy/delegate-skills", note)
        self.assertIn("skills/claude-delegate/SKILL.md", note)
        s = by_name("claude-delegate", "protected")
        self.assertEqual(s["url"], "https://github.com/amElnagdy/delegate-skills")
        self.assertEqual(s["ref"], "")
        self.assertEqual(s["class"], "protected")

    def test_codex_background_origin_unknown_and_protected(self):
        # 当前审计结论为 unknown：只断言本审计的结论，不断言任何机器上的元数据都不存在
        s = by_name("codex-background", "protected")
        self.assertEqual(s["url"], "")
        self.assertEqual(s["class"], "protected")

    def test_undeclared_local_is_unknown_not_original(self):
        s = by_name("cuda-skill")
        self.assertEqual(s["class"], "unknown")
        self.assertEqual(s["url"], "")
        self.assertNotIn("original", {x["class"] for x in SKILLS})

    def test_local_origin_evidence_is_tracked_and_refs_are_stated_in_it(self):
        # 所有证据都指向 tracked 文件，新 checkout 上必然存在
        for name, origin in M.LOCAL_ORIGINS.items():
            evidence = origin["evidence"].split("（")[0].strip()
            path = os.path.join(REPO, evidence)
            self.assertTrue(os.path.isfile(path), (name, evidence))
            if origin["ref"]:
                self.assertTrue(HEX40.match(origin["ref"]), name)
                self.assertIn(origin["ref"], read(path), name)

    def test_zotero_origin_is_recorded_in_tracked_note(self):
        note = read(os.path.join(REPO, PROVENANCE))
        self.assertIn("zotero-mcp-server", note)
        self.assertIn("0.15.0", note)
        self.assertIn("https://github.com/54yyyu/zotero-mcp", note)

    def test_declared_refs_are_only_from_documents(self):
        # 不编造 hash：没有文档记载 ref 的条目 ref 必须为空
        self.assertEqual(M.LOCAL_ORIGINS["iket-profiling"]["ref"], "")
        self.assertEqual(M.LOCAL_ORIGINS["ncu-report-skill"]["ref"], "")
        self.assertEqual(M.LOCAL_ORIGINS["zotero-cli"]["ref"], "")
        self.assertEqual(M.LOCAL_ORIGINS["claude-delegate"]["ref"], "")

    def test_no_entry_claims_local_original(self):
        for name, origin in M.LOCAL_ORIGINS.items():
            self.assertNotEqual(origin["class"], "original", name)


NESTED = "kernel-design-agents/skills/KernelWiki"
PROVENANCE = "sources/local-provenance.md"


class NestedTests(unittest.TestCase):
    def test_kernelwiki_ref_is_nested_gitlink(self):
        gitlink = M.gitlink_of(os.path.join(REPO, "kernel-design-agents"), "skills/KernelWiki")
        self.assertTrue(HEX40.match(gitlink), gitlink)
        s = by_name("KernelWiki")
        self.assertEqual(s["ref"], gitlink)
        self.assertEqual(s["class"], "nested")

    @unittest.skipUnless(os.path.isdir(os.path.join(REPO, NESTED)), "nested KernelWiki checkout absent")
    def test_kernelwiki_origin_is_nested_checkout_remote(self):
        origin = M.sh(["git", "-C", os.path.join(REPO, NESTED), "remote", "get-url", "origin"])
        self.assertTrue(origin)
        self.assertEqual(ROOTS[NESTED]["origin"], M.web(origin))
        self.assertIn(M.web(origin), by_name("KernelWiki")["url"])

    def test_nested_gitmodules_url_is_recorded_separately(self):
        self.assertIn("mit-han-lab/KernelWiki", ROOTS[NESTED]["note"] + ROOTS[NESTED].get("gitmodules", "")
                      + " " + TEXT)


class CoverageTests(unittest.TestCase):
    def test_every_install_name_is_covered(self):
        self.assertEqual(M.missing_install_names(SKILLS), [])

    def test_veloq_optional_skills_listed_in_order(self):
        names = [s["name"] for s in SKILLS if s["status"] == "optional-runtime"]
        self.assertEqual(names, ["nsys-profile-analysis", "ncu-profile-analysis"])

    def test_regeneration_is_stable_and_matches_disk(self):
        self.assertEqual(M.build_text(), TEXT)
        self.assertEqual(read(M.OUT), TEXT)

    def test_tracked_uncovered_dirs_are_not_installed(self):
        installed_sources = {s["source"] for s in SKILLS}
        for path in M.tracked_uncovered(SKILLS):
            self.assertNotIn(path, installed_sources)


if __name__ == "__main__":
    unittest.main()
