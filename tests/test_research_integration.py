"""Research 分组集成测试：外部来源 pin、skills/research 导航链接、清单与隔离安装的一致性。

检查的是真实行为：源码 HEAD 与 sources/external-sources.conf 中的 pin 一致；
隔离安装（混合模式与 --copy）后，12 个 Supervisor-Skills 与 17 个 CCFA runtime skill 都可用，
ccf-latex-templates 支撑目录随 CCFA 一起安装；skill 正文中的相对引用在安装目录内都能解析。

运行（仓库根目录；需要已检出的 repos/ccfa-skills 与 repos/supervisor-skills）:
  python3 -m unittest discover -s tests -p 'test_research_integration.py' -v

安装只写入临时目录，HOME 也指向临时目录；不联网，不安装运行时 CLI。
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUPERVISOR = os.path.join(REPO, "repos", "supervisor-skills")
CCFA = os.path.join(REPO, "repos", "ccfa-skills")
RESEARCH = os.path.join(REPO, "skills", "research")
CONF = os.path.join(REPO, "sources", "external-sources.conf")
INVENTORY = os.path.join(REPO, "skills", "routers", "agent-skill-router", "references", "inventory.tsv")
BUILD_INVENTORY = os.path.join(REPO, "skills", "routers", "agent-skill-router", "scripts", "build_inventory.py")

SUPERVISOR_PIN = "a20d76ef3979b804d7dbf599d577826957e0a8a9"
SUPERVISOR_EXPECTED = 12
CCFA_EXPECTED = 17
CCFA_SUPPORT = "ccf-latex-templates"
ISOLATED_FLAGS = ["--no-veloq", "--no-nvidia-skills", "--no-amd-skills", "--no-cursor-skills",
                  "--no-tilelang-skills"]

# skill 正文中指向本目录内资源的相对路径（references/、scripts/ 等），以及 drawio 的 requirements.txt。
SKILL_REF = re.compile(r"(?<![\w.-])((?:references|scripts|assets|examples|agents)/[\w./-]+\.[A-Za-z0-9]+"
                       r"|requirements\.txt)")
# CCFA 的跨 skill 依赖：../ccf-common/references/... 形式，相对于安装后的 skill 目录解析。
CCFA_REF = re.compile(r"\.\./(ccf-[\w-]+)/([\w./-]+\.[A-Za-z0-9]+)")


def run(args, cwd, env=None):
    proc = subprocess.run(args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout.decode("utf-8", "replace")


def git(path, *args):
    code, out = run(["git", "-C", path] + list(args), cwd=REPO)
    return code, out.strip()


def skill_dirs(root):
    """直接子目录中含 SKILL.md 的目录名（Supervisor 的 skills/<name>，CCFA 的 ccf-*）。"""
    names = []
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if os.path.isfile(os.path.join(path, "SKILL.md")):
            names.append(entry)
    return names


def conf_rows():
    rows = {}
    with open(CONF, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("|")
            rows[fields[0]] = fields
    return rows


def inventory_installed(catalog):
    names = set()
    columns = None
    with open(INVENTORY, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if columns is None:
                columns = fields
                continue
            row = dict(zip(columns, fields))
            if row["kind"] == "skill" and row["catalog"] == catalog and row["status"] == "installed":
                names.add(row["name"])
    return names


class SourcePinTest(unittest.TestCase):
    def test_conf_pins_the_verified_supervisor_commit(self):
        row = conf_rows()["supervisor-skills"]
        self.assertEqual(row[1], "https://github.com/HKUSTDial/Supervisor-Skills.git")
        self.assertEqual(row[2], SUPERVISOR_PIN)
        self.assertEqual(row[3], "skills")
        self.assertEqual(row[4], "skills")
        self.assertEqual(row[5], "none")

    def test_conf_provenance_records_each_license(self):
        provenance = conf_rows()["supervisor-skills"][6]
        self.assertIn("CC BY-NC-SA 4.0", provenance)
        self.assertIn("MIT", provenance)
        self.assertIn("CC-BY-4.0", provenance)
        self.assertIn("CC-BY-NC-SA-4.0", provenance)
        self.assertIn("handbook/", provenance)

    def test_checkout_is_detached_at_pin_with_canonical_origin(self):
        self.assertTrue(os.path.isdir(SUPERVISOR), "run bash update-repos.sh or install-external.sh")
        self.assertEqual(git(SUPERVISOR, "rev-parse", "HEAD")[1], SUPERVISOR_PIN)
        self.assertEqual(git(SUPERVISOR, "remote", "get-url", "origin")[1],
                         "https://github.com/HKUSTDial/Supervisor-Skills.git")
        code, _ = git(SUPERVISOR, "symbolic-ref", "-q", "HEAD")
        self.assertNotEqual(code, 0, "checkout should be detached at the pinned commit")

    def test_upstream_checkouts_are_unmodified(self):
        self.assertEqual(git(SUPERVISOR, "status", "--short")[1], "")
        self.assertEqual(git(CCFA, "status", "--short")[1], "")

    def test_supervisor_has_expected_skill_set(self):
        self.assertEqual(len(skill_dirs(os.path.join(SUPERVISOR, "skills"))), SUPERVISOR_EXPECTED)

    def test_ccfa_has_expected_runtime_set_and_support_folder(self):
        runtime = [d for d in os.listdir(CCFA) if d.startswith("ccf-") and
                   os.path.isfile(os.path.join(CCFA, d, "SKILL.md"))]
        self.assertEqual(len(runtime), CCFA_EXPECTED)
        self.assertTrue(os.path.isdir(os.path.join(CCFA, CCFA_SUPPORT)))
        self.assertFalse(os.path.exists(os.path.join(CCFA, CCFA_SUPPORT, "SKILL.md")))


class ResearchGroupTest(unittest.TestCase):
    def test_navigation_links_point_to_the_checkouts(self):
        for name, target in (("ccfa-skills", CCFA), ("supervisor-skills", SUPERVISOR)):
            link = os.path.join(RESEARCH, name)
            self.assertTrue(os.path.islink(link), name)
            self.assertEqual(os.readlink(link), os.path.join("..", "..", "repos", name), name)
            self.assertEqual(os.path.realpath(link), os.path.realpath(target), name)

    def test_research_readme_names_every_skill_in_the_group(self):
        with open(os.path.join(RESEARCH, "README.md"), encoding="utf-8") as f:
            text = f.read()
        names = (skill_dirs(os.path.join(SUPERVISOR, "skills")) + skill_dirs(CCFA) + [CCFA_SUPPORT])
        missing = [n for n in names if "`%s`" % n not in text]
        self.assertEqual(missing, [])

    def test_inventory_lists_each_catalog_skill_as_installed(self):
        self.assertEqual(inventory_installed("supervisor-skills"),
                         set(skill_dirs(os.path.join(SUPERVISOR, "skills"))))
        self.assertEqual(inventory_installed("ccfa-skills"), set(skill_dirs(CCFA)))

    def test_inventory_is_consistent_with_checkouts(self):
        code, out = run([sys.executable, BUILD_INVENTORY, "--check"], cwd=REPO)
        self.assertEqual(code, 0, out)


class IsolatedInstallTest(unittest.TestCase):
    """两种隔离安装：默认混合模式（symlink 到 checkout）与 --copy（真实文件）。"""

    @classmethod
    def setUpClass(cls):
        cls.dests = {}
        cls.homes = []
        for mode in ("mixed", "copy"):
            dest = tempfile.mkdtemp(prefix="agent-gpu-skills-research-%s-" % mode)
            home = tempfile.mkdtemp(prefix="agent-gpu-skills-home-")
            cls.homes.append(home)
            args = ["bash", "install.sh", "--agent", "codex", "--dest", dest] + ISOLATED_FLAGS
            if mode == "copy":
                args.append("--copy")
            code, out = run(args, cwd=REPO, env=dict(os.environ, HOME=home))
            if code != 0:
                raise AssertionError("install.sh (%s) exit %d:\n%s" % (mode, code, out[-2000:]))
            cls.dests[mode] = dest

    @classmethod
    def tearDownClass(cls):
        for dest in cls.dests.values():
            shutil.rmtree(dest, ignore_errors=True)
        for home in cls.homes:
            shutil.rmtree(home, ignore_errors=True)

    def _names(self):
        return skill_dirs(os.path.join(SUPERVISOR, "skills")), skill_dirs(CCFA)

    def test_every_supervisor_and_ccfa_skill_is_installed(self):
        supervisor, ccfa = self._names()
        for mode, dest in self.dests.items():
            for name in supervisor + ccfa:
                self.assertTrue(os.path.isfile(os.path.join(dest, name, "SKILL.md")), "%s %s" % (mode, name))

    def test_ccfa_template_folder_is_installed_beside_skills(self):
        for mode, dest in self.dests.items():
            template = os.path.join(dest, CCFA_SUPPORT, "AAAI", "aaai2026_template.tex")
            self.assertTrue(os.path.isfile(template), mode)

    def test_relative_references_resolve_inside_installed_skills(self):
        supervisor_root = os.path.join(SUPERVISOR, "skills")
        for mode, dest in self.dests.items():
            for name in skill_dirs(supervisor_root):
                with open(os.path.join(dest, name, "SKILL.md"), encoding="utf-8") as f:
                    refs = sorted(set(SKILL_REF.findall(f.read())))
                for ref in refs:
                    self.assertTrue(os.path.exists(os.path.join(dest, name, ref)),
                                    "%s %s -> %s" % (mode, name, ref))

    def test_ccfa_sibling_references_resolve_beside_installed_skills(self):
        for mode, dest in self.dests.items():
            for name in skill_dirs(CCFA):
                with open(os.path.join(dest, name, "SKILL.md"), encoding="utf-8") as f:
                    refs = sorted(set(CCFA_REF.findall(f.read())))
                for sibling, rel in refs:
                    self.assertTrue(os.path.exists(os.path.join(dest, sibling, rel)),
                                    "%s %s -> %s/%s" % (mode, name, sibling, rel))

    def test_copy_mode_has_no_symlinks_into_the_checkouts(self):
        supervisor, ccfa = self._names()
        dest = self.dests["copy"]
        for name in supervisor + ccfa + [CCFA_SUPPORT]:
            for root, dirs, files in os.walk(os.path.join(dest, name)):
                for entry in dirs + files:
                    self.assertFalse(os.path.islink(os.path.join(root, entry)),
                                     os.path.join(root, entry))

    def test_mixed_mode_links_resolve_into_the_checkouts(self):
        dest = self.dests["mixed"]
        drawio = os.path.join(dest, "drawio-reconstruction")
        for entry in ("scripts", "requirements.txt", "LICENSE"):
            link = os.path.join(drawio, entry)
            self.assertTrue(os.path.islink(link), entry)
            self.assertTrue(os.path.realpath(link).startswith(os.path.realpath(SUPERVISOR)), entry)
            self.assertTrue(os.path.exists(link), entry)

    def test_drawio_payload_is_complete_in_both_modes(self):
        for mode, dest in self.dests.items():
            drawio = os.path.join(dest, "drawio-reconstruction")
            for entry in ("scripts/check_drawio.py", "scripts/export_drawio.py", "scripts/crop_assist.py",
                          "requirements.txt", "LICENSE", "examples/data_man.drawio"):
                self.assertTrue(os.path.exists(os.path.join(drawio, entry)), "%s %s" % (mode, entry))
            with open(os.path.join(drawio, "LICENSE"), encoding="utf-8") as f:
                self.assertIn("MIT License", f.read(), mode)

    def test_mixed_mode_has_no_dangling_links(self):
        dest = self.dests["mixed"]
        dangling = []
        for root, dirs, files in os.walk(dest):
            for entry in dirs + files:
                path = os.path.join(root, entry)
                if os.path.islink(path) and not os.path.exists(path):
                    dangling.append(path)
        self.assertEqual(dangling, [])


if __name__ == "__main__":
    unittest.main()
