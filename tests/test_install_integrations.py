"""安装器与外部来源集成测试。

运行（仓库根目录）:
  python3 -m unittest discover -s tests -p 'test_install_integrations.py' -v

约定:
  - 会写文件的测试只使用临时目录：HOME 指向临时目录，--dest 指向临时目录。
  - 外部来源的“远端”用本地 git 仓库；npm 与 uv 用 PATH 中的桩程序，不访问网络、不安装真实 CLI。
  - 对真实 manifest 的断言（四个请求来源的 pin 与 checkout）只读取，不修改 repos/。
  - 缺失/失败场景全部使用 fixture，因此不会因为真实来源被更新或删除而失效。
"""

import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "skills", "routers", "agent-skill-router", "scripts"))
import build_inventory  # noqa: E402

CONF_REL = os.path.join("sources", "external-sources.conf")
SCRIPTS = ["bootstrap.sh", "install.sh", "install-external.sh", "install-rea-mcp.sh", "install-veloq.sh",
           "update-repos.sh", os.path.join("scripts", "node-preflight.sh")]
REQUESTED = {
    "sepia": ("skills", "skills", "none"),
    "wx-cli": ("skills", ".", "npm:@jackwener/wx-cli@0.3.0:wx"),
    "rea": ("skills", "runtime-skills", "npm:rea-agents@6.7.0:rea"),
    "mattpocock-skills": ("skills", "skills/engineering,skills/productivity,skills/misc", "none"),
}
GATE_FLAGS = ["--no-nvidia-skills", "--no-amd-skills", "--no-cursor-skills", "--no-ccfa-skills"]
GIT_ID = ["-c", "user.name=agent-gpu-skills-test", "-c", "user.email=test@example.invalid"]


def run(args, cwd, env=None):
    proc = subprocess.run(args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout.decode("utf-8", errors="replace")


def snapshot(path):
    """目录树快照（os.walk 不跟随目录符号链接，但会列出它们）。"""
    entries = []
    for dirpath, dirnames, filenames in os.walk(path):
        for name in dirnames + filenames:
            entries.append(os.path.relpath(os.path.join(dirpath, name), path))
    return sorted(entries)


def write(path, text, executable=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    if executable:
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)


def skill_md(name, description):
    return "---\nname: %s\ndescription: %s\n---\n\n# %s\n" % (name, description, name)


def tempdir(test, prefix):
    path = tempfile.mkdtemp(prefix=prefix)
    test.addCleanup(shutil.rmtree, path, True)
    return path


def git(cwd, *args):
    subprocess.run(["git"] + GIT_ID + list(args), cwd=cwd, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def make_remote(test, name, files):
    """创建本地 git 仓库作为“远端”，返回 (路径, commit SHA)。"""
    root = tempdir(test, "remote-%s-" % name)
    git(root, "init", "-q")
    for rel, text in files.items():
        write(os.path.join(root, rel), text)
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "fixture")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, stdout=subprocess.PIPE,
                         universal_newlines=True, check=True).stdout.strip()
    return root, sha


def copy_helpers(dest, scripts):
    """复制脚本及其 scripts/ 辅助目录（node-preflight.sh）到 fixture 根目录。"""
    for script in scripts:
        shutil.copy(os.path.join(REPO, script), os.path.join(dest, script))
    os.makedirs(os.path.join(dest, "scripts"), exist_ok=True)
    shutil.copy(os.path.join(REPO, "scripts", "node-preflight.sh"), os.path.join(dest, "scripts"))


def write_stub(path, body):
    write(path, "#!/bin/bash\n" + body, executable=True)


def node_stub(directory, version):
    write_stub(os.path.join(directory, "node"),
               'if [ "$1" = "-v" ]; then echo %s; exit 0; fi\nexit 0\n' % version)


def shq(path):
    """POSIX 单引号转义，允许路径中含空格与撇号。"""
    return "'" + path.replace("'", "'\"'\"'") + "'"


# 模拟已安装的 npm 全局 bin：参数逐个回显，并报告 PATH 中解析到的 node（用于验证启动器前置了 node 目录）。
BIN_BODY_OK = ("#!/bin/bash\n"
               "if [ \"$1\" = --version ]; then echo fixture-bin 1.0.0; exit 0; fi\n"
               "for a in \"$@\"; do echo \"ARG:[$a]\"; done\n"
               "echo \"NODE:$(command -v node)\"\n")
BIN_BODY_FAIL = "#!/bin/bash\nexit 3\n"


def write_npm_stub(stub_dir, log, prefix, bin_body, create_bin=True, create_skills=True):
    """npm 桩（不访问网络）：prefix -g 返回 prefix；install -g pkg@ver 生成 lib/node_modules/pkg/skills 与 bin/fixture-bin。"""
    write(os.path.join(stub_dir, "fixture-bin.tpl"), bin_body)
    nm = os.path.join(prefix, "lib", "node_modules")
    make_skills = ""
    if create_skills:
        make_skills = ("  mkdir -p %s/\"$name\"/skills/fixture-skill\n"
                       "  printf -- '---\\nname: fixture-skill\\n---\\n' > %s/\"$name\"/skills/fixture-skill/SKILL.md\n"
                       % (shq(nm), shq(nm)))
    make_bin = ""
    if create_bin:
        make_bin = ("  mkdir -p %s\n  cp %s %s\n  chmod 755 %s\n"
                    % (shq(os.path.join(prefix, "bin")), shq(os.path.join(stub_dir, "fixture-bin.tpl")),
                       shq(os.path.join(prefix, "bin", "fixture-bin")),
                       shq(os.path.join(prefix, "bin", "fixture-bin"))))
    script = ("#!/bin/bash\n"
              "echo \"$@\" >> %s\n"
              "if [ \"$1\" = prefix ]; then echo %s; exit 0; fi\n"
              "if [ \"$1\" = root ]; then echo %s; exit 0; fi\n"
              "if [ \"$1\" = install ]; then\n"
              "  name=\"${3%%@*}\"\n"
              "  mkdir -p %s/\"$name\"\n"
              % (shq(log), shq(prefix), shq(nm), shq(nm)))
    script += make_skills + make_bin + "  exit 0\nfi\nexit 1\n"
    write(os.path.join(stub_dir, "npm"), script, executable=True)


def conf_text(rows):
    return "# fixture\n" + "".join(row + "\n" for row in rows)


def read_conf(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n").split("|") for l in f if l.strip() and not l.startswith("#")]


class StaticManifestTest(unittest.TestCase):
    """真实 manifest：四个请求来源必须已固定、已检出，并且 skill 来源路径存在。"""

    def setUp(self):
        self.rows = {r[0]: r for r in read_conf(os.path.join(REPO, CONF_REL))}

    def test_shell_scripts_pass_bash_syntax_check(self):
        for script in SCRIPTS:
            rc, out = run(["bash", "-n", os.path.join(REPO, script)], cwd=REPO)
            self.assertEqual(rc, 0, "%s: %s" % (script, out))

    def test_requested_sources_are_pinned_with_explicit_mode_and_runtime(self):
        for name, (mode, skills_dir, runtime) in REQUESTED.items():
            self.assertIn(name, self.rows)
            row = self.rows[name]
            self.assertEqual(len(row), 7, name)
            self.assertRegex(row[2], r"^[0-9a-f]{40}$", name)
            self.assertEqual(row[3], mode, name)
            self.assertEqual(row[4], skills_dir, name)
            self.assertEqual(row[5], runtime, name)
            self.assertNotIn("pending", row[3] + row[5], name)
            self.assertNotIn("未检视", row[6], name)

    def test_manifest_fields_are_well_formed(self):
        for name, row in self.rows.items():
            self.assertEqual(len(row), 7, name)
            self.assertIn(row[3], {"pending", "skills", "none"}, name)
            self.assertTrue(row[5] == "none" or re.match(r"^(uv|npm):[^:]+:[^:]+$", row[5]), name)
            if row[5].startswith("npm:"):
                self.assertRegex(row[5], r"^npm:[^:]*@[^:]+:[^:]+$", name)

    def test_checkouts_are_at_pinned_commits(self):
        for name in REQUESTED:
            checkout = os.path.join(REPO, "repos", name)
            self.assertTrue(os.path.isdir(os.path.join(checkout, ".git")), "missing checkout: %s" % name)
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout, stdout=subprocess.PIPE,
                                  universal_newlines=True, check=True).stdout.strip()
            self.assertEqual(head, self.rows[name][2], name)

    def test_real_skill_source_paths_exist(self):
        self.assertTrue(os.path.isfile(os.path.join(REPO, "repos", "sepia", "skills", "sepia", "SKILL.md")))
        self.assertTrue(os.path.isfile(os.path.join(REPO, "repos", "sepia", "skills", "sepia-write", "SKILL.md")))
        self.assertTrue(os.path.isfile(os.path.join(REPO, "repos", "wx-cli", "SKILL.md")))
        self.assertTrue(os.path.isfile(os.path.join(REPO, "repos", "mattpocock-skills", "skills", "engineering",
                                                    "tdd", "SKILL.md")))
        for sd in REQUESTED["mattpocock-skills"][1].split(","):
            self.assertTrue(os.path.isdir(os.path.join(REPO, "repos", "mattpocock-skills", sd)), sd)

    def test_rea_npm_pin_matches_pinned_source_version(self):
        # rea 的 skill 来自 npm 包；npm 版本必须与被固定的源码版本一致。
        with open(os.path.join(REPO, "repos", "rea", ".release-please-manifest.json"), encoding="utf-8") as f:
            source_version = re.search(r'"\.":\s*"([^"]+)"', f.read()).group(1)
        self.assertEqual(REQUESTED["rea"][2], "npm:rea-agents@%s:rea" % source_version)

    def test_help_documents_dest_and_integration_flags(self):
        rc, out = run(["bash", "install.sh", "--help"], cwd=REPO)
        self.assertEqual(rc, 0)
        for flag in ("--dest", "--no-integrations", "--no-veloq"):
            self.assertIn(flag, out)
        rc, out = run(["bash", "bootstrap.sh", "--help"], cwd=REPO)
        self.assertEqual(rc, 0)
        for flag in ("--no-update", "--no-integrations", "--dest"):
            self.assertIn(flag, out)


class IsolatedInstallTest(unittest.TestCase):
    """使用真实仓库的 skill 树，但 HOME 与 --dest 都指向临时目录。"""

    def setUp(self):
        self.home = tempdir(self, "home-")
        self.outside = tempdir(self, "outside-")
        self.env = dict(os.environ, HOME=self.home)

    def test_dest_install_skips_veloq_and_never_writes_home(self):
        dest = os.path.join(tempdir(self, "dest-"), "skills")
        before = snapshot(self.home)
        rc, out = run(["bash", "install.sh", "--agent", "codex", "--dest", dest] + GATE_FLAGS,
                      cwd=REPO, env=self.env)
        self.assertEqual(snapshot(self.home), before, "HOME changed during --dest install")
        self.assertIn("跳过 VeloQ", out)
        self.assertTrue(os.path.isfile(os.path.join(dest, "cuda-skill", "SKILL.md")))
        self.assertTrue(os.path.isfile(os.path.join(dest, "agent-skill-router", "SKILL.md")))
        self.assertEqual(rc, 0, out[-2000:])

    def test_dest_installs_present_external_sources_and_reports_missing_runtime_as_optional(self):
        dest = os.path.join(tempdir(self, "dest-"), "skills")
        rc, out = run(["bash", "install.sh", "--agent", "codex", "--dest", dest, "--no-veloq"] + GATE_FLAGS,
                      cwd=REPO, env=self.env)
        self.assertTrue(os.path.isfile(os.path.join(dest, "sepia", "SKILL.md")))
        self.assertTrue(os.path.isfile(os.path.join(dest, "sepia-write", "SKILL.md")))
        self.assertTrue(os.path.isfile(os.path.join(dest, "wx-cli", "SKILL.md")))
        self.assertTrue(os.path.isfile(os.path.join(dest, "grill-me", "SKILL.md")))
        self.assertIn("可选缺失: 外部来源", out)
        self.assertNotIn("安装未完成", out)
        self.assertEqual(rc, 0, out[-2000:])

    def test_real_install_fails_when_requested_runtimes_are_missing(self):
        rc, out = run(["bash", "install.sh", "--agent", "codex", "--no-veloq"] + GATE_FLAGS,
                      cwd=REPO, env=self.env)
        self.assertNotEqual(rc, 0)
        self.assertIn("运行时命令", out)
        self.assertIn("安装未完成", out)
        self.assertNotIn("安装完成.", out)

    def test_no_integrations_flag_skips_external_section(self):
        dest = os.path.join(tempdir(self, "dest-"), "skills")
        rc, out = run(["bash", "install.sh", "--agent", "codex", "--dest", dest, "--no-veloq",
                       "--no-integrations"] + GATE_FLAGS, cwd=REPO, env=self.env)
        self.assertEqual(rc, 0, out[-2000:])
        self.assertNotIn("外部来源", out)
        self.assertFalse(os.path.exists(os.path.join(dest, "sepia")))

    def test_user_level_delegate_symlinks_survive_real_install(self):
        skills_dir = os.path.join(self.home, ".agents", "skills")
        os.makedirs(skills_dir)
        targets = {}
        for name in ("claude-delegate", "codex-background"):
            target = os.path.join(self.outside, name)
            write(os.path.join(target, "SKILL.md"), skill_md(name, "user-level"))
            os.symlink(target, os.path.join(skills_dir, name))
            targets[name] = target
        run(["bash", "install.sh", "--agent", "kimi", "--no-veloq"] + GATE_FLAGS, cwd=REPO, env=self.env)
        for name, target in targets.items():
            link = os.path.join(skills_dir, name)
            self.assertTrue(os.path.islink(link), "%s symlink was removed" % name)
            self.assertEqual(os.readlink(link), target)


class FixtureCatalogCollisionTest(unittest.TestCase):
    """临时仓库副本：外部 catalog 的同名、保留名、用户 symlink、根 SKILL.md 与 runtime-skills 缺失。"""

    def setUp(self):
        self.fixture = tempdir(self, "fixture-")
        self.outside = tempdir(self, "outside-")
        self.dest = os.path.join(tempdir(self, "dest-"), "skills")
        shutil.copy(os.path.join(REPO, "install.sh"), os.path.join(self.fixture, "install.sh"))
        write(os.path.join(self.fixture, "skills", "kernel", "cuda-skill", "SKILL.md"), skill_md("cuda-skill", "x"))
        for name in ("upstream-a", "upstream-b"):
            write(os.path.join(self.fixture, "repos", name, "skills", "group", "dup-skill", "SKILL.md"),
                  skill_md("dup-skill", name))
        write(os.path.join(self.fixture, "repos", "upstream-a", "skills", "alpha", "SKILL.md"),
              skill_md("alpha", "from upstream-a"))
        write(os.path.join(self.fixture, "repos", "upstream-b", "skills", "beta", "SKILL.md"),
              skill_md("beta", "from upstream-b"))
        write(os.path.join(self.fixture, "repos", "upstream-b", "skills", "claude-delegate", "SKILL.md"),
              skill_md("claude-delegate", "must not install"))
        # 根目录 SKILL.md（目录名 root-src，frontmatter 名 root-skill）
        write(os.path.join(self.fixture, "repos", "root-src", "SKILL.md"), skill_md("root-skill", "root"))
        # runtime-skills 缺失：不得安装，也不得伪造成功
        os.makedirs(os.path.join(self.fixture, "repos", "rea-like"))
        write(os.path.join(self.fixture, CONF_REL), conf_text([
            "upstream-a|https://example.invalid/a|UNPINNED|skills|skills|none|fixture",
            "upstream-b|https://example.invalid/b|UNPINNED|skills|skills|none|fixture",
            "root-src|https://example.invalid/r|UNPINNED|skills|.|none|fixture",
            "rea-like|https://example.invalid/rea|UNPINNED|skills|runtime-skills|npm:fake-pkg@1.0.0:fake-bin|fixture",
        ]))
        user_alpha = os.path.join(self.outside, "alpha-user")
        write(os.path.join(user_alpha, "SKILL.md"), skill_md("alpha", "user owned"))
        os.makedirs(self.dest)
        os.symlink(user_alpha, os.path.join(self.dest, "alpha"))
        self.user_alpha = user_alpha
        delegate = os.path.join(self.outside, "delegate")
        write(os.path.join(delegate, "SKILL.md"), skill_md("claude-delegate", "user-level"))
        os.symlink(delegate, os.path.join(self.dest, "claude-delegate"))
        self.delegate = delegate

    def _install(self):
        return run(["bash", os.path.join(self.fixture, "install.sh"), "--agent", "codex", "--dest", self.dest,
                    "--no-veloq", "--no-tilelang-skills"] + GATE_FLAGS, cwd=self.fixture)

    def test_first_catalog_wins_and_duplicates_are_reported_once(self):
        rc, out = self._install()
        with open(os.path.join(self.dest, "dup-skill", "SKILL.md"), encoding="utf-8") as f:
            self.assertIn("upstream-a", f.read())
        self.assertTrue(os.path.isfile(os.path.join(self.dest, "beta", "SKILL.md")))
        self.assertIn("外部来源", out)

    def test_root_skill_is_named_by_frontmatter(self):
        self._install()
        self.assertTrue(os.path.isfile(os.path.join(self.dest, "root-skill", "SKILL.md")))
        self.assertFalse(os.path.exists(os.path.join(self.dest, "root-src")))

    def test_missing_runtime_skills_root_is_skipped_and_reported(self):
        rc, out = self._install()
        self.assertIn("跳过:", out)
        self.assertFalse(os.path.exists(os.path.join(self.dest, "rea-like")))
        self.assertIn("运行时命令 'fake-bin'", out)

    def test_user_symlink_with_same_name_is_preserved(self):
        self._install()
        link = os.path.join(self.dest, "alpha")
        self.assertTrue(os.path.islink(link))
        self.assertEqual(os.readlink(link), self.user_alpha)

    def test_protected_user_level_names_are_never_replaced(self):
        rc, out = self._install()
        link = os.path.join(self.dest, "claude-delegate")
        self.assertTrue(os.path.islink(link))
        self.assertEqual(os.readlink(link), self.delegate)
        self.assertIn("用户级 skill，未替换", out)


class RuntimeInstallTest(unittest.TestCase):
    """install-external.sh 的 npm/uv 适配器，使用桩程序模拟 npm 与 CLI。"""

    def setUp(self):
        self.remote, self.sha = make_remote(self, "fixture-cli", {
            "skills/fixture-skill/SKILL.md": skill_md("fixture-skill", "fixture"),
        })
        self.fresh = tempdir(self, "fresh-")
        copy_helpers(self.fresh, ["install-external.sh"])
        self.stub = tempdir(self, "stub-bin-")
        node_stub(self.stub, "v24.22.1")
        self.global_root = tempdir(self, "npm-prefix-")
        self.log = os.path.join(self.stub, "npm.log")
        self.home = tempdir(self, "home-")

    def _conf(self, runtime, mode="skills", skills_dir="skills", ref=None):
        write(os.path.join(self.fresh, CONF_REL), conf_text([
            "fixture-cli|%s|%s|%s|%s|%s|fixture" % (self.remote, ref or self.sha, mode, skills_dir, runtime),
        ]))

    def _npm_stub(self, bin_version_ok=True, create_bin=True, create_skills=True):
        write_npm_stub(self.stub, self.log, self.global_root,
                       BIN_BODY_OK if bin_version_ok else BIN_BODY_FAIL,
                       create_bin=create_bin, create_skills=create_skills)

    def _env(self):
        return dict(os.environ, HOME=self.home, PATH=self.stub + os.pathsep + os.environ["PATH"])

    def _run(self):
        return run(["bash", "install-external.sh"], cwd=self.fresh, env=self._env())

    def test_npm_pinned_package_installs_bin_and_links_runtime_skills(self):
        self._conf("npm:fake-pkg@1.0.0:fixture-bin", skills_dir="runtime-skills")
        self._npm_stub()
        rc, out = self._run()
        self.assertEqual(rc, 0, out)
        with open(self.log, encoding="utf-8") as f:
            self.assertIn("install -g fake-pkg@1.0.0", f.read())
        link = os.path.join(self.fresh, "repos", "fixture-cli", "runtime-skills")
        self.assertTrue(os.path.islink(link))
        self.assertTrue(os.path.isfile(os.path.join(link, "fixture-skill", "SKILL.md")))
        self.assertIn("runtime OK: fixture-bin", out)

    def test_npm_spec_without_version_is_blocked_and_npm_is_not_called(self):
        self._conf("npm:fake-pkg:fixture-bin")
        self._npm_stub()
        rc, out = self._run()
        self.assertNotEqual(rc, 0)
        self.assertIn("BLOCKED: npm 规格必须固定版本", out)
        self.assertFalse(os.path.exists(self.log))

    def test_missing_bin_after_npm_install_is_failure(self):
        self._conf("npm:fake-pkg@1.0.0:fixture-bin", skills_dir="runtime-skills")
        self._npm_stub(create_bin=False)
        rc, out = self._run()
        self.assertNotEqual(rc, 0)
        self.assertIn("npm 全局 bin 中找不到 'fixture-bin'", out)

    def test_bin_whose_version_check_fails_is_failure(self):
        self._conf("npm:fake-pkg@1.0.0:fixture-bin", skills_dir="runtime-skills")
        self._npm_stub(bin_version_ok=False)
        rc, out = self._run()
        self.assertNotEqual(rc, 0)
        self.assertIn("--version' 失败", out)

    def test_package_without_skills_directory_fails_for_runtime_skills(self):
        self._conf("npm:fake-pkg@1.0.0:fixture-bin", skills_dir="runtime-skills")
        self._npm_stub(create_skills=False)
        rc, out = self._run()
        self.assertNotEqual(rc, 0)
        self.assertIn("没有 skills/ 目录", out)

    def test_npm_missing_is_blocked(self):
        if shutil.which("npm", path="/usr/bin:/bin"):
            self.skipTest("npm is available in /usr/bin:/bin; cannot simulate missing npm")
        self._conf("npm:fake-pkg@1.0.0:fixture-bin", skills_dir="runtime-skills")
        rc, out = run(["bash", "install-external.sh"], cwd=self.fresh,
                      env=dict(os.environ, HOME=self.home, PATH="/usr/bin:/bin"))
        self.assertNotEqual(rc, 0)
        self.assertIn("BLOCKED: 缺少 npm", out)

    def test_uv_runtime_is_installed_through_uv_and_verified_on_path(self):
        write(os.path.join(self.stub, "uv"), "#!/bin/bash\necho \"$@\" >> '%s'\nexit 0\n" % self.log, executable=True)
        write(os.path.join(self.stub, "fixture-cli-bin"), "#!/bin/bash\necho ok\n", executable=True)
        self._conf("uv:cli-pkg:fixture-cli-bin")
        rc, out = self._run()
        self.assertEqual(rc, 0, out)
        with open(self.log, encoding="utf-8") as f:
            self.assertIn("tool install --force", f.read())
        self.assertIn("runtime OK: fixture-cli-bin", out)


class LauncherTest(unittest.TestCase):
    """npm 运行时的 managed launcher：参数逐个转发、路径含空格与撇号、幂等不递归、保留无关文件。"""

    def setUp(self):
        self.remote, self.sha = make_remote(self, "fixture-cli", {
            "skills/fixture-skill/SKILL.md": skill_md("fixture-skill", "fixture"),
        })
        self.fresh = tempdir(self, "fresh-")
        copy_helpers(self.fresh, ["install-external.sh"])
        base = tempdir(self, "weird-")
        self.stub = os.path.join(base, "node dir's bin")
        os.makedirs(self.stub)
        node_stub(self.stub, "v24.22.1")
        self.log = os.path.join(self.stub, "npm.log")
        self.prefix = os.path.join(base, "npm prefix's")
        self.home = tempdir(self, "home-")
        self.launcher = os.path.join(self.home, ".local", "bin", "fixture-bin")
        write(os.path.join(self.fresh, CONF_REL), conf_text([
            "fixture-cli|%s|%s|none|-|npm:fake-pkg@1.0.0:fixture-bin|fixture" % (self.remote, self.sha),
        ]))
        write_npm_stub(self.stub, self.log, self.prefix, BIN_BODY_OK)
        self.env = dict(os.environ, HOME=self.home, PATH=self.stub + os.pathsep + os.environ["PATH"])

    def _install(self, *extra):
        return run(["bash", "install-external.sh"] + list(extra), cwd=self.fresh, env=self.env)

    def test_launcher_forwards_arguments_and_runs_under_selected_node_without_npm_prefix_on_path(self):
        rc, out = self._install()
        self.assertEqual(rc, 0, out)
        with open(self.launcher, encoding="utf-8") as f:
            self.assertIn("agent-gpu-skills-managed", f.read())
        # PATH 只有 ~/.local/bin 与普通工具：不含 npm 前缀，也不含 node 所在目录。
        env = dict(os.environ, HOME=self.home, PATH="%s:/usr/bin:/bin" % os.path.dirname(self.launcher))
        proc = subprocess.run([self.launcher, "a b", "it's", "--flag"], env=env, cwd=self.fresh,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        text = proc.stdout.decode("utf-8", errors="replace")
        self.assertEqual(proc.returncode, 0, text)
        self.assertIn("ARG:[a b]", text)
        self.assertIn("ARG:[it's]", text)
        self.assertIn("ARG:[--flag]", text)
        self.assertIn("NODE:" + os.path.join(self.stub, "node"), text)

    def test_repeat_install_is_idempotent_and_never_recurses(self):
        self._install()
        with open(self.launcher, "rb") as f:
            first = f.read()
        rc, out = self._install()
        self.assertEqual(rc, 0, out)
        with open(self.launcher, "rb") as f:
            self.assertEqual(f.read(), first)
        proc = subprocess.run([self.launcher, "again"], env=self.env, cwd=self.fresh,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        self.assertEqual(proc.returncode, 0)
        self.assertIn(b"ARG:[again]", proc.stdout)

    def test_foreign_launcher_is_preserved_and_reported_as_blocked(self):
        os.makedirs(os.path.dirname(self.launcher))
        write(self.launcher, "#!/bin/bash\necho UNRELATED\n", executable=True)
        rc, out = self._install()
        self.assertNotEqual(rc, 0)
        with open(self.launcher, encoding="utf-8") as f:
            self.assertEqual(f.read(), "#!/bin/bash\necho UNRELATED\n")
        self.assertIn("未替换", out)

    def test_npm_prefix_bin_that_is_the_launcher_dir_is_used_directly_without_self_reference(self):
        prefix = os.path.join(self.home, ".local")
        write_npm_stub(self.stub, self.log, prefix, BIN_BODY_OK)
        rc, out = self._install()
        self.assertEqual(rc, 0, out)
        with open(self.launcher, encoding="utf-8") as f:
            self.assertNotIn("agent-gpu-skills-managed", f.read())
        self.assertIn("无需生成", out)

    def test_no_runtime_flag_writes_no_launcher(self):
        rc, out = self._install("--no-runtime")
        self.assertEqual(rc, 0, out)
        self.assertFalse(os.path.exists(self.launcher))


class RootSymlinkSourceTest(unittest.TestCase):
    """runtime-skills 这类 symlink 源根必须被枚举；存在但没有 SKILL.md 的源不是成功。（临时 fixture，不访问网络）"""

    def setUp(self):
        self.fixture = tempdir(self, "fixture-")
        self.global_root = tempdir(self, "npm-global-")
        shutil.copy(os.path.join(REPO, "install.sh"), os.path.join(self.fixture, "install.sh"))
        write(os.path.join(self.fixture, "skills", "kernel", "cuda-skill", "SKILL.md"), skill_md("cuda-skill", "x"))
        self.skills = os.path.join(self.global_root, "lib", "node_modules", "rea-agents", "skills")
        write(os.path.join(self.skills, "reverse-engineer-anything", "SKILL.md"),
              skill_md("reverse-engineer-anything", "REA runtime skill"))
        self.link = os.path.join(self.fixture, "repos", "rea-like", "runtime-skills")
        os.makedirs(os.path.dirname(self.link))
        os.symlink(self.skills, self.link)
        write(os.path.join(self.fixture, CONF_REL), conf_text([
            "rea-like|https://example.invalid/rea|UNPINNED|skills|runtime-skills|none|fixture",
        ]))
        self.dest = os.path.join(tempdir(self, "dest-"), "skills")
        self.home = tempdir(self, "home-")

    def _dest_install(self):
        return run(["bash", os.path.join(self.fixture, "install.sh"), "--agent", "codex", "--dest", self.dest,
                    "--no-veloq", "--no-tilelang-skills"] + GATE_FLAGS, cwd=self.fixture,
                   env=dict(os.environ, HOME=self.home))

    def _real_install(self):
        return run(["bash", os.path.join(self.fixture, "install.sh"), "--agent", "codex", "--no-veloq",
                    "--no-tilelang-skills"] + GATE_FLAGS, cwd=self.fixture, env=dict(os.environ, HOME=self.home))

    def test_symlink_source_root_is_enumerated_and_installed(self):
        rc, out = self._dest_install()
        self.assertTrue(os.path.isfile(os.path.join(self.dest, "reverse-engineer-anything", "SKILL.md")), out[-1500:])
        self.assertIn("已安装 1 个 skill", out)

    def test_existing_but_empty_source_root_is_reported_missing_not_success(self):
        os.remove(self.link)
        empty = tempdir(self, "empty-src-")
        os.symlink(empty, self.link)
        rc, out = self._real_install()
        self.assertNotEqual(rc, 0)
        self.assertIn("中没有 SKILL.md", out)
        self.assertFalse(os.path.exists(os.path.join(self.home, ".codex", "skills", "reverse-engineer-anything")))

    def test_dangling_source_root_is_reported_missing(self):
        os.remove(self.link)
        os.symlink(os.path.join(self.global_root, "gone"), self.link)
        rc, out = self._real_install()
        self.assertNotEqual(rc, 0)
        self.assertIn("runtime-skills 不存在", out)


class InventoryParityTest(unittest.TestCase):
    """真实 checkout 的门禁：--dest 安装得到的名称必须与路由清单中已安装（installed）的名称一致，而不只看退出码。"""

    def test_dest_install_matches_inventory_installed_names_for_present_catalogs(self):
        inventory = os.path.join(REPO, "skills", "routers", "agent-skill-router", "references", "inventory.tsv")
        catalogs = {"local", "kernel-design-agents", "sepia", "wx-cli", "mattpocock-skills"}
        if os.path.isdir(os.path.join(REPO, "repos", "rea", "runtime-skills")):
            catalogs.add("rea")
        expected = set()
        with open(inventory, encoding="utf-8") as f:
            columns = None
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                fields = line.rstrip("\n").split("\t")
                if columns is None:
                    columns = fields
                    continue
                row = dict(zip(columns, fields))
                if row["kind"] == "skill" and row["status"] == "installed" and row["catalog"] in catalogs:
                    expected.add(row["name"])
        self.assertTrue(expected)
        dest = os.path.join(tempdir(self, "dest-"), "skills")
        run(["bash", "install.sh", "--agent", "codex", "--dest", dest, "--no-veloq", "--no-nvidia-skills",
             "--no-amd-skills", "--no-cursor-skills", "--no-ccfa-skills"], cwd=REPO,
            env=dict(os.environ, HOME=tempdir(self, "home-")))
        missing = sorted(n for n in expected if not os.path.isfile(os.path.join(dest, n, "SKILL.md")))
        self.assertEqual(missing, [])


class FreshCheckoutTest(unittest.TestCase):
    """模拟刚 clone 的仓库：没有 checkout 的外部来源，--no-update 也必须补齐。"""

    def setUp(self):
        self.remote, self.sha = make_remote(self, "fixture-catalog", {
            "skills/fresh-skill/SKILL.md": skill_md("fresh-skill", "installed from fresh checkout"),
        })
        self.fresh = tempdir(self, "fresh-")
        copy_helpers(self.fresh, [s for s in SCRIPTS if not s.startswith("scripts")])
        shutil.copytree(os.path.join(REPO, "skills", "routers", "agent-skill-router"),
                        os.path.join(self.fresh, "skills", "routers", "agent-skill-router"))
        write(os.path.join(self.fresh, "skills", "kernel", "cuda-skill", "SKILL.md"), skill_md("cuda-skill", "x"))
        self.home = tempdir(self, "home-")
        self.env = dict(os.environ, HOME=self.home)

    def _write_conf(self, rows):
        write(os.path.join(self.fresh, CONF_REL), conf_text(rows))

    def _row(self, ref=None):
        return "fixture-catalog|%s|%s|skills|skills|none|fixture" % (self.remote, ref or self.sha)

    def _bootstrap(self, *extra):
        return run(["bash", "bootstrap.sh", "--agent", "codex", "--no-veloq"] + list(extra) + GATE_FLAGS,
                   cwd=self.fresh, env=self.env)

    def test_no_update_clones_missing_source_and_pins_it(self):
        self._write_conf([self._row()])
        rc, out = self._bootstrap("--no-update")
        checkout = os.path.join(self.fresh, "repos", "fixture-catalog")
        self.assertTrue(os.path.isdir(os.path.join(checkout, ".git")), out[-2000:])
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout, stdout=subprocess.PIPE,
                              universal_newlines=True, check=True).stdout.strip()
        self.assertEqual(head, self.sha)
        self.assertTrue(os.path.isfile(os.path.join(self.home, ".codex", "skills", "fresh-skill", "SKILL.md")))
        self.assertNotEqual(rc, 0)
        self.assertIn("安装未完成", out)

    def test_rerun_is_idempotent_and_does_not_reclone(self):
        self._write_conf([self._row()])
        self._bootstrap("--no-update")
        rc, out = self._bootstrap("--no-update")
        self.assertNotIn("clone:", out)
        self.assertTrue(os.path.isfile(os.path.join(self.home, ".codex", "skills", "fresh-skill", "SKILL.md")))

    def test_no_integrations_does_not_clone_or_run_external_step(self):
        self._write_conf([self._row()])
        rc, out = self._bootstrap("--no-update", "--no-integrations")
        self.assertFalse(os.path.exists(os.path.join(self.fresh, "repos", "fixture-catalog")))
        self.assertNotIn("补齐外部来源", out)
        self.assertNotIn("fixture-catalog", out)

    def test_dest_mode_does_not_fetch_or_touch_home(self):
        self._write_conf([self._row()])
        dest = os.path.join(tempdir(self, "dest-"), "skills")
        before = snapshot(self.home)
        rc, out = run(["bash", "bootstrap.sh", "--agent", "codex", "--dest", dest, "--no-veloq"] + GATE_FLAGS,
                      cwd=self.fresh, env=self.env)
        self.assertFalse(os.path.exists(os.path.join(self.fresh, "repos", "fixture-catalog")))
        self.assertEqual(snapshot(self.home), before)
        self.assertNotIn("补齐外部来源", out)

    def test_dest_mode_never_initialises_submodules(self):
        # .gitmodules 声明了一个不存在的 submodule：--dest 下不得联网获取，只能报告缺失。
        write(os.path.join(self.fresh, ".gitmodules"),
              "[submodule \"repos/fake-sub\"]\n\tpath = repos/fake-sub\n\turl = https://example.invalid/fake.git\n")
        dest = os.path.join(tempdir(self, "dest-"), "skills")
        rc, out = run(["bash", "bootstrap.sh", "--agent", "codex", "--dest", dest, "--no-veloq"] + GATE_FLAGS,
                      cwd=self.fresh, env=self.env)
        self.assertFalse(os.path.exists(os.path.join(self.fresh, "repos", "fake-sub")))
        self.assertIn("跳过 submodule 获取（--dest 不联网）", out)

    def test_dest_and_no_integrations_never_write_a_launcher(self):
        self._write_conf([self._row()])
        dest = os.path.join(tempdir(self, "dest-"), "skills")
        run(["bash", "bootstrap.sh", "--agent", "codex", "--dest", dest, "--no-veloq"] + GATE_FLAGS,
            cwd=self.fresh, env=self.env)
        self.assertFalse(os.path.exists(os.path.join(self.home, ".local")))
        self._bootstrap("--no-update", "--no-integrations")
        self.assertFalse(os.path.exists(os.path.join(self.home, ".local")))

    def test_pending_source_is_blocked_and_fails(self):
        self._write_conf([self._row(), "unreviewed|%s|UNPINNED|pending|-|pending|fixture" % self.remote])
        rc, out = run(["bash", "install-external.sh", "--no-runtime"], cwd=self.fresh, env=self.env)
        self.assertNotEqual(rc, 0)
        self.assertIn("BLOCKED: mode=pending", out)

    def test_missing_ref_is_reported_as_failure_and_bootstrap_propagates_it(self):
        self._write_conf([self._row(ref="0" * 40)])
        rc, out = run(["bash", "install-external.sh", "--no-runtime"], cwd=self.fresh, env=self.env)
        self.assertNotEqual(rc, 0)
        self.assertIn("FAIL: 无法获取", out)
        rc, out = self._bootstrap("--no-update")
        self.assertNotEqual(rc, 0)
        self.assertIn("Bootstrap 未完成", out)
        self.assertIn("安装 skills", out)  # install.sh 仍然执行，缺失项在同一份报告中

    def test_runtime_failure_propagates_through_bootstrap_exit_status(self):
        stub = tempdir(self, "stub-")
        write(os.path.join(stub, "uv"), "#!/bin/bash\nexit 0\n", executable=True)
        self._write_conf(["fixture-catalog|%s|%s|skills|skills|uv:cli-pkg:absent-bin|fixture" % (self.remote, self.sha)])
        env = dict(self.env, PATH=stub + os.pathsep + os.environ["PATH"])
        rc, out = run(["bash", "bootstrap.sh", "--agent", "codex", "--no-update", "--no-veloq"] + GATE_FLAGS,
                      cwd=self.fresh, env=env)
        self.assertNotEqual(rc, 0)
        self.assertIn("'absent-bin' 不可用", out)
        self.assertIn("Bootstrap 未完成", out)

    def test_update_repos_external_skip_flag_skips_without_network(self):
        rc, out = run(["bash", "update-repos.sh", "--no-integrations", "external"], cwd=REPO, env=self.env)
        self.assertEqual(rc, 0, out)
        self.assertIn("跳过外部来源（--no-integrations）", out)

    def test_update_repos_external_failure_is_nonzero(self):
        for script in ("update-repos.sh", "install-external.sh"):
            shutil.copy(os.path.join(REPO, script), os.path.join(self.fresh, script))
        self._write_conf(["fixture-catalog|%s|%s|skills|skills|none|fixture" % (self.remote, "0" * 40)])
        rc, out = run(["bash", "update-repos.sh", "external"], cwd=self.fresh, env=self.env)
        self.assertNotEqual(rc, 0)
        self.assertIn("FAIL: 无法获取", out)


class RootSkillCopyTest(unittest.TestCase):
    """--copy 下根目录即 skill 的仓库（wx-cli 形态）：只复制 skill 载荷，不复制 Git 与 agent 配置。"""

    def setUp(self):
        self.fixture = tempdir(self, "fixture-")
        self.dest = os.path.join(tempdir(self, "dest-"), "skills")
        shutil.copy(os.path.join(REPO, "install.sh"), os.path.join(self.fixture, "install.sh"))
        write(os.path.join(self.fixture, "skills", "kernel", "cuda-skill", "SKILL.md"), skill_md("cuda-skill", "x"))
        root = os.path.join(self.fixture, "repos", "root-src")
        write(os.path.join(root, "SKILL.md"), skill_md("root-skill", "root"))
        write(os.path.join(root, "docs", "guide.md"), "visible support file\n")
        write(os.path.join(root, ".git", "config"), "[core]\n")
        write(os.path.join(root, ".claude", "settings.local.json"), "{}\n")
        write(os.path.join(root, ".agents", "note.md"), "agent config\n")
        write(os.path.join(self.fixture, CONF_REL), conf_text([
            "root-src|https://example.invalid/r|UNPINNED|skills|.|none|fixture",
        ]))

    def test_copy_mode_copies_only_visible_skill_payload(self):
        run(["bash", os.path.join(self.fixture, "install.sh"), "--agent", "codex", "--dest", self.dest,
             "--copy", "--no-veloq", "--no-tilelang-skills"] + GATE_FLAGS, cwd=self.fixture)
        target = os.path.join(self.dest, "root-skill")
        self.assertTrue(os.path.isfile(os.path.join(target, "SKILL.md")))
        self.assertTrue(os.path.isfile(os.path.join(target, "docs", "guide.md")))
        for hidden in (".git", ".claude", ".agents"):
            self.assertFalse(os.path.exists(os.path.join(target, hidden)), hidden)


PROTECTED_FILES = {
    "claude-delegate": ["SKILL.md", "scripts/relay.mjs"],
    "codex-background": ["SKILL.md", "scripts/bootstrap.mjs", "scripts/server.mjs"],
}


def mock_user_level(home, name, marker):
    """在临时 HOME 中模拟一个已有的用户级补丁版副本（不访问真实 HOME）。"""
    root = os.path.join(home, ".agents", "skills", name)
    for rel in PROTECTED_FILES[name]:
        write(os.path.join(root, rel), "%s %s\n" % (marker, name))
    return root


class ProtectedNameTest(unittest.TestCase):
    """受保护名（claude-delegate、codex-background）的安装策略，全部使用临时 HOME 与 --dest。"""

    def setUp(self):
        self.home = tempdir(self, "home-")
        self.outside = tempdir(self, "outside-")
        self.dest_root = tempdir(self, "dest-")
        self.env = dict(os.environ, HOME=self.home)

    def real_install(self, agent):
        return run(["bash", "install.sh", "--agent", agent, "--no-veloq"] + GATE_FLAGS, cwd=REPO, env=self.env)

    def dest_install(self, agent, dest):
        return run(["bash", "install.sh", "--agent", agent, "--dest", dest, "--no-veloq"] + GATE_FLAGS,
                   cwd=REPO, env=self.env)

    def test_existing_nonrepo_symlink_in_codex_destination_is_untouched(self):
        target = os.path.join(self.outside, "delegate")
        write(os.path.join(target, "SKILL.md"), "my own delegate\n")
        os.makedirs(os.path.join(self.home, ".codex", "skills"))
        link = os.path.join(self.home, ".codex", "skills", "claude-delegate")
        os.symlink(target, link)
        self.real_install("codex")
        self.assertTrue(os.path.islink(link))
        self.assertEqual(os.readlink(link), target)

    def test_absent_codex_destination_reuses_existing_user_level_copy_as_alias_only(self):
        user_root = mock_user_level(self.home, "claude-delegate", "MOCK USER LEVEL")
        mock_user_level(self.home, "codex-background", "MOCK USER LEVEL")
        before = snapshot(user_root)
        rc, out = self.real_install("codex")
        alias = os.path.join(self.home, ".codex", "skills", "claude-delegate")
        self.assertTrue(os.path.islink(alias), out[-1500:])
        self.assertEqual(os.readlink(alias), user_root)
        self.assertTrue(os.path.islink(os.path.join(self.home, ".codex", "skills", "codex-background")))
        self.assertEqual(snapshot(user_root), before)
        self.assertFalse(os.path.islink(user_root))
        self.assertIn("别名", out)

    def test_dest_install_never_reads_or_copies_real_home_protected_copy(self):
        mock_user_level(self.home, "claude-delegate", "MOCK USER LEVEL")
        mock_user_level(self.home, "codex-background", "MOCK USER LEVEL")
        dest = os.path.join(self.dest_root, "skills")
        self.dest_install("codex", dest)
        bundled = os.path.join(dest, "claude-delegate")
        self.assertFalse(os.path.islink(bundled))
        self.assertTrue(os.path.isfile(os.path.join(bundled, "scripts", "relay.mjs")))
        with open(os.path.join(bundled, "SKILL.md"), encoding="utf-8") as f:
            self.assertNotIn("MOCK USER LEVEL", f.read())

    def test_fresh_empty_destination_installs_local_bundled_version(self):
        dest = os.path.join(self.dest_root, "skills")
        self.dest_install("codex", dest)
        bundled = os.path.join(dest, "claude-delegate")
        self.assertTrue(os.path.isdir(bundled) and not os.path.islink(bundled))
        with open(os.path.join(REPO, "skills", "other", "claude-delegate", "SKILL.md"), encoding="utf-8") as f:
            expected = f.read()
        with open(os.path.join(bundled, "SKILL.md"), encoding="utf-8") as f:
            self.assertEqual(f.read(), expected)
        self.assertTrue(os.path.isfile(os.path.join(dest, "codex-background", "scripts", "bootstrap.mjs")))

    def test_external_catalog_cannot_occupy_protected_names(self):
        fixture = tempdir(self, "fixture-")
        shutil.copy(os.path.join(REPO, "install.sh"), os.path.join(fixture, "install.sh"))
        write(os.path.join(fixture, "skills", "kernel", "cuda-skill", "SKILL.md"), skill_md("cuda-skill", "x"))
        write(os.path.join(fixture, "repos", "catalog", "skills", "claude-delegate", "SKILL.md"),
              "CATALOG COPY\n")
        write(os.path.join(fixture, CONF_REL), conf_text([
            "catalog|https://example.invalid/c|UNPINNED|skills|skills|none|fixture",
        ]))
        dest = os.path.join(self.dest_root, "skills")
        out = run(["bash", os.path.join(fixture, "install.sh"), "--agent", "codex", "--dest", dest,
                   "--no-veloq", "--no-tilelang-skills"] + GATE_FLAGS, cwd=fixture, env=self.env)[1]
        self.assertFalse(os.path.exists(os.path.join(dest, "claude-delegate")))
        # 本地保护名先于 catalog 登记，因此 catalog 的同名候选被计为“跳过同名/保留”。
        self.assertIn("已安装 0 个 skill（跳过同名/保留 1 个）", out)

    def test_background_is_only_installed_for_codex(self):
        dest_claude = os.path.join(self.dest_root, "claude")
        self.dest_install("claude", dest_claude)
        self.assertTrue(os.path.isdir(os.path.join(dest_claude, "claude-delegate")))
        self.assertFalse(os.path.exists(os.path.join(dest_claude, "codex-background")))
        dest_codex = os.path.join(self.dest_root, "codex")
        self.dest_install("codex", dest_codex)
        self.assertTrue(os.path.isdir(os.path.join(dest_codex, "codex-background")))

    def test_macos_hint_does_not_tell_patched_users_to_replace_registration(self):
        if sys.platform != "darwin":
            self.skipTest("macOS-specific hint")
        rc, out = self.real_install("codex")
        self.assertIn("不要替换", out)
        self.assertNotIn('node "', out)


class ReaMcpTest(unittest.TestCase):
    """install-rea-mcp.sh：只为 codex 注册，使用 REA 自身的 setup；Node 预检；失败必须冒泡。"""

    def setUp(self):
        self.fresh = tempdir(self, "fresh-")
        copy_helpers(self.fresh, ["install-rea-mcp.sh"])
        self.bin = tempdir(self, "bin-")
        self.home = tempdir(self, "home-")
        self.log = os.path.join(self.bin, "rea.log")
        self.delegate_target = tempdir(self, "delegate-")
        os.makedirs(os.path.join(self.home, ".agents", "skills"))
        os.symlink(self.delegate_target, os.path.join(self.home, ".agents", "skills", "claude-delegate"))
        write(os.path.join(self.home, ".codex", "config.toml"),
              "[mcp_servers.other]\ncommand = \"other\"\n")
        self.config_before = snapshot(self.home)
        with open(os.path.join(self.home, ".codex", "config.toml"), encoding="utf-8") as f:
            self.config_text = f.read()
        node_stub(self.bin, "v24.22.1")
        write(os.path.join(self.fresh, CONF_REL), conf_text([
            "rea-fix|https://example.invalid/rea|UNPINNED|skills|runtime-skills|npm:rea-agents@6.7.0:fake-rea|fixture",
        ]))
        self._rea(exit_code=0)

    def _rea(self, exit_code=0):
        write_stub(os.path.join(self.bin, "fake-rea"),
                   'echo "$@" >> \'%s\'\nif [ "$1" = setup ]; then echo \'{"status":"ok"}\'; exit %d; fi\n'
                   % (self.log, exit_code))

    def _env(self, path=None):
        return dict(os.environ, HOME=self.home, PATH=self.bin + os.pathsep + os.environ["PATH"])

    def _run(self, *args):
        return run(["bash", "install-rea-mcp.sh"] + list(args), cwd=self.fresh, env=self._env())

    def test_registers_codex_with_scoped_noninteractive_command(self):
        rc, out = self._run("--agent", "codex")
        self.assertEqual(rc, 0, out)
        with open(self.log, encoding="utf-8") as f:
            self.assertEqual(f.read().strip(), "setup --client codex --skill=false --yes --json")

    def test_other_agents_are_not_registered(self):
        rc, out = self._run("--agent", "cursor")
        self.assertEqual(rc, 0)
        self.assertIn("仅注册到 codex", out)
        self.assertFalse(os.path.exists(self.log))

    def test_unrelated_codex_config_and_delegate_symlink_are_untouched(self):
        rc, out = self._run("--agent", "codex")
        self.assertEqual(rc, 0, out)
        with open(os.path.join(self.home, ".codex", "config.toml"), encoding="utf-8") as f:
            self.assertEqual(f.read(), self.config_text)
        link = os.path.join(self.home, ".agents", "skills", "claude-delegate")
        self.assertEqual(os.readlink(link), self.delegate_target)

    def test_setup_failure_propagates_as_nonzero(self):
        self._rea(exit_code=3)
        rc, out = self._run("--agent", "codex")
        self.assertNotEqual(rc, 0)
        self.assertIn("rea setup 失败", out)

    def test_incompatible_node_is_blocked_before_registration(self):
        node_stub(self.bin, "v24.10.0")
        rc, out = self._run("--agent", "codex")
        self.assertNotEqual(rc, 0)
        self.assertIn("不满足已安装 rea-agents 包的 engines", out)
        self.assertIn("PATH=$HOME/.nvm/versions/node/v24.22.1/bin", out)
        self.assertFalse(os.path.exists(self.log))

    def test_node_preflight_follows_published_rea_engines_boundaries(self):
        # 已发布 rea-agents@6.7.0 的 engines: ^22.19.0 || ^24.11.0 || >=26.0.0
        accepted = ("v22.19.0", "v24.11.0", "v24.21.0", "v26.0.0", "v27.0.0")
        rejected = ("v21.9.0", "v22.18.0", "v23.5.0", "v24.10.0", "v25.1.0")
        for version in accepted:
            node_stub(self.bin, version)
            rc, out = self._run("--agent", "codex")
            self.assertEqual(rc, 0, "%s should be accepted: %s" % (version, out))
        if os.path.exists(self.log):
            os.remove(self.log)
        for version in rejected:
            node_stub(self.bin, version)
            rc, out = self._run("--agent", "codex")
            self.assertNotEqual(rc, 0, "%s should be rejected" % version)
            self.assertIn("不满足已安装 rea-agents 包的 engines", out)
            self.assertFalse(os.path.exists(self.log))

    def test_missing_rea_binary_fails_when_manifest_requests_it(self):
        os.remove(os.path.join(self.bin, "fake-rea"))
        env = dict(os.environ, HOME=self.home, PATH="/usr/bin:/bin")
        rc, out = run(["bash", "install-rea-mcp.sh", "--agent", "codex"], cwd=self.fresh, env=env)
        self.assertNotEqual(rc, 0)
        self.assertIn("找不到 'fake-rea'", out)

    def test_no_rea_row_means_nothing_to_register(self):
        write(os.path.join(self.fresh, CONF_REL), conf_text([
            "other|https://example.invalid/o|UNPINNED|none|-|none|fixture",
        ]))
        rc, out = self._run("--agent", "codex")
        self.assertEqual(rc, 0)
        self.assertIn("跳过", out)
        self.assertFalse(os.path.exists(self.log))


class BootstrapReaAndInventoryTest(unittest.TestCase):
    """单次 bootstrap 的 REA 注册门控，以及新来源在同一次运行中即可被路由。"""

    def setUp(self):
        self.remote, self.sha = make_remote(self, "fixture-catalog", {
            "skills/fresh-skill/SKILL.md": skill_md("fresh-skill", "installed from a single bootstrap run"),
        })
        self.fresh = tempdir(self, "fresh-")
        copy_helpers(self.fresh, [s for s in SCRIPTS if not s.startswith("scripts")])
        shutil.copytree(os.path.join(REPO, "skills", "routers", "agent-skill-router"),
                        os.path.join(self.fresh, "skills", "routers", "agent-skill-router"))
        write(os.path.join(self.fresh, "skills", "kernel", "cuda-skill", "SKILL.md"), skill_md("cuda-skill", "x"))
        self.home = tempdir(self, "home-")
        self.bin = tempdir(self, "bin-")
        node_stub(self.bin, "v24.22.1")
        self.rea_log = os.path.join(self.bin, "rea.log")
        self.env = dict(os.environ, HOME=self.home, PATH=self.bin + os.pathsep + os.environ["PATH"])

    def _conf(self, rows):
        write(os.path.join(self.fresh, CONF_REL), conf_text(rows))

    def _bootstrap(self, *extra):
        return run(["bash", "bootstrap.sh", "--agent", "codex", "--no-veloq"] + list(extra) + GATE_FLAGS,
                   cwd=self.fresh, env=self.env)

    def _route(self, query):
        return run([sys.executable, os.path.join(self.fresh, "skills", "routers", "agent-skill-router",
                                                 "scripts", "route_skill.py"), query, "--json"],
                   cwd=self.fresh, env=self.env)

    def test_single_bootstrap_run_makes_new_source_routable(self):
        self._conf(["fixture-catalog|%s|%s|skills|skills|none|fixture" % (self.remote, self.sha)])
        self._bootstrap("--no-update")
        rc, out = self._route("用 fresh-skill skill 处理这个任务")
        self.assertEqual(rc, 0, out)
        self.assertIn('"status": "installable"', out)
        self.assertIn('"skill": "fresh-skill"', out)

    def test_inventory_regeneration_is_reported_in_bootstrap_output(self):
        self._conf(["fixture-catalog|%s|%s|skills|skills|none|fixture" % (self.remote, self.sha)])
        rc, out = self._bootstrap("--no-update")
        self.assertIn("路由清单", out)
        self.assertIn("已写入", out)

    def test_rea_mcp_is_skipped_by_no_integrations_and_dest(self):
        # install.sh 的验证会调用 fake-rea --version，因此只断言没有 setup 调用。
        self._conf(["rea-fix|https://example.invalid/rea|UNPINNED|skills|runtime-skills|npm:rea-agents@6.7.0:fake-rea|fixture"])
        write_stub(os.path.join(self.bin, "fake-rea"), 'echo "$@" >> \'%s\'\nexit 0\n' % self.rea_log)
        self._bootstrap("--no-update", "--no-integrations")
        self.assertNotIn("setup", self._log_text())
        dest = os.path.join(tempdir(self, "dest-"), "skills")
        run(["bash", "bootstrap.sh", "--agent", "codex", "--dest", dest, "--no-veloq"] + GATE_FLAGS,
            cwd=self.fresh, env=self.env)
        self.assertNotIn("setup", self._log_text())

    def _log_text(self):
        if not os.path.exists(self.rea_log):
            return ""
        with open(self.rea_log, encoding="utf-8") as f:
            return f.read()

    def test_dest_bootstrap_does_not_modify_tracked_inventory(self):
        inventory = os.path.join(self.fresh, "skills", "routers", "agent-skill-router", "references",
                                 "inventory.tsv")
        with open(inventory, "rb") as f:
            before = f.read()
        self._conf(["fixture-catalog|%s|%s|skills|skills|none|fixture" % (self.remote, self.sha)])
        dest = os.path.join(tempdir(self, "dest-"), "skills")
        run(["bash", "bootstrap.sh", "--agent", "codex", "--dest", dest, "--no-veloq"] + GATE_FLAGS,
            cwd=self.fresh, env=self.env)
        with open(inventory, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_rea_mcp_failure_propagates_through_bootstrap(self):
        self._conf(["rea-fix|%s|%s|skills|runtime-skills|npm:rea-agents@6.7.0:fake-rea|fixture"
                    % (self.remote, self.sha)])
        self._stub_npm_and_rea(setup_exit=3)
        rc, out = self._bootstrap("--no-update")
        self.assertNotEqual(rc, 0)
        self.assertIn("rea setup 失败", out)
        self.assertIn("Bootstrap 未完成", out)

    def _stub_npm_and_rea(self, setup_exit):
        global_root = tempdir(self, "npm-global-")
        pkg = os.path.join(global_root, "node_modules")
        write_stub(os.path.join(self.bin, "npm"),
                   'echo "$@" >> \'%s\'\nif [ "$1" = root ]; then echo \'%s\'; exit 0; fi\n'
                   'if [ "$1" = install ]; then\n  mkdir -p \'%s\'/rea-agents/skills/fixture-skill\n'
                   '  printf -- \'---\\nname: fixture-skill\\n---\\n\' > \'%s\'/rea-agents/skills/fixture-skill/SKILL.md\n'
                   '  exit 0\nfi\nexit 1\n' % (self.rea_log, pkg, pkg, pkg))
        write_stub(os.path.join(self.bin, "fake-rea"),
                   'echo "$@" >> \'%s\'\nif [ "$1" = setup ]; then exit %d; fi\nexit 0\n'
                   % (self.rea_log, setup_exit))


if __name__ == "__main__":
    unittest.main()
