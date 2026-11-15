"""独立安装脚本 install-agent-desktop.sh 的离线测试。

运行（仓库根目录）:
  python3 -m unittest discover -s tests -p 'test_agent_desktop_install.py' -v

约定:
  - 所有写入只发生在临时目录：HOME、--dest、以及复制到临时目录的脚本与 checkout。
  - 上游用本地 bare 仓库伪造；npm 用 PATH 中的桩程序，不访问网络、不安装真实包。
  - 复制到临时目录的脚本只把 UPSTREAM_URL 与 UPSTREAM_SHA 换成夹具仓库的值；
    pin 本身由 ScriptStaticTests 直接读取仓库中的脚本。
  - 真实 checkout 测试只读：固定提交不在本地 repos/agent-desktop 中时跳过，不联网获取。
"""

import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = "install-agent-desktop.sh"
UPSTREAM_URL = "https://github.com/vercel-labs/agent-desktop.git"
PINNED_SHA = "b0b7277bea09b770ef9d1fb45aad667bc432cd57"
MAIN_SHA = "8ffecc1d3d088727f8d84baf4dde0224aab2e8ad"
PINNED_VERSION = "1.3.1"
SKILL_FILES = [
    "SKILL.md",
    "references/commands-interaction.md",
    "references/commands-observation.md",
    "references/commands-system.md",
    "references/workflows.md",
]
UNIFIED_FILES = ["install.sh", "bootstrap.sh", "install-external.sh", "install-rea-mcp.sh",
                 "install-veloq.sh", "update-repos.sh", os.path.join("sources", "external-sources.conf"),
                 os.path.join("scripts", "build_sources.py")]
GIT_ID = ["-c", "user.name=agent-gpu-skills-test", "-c", "user.email=test@example.invalid"]

# 桩 npm：记录参数；在 --prefix 下生成包、原生二进制和 .bin 入口。环境变量可制造失败场景。
NPM_STUB = """#!/bin/bash
echo "$@" >> "$NPM_LOG"
for a in "$@"; do
    case "$a" in
        --allow-scripts*)
            echo "npm error code EALLOWSCRIPTS" >&2
            echo "npm error --allow-scripts is not allowed in project-scoped installs" >&2
            exit 1 ;;
    esac
done
if [ -n "${FAKE_NPM_FAIL:-}" ]; then echo "fake npm failure" >&2; exit 1; fi
prefix=""
while [ $# -gt 0 ]; do
    case "$1" in
        --prefix) prefix="$2"; shift 2 ;;
        *) shift ;;
    esac
done
pkg="$prefix/node_modules/agent-desktop"
mkdir -p "$pkg/lib" "$pkg/bin" "$prefix/node_modules/.bin"
printf '{"name":"agent-desktop","version":"%s"}\\n' "${FAKE_NPM_VERSION:-1.3.1}" > "$pkg/package.json"
cat > "$pkg/lib/platform.js" <<'JS'
module.exports = { resolve: (p, a) => ({ released: true, binaryName: "agent-desktop-fake-native" }) };
JS
if [ -z "${FAKE_NPM_NO_NATIVE:-}" ]; then echo native > "$pkg/bin/agent-desktop-fake-native"; fi
cat > "$prefix/node_modules/.bin/agent-desktop" <<'SH'
#!/bin/bash
touch "$(dirname "$0")/../agent-desktop/ran"
echo "fake-agent-desktop"
for a in "$@"; do echo "arg:$a"; done
SH
chmod +x "$prefix/node_modules/.bin/agent-desktop"
"""


def tempdir(test, prefix):
    path = tempfile.mkdtemp(prefix=prefix)
    test.addCleanup(shutil.rmtree, path, True)
    return path


def git(cwd, *args):
    proc = subprocess.run(["git", *args], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise AssertionError("git %s failed: %s" % (" ".join(args), proc.stdout.decode()))
    return proc.stdout.decode("utf-8")


def write(path, text, executable=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    if executable:
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)


def snapshot(path):
    """目录树快照（相对路径，含目录；不跟随符号链接）。路径不存在时返回空列表。"""
    if not os.path.lexists(path):
        return []
    entries = []
    for dirpath, dirnames, filenames in os.walk(path):
        for name in dirnames + filenames:
            entries.append(os.path.relpath(os.path.join(dirpath, name), path))
    return sorted(entries)


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def run_script(root, args, env):
    proc = subprocess.run(["bash", os.path.join(root, SCRIPT), *args], cwd=root, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout.decode("utf-8", errors="replace")


def make_upstream(test, extra_files=()):
    """夹具上游：提交 skills/agent-desktop（与 release 文件集合一致）与 LICENSE，并导出为 bare 远端。返回 (远端路径, SHA)。"""
    root = tempdir(test, "upstream-")
    src = os.path.join(root, "src")
    os.makedirs(src)
    git(src, "init", "-q")
    for rel in SKILL_FILES + list(extra_files):
        write(os.path.join(src, "skills", "agent-desktop", rel), "# fixture %s\n" % rel)
    write(os.path.join(src, "LICENSE"), "Apache License\nVersion 2.0, January 2004\n")
    git(src, "add", "-A")
    git(src, *GIT_ID, "commit", "-q", "-m", "fixture")
    remote = os.path.join(root, "remote.git")
    git(root, "clone", "-q", "--bare", src, remote)
    return remote, git(src, "rev-parse", "HEAD").strip()


def make_root(test, remote, sha, checkout=True):
    """复制脚本到临时根目录；只在副本中替换 URL 与 SHA。checkout=True 时预先 clone 夹具远端到 repos/agent-desktop。"""
    root = tempdir(test, "root-")
    with open(os.path.join(REPO, SCRIPT), encoding="utf-8") as f:
        text = f.read()
    replaced = text.replace('UPSTREAM_URL="%s"' % UPSTREAM_URL, 'UPSTREAM_URL="%s"' % remote)
    replaced = replaced.replace('UPSTREAM_SHA="%s"' % PINNED_SHA, 'UPSTREAM_SHA="%s"' % sha)
    if replaced.count(remote) != 1 or sha not in replaced:
        raise AssertionError("fixture substitution failed")
    write(os.path.join(root, SCRIPT), replaced, executable=True)
    os.makedirs(os.path.join(root, "scripts"))
    shutil.copy(os.path.join(REPO, "scripts", "node-preflight.sh"), os.path.join(root, "scripts"))
    if checkout:
        os.makedirs(os.path.join(root, "repos"))
        git(root, "clone", "-q", remote, os.path.join(root, "repos", "agent-desktop"))
    return root


def base_env(home, bindir, npm_log):
    env = dict(os.environ)
    env.update({"HOME": home, "PATH": bindir + os.pathsep + os.environ["PATH"], "NPM_LOG": npm_log})
    for key in ("FAKE_NPM_FAIL", "FAKE_NPM_VERSION", "FAKE_NPM_NO_NATIVE"):
        env.pop(key, None)
    return env


class ScriptStaticTests(unittest.TestCase):
    def setUp(self):
        self.text = read(os.path.join(REPO, SCRIPT))

    def test_bash_syntax_passes(self):
        for rel in (SCRIPT, os.path.join("scripts", "node-preflight.sh")):
            proc = subprocess.run(["bash", "-n", os.path.join(REPO, rel)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(proc.returncode, 0, proc.stdout.decode())

    def test_npm_version_and_release_sha_are_pinned(self):
        self.assertIn('NPM_VERSION="%s"' % PINNED_VERSION, self.text)
        self.assertIn('UPSTREAM_SHA="%s"' % PINNED_SHA, self.text)
        self.assertNotIn(MAIN_SHA, self.text, "skill must come from the release commit, not main")

    def test_npm_runtime_is_private_prefix_never_global(self):
        self.assertIn('npm install --prefix "$RUNTIME_DIR" --ignore-scripts', self.text)
        self.assertIn('"$NPM_SPEC"', self.text)
        self.assertNotRegex(self.text, r"(?m)^[^#\n]*--allow-scripts")
        self.assertNotRegex(self.text, r"npm\s+install\s+[^\n]*\s-g\b")
        self.assertNotRegex(self.text, r"npm\s+install\s+[^\n]*--global\b")

    def test_unified_installers_do_not_reference_the_standalone_script(self):
        for rel in UNIFIED_FILES:
            with open(os.path.join(REPO, rel), encoding="utf-8") as f:
                body = f.read()
            self.assertNotIn("agent-desktop", body, rel)
            self.assertNotIn("install-agent-desktop", body, rel)

    def test_node_preflight_has_agent_desktop_engine_rule(self):
        preflight = read(os.path.join(REPO, "scripts", "node-preflight.sh"))
        self.assertIn("agent-desktop)", preflight)
        self.assertIn("agent-desktop 要求的 >=18", preflight)

    def test_help_documents_agent_and_dest(self):
        proc = subprocess.run(["bash", os.path.join(REPO, SCRIPT), "--help"], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
        self.assertEqual(proc.returncode, 0)
        out = proc.stdout.decode("utf-8")
        self.assertIn("--agent", out)
        self.assertIn("--dest", out)


class RealCheckoutTests(unittest.TestCase):
    """读取真实 repos/agent-desktop 的固定提交，验证 skill 字节与 git 对象完全一致（只读，不联网）。"""

    def setUp(self):
        probe = subprocess.run(["git", "-C", os.path.join(REPO, "repos", "agent-desktop"), "cat-file", "-e",
                                PINNED_SHA + "^{commit}"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if probe.returncode != 0:
            self.skipTest("本地 repos/agent-desktop 中没有固定提交 %s" % PINNED_SHA[:12])
        self.home = tempdir(self, "home-")
        self.dest = os.path.join(tempdir(self, "dest-"), "skills")
        self.bindir = tempdir(self, "bin-")
        self.npm_log = os.path.join(self.bindir, "npm.log")

    def test_dest_install_matches_release_commit_objects_byte_for_byte(self):
        env = base_env(self.home, self.bindir, self.npm_log)
        rc, out = run_script(REPO, ["--dest", self.dest], env)
        self.assertEqual(rc, 0, out)
        skill = os.path.join(self.dest, "agent-desktop")
        expected = sorted(SKILL_FILES + ["LICENSE", ".agent-desktop-standalone-installer"])
        self.assertEqual(snapshot(skill), sorted(expected + ["references"]), out)
        for rel in SKILL_FILES:
            blob = subprocess.run(["git", "-C", os.path.join(REPO, "repos", "agent-desktop"), "cat-file", "blob",
                                   "%s:skills/agent-desktop/%s" % (PINNED_SHA, rel)], stdout=subprocess.PIPE).stdout
            with open(os.path.join(skill, rel), "rb") as f:
                self.assertEqual(f.read(), blob, rel)
        self.assertFalse(os.path.lexists(self.npm_log), "--dest must not run npm")
        self.assertEqual(snapshot(self.home), [], "--dest must not write HOME")


class DestInstallTests(unittest.TestCase):
    def setUp(self):
        self.remote, self.sha = make_upstream(self, extra_files=())
        self.root = make_root(self, self.remote, self.sha)
        self.home = tempdir(self, "home-")
        self.bindir = tempdir(self, "bin-")
        self.npm_log = os.path.join(self.bindir, "npm.log")
        self.dest = os.path.join(tempdir(self, "dest-"), "skills")
        self.skill = os.path.join(self.dest, "agent-desktop")

    def _run(self):
        return run_script(self.root, ["--dest", self.dest], base_env(self.home, self.bindir, self.npm_log))

    def test_dest_installs_complete_skill_with_license_and_marker(self):
        rc, out = self._run()
        self.assertEqual(rc, 0, out)
        for rel in SKILL_FILES:
            self.assertEqual(read(os.path.join(self.skill, rel)), "# fixture %s\n" % rel, rel)
        self.assertIn("Apache License", read(os.path.join(self.skill, "LICENSE")))
        marker = read(os.path.join(self.skill, ".agent-desktop-standalone-installer"))
        self.assertIn("package: agent-desktop@1.3.1", marker)
        self.assertIn(self.sha, marker)

    def test_rerun_is_clean_and_leaves_no_staging_directory(self):
        self.assertEqual(self._run()[0], 0)
        before = snapshot(self.dest)
        rc, out = self._run()
        self.assertEqual(rc, 0, out)
        self.assertEqual(snapshot(self.dest), before)
        self.assertFalse([n for n in os.listdir(self.dest) if n.startswith(".agent-desktop-stage")])

    def test_user_owned_skill_directory_is_refused_and_untouched(self):
        write(os.path.join(self.skill, "my-notes.md"), "keep me\n")
        rc, out = self._run()
        self.assertNotEqual(rc, 0, out)
        self.assertIn("BLOCKED", out)
        self.assertEqual(read(os.path.join(self.skill, "my-notes.md")), "keep me\n")
        self.assertFalse(os.path.lexists(os.path.join(self.skill, "SKILL.md")))

    def test_symlink_at_skill_target_is_refused_and_kept(self):
        elsewhere = tempdir(self, "elsewhere-")
        os.makedirs(self.dest)
        os.symlink(elsewhere, self.skill)
        rc, out = self._run()
        self.assertNotEqual(rc, 0, out)
        self.assertTrue(os.path.islink(self.skill))
        self.assertEqual(os.readlink(self.skill), elsewhere)

    def test_unknown_agent_is_rejected(self):
        proc = subprocess.run(["bash", os.path.join(self.root, SCRIPT), "--agent", "nope"],
                              env=base_env(self.home, self.bindir, self.npm_log), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("未知 agent", proc.stdout.decode())

    def test_dirty_checkout_is_refused_before_any_write(self):
        write(os.path.join(self.root, "repos", "agent-desktop", "untracked.txt"), "x\n")
        rc, out = self._run()
        self.assertNotEqual(rc, 0, out)
        self.assertIn("未提交或未跟踪的改动", out)
        self.assertFalse(os.path.lexists(self.dest))

    def test_wrong_origin_is_refused(self):
        git(os.path.join(self.root, "repos", "agent-desktop"), "remote", "set-url", "origin", "/elsewhere.git")
        rc, out = self._run()
        self.assertNotEqual(rc, 0, out)
        self.assertIn("origin", out)
        self.assertFalse(os.path.lexists(self.dest))

    def test_plain_directory_at_checkout_path_is_refused_and_kept(self):
        checkout = os.path.join(self.root, "repos", "agent-desktop")
        shutil.rmtree(checkout)
        write(os.path.join(checkout, "user-file.txt"), "not a checkout\n")
        rc, out = self._run()
        self.assertNotEqual(rc, 0, out)
        self.assertIn("不是独立 git checkout", out)
        self.assertEqual(read(os.path.join(checkout, "user-file.txt")), "not a checkout\n")

    def test_upstream_tree_with_extra_reference_fails_before_writing(self):
        remote, sha = make_upstream(self, extra_files=["references/extra.md"])
        root = make_root(self, remote, sha)
        rc, out = run_script(root, ["--dest", self.dest], base_env(self.home, self.bindir, self.npm_log))
        self.assertNotEqual(rc, 0, out)
        self.assertIn("文件与本脚本不一致", out)
        self.assertIn("references/extra.md", out)
        self.assertFalse(os.path.lexists(self.dest))

    def test_pinned_commit_missing_from_remote_is_a_failure(self):
        root = make_root(self, self.remote, "0" * 40)
        rc, out = run_script(root, ["--dest", self.dest], base_env(self.home, self.bindir, self.npm_log))
        self.assertNotEqual(rc, 0, out)
        self.assertIn("无法获取", out)
        self.assertFalse(os.path.lexists(self.dest))

    def test_missing_checkout_is_created_and_only_pinned_commit_fetched(self):
        root = make_root(self, self.remote, self.sha, checkout=False)
        rc, out = run_script(root, ["--dest", self.dest], base_env(self.home, self.bindir, self.npm_log))
        self.assertEqual(rc, 0, out)
        self.assertTrue(os.path.isdir(os.path.join(root, "repos", "agent-desktop", ".git")))
        self.assertEqual(read(os.path.join(self.dest, "agent-desktop", "SKILL.md")), "# fixture SKILL.md\n")

    def test_dest_install_never_writes_home_or_runs_npm(self):
        rc, out = self._run()
        self.assertEqual(rc, 0, out)
        self.assertEqual(snapshot(self.home), [])
        self.assertFalse(os.path.lexists(self.npm_log))


class RealModeTests(unittest.TestCase):
    """真实安装路径：HOME 与 npm 都是临时桩，验证写入范围、launcher、冲突与失败处理。"""

    def setUp(self):
        self.remote, self.sha = make_upstream(self)
        self.root = make_root(self, self.remote, self.sha)
        self.home = tempdir(self, "home-")
        self.bindir = tempdir(self, "bin-")
        write(os.path.join(self.bindir, "npm"), NPM_STUB, executable=True)
        self.npm_log = os.path.join(self.bindir, "npm.log")
        self.skill = os.path.join(self.home, ".codex", "skills", "agent-desktop")
        self.runtime = os.path.join(self.home, ".local", "share", "agent-gpu-skills", "agent-desktop", "1.3.1")
        self.launcher = os.path.join(self.home, ".local", "bin", "agent-desktop")

    def _run(self, extra_env=None, args=()):
        env = base_env(self.home, self.bindir, self.npm_log)
        env.update(extra_env or {})
        return run_script(self.root, [*args], env)

    def _npm_calls(self):
        return read(self.npm_log).splitlines() if os.path.exists(self.npm_log) else []

    def test_real_install_pins_npm_version_and_writes_only_owned_paths(self):
        rc, out = self._run()
        self.assertEqual(rc, 0, out)
        calls = self._npm_calls()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], "install --prefix %s --ignore-scripts --no-save --no-audit --no-fund agent-desktop@1.3.1"
                         % self.runtime)
        self.assertNotIn("--allow-scripts", calls[0])
        self.assertNotIn(" -g", calls[0])
        self.assertTrue(os.path.isfile(os.path.join(self.skill, "SKILL.md")))
        self.assertTrue(os.path.isfile(os.path.join(self.runtime, ".agent-desktop-standalone-installer")))
        self.assertTrue(os.access(self.launcher, os.X_OK))
        self.assertEqual(sorted(os.listdir(self.home)), [".codex", ".local"])
        self.assertFalse(os.path.lexists(os.path.join(self.runtime, "node_modules", "agent-desktop", "ran")),
                         "the installer must not run agent-desktop")

    def test_launcher_forwards_arguments_verbatim(self):
        self.assertEqual(self._run()[0], 0)
        proc = subprocess.run([self.launcher, "a b", "c'd"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              env=base_env(self.home, self.bindir, self.npm_log))
        out = proc.stdout.decode()
        self.assertEqual(proc.returncode, 0, out)
        self.assertIn("arg:a b", out)
        self.assertIn("arg:c'd", out)

    def test_rerun_is_clean(self):
        self.assertEqual(self._run()[0], 0)
        before = snapshot(self.home)
        rc, out = self._run()
        self.assertEqual(rc, 0, out)
        self.assertEqual(len(self._npm_calls()), 2)
        self.assertEqual(snapshot(self.home), before)

    def test_user_launcher_is_refused_and_no_write_happens(self):
        write(self.launcher, "#!/bin/bash\necho my own tool\n", executable=True)
        rc, out = self._run()
        self.assertNotEqual(rc, 0, out)
        self.assertIn("不是本脚本生成的 launcher", out)
        self.assertEqual(read(self.launcher), "#!/bin/bash\necho my own tool\n")
        self.assertEqual(self._npm_calls(), [])
        self.assertFalse(os.path.lexists(self.skill))

    def test_user_launcher_symlink_is_refused(self):
        elsewhere = os.path.join(tempdir(self, "tool-"), "tool")
        write(elsewhere, "#!/bin/bash\n", executable=True)
        os.makedirs(os.path.dirname(self.launcher))
        os.symlink(elsewhere, self.launcher)
        rc, out = self._run()
        self.assertNotEqual(rc, 0, out)
        self.assertTrue(os.path.islink(self.launcher))
        self.assertEqual(self._npm_calls(), [])

    def test_unowned_runtime_directory_is_refused_before_npm(self):
        write(os.path.join(self.runtime, "user-data.txt"), "keep\n")
        rc, out = self._run()
        self.assertNotEqual(rc, 0, out)
        self.assertIn("BLOCKED", out)
        self.assertEqual(read(os.path.join(self.runtime, "user-data.txt")), "keep\n")
        self.assertEqual(self._npm_calls(), [])

    def test_npm_failure_is_nonzero(self):
        rc, out = self._run({"FAKE_NPM_FAIL": "1"})
        self.assertNotEqual(rc, 0, out)
        self.assertIn("npm install 失败", out)
        self.assertFalse(os.path.lexists(self.launcher))

    def test_installed_version_mismatch_is_failure(self):
        rc, out = self._run({"FAKE_NPM_VERSION": "1.3.0"})
        self.assertNotEqual(rc, 0, out)
        self.assertIn("版本为 '1.3.0'", out)
        self.assertFalse(os.path.lexists(self.launcher))

    def test_missing_native_binary_is_failure(self):
        rc, out = self._run({"FAKE_NPM_NO_NATIVE": "1"})
        self.assertNotEqual(rc, 0, out)
        self.assertIn("原生二进制", out)
        self.assertFalse(os.path.lexists(self.launcher))

    def test_old_node_is_blocked_before_any_write(self):
        write(os.path.join(self.bindir, "node"), "#!/bin/bash\necho v16.20.0\n", executable=True)
        rc, out = self._run()
        self.assertNotEqual(rc, 0, out)
        self.assertIn("低于 agent-desktop 要求的 >=18", out)
        self.assertFalse(os.path.lexists(self.skill))
        self.assertEqual(self._npm_calls(), [])

    def test_missing_npm_is_blocked(self):
        if shutil.which("npm", path="/usr/bin:/bin"):
            self.skipTest("npm is available in /usr/bin:/bin; cannot simulate missing npm")
        bare = tempdir(self, "bare-bin-")
        os.symlink(os.path.realpath(shutil.which("node")), os.path.join(bare, "node"))
        env = base_env(self.home, bare, self.npm_log)
        env["PATH"] = bare + os.pathsep + "/usr/bin:/bin"
        rc, out = run_script(self.root, [], env)
        self.assertNotEqual(rc, 0, out)
        self.assertIn("缺少 npm", out)
        self.assertFalse(os.path.lexists(self.skill))


if __name__ == "__main__":
    unittest.main(verbosity=2)
