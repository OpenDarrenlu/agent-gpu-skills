"""Bounded behavioral checks; fixture artifacts and temporary scripts stay in devlog."""

import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess
import signal
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
SKILL = REPO / "skills/other/brain-and-hands"
DEVLOG = Path(os.environ.get("BAH_TEST_DEVLOG", str(REPO.parent / "devlog")))
DEVLOG.mkdir(exist_ok=True)
sys.path.insert(0, str(SKILL / "scripts"))
import route_task
import run_simple_test


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="bah-model-test-", dir=DEVLOG)
        self.addCleanup(self.tmp.cleanup)
        self.catalog = Path(self.tmp.name) / "models.json"
        self.catalog.write_text(json.dumps({"models": [
            {"slug": "gpt-5.6-sol"}, {"slug": "gpt-6.1-sol"}, {"slug": "gpt-6.10-sol"},
            {"slug": "gpt-9-sol", "visibility": "hide"},
            {"slug": "gpt-10-sol", "supported_in_api": False},
            {"slug": "gpt-6-astra"}, {"slug": "gpt-6.1-terra"}, {"slug": "gpt-11-luna"}]}))

    def route(self, host="codex", task="test", **kwargs):
        return route_task.resolve(host, task, self.catalog, environ={}, **kwargs)

    def test_latest_version_is_numeric_and_available(self):
        self.assertEqual(self.route()["model"], "gpt-6.10-sol")

    def test_debug_stays_in_strong_main_session(self):
        self.assertEqual(self.route(task="debug")["model"], "gpt-6-astra")
        self.assertEqual(self.route(task="debug")["session"], "main")

    def test_scout_uses_terra_not_newer_luna(self):
        self.assertEqual(self.route(task="scout")["model"], "gpt-6.1-terra")

    def test_explicit_pin_is_preserved(self):
        self.assertEqual(self.route(model="gpt-5.6-sol")["model"], "gpt-5.6-sol")

    def test_wrong_family_pin_is_rejected(self):
        with self.assertRaises(ValueError):
            self.route(model="gpt-6-astra")

    def test_environment_pin_can_be_overridden_explicitly(self):
        result = route_task.resolve("codex", "test", self.catalog, "gpt-6.1-sol",
                                    {"BAH_CODEX_SOL_MODEL": "gpt-5.6-sol"})
        self.assertEqual(result["model"], "gpt-6.1-sol")

    def test_simple_test_uses_fresh_claude_from_both_hosts(self):
        for host in ("codex", "claude"):
            result = self.route(host, "simple-test")
            self.assertEqual((result["host"], result["family"], result["session"]),
                             ("claude", "noir", "fresh-cli"))

    def test_noir_is_not_silently_replaced_with_gris(self):
        with self.assertRaises(ValueError):
            self.route(task="simple-test", model="a7n-gris")

    def test_noir_family_cannot_be_faked_by_a_substring(self):
        for model in ("not-a-noir-model", "a7n-noir-impostor", "a7n-gris-noir"):
            with self.assertRaises(ValueError):
                self.route(task="simple-test", model=model)
        self.assertTrue(route_task.is_noir("anthropic/a7n-noir[1m]"))
        self.assertTrue(route_task.is_noir("a7n-noir"))

    def test_claude_uses_family_aliases(self):
        for task, model in (("debug", "opus"), ("test", "sonnet"), ("scout", "haiku")):
            self.assertEqual(self.route("claude", task)["model"], model)

    def test_missing_family_fails_without_fallback(self):
        self.catalog.write_text('{"models":[{"slug":"gpt-9-luna"}]}')
        with self.assertRaises(ValueError):
            self.route()

    def test_missing_catalog_is_actionable(self):
        self.catalog.unlink()
        with self.assertRaisesRegex(ValueError, "models-file"):
            self.route()


FAKE_CLAUDE = '''#!/usr/bin/env python3
import json, os, pathlib, sys, time
args = sys.argv[1:]
prompt = sys.stdin.read()
pathlib.Path("capture.json").write_text(json.dumps({"args":args,"prompt":prompt,
    "nested_marker":os.environ.get("CLAUDECODE"),
    "subagent_model":os.environ.get("CLAUDE_CODE_SUBAGENT_MODEL"),
    "config_dir":os.environ.get("CLAUDE_CONFIG_DIR")}))
mode = pathlib.Path("mode.txt").read_text() if pathlib.Path("mode.txt").exists() else "ok"
if mode == "timeout": time.sleep(30)
if mode == "malformed": print("not a model result"); sys.exit(0)
model = "a7n-gris" if mode == "wrong-model" else "anthropic/a7n-noir[1m]"
if mode == "wrong-id": model = "not-a-noir-model"
print(json.dumps({"type":"result","is_error":mode == "error", "result":"Observed report. " + os.environ.get("ANTHROPIC_API_KEY", ""),
    "modelUsage":None if mode == "null-usage" else {model:{"inputTokens":10,"outputTokens":5}}}))
'''


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="bah-runner-test-", dir=DEVLOG)
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.work = self.base / "work"
        self.work.mkdir()
        self.fake = self.base / "claude-fake"
        self.fake.write_text(FAKE_CLAUDE)
        self.fake.chmod(0o755)
        self.settings = self.base / "settings.json"
        self.settings.write_text('{"env":{"ANTHROPIC_API_KEY":"dummy-private-key","ANTHROPIC_BASE_URL":"https://example.invalid"}}')

    def invoke(self, code="print('1 test passed')", mode="ok", extra=()):
        (self.work / "mode.txt").write_text(mode)
        artifact = self.base / ("artifacts-" + str(len(list(self.base.iterdir()))))
        command = [sys.executable, str(SKILL / "scripts/run_simple_test.py"),
                   "--cwd", str(self.work), "--artifacts", str(artifact), "--claude-bin", str(self.fake),
                   "--settings-file", str(self.settings), *extra,
                   "--", sys.executable, "-c", code]
        env = dict(os.environ)
        for key in run_simple_test.AUTH_KEYS:
            env.pop(key, None)
        env.update(CLAUDECODE="parent-claude-session", CLAUDE_CODE_SUBAGENT_MODEL="opus")
        result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=15)
        return result, json.loads((artifact / "run.json").read_text()), artifact

    def test_real_command_status_and_no_shell_expansion(self):
        result, record, artifact = self.invoke("print('literal $(touch INJECTED)')")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(record["test"]["exit_code"], 0)
        self.assertIn("$(touch INJECTED)", (artifact / "test.stdout").read_text())
        self.assertFalse((self.work / "INJECTED").exists())

    def test_failing_test_is_not_overridden_by_a_model_report(self):
        result, record, _ = self.invoke("import sys; print('1 failed'); sys.exit(7)")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(record["test"]["exit_code"], 7)
        self.assertTrue(record["session"]["valid_noir_result"])

    def test_nested_claude_environment_and_customizations_are_isolated(self):
        result, record, _ = self.invoke()
        self.assertEqual(result.returncode, 0)
        capture = json.loads((self.work / "capture.json").read_text())
        self.assertIsNone(capture["nested_marker"])
        self.assertIsNone(capture["subagent_model"])
        self.assertFalse(Path(capture["config_dir"]).exists())
        args = capture["args"]
        self.assertEqual(args[args.index("--tools") + 1], "")
        self.assertEqual(args[args.index("--setting-sources") + 1], "")
        self.assertIn("--bare", args)
        self.assertIn("--strict-mcp-config", args)
        self.assertNotIn("--resume", args)
        self.assertEqual(args[args.index("--session-id") + 1], record["session_id"])

    def test_each_run_has_a_new_session_id(self):
        _, first, _ = self.invoke()
        _, second, _ = self.invoke()
        self.assertNotEqual(first["session_id"], second["session_id"])

    def test_provider_credentials_are_not_in_output_artifacts(self):
        result, _, artifact = self.invoke("print('dummy-private-key')")
        self.assertNotIn("dummy-private-key", result.stdout + result.stderr)
        for name in ("run.json", "session.jsonl", "session.stderr", "noir-report.txt", "test.stdout", "test.stderr"):
            self.assertNotIn("dummy-private-key", (artifact / name).read_text())
        capture = json.loads((self.work / "capture.json").read_text())
        self.assertNotIn("dummy-private-key", capture["prompt"])

    def test_malformed_or_wrong_model_is_not_success(self):
        for mode in ("malformed", "wrong-model", "wrong-id", "null-usage", "error"):
            result, record, _ = self.invoke(mode=mode)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(record["session"]["valid_noir_result"])
            self.assertNotIn("Traceback", result.stderr)

    def test_model_launch_failure_does_not_overwrite_the_test_failure(self):
        result, record, _ = self.invoke("import os,sys; os.unlink(" + repr(str(self.fake)) + "); sys.exit(7)")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(record["test"]["exit_code"], 7)
        self.assertEqual(record["session"]["exit_code"], 2)
        self.assertFalse(record["session"]["valid_noir_result"])

    def test_test_timeout_is_distinct_and_stops_its_process(self):
        result, record, _ = self.invoke("import time; time.sleep(30)", extra=("--test-timeout", "0.1"))
        self.assertEqual(result.returncode, 124)
        self.assertTrue(record["test"]["timed_out"])

    def test_model_timeout_preserves_passing_test(self):
        result, record, _ = self.invoke(mode="timeout", extra=("--session-timeout", "0.1"))
        self.assertEqual(result.returncode, 2)
        self.assertEqual(record["test"]["exit_code"], 0)
        self.assertTrue(record["session"]["timed_out"])

    def test_refuses_artifacts_inside_project(self):
        result = subprocess.run([sys.executable, str(SKILL / "scripts/run_simple_test.py"),
                                 "--cwd", str(self.work), "--artifacts", str(self.work / "bad"),
                                 "--", sys.executable, "-c", "print(1)"], capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.work / "bad").exists())

    def test_caller_interruption_reaps_only_the_test_child(self):
        artifacts = self.base / "interrupted"
        child_code = "import os,pathlib,time; pathlib.Path('child.pid').write_text(str(os.getpid())); time.sleep(30)"
        process = subprocess.Popen([sys.executable, str(SKILL / "scripts/run_simple_test.py"),
            "--cwd", str(self.work), "--artifacts", str(artifacts), "--claude-bin", str(self.fake),
            "--settings-file", str(self.settings), "--", sys.executable, "-c", child_code],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(lambda: process.poll() is None and process.kill())
        pidfile = self.work / "child.pid"
        deadline = time.monotonic() + 5
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(pidfile.exists())
        child_pid = int(pidfile.read_text())
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=5)
        self.assertEqual(process.returncode, 130)
        with self.assertRaises(ProcessLookupError):
            os.kill(child_pid, 0)
        self.assertNotEqual(json.loads((artifacts / "run.json").read_text()).get("state"), "complete")


class InstallationTests(unittest.TestCase):
    def test_unified_install_reuses_the_single_skill_payload(self):
        with tempfile.TemporaryDirectory(prefix="bah-unified-install-test-", dir=DEVLOG) as directory:
            root = Path(directory) / "repo"
            source = root / "skills/other/brain-and-hands"
            shutil.copytree(SKILL, source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            (root / "skills/kernel/cuda-skill").mkdir(parents=True)
            script = (REPO / "install.sh").read_text()
            script = re.sub(r"^SKILL_NAMES=\(.*?^\)", "SKILL_NAMES=(brain-and-hands)", script, flags=re.S | re.M)
            script = re.sub(r"^SKILL_DIRS=\(.*?^\)", "SKILL_DIRS=(skills/other/brain-and-hands)", script, flags=re.S | re.M)
            # Exercise the real install path, excluding unrelated full-bundle verification.
            call = '\ninstall_to_agent "$AGENT"\n'
            script = script[:script.index(call) + len(call)]
            (root / "install.sh").write_text(script)
            shutil.copyfile(REPO / "install-brain-and-hands.sh", root / "install-brain-and-hands.sh")
            for host in ("codex", "claude"):
                target = Path(directory) / host
                result = subprocess.run(["bash", str(root / "install.sh"), "--agent", host, "--dest", str(target),
                    "--no-veloq", "--no-tilelang-skills", "--no-nvidia-skills", "--no-amd-skills",
                    "--no-cursor-skills", "--no-ccfa-skills", "--no-integrations"], capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout[-3000:])
                installed = target / "brain-and-hands"
                self.assertEqual((installed / ".brain-and-hands-source").read_text().strip(), str(source))
                self.assertFalse((installed / ".claude-plugin").exists())
                subprocess.run(["bash", str(root / "install-brain-and-hands.sh"), "--agent", host,
                                "--dest", str(target)], check=True, capture_output=True, timeout=5)

    def test_isolated_install_for_both_hosts_preserves_complete_skill(self):
        with tempfile.TemporaryDirectory(prefix="bah-install-test-", dir=DEVLOG) as directory:
            for host in ("codex", "claude"):
                target = Path(directory) / host
                command = ["bash", str(REPO / "install-brain-and-hands.sh"), "--agent", host, "--dest", str(target)]
                subprocess.run(command, check=True, capture_output=True)
                subprocess.run(command, check=True, capture_output=True)
                installed = target / "brain-and-hands"
                self.assertEqual((installed / "SKILL.md").read_text(), (SKILL / "SKILL.md").read_text())
                self.assertTrue((installed / "references/models.md").is_file())
                self.assertFalse((installed / ".claude-plugin").exists())
                result = subprocess.run([sys.executable, str(installed / "scripts/route_task.py"),
                                         "--host", host, "--task", "simple-test"], capture_output=True, text=True, check=True)
                self.assertEqual(json.loads(result.stdout)["family"], "noir")
                subprocess.run(command + ["--uninstall"], check=True, capture_output=True)
                self.assertFalse(installed.exists())

    def test_preserves_existing_unmanaged_skill(self):
        with tempfile.TemporaryDirectory(prefix="bah-preserve-test-", dir=DEVLOG) as directory:
            target = Path(directory) / "brain-and-hands"
            target.mkdir()
            (target / "SKILL.md").write_text("user content")
            result = subprocess.run(["bash", str(REPO / "install-brain-and-hands.sh"), "--agent", "codex", "--dest", directory], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((target / "SKILL.md").read_text(), "user content")

    def test_hook_returns_json_without_launching_sessions(self):
        result = subprocess.run(["bash", str(SKILL / "hooks/session-start.sh")], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["hookSpecificOutput"]["hookEventName"], "SessionStart")

    def test_local_registration_refresh_does_not_walk_sources(self):
        script = REPO / "skills/routers/agent-skill-router/scripts/build_inventory.py"
        spec = importlib.util.spec_from_file_location("bah_inventory", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with mock.patch.object(module, "build_catalogs", side_effect=AssertionError("must not enumerate catalogs")):
            rows = module.refresh_local_skill(str(REPO), "brain-and-hands")
        row = next(row for row in rows if row[:3] == ("skill", "local", "brain-and-hands"))
        self.assertEqual(row[3], "installed")

    def test_source_refresh_preserves_other_refs_without_checkout_calls(self):
        script = REPO / "scripts/build_sources.py"
        spec = importlib.util.spec_from_file_location("bah_sources", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        before = (REPO / "SOURCES.md").read_text()
        with mock.patch.object(module, "checkout_roots", side_effect=AssertionError("no checkout calls")), \
             mock.patch.object(module, "sh", side_effect=AssertionError("no git enumeration")):
            after = module.refresh_local_source("brain-and-hands")
        for row in before.splitlines():
            if row.startswith("| `") and "brain-and-hands" not in row:
                self.assertIn(row, after)
        self.assertEqual(module.refresh_local_source("brain-and-hands"), after)


if __name__ == "__main__":
    unittest.main()
