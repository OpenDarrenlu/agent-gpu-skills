"""Tests for collect_evidence.py.

Most behavioural tests drive the collector against small executable stubs so
they run with no CUDA toolkit and no GPU. One optional integration test
compiles real cubin/fatbin artifacts offline and is skipped only when the
required toolkit binaries (nvcc, cuobjdump) are genuinely missing.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "collect_evidence.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("collect_evidence_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CE = _load_module()

# --- stub tools (executed via their shebang in the test environment) --------

CUOBJDUMP_STUB = textwrap.dedent(
    """\
    import os, sys, time
    args = sys.argv[1:]
    def has(flag):
        return flag in args
    if has("--version"):
        sys.stdout.write("cuobjdump stub release 13.1\\n"); sys.exit(0)
    if has("--list-elf"):
        sys.stdout.write("ELF file 1: stub.1.sm_80.cubin\\n")
        sys.stdout.write("ELF file 2: stub.2.sm_90.cubin\\n")
        sys.exit(0)
    if has("--list-ptx"):
        sys.exit(0)
    if has("--extract-elf"):
        mode = os.environ.get("STUB_EXTRACT", "ok2")
        if mode == "error":
            sys.stderr.write("extract failed\\n"); sys.exit(1)
        if mode == "empty":
            sys.exit(0)
        for name in ("stub.1.sm_80.cubin", "stub.2.sm_90.cubin"):
            with open(os.path.join(os.getcwd(), name), "wb") as fh:
                fh.write(b"\\x7fELF" + b"\\x00" * 60)
        sys.exit(0)
    if has("--dump-sass"):
        mode = os.environ.get("STUB_CUOBJDUMP_SASS", "sass")
        if mode == "timeout":
            sys.stdout.write("\\t/*0000*/ FADD R0, R0, R0 ;\\n")
            sys.stdout.flush()
            time.sleep(float(os.environ.get("STUB_SLEEP", "10"))); sys.exit(0)
        if mode == "error":
            sys.stderr.write("sass failed\\n"); sys.exit(1)
        if "--function" in args:
            requested = args[args.index("--function") + 1]
            if requested != os.environ.get("STUB_FUNC", "present_kernel"):
                sys.exit(0)
        if mode == "empty":
            sys.exit(0)
        sys.stdout.write("\\t/*0000*/ FADD R0, R0, R0 ;\\n"); sys.exit(0)
    sys.stdout.write("stub-evidence\\n"); sys.exit(0)
    """
)

NVDISASM_STUB = textwrap.dedent(
    """\
    import os, sys, time
    args = sys.argv[1:]
    def has(flag):
        return flag in args
    if has("--version"):
        sys.stdout.write("nvdisasm stub release 13.1\\n"); sys.exit(0)
    if has("--help"):
        text = "Usage: nvdisasm\\n--print-code\\n--print-life-ranges\\n"
        if os.environ.get("STUB_NVD_JSON", "1") == "1":
            text += "--emit-json  Emit instructions as JSON\\n"
        sys.stdout.write(text); sys.exit(0)
    if has("--emit-json"):
        sys.stdout.write('{"SchemaVersion": 1}\\n'); sys.exit(0)
    if has("--output-control-flow-graph-with-basic-blocks"):
        if os.environ.get("STUB_NVD_CFG", "ok") == "error":
            sys.stderr.write("cfg failed\\n"); sys.exit(1)
        sys.stdout.write("digraph cfg {}\\n"); sys.exit(0)
    if has("--print-life-ranges"):
        sys.stdout.write("life ranges\\n"); sys.exit(0)
    if has("--print-code"):
        mode = os.environ.get("STUB_NVD_CODE", "sass")
        if mode == "timeout":
            time.sleep(float(os.environ.get("STUB_SLEEP", "10"))); sys.exit(0)
        if mode == "error":
            sys.stderr.write("print-code failed\\n"); sys.exit(1)
        if mode == "empty":
            sys.exit(0)
        sys.stdout.write("\\t/*0000*/ FADD R0, R0, R0 ;\\n"); sys.exit(0)
    sys.exit(0)
    """
)

CUDA_SOURCE = textwrap.dedent(
    """\
    extern "C" __global__ void add_kernel(const float* a, const float* b,
                                          float* c, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) c[i] = a[i] + b[i];
    }
    extern "C" __global__ void scale_kernel(float* data, float s, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) data[i] *= s;
    }
    """
)


def _write_stub(bindir: Path, name: str, source: str) -> None:
    # Absolute interpreter shebang so the stub runs even when the collector is
    # invoked with a restricted PATH that lacks /usr/bin/env.
    path = bindir / name
    path.write_text("#!{}\n{}".format(sys.executable, source), encoding="utf-8")
    path.chmod(0o755)


def _elf_header(e_machine: int) -> bytes:
    header = bytearray(64)
    header[0:4] = b"\x7fELF"
    header[4] = 2  # 64-bit
    header[5] = 1  # little-endian
    header[6] = 1  # version
    header[16:18] = (2).to_bytes(2, "little")  # ET_EXEC
    header[18:20] = e_machine.to_bytes(2, "little")
    return bytes(header)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class CollectorTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.bindir = self.tmp / "bin"
        self.bindir.mkdir()
        _write_stub(self.bindir, "cuobjdump", CUOBJDUMP_STUB)
        _write_stub(self.bindir, "nvdisasm", NVDISASM_STUB)

    def stub_env(self, **extra):
        env = dict(os.environ)
        env["PATH"] = str(self.bindir) + os.pathsep + env.get("PATH", "")
        env.update({key: str(value) for key, value in extra.items()})
        return env

    def empty_env(self):
        empty = self.tmp / "emptybin"
        empty.mkdir(exist_ok=True)
        env = dict(os.environ)
        env["PATH"] = str(empty)
        return env

    def subset_env(self, cuobjdump=True, nvdisasm=True, **extra):
        """PATH limited to a fresh dir holding only the requested stubs."""
        directory = Path(tempfile.mkdtemp(dir=str(self.tmp)))
        if cuobjdump:
            _write_stub(directory, "cuobjdump", CUOBJDUMP_STUB)
        if nvdisasm:
            _write_stub(directory, "nvdisasm", NVDISASM_STUB)
        env = dict(os.environ)
        env["PATH"] = str(directory)
        env.update({key: str(value) for key, value in extra.items()})
        return env

    def make_cubin(self, name="input.cubin") -> Path:
        path = self.tmp / name
        path.write_bytes(_elf_header(CE.EM_CUDA) + b"payload-bytes")
        return path

    def make_fatbin(self, name="input.fatbin") -> Path:
        path = self.tmp / name
        path.write_bytes(b"\x50\xed\x55\xba" + b"\x00" * 60 + b"payload")
        return path

    def run_collector(self, input_path, out_dir, env, *extra):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(input_path),
                "--out",
                str(out_dir),
                *extra,
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

    def manifest(self, out_dir: Path):
        return json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))

    def command_record(self, manifest, name):
        for record in manifest["commands"]:
            if record["name"] == name:
                return record
        return None


class UnitTest(CollectorTestBase):
    def test_classify_detects_cubin_by_e_machine(self):
        path = self.make_cubin()
        info = CE.classify_input(str(path))
        self.assertTrue(info["is_cuda_cubin"])
        self.assertEqual(info["elf"]["e_machine"], 190)

    def test_classify_host_elf_is_not_cubin(self):
        path = self.tmp / "host.elf"
        path.write_bytes(_elf_header(62))  # x86-64
        info = CE.classify_input(str(path))
        self.assertFalse(info["is_cuda_cubin"])
        self.assertEqual(info["kind"], "elf-other")

    def test_classify_fatbin_container(self):
        info = CE.classify_input(str(self.make_fatbin()))
        self.assertFalse(info["is_cuda_cubin"])
        self.assertEqual(info["kind"], "fatbin")

    def test_has_disassembly_marker(self):
        self.assertTrue(CE.has_disassembly("\t/*0010*/ IMAD R0 ;"))
        self.assertFalse(CE.has_disassembly("no instructions here"))


class PreflightTest(CollectorTestBase):
    def test_help_documents_status_and_exit_codes(self):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("complete", proc.stdout)
        self.assertIn("partial", proc.stdout)
        self.assertIn("failed", proc.stdout)

    def test_missing_input_is_invalid_and_creates_no_output(self):
        out = self.tmp / "out"
        proc = self.run_collector(self.tmp / "nope.cubin", out, self.stub_env())
        self.assertEqual(proc.returncode, CE.EXIT_INVALID)
        self.assertFalse(out.exists())

    def test_directory_input_rejected(self):
        d = self.tmp / "adir"
        d.mkdir()
        out = self.tmp / "out"
        proc = self.run_collector(d, out, self.stub_env())
        self.assertEqual(proc.returncode, CE.EXIT_INVALID)
        self.assertFalse(out.exists())

    def test_refuses_existing_output_directory(self):
        cubin = self.make_cubin()
        out = self.tmp / "out"
        out.mkdir()  # pre-existing, even though empty
        proc = self.run_collector(cubin, out, self.stub_env())
        self.assertEqual(proc.returncode, CE.EXIT_INVALID)
        self.assertFalse((out / "manifest.json").exists())
        self.assertEqual(list(out.iterdir()), [])

    def test_nonpositive_timeout_rejected(self):
        cubin = self.make_cubin()
        out = self.tmp / "out"
        proc = self.run_collector(cubin, out, self.stub_env(), "--timeout", "0")
        self.assertEqual(proc.returncode, CE.EXIT_INVALID)
        self.assertFalse(out.exists())

    def test_nonfinite_timeout_rejected(self):
        cubin = self.make_cubin()
        for value in ("inf", "nan"):
            out = self.tmp / ("out_" + value)
            proc = self.run_collector(cubin, out, self.stub_env(), "--timeout", value)
            self.assertEqual(proc.returncode, CE.EXIT_INVALID)
            self.assertFalse(out.exists())

    def test_empty_function_rejected_before_output(self):
        cubin = self.make_cubin()
        out = self.tmp / "out"
        proc = self.run_collector(cubin, out, self.stub_env(), "--function", "   ")
        self.assertEqual(proc.returncode, CE.EXIT_INVALID)
        self.assertFalse(out.exists())

    def test_empty_extract_selector_rejected_before_output(self):
        cubin = self.make_cubin()
        out = self.tmp / "out"
        proc = self.run_collector(cubin, out, self.stub_env(), "--extract", "")
        self.assertEqual(proc.returncode, CE.EXIT_INVALID)
        self.assertFalse(out.exists())


class StubBehaviourTest(CollectorTestBase):
    def test_missing_tools_fail_but_write_manifest(self):
        cubin = self.make_cubin()
        out = self.tmp / "out"
        proc = self.run_collector(cubin, out, self.empty_env())
        self.assertEqual(proc.returncode, CE.EXIT_FAILED)
        manifest = self.manifest(out)
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["toolkit_versions"]["cuobjdump"]["status"], "missing")
        self.assertEqual(manifest["toolkit_versions"]["nvdisasm"]["status"], "missing")
        self.assertEqual(self.command_record(manifest, "dump-sass")["status"], "skipped")
        self.assertEqual(self.command_record(manifest, "print-code")["status"], "skipped")

    def test_cubin_complete_path(self):
        cubin = self.make_cubin()
        out = self.tmp / "out"
        proc = self.run_collector(cubin, out, self.stub_env())
        self.assertEqual(proc.returncode, CE.EXIT_COMPLETE)
        manifest = self.manifest(out)
        self.assertEqual(manifest["status"], "complete")
        self.assertTrue(manifest["input_classification"]["is_cuda_cubin"])
        self.assertTrue(manifest["disassembly"]["cuobjdump_sass_present"])
        self.assertTrue(manifest["disassembly"]["nvdisasm_code_present"])

    def test_function_found(self):
        cubin = self.make_cubin()
        out = self.tmp / "out"
        proc = self.run_collector(
            cubin, out, self.stub_env(STUB_FUNC="present_kernel"),
            "--function", "present_kernel",
        )
        manifest = self.manifest(out)
        self.assertTrue(manifest["disassembly"]["function_found"])
        self.assertEqual(manifest["disassembly"]["function_requested"], "present_kernel")

    def test_function_miss_is_partial_and_nvdisasm_does_not_mask_it(self):
        cubin = self.make_cubin()
        out = self.tmp / "out"
        proc = self.run_collector(
            cubin, out, self.stub_env(STUB_FUNC="present_kernel"),
            "--function", "ghost_kernel",
        )
        self.assertEqual(proc.returncode, CE.EXIT_PARTIAL)
        manifest = self.manifest(out)
        self.assertEqual(manifest["status"], "partial")
        self.assertFalse(manifest["disassembly"]["function_found"])
        # nvdisasm still produced whole-cubin SASS, but it must not count as
        # the requested function being found.
        self.assertTrue(manifest["disassembly"]["nvdisasm_code_present"])
        self.assertIn("not filtered", manifest["disassembly"]["nvdisasm_scope"])

    def test_empty_disassembly_is_failure(self):
        cubin = self.make_cubin()
        out = self.tmp / "out"
        proc = self.run_collector(
            cubin, out,
            self.stub_env(STUB_CUOBJDUMP_SASS="empty", STUB_NVD_CODE="empty"),
        )
        self.assertEqual(proc.returncode, CE.EXIT_FAILED)
        self.assertEqual(self.manifest(out)["status"], "failed")

    def test_nonzero_sass_tool_recorded_and_partial(self):
        cubin = self.make_cubin()
        out = self.tmp / "out"
        proc = self.run_collector(
            cubin, out, self.stub_env(STUB_CUOBJDUMP_SASS="error"),
        )
        self.assertEqual(proc.returncode, CE.EXIT_PARTIAL)
        manifest = self.manifest(out)
        self.assertEqual(self.command_record(manifest, "dump-sass")["status"], "error")
        self.assertTrue(manifest["disassembly"]["nvdisasm_code_present"])

    def test_timeout_recorded(self):
        cubin = self.make_cubin()
        out = self.tmp / "out"
        proc = self.run_collector(
            cubin, out,
            self.stub_env(STUB_CUOBJDUMP_SASS="timeout", STUB_SLEEP="10"),
            "--timeout", "1",
        )
        manifest = self.manifest(out)
        record = self.command_record(manifest, "dump-sass")
        self.assertEqual(record["status"], "timeout")
        self.assertGreater(record["stdout_bytes"], 0)  # partial output retained
        self.assertEqual(manifest["status"], "partial")

    def test_cuobjdump_empty_sass_with_nvdisasm_is_partial(self):
        cubin = self.make_cubin()
        out = self.tmp / "out"
        proc = self.run_collector(
            cubin, out, self.stub_env(STUB_CUOBJDUMP_SASS="empty"),
        )
        self.assertEqual(proc.returncode, CE.EXIT_PARTIAL)
        manifest = self.manifest(out)
        self.assertEqual(manifest["status"], "partial")
        self.assertFalse(manifest["disassembly"]["cuobjdump_sass_present"])
        self.assertTrue(manifest["disassembly"]["nvdisasm_code_present"])

    def test_nvdisasm_missing_is_partial_when_cuobjdump_has_sass(self):
        cubin = self.make_cubin()
        out = self.tmp / "out"
        proc = self.run_collector(cubin, out, self.subset_env(nvdisasm=False))
        self.assertEqual(proc.returncode, CE.EXIT_PARTIAL)
        manifest = self.manifest(out)
        self.assertEqual(manifest["toolkit_versions"]["nvdisasm"]["status"], "missing")
        self.assertTrue(manifest["disassembly"]["cuobjdump_sass_present"])

    def test_cuobjdump_missing_still_gathers_nvdisasm(self):
        cubin = self.make_cubin()
        out = self.tmp / "out"
        proc = self.run_collector(cubin, out, self.subset_env(cuobjdump=False))
        self.assertEqual(proc.returncode, CE.EXIT_PARTIAL)
        manifest = self.manifest(out)
        self.assertEqual(manifest["toolkit_versions"]["cuobjdump"]["status"], "missing")
        self.assertTrue(manifest["disassembly"]["nvdisasm_code_present"])
        self.assertEqual(self.command_record(manifest, "print-code")["status"], "ok")

    def test_cuobjdump_missing_with_function_cannot_pass(self):
        cubin = self.make_cubin()
        out = self.tmp / "out"
        proc = self.run_collector(
            cubin, out, self.subset_env(cuobjdump=False), "--function", "add_kernel",
        )
        self.assertNotEqual(proc.returncode, CE.EXIT_COMPLETE)
        manifest = self.manifest(out)
        self.assertFalse(manifest["disassembly"]["function_found"])

    def test_toolkit_records_ptxas_key(self):
        cubin = self.make_cubin()
        out = self.tmp / "out"
        self.run_collector(cubin, out, self.stub_env())
        self.assertIn("ptxas", self.manifest(out)["toolkit_versions"])

    def test_noncubin_skips_nvdisasm(self):
        fatbin = self.make_fatbin()
        out = self.tmp / "out"
        proc = self.run_collector(fatbin, out, self.stub_env())
        manifest = self.manifest(out)
        self.assertFalse(manifest["input_classification"]["is_cuda_cubin"])
        pc = self.command_record(manifest, "print-code")
        self.assertEqual(pc["status"], "skipped")
        self.assertIn("cubin", pc["reason"])
        self.assertTrue(manifest["disassembly"]["cuobjdump_sass_present"])

    def test_extraction_success_inventories_hashes(self):
        fatbin = self.make_fatbin()
        out = self.tmp / "out"
        proc = self.run_collector(
            fatbin, out, self.stub_env(STUB_EXTRACT="ok2"), "--extract", "sm_",
        )
        self.assertEqual(proc.returncode, CE.EXIT_COMPLETE)
        manifest = self.manifest(out)
        extraction = manifest["extraction"]
        self.assertEqual(extraction["status"], "ok")
        self.assertGreaterEqual(len(extraction["files"]), 2)
        for entry in extraction["files"]:
            self.assertIn("sha256", entry)
            self.assertEqual(len(entry["sha256"]), 64)
        self.assertTrue(any("separately" in note for note in manifest["notes"]))

    def test_extraction_empty_is_not_pass(self):
        fatbin = self.make_fatbin()
        out = self.tmp / "out"
        proc = self.run_collector(
            fatbin, out, self.stub_env(STUB_EXTRACT="empty"), "--extract", "none",
        )
        self.assertNotEqual(proc.returncode, CE.EXIT_COMPLETE)
        self.assertEqual(self.manifest(out)["extraction"]["status"], "empty")

    def test_extraction_failure_is_not_pass(self):
        fatbin = self.make_fatbin()
        out = self.tmp / "out"
        proc = self.run_collector(
            fatbin, out, self.stub_env(STUB_EXTRACT="error"), "--extract", "sm_",
        )
        self.assertNotEqual(proc.returncode, CE.EXIT_COMPLETE)
        self.assertEqual(self.manifest(out)["extraction"]["status"], "failed")

    def test_missing_emit_json_capability_not_a_failure(self):
        cubin = self.make_cubin()
        out = self.tmp / "out"
        proc = self.run_collector(cubin, out, self.stub_env(STUB_NVD_JSON="0"))
        self.assertEqual(proc.returncode, CE.EXIT_COMPLETE)
        record = self.command_record(self.manifest(out), "emit-json")
        self.assertEqual(record["status"], "skipped")
        self.assertIn("advertise", record["reason"])

    def test_cfg_failure_does_not_erase_sass(self):
        cubin = self.make_cubin()
        out = self.tmp / "out"
        proc = self.run_collector(cubin, out, self.stub_env(STUB_NVD_CFG="error"))
        manifest = self.manifest(out)
        self.assertEqual(self.command_record(manifest, "cfg")["status"], "error")
        self.assertTrue(manifest["disassembly"]["nvdisasm_code_present"])
        self.assertEqual(manifest["status"], "partial")

    def test_path_with_spaces_and_metacharacters_no_shell(self):
        weird_dir = self.tmp / "weird dir"
        weird_dir.mkdir()
        name = "in ; $(touch pwned) & name.cubin"
        cubin = weird_dir / name
        cubin.write_bytes(_elf_header(CE.EM_CUDA) + b"payload")
        expected = _sha256(cubin)
        out = self.tmp / "out"
        proc = self.run_collector(cubin, out, self.stub_env())
        self.assertEqual(proc.returncode, CE.EXIT_COMPLETE)
        manifest = self.manifest(out)
        self.assertEqual(manifest["input"]["sha256"], expected)
        self.assertTrue(manifest["input"]["path_abs"].endswith(name))
        # No shell expansion of the metacharacters anywhere.
        self.assertFalse((weird_dir / "pwned").exists())
        self.assertFalse((Path.cwd() / "pwned").exists())

    def test_input_not_modified(self):
        cubin = self.make_cubin()
        before = _sha256(cubin)
        out = self.tmp / "out"
        self.run_collector(cubin, out, self.stub_env())
        self.assertEqual(_sha256(cubin), before)
        self.assertEqual(self.manifest(out)["input"]["sha256"], before)


@unittest.skipUnless(
    shutil.which("nvcc") and shutil.which("cuobjdump"),
    "requires an installed CUDA toolkit (nvcc, cuobjdump)",
)
class RealOfflineIntegrationTest(unittest.TestCase):
    """Compiles real artifacts offline; no GPU runtime is used or claimed."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.nvcc = shutil.which("nvcc")
        self.cuobjdump = shutil.which("cuobjdump")
        self.src = self.tmp / "kernels.cu"
        self.src.write_text(CUDA_SOURCE, encoding="utf-8")

    def _compile(self, args):
        proc = subprocess.run(
            [self.nvcc, *args], capture_output=True, text=True, check=False
        )
        # A compile failure is a real failure, never a skip.
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def run_collector(self, input_path, out_dir, *extra):
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(input_path), "--out", str(out_dir), *extra],
            capture_output=True,
            text=True,
            check=False,
        )

    def manifest(self, out_dir):
        return json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))

    def read_command_stdout(self, out_dir, manifest, name):
        for record in manifest["commands"]:
            if record["name"] == name and record["stdout_path"]:
                return (out_dir / record["stdout_path"]).read_text(
                    encoding="utf-8", errors="replace"
                )
        return ""

    def test_cubin_and_fatbin_evidence(self):
        cubin = self.tmp / "kernels.cubin"
        fatbin = self.tmp / "kernels.fatbin"
        self._compile(["-cubin", "-arch=sm_80", "-o", str(cubin), str(self.src)])
        self._compile(
            [
                "-fatbin",
                "-gencode", "arch=compute_80,code=sm_80",
                "-gencode", "arch=compute_90,code=sm_90",
                "-o", str(fatbin), str(self.src),
            ]
        )

        # --- bare cubin ---
        cubin_sha = _sha256(cubin)
        out_cubin = self.tmp / "out_cubin"
        proc = self.run_collector(cubin, out_cubin)
        self.assertIn(proc.returncode, (CE.EXIT_COMPLETE, CE.EXIT_PARTIAL))
        manifest = self.manifest(out_cubin)
        self.assertIn(manifest["status"], ("complete", "partial"))
        self.assertEqual(manifest["input"]["sha256"], cubin_sha)
        self.assertEqual(manifest["input"]["size_bytes"], cubin.stat().st_size)
        self.assertTrue(manifest["input_classification"]["is_cuda_cubin"])
        self.assertEqual(manifest["input_classification"]["elf"]["e_machine"], 190)
        self.assertTrue(manifest["disassembly"]["cuobjdump_sass_present"])

        symbols = self.read_command_stdout(out_cubin, manifest, "dump-elf-symbols")
        self.assertIn("add_kernel", symbols)
        self.assertIn("scale_kernel", symbols)
        res = self.command_record_status(manifest, "dump-resource-usage")
        self.assertEqual(res, "ok")

        if shutil.which("nvdisasm"):
            self.assertTrue(manifest["disassembly"]["nvdisasm_code_present"])

        # input is untouched
        self.assertEqual(_sha256(cubin), cubin_sha)

        # --- cubin with function filter ---
        out_fn = self.tmp / "out_fn"
        self.run_collector(cubin, out_fn, "--function", "add_kernel")
        man_fn = self.manifest(out_fn)
        self.assertTrue(man_fn["disassembly"]["function_found"])
        sass = self.read_command_stdout(out_fn, man_fn, "dump-sass")
        self.assertIn("add_kernel", sass)

        # --- fatbin with extraction ---
        fatbin_sha = _sha256(fatbin)
        out_fat = self.tmp / "out_fat"
        self.run_collector(fatbin, out_fat, "--extract", "sm_80")
        man_fat = self.manifest(out_fat)
        self.assertFalse(man_fat["input_classification"]["is_cuda_cubin"])
        self.assertEqual(man_fat["input_classification"]["kind"], "fatbin")

        list_elf = self.read_command_stdout(out_fat, man_fat, "list-elf")
        self.assertIn("sm_80", list_elf)
        self.assertIn("sm_90", list_elf)  # multi-image fatbin

        pc = next(c for c in man_fat["commands"] if c["name"] == "print-code")
        self.assertEqual(pc["status"], "skipped")
        self.assertIn("cubin", pc["reason"])

        self.assertEqual(man_fat["extraction"]["status"], "ok")
        self.assertTrue(man_fat["extraction"]["files"])
        for entry in man_fat["extraction"]["files"]:
            self.assertEqual(len(entry["sha256"]), 64)
            self.assertGreater(entry["size_bytes"], 0)

        self.assertEqual(_sha256(fatbin), fatbin_sha)

    def command_record_status(self, manifest, name):
        for record in manifest["commands"]:
            if record["name"] == name:
                return record["status"]
        return None


if __name__ == "__main__":
    unittest.main()
