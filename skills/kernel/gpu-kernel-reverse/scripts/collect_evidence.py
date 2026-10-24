#!/usr/bin/env python3
"""Collect static reverse-engineering evidence from an NVIDIA GPU binary.

This tool runs the installed CUDA binary utilities (``cuobjdump`` /
``nvdisasm``) on a supplied cubin, fatbin, host executable, or shared library
and saves their raw output plus a ``manifest.json``. It is an evidence
collector, NOT a decompiler and NOT a verifier:

* It never executes the supplied binary and never modifies the input.
* All subprocesses are spawned from argv lists (no shell), so spaces and
  shell metacharacters in paths are passed literally.
* Status describes how much *evidence was collected*. It never claims
  semantic, GPU-correctness, or performance verification.

What it runs (each as a separate command, output saved verbatim):

* ``cuobjdump --list-elf`` and ``cuobjdump --list-ptx`` for container
  inventory (separate calls, because list/extract options override others).
* ``cuobjdump --dump-elf``, ``--dump-elf-symbols``, ``--dump-resource-usage``,
  ``--dump-ptx`` and ``--dump-sass``. ``--function`` (if given) filters the
  ``--dump-sass`` call only; function filtering is a cuobjdump feature.
* For a raw cubin only (an ELF whose ``e_machine`` == 190 / EM_CUDA, detected
  from the header, not the filename): ``nvdisasm --print-code
  --print-instruction-encoding``, ``--output-control-flow-graph-with-basic-blocks``,
  ``--print-life-ranges``, and ``--emit-json`` *only if* the local
  ``nvdisasm --help`` advertises it. nvdisasm disassembles the whole cubin and
  is not filtered by ``--function``; its success therefore cannot confirm a
  requested function. nvdisasm is never pointed at a host ELF or fatbin.
* No attempt is made to reconstruct PTX from SASS.

``--extract`` (explicit opt-in) runs ``cuobjdump --extract-elf SELECTOR
<absolute-input>`` with the working directory set to a fresh extraction
subdirectory, then inventories and hashes whatever files appeared. A selector
is a substring and may match multiple images. Extracted files are NOT
analyzed automatically and the first is NOT auto-selected; collect each
selected cubin separately with this tool. An extraction that fails or selects
nothing is recorded as a failure.

Status / exit codes (see ``manifest.json`` for the reason):

    complete (exit 0) : all requested evidence collected (collection only)
    partial  (exit 3) : some evidence collected, with caveats
    failed   (exit 1) : no disassembly evidence could be collected
    (exit 2)          : invalid invocation; no output directory is created

Notes on partial vs. complete: empty SASS or an absent requested function does
not count as successful disassembly; a missing optional JSON capability is not
a failure; a missing nvdisasm (for a cubin) is partial when cuobjdump still
yields SASS; a missing cuobjdump does not stop nvdisasm evidence; optional
CFG / life-range failures do not erase available SASS.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

EM_CUDA = 190
_CHUNK = 1 << 20

# A presence heuristic (NOT a SASS parser): cuobjdump --dump-sass and
# nvdisasm --print-code both prefix instructions with an offset comment like
# /*0000*/. Finding one means real disassembly was emitted.
import re

_SASS_MARKER = re.compile(r"/\*[0-9a-fA-F]{3,}\*/")

DISCLAIMER = (
    "Evidence collection only. Status reflects which tool outputs were "
    "gathered; it does NOT verify kernel semantics, GPU correctness, or "
    "performance. Disassembly presence is detected by an offset-comment "
    "heuristic, not a full SASS parser."
)

# Exit codes (also in the module docstring / --help).
EXIT_COMPLETE = 0
EXIT_FAILED = 1
EXIT_INVALID = 2
EXIT_PARTIAL = 3

_STATUS_EXIT = {"complete": EXIT_COMPLETE, "partial": EXIT_PARTIAL, "failed": EXIT_FAILED}


def _usage_error(message: str) -> int:
    sys.stderr.write("error: {}\n".format(message))
    return EXIT_INVALID


def sha256_file(path: str) -> Tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def sha256_bytes_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def classify_input(path: str) -> Dict[str, Any]:
    """Classify the input from its header bytes, not its filename.

    A raw CUDA cubin is an ELF whose e_machine is 190 (EM_CUDA); only such a
    file may be passed directly to nvdisasm.
    """
    with open(path, "rb") as handle:
        header = handle.read(64)
    info: Dict[str, Any] = {
        "is_cuda_cubin": False,
        "kind": "unknown",
        "leading_bytes_hex": header[:8].hex(),
    }
    if header[:4] == b"\x7fELF":
        ei_class = header[4] if len(header) > 4 else None
        ei_data = header[5] if len(header) > 5 else None
        endian = {1: "little", 2: "big"}.get(ei_data)
        e_type = None
        e_machine = None
        if endian and len(header) >= 20:
            e_type = int.from_bytes(header[16:18], endian)
            e_machine = int.from_bytes(header[18:20], endian)
        info.update(
            {
                "elf": {
                    "ei_class": ei_class,
                    "ei_data": ei_data,
                    "e_type": e_type,
                    "e_machine": e_machine,
                }
            }
        )
        if e_machine == EM_CUDA:
            info["is_cuda_cubin"] = True
            info["kind"] = "cuda-cubin"
        else:
            info["kind"] = "elf-other"  # host executable / shared library / etc.
    elif header[:4] == b"\x50\xed\x55\xba":
        info["kind"] = "fatbin"  # NVIDIA fatbinary container
    elif header[:8] == b"!<arch>\n":
        info["kind"] = "archive"
    return info


_SCAN_OVERLAP = 256  # bytes carried between chunks so a marker is not split


def has_disassembly(text: str) -> bool:
    return bool(_SASS_MARKER.search(text))


def _file_has_disassembly(path: str) -> bool:
    """Scan a saved stdout file chunk-by-chunk for an instruction marker.

    Uses latin-1 so every byte maps to a character (the marker is ASCII) and a
    small overlap so a marker spanning a chunk boundary is still found. This is
    a presence heuristic, not a SASS parser, and never loads the whole file.
    """
    overlap = ""
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(_CHUNK)
                if not chunk:
                    break
                window = overlap + chunk.decode("latin-1")
                if _SASS_MARKER.search(window):
                    return True
                overlap = window[-_SCAN_OVERLAP:]
    except OSError:
        return False
    return False


def _file_contains(path: str, token: bytes) -> bool:
    """Chunk-scan a saved file for a literal byte token (e.g. an option name)."""
    overlap = b""
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(_CHUNK)
                if not chunk:
                    break
                if token in overlap + chunk:
                    return True
                overlap = (overlap + chunk)[-len(token):]
    except OSError:
        return False
    return False


def _read_head(path: str, limit: int) -> str:
    try:
        with open(path, "rb") as handle:
            data = handle.read(limit)
    except OSError:
        return ""
    return data.decode("utf-8", "replace").strip()


class Command:
    """Record for one subprocess (or a skipped step)."""

    def __init__(self, name: str, argv: List[str], cwd: str):
        self.name = name
        self.argv = argv
        self.cwd = cwd
        self.status = "skipped"
        self.returncode: Optional[int] = None
        self.timeout_s: Optional[float] = None
        self.stdout_rel: Optional[str] = None
        self.stderr_rel: Optional[str] = None
        self.stdout_bytes = 0
        self.stderr_bytes = 0
        self.reason: Optional[str] = None
        self.sass_present = False
        self.head_text = ""

    def to_dict(self) -> Dict[str, Any]:
        record = {
            "name": self.name,
            "argv": self.argv,
            "cwd": self.cwd,
            "status": self.status,
            "returncode": self.returncode,
            "timeout_s": self.timeout_s,
            "stdout_path": self.stdout_rel,
            "stderr_path": self.stderr_rel,
            "stdout_bytes": self.stdout_bytes,
            "stderr_bytes": self.stderr_bytes,
        }
        if self.reason is not None:
            record["reason"] = self.reason
        return record

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _skipped(name: str, argv: List[str], reason: str) -> Command:
    cmd = Command(name, argv, os.getcwd())
    cmd.status = "skipped"
    cmd.reason = reason
    return cmd


def run_command(
    name: str,
    argv: List[str],
    out_dir: str,
    commands_dir: str,
    timeout: float,
    cwd: Optional[str] = None,
    scan_sass: bool = False,
    head_bytes: int = 0,
) -> Command:
    """Run a tool, streaming stdout/stderr straight to files (bounded RAM).

    The raw files are kept complete. Only a bounded head (``head_bytes``) is
    read back into memory, and SASS presence is detected by scanning the saved
    file in chunks, so a multi-image library dump never lives in memory.
    """
    cmd = Command(name, argv, cwd or os.getcwd())
    cmd.timeout_s = timeout
    stdout_path = os.path.join(commands_dir, name + ".stdout")
    stderr_path = os.path.join(commands_dir, name + ".stderr")

    try:
        with open(stdout_path, "wb") as out_handle, open(stderr_path, "wb") as err_handle:
            try:
                proc = subprocess.Popen(argv, stdout=out_handle, stderr=err_handle, cwd=cwd)
            except FileNotFoundError:
                cmd.status = "missing"
                cmd.reason = "executable not found: {}".format(argv[0])
            except OSError as exc:
                cmd.status = "error"
                cmd.reason = "os error: {}".format(exc)
            else:
                try:
                    proc.wait(timeout=timeout)
                    cmd.returncode = proc.returncode
                    cmd.status = "ok" if proc.returncode == 0 else "error"
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    cmd.status = "timeout"
                    cmd.reason = "exceeded {}s (partial output kept)".format(timeout)
    except OSError as exc:
        cmd.status = "error"
        cmd.reason = "cannot write output files: {}".format(exc)

    if cmd.status in ("ok", "error", "timeout") and os.path.exists(stdout_path):
        cmd.stdout_rel = os.path.relpath(stdout_path, out_dir)
        cmd.stderr_rel = os.path.relpath(stderr_path, out_dir)
        cmd.stdout_bytes = os.path.getsize(stdout_path)
        cmd.stderr_bytes = os.path.getsize(stderr_path)
        if scan_sass:
            cmd.sass_present = _file_has_disassembly(stdout_path)
        if head_bytes:
            cmd.head_text = _read_head(stdout_path, head_bytes)
    else:
        # Missing/unstartable: do not leave empty files masquerading as output.
        for path in (stdout_path, stderr_path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
    return cmd


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="collect_evidence.py",
        description=(
            "Collect static reverse-engineering evidence from an NVIDIA GPU "
            "binary using cuobjdump / nvdisasm. Never executes the input, "
            "never modifies it, and never claims semantic/GPU/performance "
            "verification."
        ),
        epilog=(
            "Status / exit codes: complete=0 (all requested evidence "
            "collected), partial=3 (evidence with caveats, e.g. empty SASS, "
            "requested function missing, missing tool, optional CFG/JSON "
            "failure, failed extraction), failed=1 (no disassembly evidence), "
            "2=invalid invocation (no output directory is created). Status is "
            "about evidence collection only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="path to the cubin/fatbin/executable/library")
    parser.add_argument(
        "--out",
        required=True,
        help="output directory for evidence; must NOT already exist",
    )
    parser.add_argument(
        "--function",
        default=None,
        help="mangled entry name to filter cuobjdump --dump-sass (cuobjdump only)",
    )
    parser.add_argument(
        "--extract",
        default=None,
        help="cuobjdump --extract-elf selector substring (explicit opt-in)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="per-subprocess timeout in seconds (default 60)",
    )
    return parser.parse_args(argv)


def _inventory_dir(directory: str) -> List[Dict[str, Any]]:
    files = []
    for root, _dirs, names in os.walk(directory):
        for fname in sorted(names):
            full = os.path.join(root, fname)
            if not os.path.isfile(full):
                continue
            try:
                sha = sha256_bytes_of_file(full)
                size = os.path.getsize(full)
            except OSError as exc:
                sha = None
                size = None
                files.append(
                    {
                        "name": os.path.relpath(full, directory),
                        "error": str(exc),
                    }
                )
                continue
            files.append(
                {
                    "name": os.path.relpath(full, directory),
                    "size_bytes": size,
                    "sha256": sha,
                }
            )
    return files


def main(argv: List[str] = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    # ---- preflight (no output directory is created if any check fails) ----
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        return _usage_error("--timeout must be a finite positive number of seconds")

    if args.function is not None and not args.function.strip():
        return _usage_error("--function must not be empty or whitespace")

    if args.extract is not None and not args.extract.strip():
        return _usage_error("--extract selector must not be empty or whitespace")

    input_path = args.input
    if not os.path.exists(input_path):
        return _usage_error("input does not exist: {}".format(input_path))
    if not os.path.isfile(input_path):
        return _usage_error("input is not a regular file: {}".format(input_path))

    out_dir = args.out
    if os.path.lexists(out_dir):
        return _usage_error(
            "refusing to use an existing output path (evidence must not be "
            "overwritten): {}".format(out_dir)
        )

    input_abs = os.path.abspath(input_path)
    try:
        input_sha, input_size = sha256_file(input_abs)
    except OSError as exc:
        return _usage_error("cannot read input: {}".format(exc))

    try:
        os.makedirs(out_dir, exist_ok=False)
    except OSError as exc:
        return _usage_error("cannot create output directory: {}".format(exc))
    commands_dir = os.path.join(out_dir, "commands")
    os.makedirs(commands_dir, exist_ok=False)

    classification = classify_input(input_abs)
    is_cubin = classification["is_cuda_cubin"]

    cuobjdump = shutil.which("cuobjdump")
    nvdisasm = shutil.which("nvdisasm")
    nvcc = shutil.which("nvcc")
    ptxas = shutil.which("ptxas")

    def run(
        name: str,
        argv_list: List[str],
        cwd: Optional[str] = None,
        scan_sass: bool = False,
        head_bytes: int = 0,
    ) -> Command:
        return run_command(
            name,
            argv_list,
            out_dir,
            commands_dir,
            args.timeout,
            cwd,
            scan_sass=scan_sass,
            head_bytes=head_bytes,
        )

    commands: List[Command] = []
    notes: List[str] = []

    # ---- toolkit versions ----
    toolkit: Dict[str, Any] = {}
    for tool_name, resolved in (
        ("cuobjdump", cuobjdump),
        ("nvdisasm", nvdisasm),
        ("nvcc", nvcc),
        ("ptxas", ptxas),
    ):
        if resolved is None:
            toolkit[tool_name] = {"path": None, "status": "missing"}
            continue
        vcmd = run(tool_name + "-version", [resolved, "--version"], head_bytes=600)
        commands.append(vcmd)
        toolkit[tool_name] = {
            "path": resolved,
            "status": vcmd.status,
            "version_text": vcmd.head_text,
        }

    # ---- cuobjdump inventory + evidence ----
    cuobjdump_cmds: Dict[str, Command] = {}

    def cuobjdump_step(name: str, extra: List[str], scan_sass: bool = False) -> Command:
        argv_list = [cuobjdump] + extra + [input_abs]
        cmd = run(name, argv_list, scan_sass=scan_sass)
        commands.append(cmd)
        cuobjdump_cmds[name] = cmd
        return cmd

    if cuobjdump is None:
        notes.append("cuobjdump not found: no ELF/symbol/resource/container evidence")
        for name in (
            "list-elf",
            "list-ptx",
            "dump-elf",
            "dump-elf-symbols",
            "dump-resource-usage",
            "dump-ptx",
            "dump-sass",
        ):
            skip = _skipped(name, ["cuobjdump"], "cuobjdump not found")
            commands.append(skip)
            cuobjdump_cmds[name] = skip
    else:
        cuobjdump_step("list-elf", ["--list-elf"])
        cuobjdump_step("list-ptx", ["--list-ptx"])
        cuobjdump_step("dump-elf", ["--dump-elf"])
        cuobjdump_step("dump-elf-symbols", ["--dump-elf-symbols"])
        cuobjdump_step("dump-resource-usage", ["--dump-resource-usage"])
        cuobjdump_step("dump-ptx", ["--dump-ptx"])
        sass_extra = ["--dump-sass"]
        if args.function:
            sass_extra += ["--function", args.function]
        cuobjdump_step("dump-sass", sass_extra, scan_sass=True)

    # ---- nvdisasm (raw cubin only) ----
    nvdisasm_cmds: Dict[str, Command] = {}
    emit_json_advertised = False

    def record_nvd_skip(names: List[str], reason: str) -> None:
        for name in names:
            skip = _skipped(name, ["nvdisasm"], reason)
            commands.append(skip)
            nvdisasm_cmds[name] = skip

    nvd_disasm_names = ["print-code", "cfg", "life-ranges", "emit-json"]
    if not is_cubin:
        reason = (
            "input is not a raw CUDA cubin (e_machine != {}); nvdisasm accepts "
            "a single cubin only. Extract a cubin and collect it "
            "separately.".format(EM_CUDA)
        )
        record_nvd_skip(nvd_disasm_names, reason)
        if classification["kind"] in ("fatbin", "elf-other", "archive"):
            notes.append(
                "nvdisasm direct disassembly skipped: input is a {}. Use "
                "--extract and run this collector on each extracted "
                "cubin.".format(classification["kind"])
            )
    elif nvdisasm is None:
        record_nvd_skip(nvd_disasm_names, "nvdisasm not found")
        notes.append("nvdisasm not found: no direct cubin disassembly/CFG/liveness")
    else:
        help_cmd = run("nvdisasm-help", [nvdisasm, "--help"])
        commands.append(help_cmd)
        emit_json_advertised = help_cmd.stdout_rel is not None and _file_contains(
            os.path.join(out_dir, help_cmd.stdout_rel), b"--emit-json"
        )

        pc = run(
            "print-code",
            [nvdisasm, "--print-code", "--print-instruction-encoding", input_abs],
            scan_sass=True,
        )
        commands.append(pc)
        nvdisasm_cmds["print-code"] = pc

        cfg = run(
            "cfg",
            [nvdisasm, "--output-control-flow-graph-with-basic-blocks", input_abs],
        )
        commands.append(cfg)
        nvdisasm_cmds["cfg"] = cfg

        life = run("life-ranges", [nvdisasm, "--print-life-ranges", input_abs])
        commands.append(life)
        nvdisasm_cmds["life-ranges"] = life

        if emit_json_advertised:
            ej = run("emit-json", [nvdisasm, "--emit-json", input_abs])
            commands.append(ej)
            nvdisasm_cmds["emit-json"] = ej
        else:
            skip = _skipped(
                "emit-json",
                [nvdisasm, "--emit-json"],
                "local nvdisasm --help does not advertise --emit-json",
            )
            commands.append(skip)
            nvdisasm_cmds["emit-json"] = skip
            notes.append("nvdisasm --emit-json not advertised locally (optional)")

    # ---- disassembly presence ----
    sass_cmd = cuobjdump_cmds.get("dump-sass")
    cuobjdump_sass_ok = bool(sass_cmd and sass_cmd.ok and sass_cmd.sass_present)
    pc_cmd = nvdisasm_cmds.get("print-code")
    nvdisasm_code_ok = bool(pc_cmd and pc_cmd.ok and pc_cmd.sass_present)

    if args.function is None:
        function_found: Optional[bool] = None
    else:
        function_found = cuobjdump_sass_ok  # filtered dump is the only authority
        if cuobjdump is None:
            notes.append(
                "requested --function cannot be confirmed: cuobjdump (which "
                "performs function filtering) is unavailable"
            )

    nvdisasm_scope = (
        "entire cubin (not filtered by --function)"
        if is_cubin
        else "n/a: input is not a raw cubin"
    )

    disassembly = {
        "cuobjdump_sass_present": cuobjdump_sass_ok,
        "nvdisasm_code_present": nvdisasm_code_ok,
        "function_requested": args.function,
        "function_found": function_found,
        "nvdisasm_scope": nvdisasm_scope,
    }

    # ---- extraction (explicit opt-in) ----
    extraction: Dict[str, Any] = {"requested": args.extract is not None}
    if args.extract is not None:
        extraction["selector"] = args.extract
        if cuobjdump is None:
            extraction["status"] = "failed"
            extraction["reason"] = "cuobjdump not found"
            skip = _skipped(
                "extract-elf",
                ["cuobjdump", "--extract-elf", args.extract, input_abs],
                "cuobjdump not found",
            )
            commands.append(skip)
        else:
            extract_dir = os.path.join(out_dir, "extracted")
            os.makedirs(extract_dir, exist_ok=False)
            extraction["dir"] = os.path.relpath(extract_dir, out_dir)
            ecmd = run(
                "extract-elf",
                [cuobjdump, "--extract-elf", args.extract, input_abs],
                cwd=extract_dir,
            )
            commands.append(ecmd)
            files = _inventory_dir(extract_dir)
            extraction["files"] = files
            had_errors = any("error" in entry for entry in files)
            if ecmd.status != "ok":
                extraction["status"] = "failed"
                extraction["reason"] = "extract-elf {}".format(ecmd.status)
            elif not files:
                extraction["status"] = "empty"
                extraction["reason"] = "selector matched no image"
            elif had_errors:
                extraction["status"] = "failed"
                extraction["reason"] = "could not inventory/hash some extracted files"
            else:
                extraction["status"] = "ok"
                notes.append(
                    "extraction produced {} file(s); collect each selected "
                    "cubin separately with this tool (nothing was analyzed "
                    "automatically)".format(len(files))
                )

    # ---- status ----
    any_sass = cuobjdump_sass_ok or nvdisasm_code_ok
    reasons: List[str] = []
    if not any_sass:
        status = "failed"
        reasons.append("no SASS disassembly was obtained")
    else:
        status = "complete"
        bad = ("error", "timeout", "missing")  # a started tool that did not succeed
        if args.function is not None and not cuobjdump_sass_ok:
            status = "partial"
            reasons.append(
                "requested function not found in cuobjdump --dump-sass "
                "(nvdisasm covers the whole cubin and cannot confirm it)"
            )
        elif args.function is None and sass_cmd is not None and sass_cmd.ok and not cuobjdump_sass_ok:
            # cuobjdump ran cleanly but emitted no disassembly for this input.
            status = "partial"
            reasons.append("cuobjdump --dump-sass produced no disassembly")
        if is_cubin and not nvdisasm_code_ok:
            status = "partial"
            reasons.append("nvdisasm direct cubin disassembly unavailable or failed")
        if cuobjdump is None:
            status = "partial"
            reasons.append("cuobjdump unavailable (no ELF/symbol/resource evidence)")
        else:
            # PTX commands count only when they error/time out; a clean empty
            # dump (no retained PTX) is normal and is not penalised here.
            for core in (
                "list-elf",
                "list-ptx",
                "dump-elf",
                "dump-elf-symbols",
                "dump-resource-usage",
                "dump-ptx",
                "dump-sass",
            ):
                cmd = cuobjdump_cmds.get(core)
                if cmd is not None and cmd.status in bad:
                    status = "partial"
                    reasons.append("cuobjdump {} {}".format(core, cmd.status))
        if is_cubin and nvdisasm is not None:
            for aux in ("cfg", "life-ranges"):
                cmd = nvdisasm_cmds.get(aux)
                if cmd is not None and cmd.status in bad:
                    status = "partial"
                    reasons.append(
                        "nvdisasm {} {} (does not erase available SASS)".format(
                            aux, cmd.status
                        )
                    )
            ej = nvdisasm_cmds.get("emit-json")
            if emit_json_advertised and ej is not None and ej.status in bad:
                status = "partial"
                reasons.append("nvdisasm --emit-json {}".format(ej.status))
        if extraction.get("requested") and extraction.get("status") != "ok":
            status = "partial"
            reasons.append("extraction {}".format(extraction.get("status")))

    status_reason = "; ".join(reasons) if reasons else (
        "all requested evidence collected (collection only; no semantic/GPU/"
        "performance verification)"
    )

    manifest = {
        "tool": "collect_evidence",
        "schema": 1,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": status,
        "status_reason": status_reason,
        "exit_code": _STATUS_EXIT[status],
        "disclaimer": DISCLAIMER,
        "input": {
            "path_abs": input_abs,
            "size_bytes": input_size,
            "sha256": input_sha,
        },
        "input_classification": classification,
        "invocation": {
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
            "timeout_s": args.timeout,
            "function_filter": args.function,
            "extract_selector": args.extract,
            "python": sys.version,
        },
        "toolkit_versions": toolkit,
        "disassembly": disassembly,
        "extraction": extraction,
        "commands": [cmd.to_dict() for cmd in commands],
        "notes": notes,
    }

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return _STATUS_EXIT[status]


if __name__ == "__main__":
    sys.exit(main())
