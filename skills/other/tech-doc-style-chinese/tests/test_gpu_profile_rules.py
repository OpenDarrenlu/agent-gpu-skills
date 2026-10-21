"""Tests for the GPU / AI Infra rules enabled by ``--profile gpu``."""

from __future__ import annotations

import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "lint_copy_rules.py"
LINTER = runpy.run_path(str(SCRIPT), run_name="lint_copy_rules_gpu_test")
scan_markdown = LINTER["scan_markdown"]


class GpuProfileTest(unittest.TestCase):
    def scan(self, text: str, profile: str = "gpu"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.md"
            path.write_text(text, encoding="utf-8")
            return scan_markdown(path, profile)

    def kinds(self, text: str, profile: str = "gpu"):
        return [item.kind for item in self.scan(text, profile)]

    def test_base_profile_ignores_gpu_rules(self):
        text = "使用 cuda 和 kv cache，耗时 200us，零开销。"
        self.assertEqual(self.scan(text, "base"), [])
        self.assertTrue(self.scan(text, "gpu"))

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            self.scan("正文。", "unknown")

    def test_casing_rules(self):
        findings = self.scan("基于 cutlass 与 sglang，在 hopper 上测试 flash attention 和 kv cache。")
        messages = " ".join(item.message for item in findings if item.kind == "gpu-casing")
        for expected in ("CUTLASS", "SGLang", "Hopper", "FlashAttention", "KV cache"):
            with self.subTest(expected=expected):
                self.assertIn(expected, messages)

    def test_official_casing_passes(self):
        text = "基于 CUTLASS、CuTe DSL、SGLang、vLLM、FlashAttention 和 KV cache，在 B200 上使用 BF16 与 FP8。"
        self.assertEqual(self.kinds(text), [])

    def test_identifiers_and_code_are_not_casing_findings(self):
        text = "调用 `torch.cuda.synchronize()`，参见 cute::make_layout、triton.jit 和 sm_90a。"
        self.assertNotIn("gpu-casing", self.kinds(text))

    def test_unit_spacing(self):
        findings = [item for item in self.scan("耗时 200us，容量 80GB，带宽 3.2TB/s。") if item.kind == "unit-spacing"]
        self.assertEqual(len(findings), 3)
        self.assertIn("200 us", findings[0].message)
        self.assertEqual(self.kinds("耗时 200 µs，容量 80 GB，128B swizzle。"), [])

    def test_speedup_and_shape(self):
        kinds = self.kinds("加速 1.3x，tile 为 128x128x64，地址 0x10。")
        self.assertIn("speedup", kinds)
        self.assertIn("shape", kinds)
        self.assertEqual(kinds.count("shape"), 1)
        self.assertEqual(self.kinds("加速 1.3×，tile 为 128×128×64。"), [])

    def test_overclaim_and_vague_are_warnings(self):
        findings = self.scan("通信计算零开销重叠，性能显著提升。")
        severities = {item.kind: item.severity for item in findings}
        self.assertEqual(severities.get("overclaim"), "warning")
        self.assertEqual(severities.get("vague"), "warning")

    def test_domain_typo_is_error_and_context_is_warning(self):
        findings = self.scan("带宽是瓶劲。规约结果通过通讯库传输。")
        by_kind = [(item.kind, item.severity) for item in findings]
        self.assertIn(("typo", "error"), by_kind)
        self.assertEqual([k for k, s in by_kind if s == "warning"], ["context", "context"])

    def test_cjk_spacing(self):
        findings = [item for item in self.scan("这个kernel耗时3 ms。") if item.kind == "spacing"]
        self.assertEqual(len(findings), 1)
        self.assertIn("3 处", findings[0].message)
        self.assertEqual(self.kinds("这个 kernel 耗时 3 ms。运行 `nvcc` 编译。"), [])

    def test_consistency_reports_minority_variant_once(self):
        text = "第一处空泡。\n\n第二处空泡。\n\n第三处气泡，第四处气泡。"
        findings = [item for item in self.scan(text) if item.kind == "consistency"]
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].line, 5)
        self.assertIn("气泡", findings[0].message)

    def test_consistency_ignores_code(self):
        text = "流水线空泡。\n\n```text\n气泡\n```\n"
        self.assertNotIn("consistency", self.kinds(text))

    def test_cli_profile_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.md"
            path.write_text("瓶劲在 HBM。\n", encoding="utf-8")
            base = subprocess.run(
                [sys.executable, str(SCRIPT), str(path)],
                check=False, capture_output=True, text=True,
            )
            gpu = subprocess.run(
                [sys.executable, str(SCRIPT), "--profile", "gpu", str(path)],
                check=False, capture_output=True, text=True,
            )
        self.assertEqual(base.returncode, 0)
        self.assertEqual(gpu.returncode, 2)
        self.assertIn("瓶颈", gpu.stdout)


if __name__ == "__main__":
    unittest.main()
