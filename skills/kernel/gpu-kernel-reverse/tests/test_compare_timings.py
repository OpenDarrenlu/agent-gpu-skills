"""Tests for compare_timings.py (pure-stdlib, no GPU, no external tools)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_timings.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_timings_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CT = _load_module()


def _case(name, ref, cand):
    return {"name": name, "reference_us": ref, "candidate_us": cand}


def _doc(expected, cases):
    return {"expected_cases": expected, "cases": cases}


class VerdictTest(unittest.TestCase):
    def analyze(self, data, **kw):
        kw.setdefault("margin", 0.05)
        kw.setdefault("min_pairs", 20)
        kw.setdefault("bootstrap", 1000)
        kw.setdefault("seed", 0)
        return CT.analyze(data, **kw)

    def test_pass_when_candidate_not_slower(self):
        data = _doc(["a"], [_case("a", [10.0] * 20, [10.0] * 20)])
        report = self.analyze(data)
        self.assertEqual(report["cases"][0]["verdict"], "pass")
        self.assertEqual(report["verdict"], "pass")
        self.assertEqual(report["exit_code"], CT.EXIT_PASS)

    def test_faster_candidate_passes(self):
        data = _doc(["a"], [_case("a", [10.0] * 20, [8.0] * 20)])
        report = self.analyze(data)
        self.assertEqual(report["cases"][0]["verdict"], "pass")
        self.assertLess(report["cases"][0]["ratio"], 1.0)

    def test_regression_when_candidate_slower(self):
        data = _doc(["a"], [_case("a", [10.0] * 20, [13.0] * 20)])
        report = self.analyze(data)
        self.assertEqual(report["cases"][0]["verdict"], "regression")
        self.assertEqual(report["verdict"], "regression")
        self.assertEqual(report["exit_code"], CT.EXIT_REGRESSION)
        self.assertGreater(report["cases"][0]["ci_lower"], report["threshold"])

    def test_inconclusive_when_interval_straddles_threshold(self):
        data = _doc(["a"], [_case("a", [10.0] * 20, [10.0] * 10 + [11.0] * 10)])
        report = self.analyze(data)
        case = report["cases"][0]
        self.assertEqual(case["verdict"], "inconclusive")
        self.assertLessEqual(case["ci_lower"], report["threshold"])
        self.assertGreater(case["ci_upper"], report["threshold"])
        self.assertEqual(report["exit_code"], CT.EXIT_INCONCLUSIVE)

    def test_insufficient_pairs_cannot_pass(self):
        data = _doc(["a"], [_case("a", [10.0] * 5, [10.0] * 5)])
        report = self.analyze(data, min_pairs=20)
        self.assertEqual(report["cases"][0]["verdict"], "insufficient")
        self.assertEqual(report["exit_code"], CT.EXIT_INSUFFICIENT)

    def test_overall_prefers_regression_over_other_nonpass(self):
        data = _doc(
            ["ok", "bad"],
            [
                _case("ok", [10.0] * 20, [10.0] * 20),
                _case("bad", [10.0] * 20, [13.0] * 20),
            ],
        )
        report = self.analyze(data)
        self.assertEqual(report["verdict"], "regression")

    def test_overall_insufficient_when_no_regression(self):
        data = _doc(
            ["ok", "few"],
            [
                _case("ok", [10.0] * 20, [10.0] * 20),
                _case("few", [10.0] * 5, [10.0] * 5),
            ],
        )
        report = self.analyze(data, min_pairs=20)
        self.assertEqual(report["verdict"], "insufficient")
        self.assertEqual(report["exit_code"], CT.EXIT_INSUFFICIENT)

    def test_reports_medians_counts_and_ratio(self):
        data = _doc(["a"], [_case("a", [10.0, 12.0, 14.0], [11.0, 13.0, 15.0])])
        case = self.analyze(data, min_pairs=2)["cases"][0]
        self.assertEqual(case["pairs"], 3)
        self.assertEqual(case["reference_median_us"], 12.0)
        self.assertEqual(case["candidate_median_us"], 13.0)
        self.assertAlmostEqual(case["ratio"], 13.0 / 12.0)

    def test_paired_bootstrap_keeps_indices_paired(self):
        # candidate is exactly half of each paired reference, so every paired
        # resample must yield ratio 0.5 regardless of which indices are drawn.
        ref = [float(i) for i in range(1, 21)]
        cand = [0.5 * v for v in ref]
        case = self.analyze(_doc(["a"], [_case("a", ref, cand)]))["cases"][0]
        self.assertEqual(case["ratio"], 0.5)
        self.assertEqual(case["ci_lower"], 0.5)
        self.assertEqual(case["ci_upper"], 0.5)
        self.assertEqual(case["verdict"], "pass")

    def test_large_finite_times_do_not_overflow(self):
        data = _doc(["a"], [_case("a", [1e308] * 20, [1e308] * 20)])
        case = self.analyze(data)["cases"][0]
        self.assertEqual(case["ratio"], 1.0)
        self.assertEqual(case["ci_upper"], 1.0)
        self.assertEqual(case["verdict"], "pass")


class DeterminismTest(unittest.TestCase):
    def test_same_seed_reproduces_interval(self):
        data = _doc(["a"], [_case("a", [10.0] * 20, [10.0] * 10 + [11.0] * 10)])
        first = CT.analyze(data, margin=0.05, min_pairs=20, bootstrap=1000, seed=7)
        second = CT.analyze(data, margin=0.05, min_pairs=20, bootstrap=1000, seed=7)
        self.assertEqual(first["cases"][0]["ci_lower"], second["cases"][0]["ci_lower"])
        self.assertEqual(first["cases"][0]["ci_upper"], second["cases"][0]["ci_upper"])

    def test_case_order_does_not_change_per_case_interval(self):
        a = _case("a", [10.0] * 20, [10.0] * 10 + [11.0] * 10)
        b = _case("b", [10.0] * 20, [10.0] * 10 + [12.0] * 10)
        forward = CT.analyze(_doc(["a", "b"], [a, b]), margin=0.05, min_pairs=20,
                             bootstrap=1000, seed=3)
        reversed_cases = CT.analyze(_doc(["a", "b"], [b, a]), margin=0.05, min_pairs=20,
                                    bootstrap=1000, seed=3)
        fwd = {c["name"]: c["ci_upper"] for c in forward["cases"]}
        rev = {c["name"]: c["ci_upper"] for c in reversed_cases["cases"]}
        self.assertEqual(fwd, rev)


class ValidationTest(unittest.TestCase):
    def analyze(self, data, **kw):
        kw.setdefault("margin", 0.05)
        kw.setdefault("min_pairs", 20)
        kw.setdefault("bootstrap", 1000)
        kw.setdefault("seed", 0)
        return CT.analyze(data, **kw)

    def assertInvalid(self, data, **kw):
        with self.assertRaises(CT.InvalidInput):
            self.analyze(data, **kw)

    def test_missing_expected_case(self):
        self.assertInvalid(_doc(["a", "b"], [_case("a", [1.0], [1.0])]))

    def test_unexpected_case(self):
        self.assertInvalid(_doc(["a"], [_case("a", [1.0], [1.0]),
                                        _case("b", [1.0], [1.0])]))

    def test_duplicate_case(self):
        self.assertInvalid(_doc(["a"], [_case("a", [1.0], [1.0]),
                                        _case("a", [1.0], [1.0])]))

    def test_duplicate_expected_name(self):
        self.assertInvalid(_doc(["a", "a"], [_case("a", [1.0], [1.0])]))

    def test_empty_expected_cases(self):
        self.assertInvalid(_doc([], []))

    def test_empty_array(self):
        self.assertInvalid(_doc(["a"], [_case("a", [], [])]))

    def test_unequal_arrays(self):
        self.assertInvalid(_doc(["a"], [_case("a", [1.0, 2.0], [1.0])]))

    def test_bool_is_not_a_timing(self):
        self.assertInvalid(_doc(["a"], [_case("a", [True], [1.0])]), min_pairs=2)

    def test_nan_rejected(self):
        self.assertInvalid(_doc(["a"], [_case("a", [float("nan")], [1.0])]), min_pairs=2)

    def test_infinity_rejected(self):
        self.assertInvalid(_doc(["a"], [_case("a", [float("inf")], [1.0])]), min_pairs=2)

    def test_zero_rejected(self):
        self.assertInvalid(_doc(["a"], [_case("a", [0.0], [1.0])]), min_pairs=2)

    def test_negative_rejected(self):
        self.assertInvalid(_doc(["a"], [_case("a", [-1.0], [1.0])]), min_pairs=2)

    def test_string_number_rejected(self):
        self.assertInvalid(_doc(["a"], [_case("a", ["1.0"], [1.0])]), min_pairs=2)

    def test_non_object_top_level(self):
        self.assertInvalid([1, 2, 3])

    def test_bad_margin(self):
        self.assertInvalid(_doc(["a"], [_case("a", [1.0], [1.0])]), margin=-0.1, min_pairs=2)

    def test_nonfinite_margin(self):
        self.assertInvalid(_doc(["a"], [_case("a", [1.0], [1.0])]),
                           margin=float("inf"), min_pairs=2)

    def test_min_pairs_too_small(self):
        self.assertInvalid(_doc(["a"], [_case("a", [1.0], [1.0])]), min_pairs=1)

    def test_bootstrap_not_positive(self):
        self.assertInvalid(_doc(["a"], [_case("a", [1.0], [1.0])]), bootstrap=0, min_pairs=2)

    def test_bootstrap_below_minimum_rejected(self):
        self.assertInvalid(_doc(["a"], [_case("a", [1.0], [1.0])]), bootstrap=1, min_pairs=2)
        self.assertInvalid(_doc(["a"], [_case("a", [1.0], [1.0])]), bootstrap=999, min_pairs=2)

    def test_negative_seed(self):
        self.assertInvalid(_doc(["a"], [_case("a", [1.0], [1.0])]), seed=-1, min_pairs=2)

    def test_huge_unrepresentable_int_rejected(self):
        self.assertInvalid(_doc(["a"], [_case("a", [10 ** 400], [1.0])]), min_pairs=2)

    def test_ratio_overflow_rejected(self):
        # tiny reference vs huge candidate -> ratio overflows to infinity.
        self.assertInvalid(
            _doc(["a"], [_case("a", [5e-324] * 20, [1e308] * 20)]),
        )


class CliTest(unittest.TestCase):
    def run_cli(self, data, *flags):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timings.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), str(path), *flags],
                capture_output=True,
                text=True,
                check=False,
            )
        return proc

    def test_help_exits_zero_and_documents_exit_codes(self):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("Exit codes", proc.stdout)
        self.assertIn("regression", proc.stdout)

    def test_cli_pass_exit_zero(self):
        data = _doc(["a"], [_case("a", [10.0] * 20, [10.0] * 20)])
        proc = self.run_cli(data, "--bootstrap", "1000")
        self.assertEqual(proc.returncode, CT.EXIT_PASS)
        report = json.loads(proc.stdout)
        self.assertEqual(report["verdict"], "pass")

    def test_cli_regression_exit_code(self):
        data = _doc(["a"], [_case("a", [10.0] * 20, [13.0] * 20)])
        proc = self.run_cli(data, "--bootstrap", "1000")
        self.assertEqual(proc.returncode, CT.EXIT_REGRESSION)

    def test_cli_inconclusive_exit_code(self):
        data = _doc(["a"], [_case("a", [10.0] * 20, [10.0] * 10 + [11.0] * 10)])
        proc = self.run_cli(data, "--bootstrap", "2000")
        self.assertEqual(proc.returncode, CT.EXIT_INCONCLUSIVE)

    def test_cli_insufficient_exit_code(self):
        data = _doc(["a"], [_case("a", [10.0] * 5, [10.0] * 5)])
        proc = self.run_cli(data, "--bootstrap", "1000")
        self.assertEqual(proc.returncode, CT.EXIT_INSUFFICIENT)

    def test_cli_invalid_data_exit_one_with_json(self):
        data = _doc(["a", "b"], [_case("a", [1.0], [1.0])])
        proc = self.run_cli(data, "--min-pairs", "2", "--bootstrap", "1000")
        self.assertEqual(proc.returncode, CT.EXIT_INVALID)
        report = json.loads(proc.stdout)
        self.assertEqual(report["verdict"], "invalid")
        self.assertIn("error", report)

    def test_cli_missing_file_exit_one(self):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "/no/such/timings.json"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, CT.EXIT_INVALID)
        self.assertEqual(json.loads(proc.stdout)["verdict"], "invalid")

    def test_cli_reports_timing_only_limitation(self):
        data = _doc(["a"], [_case("a", [10.0] * 20, [10.0] * 20)])
        proc = self.run_cli(data, "--bootstrap", "1000")
        report = json.loads(proc.stdout)
        self.assertIn("Timing-only", report["limitations"])


if __name__ == "__main__":
    unittest.main()
