#!/usr/bin/env python3
"""Paired timing comparison helper for GPU kernel reverse engineering.

This script ONLY evaluates timing numbers that the caller has already
measured. It performs no GPU execution, does not launch any binary, and
cannot prove that the two sets of timings are correct, comparable, or
independently sampled. Establishing a trustworthy GPU environment, a frozen
set of expected cases, correctness of the candidate, and comparable
measurement conditions is the caller's responsibility (see the skill's
validation-and-performance reference). A ``pass`` here is a timing-only gate,
not a ``performance-matched`` conclusion.

Input JSON (positional argument)::

    {
      "expected_cases": ["shape-A", "shape-B"],
      "cases": [
        {"name": "shape-A",
         "reference_us": [12.1, 12.0, ...],
         "candidate_us": [12.0, 12.1, ...]}
      ]
    }

``expected_cases`` must be a non-empty list of unique, non-empty case names
frozen before measurement. ``cases`` must cover exactly those names, with no
missing, duplicate, or unexpected entries. Each ``reference_us`` /
``candidate_us`` pair must be non-empty, equal length, and contain only
positive, finite numbers (booleans, NaN, infinity, zero and negatives are
rejected).

Timing values must be finite, positive numbers that are representable as
Python floats; unrepresentably large integers, booleans, NaN, infinity, zero
and negatives are rejected. For each case the script estimates
``median(candidate) / median(reference)`` and a 95% percentile confidence
interval from a paired bootstrap (indices are resampled together, so each
reference/candidate pair stays paired). The bootstrap RNG is deterministic
given ``--seed`` and the case name, and uses at least 1000 draws so a 95%
interval is not fabricated from a handful of points.

Per-case verdicts::

    insufficient : fewer than --min-pairs pairs (cannot pass regardless of CI)
    pass         : CI upper bound <= 1 + margin (faster also passes)
    regression   : CI lower bound  > 1 + margin (measured slowdown)
    inconclusive : CI straddles the threshold (improve/extend measurement)

The report (JSON) is written to stdout. Exit code::

    0  every expected case passed
    1  invalid input data or options (a JSON error report is printed)
    2  command-line usage error (argparse)
    3  at least one regression
    4  at least one inconclusive case (no regression)
    5  at least one insufficient case (no regression / inconclusive)

Each result is a per-case interval. The bootstrap assumes the supplied pairs
are independent, which this script does not verify, and the per-case intervals
are NOT a simultaneous 95% guarantee over the whole suite (that needs
multiple-comparison correction).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from typing import Any, Dict, List, Tuple

CI_LEVEL = 0.95
_LOW_Q = (1.0 - CI_LEVEL) / 2.0
_HIGH_Q = 1.0 - _LOW_Q

METHOD = (
    "Per-case ratio = median(candidate_us) / median(reference_us). "
    "95% percentile confidence interval from a paired bootstrap that resamples "
    "paired indices with replacement and recomputes the ratio of medians. RNG "
    "is deterministic given --seed and the case name."
)
LIMITATIONS = (
    "Timing-only. This tool does not run any GPU or binary, does not verify "
    "correctness, comparability, or sample independence, and does not check "
    "measurement conditions. Each result is a per-case interval; the bootstrap "
    "assumes the supplied pairs are independent. These are not a simultaneous "
    "suite-wide 95% guarantee (that needs multiple-comparison correction). A "
    "'pass' is a timing gate, not a 'performance-matched' conclusion."
)

MIN_BOOTSTRAP = 1000

# Exit codes (also documented in the module docstring / --help).
EXIT_PASS = 0
EXIT_INVALID = 1
EXIT_REGRESSION = 3
EXIT_INCONCLUSIVE = 4
EXIT_INSUFFICIENT = 5

# Priority order used to pick the overall verdict and exit code.
_VERDICT_EXIT = {
    "pass": EXIT_PASS,
    "regression": EXIT_REGRESSION,
    "inconclusive": EXIT_INCONCLUSIVE,
    "insufficient": EXIT_INSUFFICIENT,
}
_VERDICT_PRIORITY = ["regression", "inconclusive", "insufficient", "pass"]


class InvalidInput(Exception):
    """Raised for malformed input data or options (maps to exit code 1)."""


def _as_timing(value: Any):
    """Return a finite positive float for a valid timing, else ``None``.

    Booleans are rejected, and integers too large to represent as a finite
    float (e.g. ``10**400``) are rejected rather than becoming infinity.
    """
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _safe_median(values: List[float]) -> float:
    """Median that avoids overflow for even counts near the float maximum.

    ``statistics.median`` averages the two middle values as ``(a + b) / 2``,
    which overflows to ``inf`` for large inputs. ``a + (b - a) / 2`` stays
    within ``[a, b]`` for sorted ``a <= b``.
    """
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    low, high = ordered[mid - 1], ordered[mid]
    return low + (high - low) / 2.0


def _finite_ratio(name: str, numerator: float, denominator: float) -> float:
    """Compute a ratio, rejecting overflow/underflow to a non-finite or zero."""
    ratio = numerator / denominator
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise InvalidInput(
            "case {!r} produced a non-finite or non-positive ratio "
            "({!r}); timings are too extreme to compare".format(name, ratio)
        )
    return ratio


def _percentile(sorted_vals: List[float], q: float) -> float:
    """Linear-interpolation percentile of an already-sorted list."""
    m = len(sorted_vals)
    if m == 0:
        raise ValueError("empty distribution")
    if m == 1:
        return sorted_vals[0]
    pos = q * (m - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[int(pos)]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _case_seed(seed: int, name: str) -> int:
    """Deterministic per-case seed, independent of case ordering."""
    digest = hashlib.sha256("{}\x00{}".format(seed, name).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def validate_options(margin: float, min_pairs: int, bootstrap: int, seed: int) -> None:
    if isinstance(margin, bool) or not isinstance(margin, (int, float)):
        raise InvalidInput("margin must be a number")
    if not math.isfinite(margin) or margin < 0:
        raise InvalidInput("margin must be finite and non-negative")
    if isinstance(min_pairs, bool) or not isinstance(min_pairs, int):
        raise InvalidInput("min-pairs must be an integer")
    if min_pairs < 2:
        raise InvalidInput("min-pairs must be >= 2")
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int):
        raise InvalidInput("bootstrap must be an integer")
    if bootstrap < MIN_BOOTSTRAP:
        raise InvalidInput(
            "bootstrap must be at least {} draws so a 95% interval is not "
            "fabricated from too few points".format(MIN_BOOTSTRAP)
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise InvalidInput("seed must be an integer")
    if seed < 0:
        raise InvalidInput("seed must be a non-negative integer")


def _validate_cases(data: Any) -> Tuple[List[str], Dict[str, Dict[str, List[float]]]]:
    if not isinstance(data, dict):
        raise InvalidInput("top-level JSON must be an object")

    expected = data.get("expected_cases")
    if not isinstance(expected, list) or not expected:
        raise InvalidInput("expected_cases must be a non-empty list")
    seen = set()
    for name in expected:
        if isinstance(name, bool) or not isinstance(name, str) or not name:
            raise InvalidInput("expected_cases entries must be non-empty strings")
        if name in seen:
            raise InvalidInput("expected_cases contains duplicate name: {!r}".format(name))
        seen.add(name)

    cases = data.get("cases")
    if not isinstance(cases, list):
        raise InvalidInput("cases must be a list")

    by_name: Dict[str, Dict[str, List[float]]] = {}
    for entry in cases:
        if not isinstance(entry, dict):
            raise InvalidInput("each case must be an object")
        name = entry.get("name")
        if isinstance(name, bool) or not isinstance(name, str) or not name:
            raise InvalidInput("each case needs a non-empty string name")
        if name not in seen:
            raise InvalidInput("unexpected case not in expected_cases: {!r}".format(name))
        if name in by_name:
            raise InvalidInput("duplicate case in cases: {!r}".format(name))

        ref = entry.get("reference_us")
        cand = entry.get("candidate_us")
        normalized = {}
        for label, arr in (("reference_us", ref), ("candidate_us", cand)):
            if not isinstance(arr, list) or not arr:
                raise InvalidInput(
                    "case {!r} {} must be a non-empty array".format(name, label)
                )
            values = []
            for value in arr:
                number = _as_timing(value)
                if number is None:
                    raise InvalidInput(
                        "case {!r} {} has a value that is not a finite positive "
                        "representable number: {!r}".format(name, label, value)
                    )
                values.append(number)
            normalized[label] = values
        if len(normalized["reference_us"]) != len(normalized["candidate_us"]):
            raise InvalidInput(
                "case {!r} reference_us ({}) and candidate_us ({}) must be "
                "equal length".format(
                    name,
                    len(normalized["reference_us"]),
                    len(normalized["candidate_us"]),
                )
            )
        by_name[name] = normalized

    missing = [n for n in expected if n not in by_name]
    if missing:
        raise InvalidInput("missing expected cases: {}".format(", ".join(missing)))

    return list(expected), by_name


def evaluate_case(
    name: str,
    ref: List[float],
    cand: List[float],
    *,
    threshold: float,
    min_pairs: int,
    bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    n = len(ref)
    med_ref = _safe_median(ref)
    med_cand = _safe_median(cand)
    point_ratio = _finite_ratio(name, med_cand, med_ref)

    rng = random.Random(_case_seed(seed, name))
    ratios: List[float] = []
    for _ in range(bootstrap):
        idxs = [rng.randrange(n) for _ in range(n)]
        r = _safe_median([ref[i] for i in idxs])
        c = _safe_median([cand[i] for i in idxs])
        ratios.append(_finite_ratio(name, c, r))
    ratios.sort()
    ci_lower = _percentile(ratios, _LOW_Q)
    ci_upper = _percentile(ratios, _HIGH_Q)

    if n < min_pairs:
        verdict = "insufficient"
    elif ci_upper <= threshold:
        verdict = "pass"
    elif ci_lower > threshold:
        verdict = "regression"
    else:
        verdict = "inconclusive"

    return {
        "name": name,
        "pairs": n,
        "reference_median_us": med_ref,
        "candidate_median_us": med_cand,
        "ratio": point_ratio,
        "ci_level": CI_LEVEL,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "verdict": verdict,
    }


def analyze(
    data: Any,
    *,
    margin: float,
    min_pairs: int,
    bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    """Validate ``data`` and return a report dict (raises InvalidInput)."""
    validate_options(margin, min_pairs, bootstrap, seed)
    expected, by_name = _validate_cases(data)
    threshold = 1.0 + margin

    case_reports = [
        evaluate_case(
            name,
            by_name[name]["reference_us"],
            by_name[name]["candidate_us"],
            threshold=threshold,
            min_pairs=min_pairs,
            bootstrap=bootstrap,
            seed=seed,
        )
        for name in expected
    ]

    verdicts = {c["verdict"] for c in case_reports}
    overall = "pass"
    for candidate in _VERDICT_PRIORITY:
        if candidate in verdicts:
            overall = candidate
            break

    report = {
        "tool": "compare_timings",
        "verdict": overall,
        "exit_code": _VERDICT_EXIT[overall],
        "threshold": threshold,
        "options": {
            "margin": margin,
            "min_pairs": min_pairs,
            "bootstrap": bootstrap,
            "seed": seed,
        },
        "method": METHOD,
        "limitations": LIMITATIONS,
        "cases": case_reports,
    }
    return report


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="compare_timings.py",
        description=(
            "Timing-only paired bootstrap gate for reverse-engineered kernels. "
            "Does not run any GPU/binary and does not prove correctness, "
            "comparability, or sample independence."
        ),
        epilog=(
            "Exit codes: 0 all cases pass; 1 invalid data/options; 2 CLI usage "
            "error; 3 regression; 4 inconclusive; 5 insufficient pairs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="path to the timing JSON report")
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="allowed slowdown fraction; threshold = 1 + margin (default 0.05)",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=20,
        dest="min_pairs",
        help="minimum paired batches required to pass (default 20, must be >= 2)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="number of bootstrap resamples per case (default 10000, min 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="deterministic RNG seed, non-negative integer (default 0)",
    )
    return parser.parse_args(argv)


def _load_json(path: str) -> Any:
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except FileNotFoundError:
        raise InvalidInput("input file not found: {}".format(path))
    except IsADirectoryError:
        raise InvalidInput("input path is a directory: {}".format(path))
    except OSError as exc:
        raise InvalidInput("cannot read input file: {}".format(exc))
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidInput("input is not valid UTF-8: {}".format(exc))
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidInput("invalid JSON: {}".format(exc))


def main(argv: List[str] = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        data = _load_json(args.input)
        report = analyze(
            data,
            margin=args.margin,
            min_pairs=args.min_pairs,
            bootstrap=args.bootstrap,
            seed=args.seed,
        )
    except InvalidInput as exc:
        error_report = {
            "tool": "compare_timings",
            "verdict": "invalid",
            "exit_code": EXIT_INVALID,
            "error": str(exc),
            "limitations": LIMITATIONS,
        }
        print(json.dumps(error_report, indent=2, allow_nan=False))
        return EXIT_INVALID

    print(json.dumps(report, indent=2, allow_nan=False))
    return int(report["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
