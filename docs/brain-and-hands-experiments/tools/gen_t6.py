#!/usr/bin/env python3
"""T6: 1000 unique functions (no repeated template bodies), 30 with code that contradicts the docstring. usage: gen_t6.py OUT_DIR"""
import json, os, random, sys
OUT = sys.argv[1]; R = random.Random(6006)
NAMES = "a b c d x y z m n p q u v w k lo hi t s r amount rate items vals total base step cap".split()
VERBS = "calc get make adjust fold scan pick score fit trim mix cut tally sift".split()
NOUNS = "price cart batch label queue stock route quota grade panel token lane slot ratio shard".split()
def ps(n): return R.sample(NAMES, n)
def S1():
    a, b, c = ps(3); o1 = R.choice([('+', 'sum of {a} and {b}'), ('-', 'difference of {a} and {b}'), ('*', 'product of {a} and {b}')])
    o2 = R.choice([('*', 'multiplied by {k}'), ('+', 'increased by {k}'), ('-', 'decreased by {k}')]); k = R.randint(2, 9)
    o3 = R.choice([('-', 'minus {c}'), ('+', 'plus {c}')])
    mk = lambda k, o1, o2, o3: f"({a} {o1} {b}) {o2} {k} {o3} {c}"
    doc = f"Return the {o1[1]}, {o2[1]}, {o3[1]}.".format(a=a, b=b, k=k, c=c)
    flip = lambda o: {'+': '-', '-': '+', '*': '+'}[o]
    bug = R.choice(['k', 'o1', 'o3'])
    badv = mk(k + 1 if bug == 'k' else k, flip(o1[0]) if bug == 'o1' else o1[0], o2[0], flip(o3[0]) if bug == 'o3' else o3[0])
    return f"{a}, {b}, {c}", doc, mk(k, o1[0], o2[0], o3[0]), badv
def S2():
    xs, t = ps(2); w, op = R.choice([('greater than', '>'), ('at least', '>='), ('less than', '<'), ('at most', '<=')])
    bad = {'>': '>=', '>=': '>', '<': '<=', '<=': '<'}[op]
    f = lambda o: f"[v for v in {xs} if v {o} {t}]"
    return f"{xs}, {t}", R.choice([f"Return the items of {xs} that are {w} {t}, in their original order.", f"Keep only the entries of {xs} {w} {t}; order is preserved."]), f(op), f(bad)
def S3():
    xs, n = ps(2); w, e = R.choice([('even', 'v % 2 == 0'), ('odd', 'v % 2 == 1'), ('positive', 'v > 0'), ('negative', 'v < 0')])
    bade = {'v % 2 == 0': 'v % 2 == 1', 'v % 2 == 1': 'v % 2 == 0', 'v > 0': 'v >= 0', 'v < 0': 'v <= 0'}[e]
    f = lambda sl, ee: f"sum(1 for v in {xs}[:{sl}] if {ee})"
    bug = R.choice(['slice', 'cond'])
    return f"{xs}, {n}", f"Count how many of the first {n} items of {xs} are {w}.", f(n, e), f(n + '+1' if bug == 'slice' else n, bade if bug == 'cond' else e)
def S4():
    p, = ps(1); sep = R.choice([',', ';', '|', '/']); bs = R.choice([' ' + sep, sep + ' ', sep + sep])
    return f"{p}", f"Join the strings of {p} with '{sep}' between them, skipping empty strings.", f"'{sep}'.join(s for s in {p} if s)", R.choice([f"'{bs}'.join(s for s in {p} if s)", f"'{sep}'.join({p})"])
def S5():
    d, k, dv = ps(3)
    return f"{d}, {k}, {dv}", f"Return {d}[{k}] when {k} is present in {d}, otherwise {dv}.", f"{d}[{k}] if {k} in {d} else {dv}", f"{d}.get({k}) or {dv}"
def S6():
    lo, hi = ps(2)
    return f"{lo}, {hi}", f"Sum the integers from {lo} to {hi}, both ends included.", f"sum(range({lo}, {hi} + 1))", f"sum(range({lo}, {hi}))"
def S7():
    a, b, cap = ps(3); w, fn = R.choice([('smaller', 'min'), ('larger', 'max')]); bad = 'max' if fn == 'min' else 'min'
    return f"{a}, {b}, {cap}", f"Return the {w} of {a} and {b}, but never more than {cap}.", f"min({fn}({a}, {b}), {cap})", f"min({bad}({a}, {b}), {cap})" if fn == 'min' else R.choice([f"min({bad}({a}, {b}), {cap})", f"{fn}({fn}({a}, {b}), {cap})"])
def S8():
    amt, pct = ps(2)
    return f"{amt}, {pct}", f"Return {pct} percent of {amt}, rounded down to an int.", f"int({amt} * {pct} / 100)", R.choice([f"round({amt} * {pct} / 100)", f"int({amt} * {pct} / 10)"])
def S9():
    n, = ps(1); k, m = R.sample([3, 4, 5, 6, 7, 9], 2)
    return f"{n}", f"Return True only if {n} is divisible by both {k} and {m}.", f"{n} % {k} == 0 and {n} % {m} == 0", R.choice([f"{n} % {k} == 0 or {n} % {m} == 0", f"{n} % {k} == 0 and {n} % {m} == 1"])
def S10():
    s, = ps(1); w, fn = R.choice([('upper', 'upper'), ('lower', 'lower')]); bad = 'lower' if fn == 'upper' else 'upper'
    return f"{s}", f"Return {s} in {w} case with spaces removed from both ends.", f"{s}.strip().{fn}()", R.choice([f"{s}.strip().{bad}()", f"{s}.lstrip().{fn}()"])
def S11():
    xs, dv = ps(2)
    return f"{xs}, {dv}", f"Return the last item of {xs}, or {dv} when {xs} is empty.", f"{xs}[-1] if {xs} else {dv}", R.choice([f"{xs}[0] if {xs} else {dv}", f"{xs}[-1] if len({xs}) > 1 else {dv}"])
def S12():
    xs, = ps(1)
    return f"{xs}", f"Return the running totals of {xs}: entry i is the sum of the first i+1 items.", f"list(__import__('itertools').accumulate({xs}))", R.choice([f"[0] + list(__import__('itertools').accumulate({xs}))", f"list(__import__('itertools').accumulate({xs}))[:-1]"])
SH = [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12]
N_MOD, PER = 40, 25
bugs = set(R.sample(range(N_MOD * PER), 30)); truth = []; idx = 0
os.makedirs(f"{OUT}/t6_audit/lib", exist_ok=True)
for m in range(N_MOD):
    out = [f'"""Module {m:02d}."""\n']
    for _ in range(PER):
        args, doc, good, bad = R.choice(SH)()
        while bad == good: args, doc, good, bad = R.choice(SH)()
        name = f"{R.choice(VERBS)}_{R.choice(NOUNS)}_{idx}"
        out.append(f"\ndef {name}({args}):\n    \"\"\"{doc}\"\"\"\n    return {bad if idx in bugs else good}\n")
        if idx in bugs: truth.append(f"mod_{m:02d}.{name}")
        idx += 1
    open(f"{OUT}/t6_audit/lib/mod_{m:02d}.py", 'w').write(''.join(out))
open(f"{OUT}/t6_audit/lib/__init__.py", 'w').write('')
json.dump({"t1": sorted(truth)}, open(f"{OUT}/truth.json", 'w'), indent=1)
print(len(truth), 'planted of', idx)
