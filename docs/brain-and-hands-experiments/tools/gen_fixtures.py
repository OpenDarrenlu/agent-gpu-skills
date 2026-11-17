#!/usr/bin/env python3
"""Deterministic task fixtures for the brain-and-hands experiments.

usage: gen_fixtures.py OUT_DIR   -> OUT_DIR/{t1_audit,t2_logs,t3_rename,t4_fix}/ + OUT_DIR/truth.json
"""
import json, os, random, sys, textwrap

OUT = sys.argv[1]
rnd = random.Random(20261117)

def w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)

# ---------------------------------------------------------------- T1: docstring-vs-behaviour audit
# name, args, docstring, correct body, buggy body
T = [
 ("clamp", "x, lo, hi", "Clamp x into the closed interval [lo, hi]. Raises ValueError if lo > hi.",
  "if lo > hi:\n    raise ValueError('lo > hi')\nreturn max(lo, min(x, hi))", "return max(lo, min(x, hi))"),
 ("chunk", "seq, n", "Split seq into lists of length n; the last chunk may be shorter.",
  "return [list(seq[i:i + n]) for i in range(0, len(seq), n)]",
  "return [list(seq[i:i + n]) for i in range(0, len(seq) - n + 1, n)]"),
 ("percent", "part, whole", "Return part/whole as a percentage rounded to 2 decimals; 0.0 when whole is 0.",
  "if not whole:\n    return 0.0\nreturn round(part / whole * 100, 2)", "return round(part / whole * 100, 2)"),
 ("days_between", "a, b", "Whole days from date a to date b (negative when b is earlier).",
  "return (b - a).days", "return abs((b - a).days)"),
 ("slug", "text", "Lowercase text and join its words with hyphens.",
  "return '-'.join(text.lower().split())", "return '_'.join(text.lower().split())"),
 ("median", "xs", "Median of a non-empty list; mean of the two middle values for even length.",
  "s = sorted(xs)\nm = len(s) // 2\nreturn s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2",
  "s = sorted(xs)\nreturn s[len(s) // 2]"),
 ("take", "items, n", "Return the first n items, or the whole list if n exceeds its length. n must be >= 0.",
  "if n < 0:\n    raise ValueError('n < 0')\nreturn list(items[:n])",
  "if n < 0:\n    raise ValueError('n < 0')\nif n > len(items):\n    raise IndexError(n)\nreturn list(items[:n])"),
 ("discount", "price, pct", "Price after a pct percent discount (pct in 0..100), rounded to cents.",
  "return round(price * (1 - pct / 100), 2)", "return round(price * (1 - pct), 2)"),
 ("last_word", "text", "Return the last whitespace-separated word, or '' for blank text.",
  "parts = text.split()\nreturn parts[-1] if parts else ''", "parts = text.split()\nreturn parts[-1] if parts else None"),
 ("weighted_mean", "values, weights", "Weighted mean of values; the weights need not sum to 1.",
  "return sum(v * w for v, w in zip(values, weights)) / sum(weights)",
  "return sum(v * w for v, w in zip(values, weights)) / len(values)"),
 ("is_leap", "y", "True for leap years: divisible by 4, except centuries not divisible by 400.",
  "return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)", "return y % 4 == 0 and y % 100 != 0"),
 ("pad_left", "s, width", "Left-pad s with zeros to width; never truncates a longer string.",
  "return s.rjust(width, '0')", "return s.zfill(width)[-width:]"),
 ("dedupe", "xs", "Remove duplicates, keeping the order of first occurrences.",
  "seen = set()\nreturn [x for x in xs if not (x in seen or seen.add(x))]", "return list(set(xs))"),
 ("truncate", "text, n", "Cut text to at most n characters, ending with '...' when cut; the result including dots is at most n.",
  "return text if len(text) <= n else text[:n - 3] + '...'", "return text if len(text) <= n else text[:n] + '...'"),
 ("rolling_sum", "xs, k", "Sum of each window of k consecutive items; an empty list if k > len(xs).",
  "return [sum(xs[i:i + k]) for i in range(len(xs) - k + 1)]",
  "return [sum(xs[i:i + k]) for i in range(len(xs) - k)]"),
 ("bucket", "v, edges", "Index of the bucket for v given ascending edges; a value equal to an edge goes to the upper bucket.",
  "import bisect\nreturn bisect.bisect_right(edges, v)", "import bisect\nreturn bisect.bisect_left(edges, v)"),
 ("merge_dicts", "*ds", "Merge dicts left to right; later values win.",
  "out = {}\nfor d in ds:\n    out.update(d)\nreturn out", "out = {}\nfor d in reversed(ds):\n    out.update(d)\nreturn out"),
 ("to_celsius", "f", "Convert Fahrenheit to Celsius, rounded to 1 decimal.",
  "return round((f - 32) * 5 / 9, 1)", "return round((f - 32) * 9 / 5, 1)"),
 ("initials", "name", "Uppercase initials of each word, e.g. 'ada lovelace' -> 'AL'.",
  "return ''.join(p[0].upper() for p in name.split())", "return ''.join(p[0] for p in name.split())"),
 ("split_pair", "s", "Split 'k=v' at the first '=' only; v may itself contain '='.",
  "k, _, v = s.partition('=')\nreturn k, v", "k, v = s.split('=')[:2]\nreturn k, v"),
 ("remaining", "stock, sold", "Units left after selling; never negative.",
  "return max(0, stock - sold)", "return stock - sold"),
 ("round_half_up", "x", "Round half away from zero to an int: 2.5 -> 3, -2.5 -> -3.",
  "import math\nreturn int(math.copysign(math.floor(abs(x) + 0.5), x))", "return round(x)"),
 ("ceil_div", "a, b", "Ceiling of a / b for positive ints.",
  "return -(-a // b)", "return a // b + 1"),
 ("squash_ws", "s", "Collapse every run of whitespace to one space and strip both ends.",
  "return ' '.join(s.split())", "return s.strip().replace('  ', ' ')"),
]
NOUNS = ["order", "cart", "invoice", "sku", "ship", "tax", "user", "ledger", "refund", "batch", "ticket", "queue",
         "region", "promo", "audit", "store", "price", "vendor", "route", "coupon", "report", "event", "pack", "stock",
         "bundle", "carrier", "catalog", "channel", "credit", "dock", "export", "fee", "gift", "hub", "import", "item",
         "journal", "kit", "label", "lot", "margin", "note", "offer", "payment", "quote", "receipt", "supplier", "terms",
         "unit", "voucher", "wallet", "warehouse", "yield", "zone", "agent", "basket"]

def indent(body, n=4):
    return textwrap.indent(body, ' ' * n)

plant = {}            # (module idx, template idx) -> True
mods = list(range(len(NOUNS)))
rnd.shuffle(mods)
tmpl_order = list(range(len(T)))
rnd.shuffle(tmpl_order)
for k in range(10):
    plant[(mods[k], tmpl_order[k])] = True
t1_truth = []
for mi, noun in enumerate(NOUNS):
    chosen = [(mi * 5 + j * 3) % len(T) for j in range(16)]
    chosen = list(dict.fromkeys(chosen + [t for (m, t) in plant if m == mi]))
    src = [f'"""{noun.capitalize()} helpers."""\n']
    for ti in chosen:
        base, args, doc, good, bad = T[ti]
        buggy = (mi, ti) in plant
        name = f"{base}_{noun}"
        src.append(f"\ndef {name}({args}):\n    \"\"\"{doc}\"\"\"\n{indent(bad if buggy else good)}\n")
        if buggy:
            t1_truth.append(f"{noun}_utils.{name}")
    w(f"{OUT}/t1_audit/orderkit/{noun}_utils.py", ''.join(src))
w(f"{OUT}/t1_audit/orderkit/__init__.py", "")

# ---------------------------------------------------------------- T2: incident logs
def ts(h, m, s):
    return f"2026-11-14T{h:02d}:{m:02d}:{s:02d}Z"

def rid():
    return "req-" + ''.join(rnd.choice("0123456789abcdef") for _ in range(8))

PATHS = ["/api/cart", "/api/orders", "/api/search", "/api/profile", "/static/app.js", "/api/checkout", "/api/items"]
hosts = ["web-01", "web-02", "web-03", "web-04"]
lines = {h: [] for h in hosts + ["db-primary", "cache-01", "deploy", "lb-01"]}
def emit(f, t, level, msg):
    lines[f].append((t, f"{t} {level:<5} {msg}"))
FIRST_5XX = "req-c0ffee42"
for sec in range(0, 4 * 3600, 9):          # 08:00 .. 12:00
    t = 8 * 3600 + sec
    h, m, s = t // 3600, (t % 3600) // 60, t % 60
    for host in hosts:
        if rnd.random() < 0.55:
            r = rid(); p = rnd.choice(PATHS); ms = int(rnd.lognormvariate(4.2, 0.5))
            emit(host, ts(h, m, s), "INFO", f"{r} GET {p} 200 {ms}ms")
    if rnd.random() < 0.5:
        emit("lb-01", ts(h, m, s), "INFO", f"{rid()} upstream={rnd.choice(hosts)} 200")
    if rnd.random() < 0.4:
        emit("cache-01", ts(h, m, s), "INFO", f"hit_ratio={rnd.uniform(0.91, 0.97):.2f} evictions={rnd.randint(0, 9)}")
    if rnd.random() < 0.3:
        emit("db-primary", ts(h, m, s), "INFO", f"pool in_use={rnd.randint(12, 38)}/100 qps={rnd.randint(180, 420)}")
# decoys
emit("web-02", ts(9, 12, 3), "ERROR", "req-0badc0de GET /healthz 500 3ms probe failed: disk 91% (resolved 09:14)")
emit("web-02", ts(9, 14, 1), "INFO", "disk cleanup done, usage 62%")
for k in range(25):
    t = 8 * 3600 + rnd.randint(0, 4 * 3600)
    h, m, s = t // 3600, (t % 3600) // 60, t % 60
    emit(rnd.choice(hosts), ts(h, m, s), "WARN", f"{rid()} GET /api/search upstream timeout, retry ok 200 {rnd.randint(900, 1400)}ms")
emit("deploy", ts(8, 0, 12), "INFO", "config set cache.ttl_seconds=600 (was 300) by deploy-bot release=2026.11.2 hosts=cache-01")
emit("deploy", ts(10, 42, 7), "INFO", "config set cache.ttl_seconds=0 (was 600) by deploy-bot release=2026.11.3 hosts=cache-01")
emit("deploy", ts(10, 51, 40), "INFO", "release=2026.11.3 finished, 4/4 hosts ok")
for s in range(10, 600, 7):
    emit("cache-01", ts(10, 42 + s // 60, s % 60), "WARN", f"hit_ratio={max(0.0, 0.9 - s / 80):.2f} evictions={rnd.randint(300, 900)}")
emit("db-primary", ts(10, 43, 31), "ERROR", "pool exhausted: in_use=100/100, waiting=57")
for k, (m, s) in enumerate([(43, 34), (43, 35), (43, 36), (43, 40), (43, 41)]):
    emit("web-04" if k != 2 else "web-01", ts(10, m, s),
         "ERROR", f"{FIRST_5XX if k == 0 else rid()} GET /api/checkout 503 30001ms db pool timeout")
for s in range(0, 500, 3):
    host = rnd.choice(hosts)
    emit(host, ts(10, 44 + s // 60, s % 60), "ERROR", f"{rid()} GET {rnd.choice(PATHS)} 503 {rnd.randint(2000, 30000)}ms db pool timeout")
for f, rows in lines.items():
    rows.sort()
    w(f"{OUT}/t2_logs/{f}.log", '\n'.join(r[1] for r in rows) + '\n')
t2_truth = {"request": FIRST_5XX, "config": "cache.ttl_seconds=0", "host": "web-04"}

# ---------------------------------------------------------------- T3: rename + argument swap
CALLS = [
    "format_money({a}, {c})",
    "format_money(amount={a}, currency={c})",
    "format_money({a}, currency={c})",
    "format_money(currency={c}, amount={a})",
    "money.format_money({a}, {c})",
    "fm({a}, {c})",
    "format_money(\n        {a},\n        {c},\n    )",
    "format_money({a})",
]
w(f"{OUT}/t3_rename/core/__init__.py", "")
w(f"{OUT}/t3_rename/core/money.py", textwrap.dedent('''\
    def format_money(amount, currency="USD"):
        """Format an amount in cents, e.g. format_money(1250, "USD") -> "USD 12.50"."""
        return f"{currency} {amount / 100:.2f}"
    '''))
n_calls = 0
t3_sites = []
for i in range(20):
    imp = ["from core.money import format_money", "from core.money import format_money as fm",
           "from core import money"][i % 3]
    body = [imp, ""]
    for j in range(2 + i % 3):
        style = (i + j) % len(CALLS)
        if "money." in CALLS[style] and "import money" not in imp:
            style = 0
        if CALLS[style].startswith("fm(") and "as fm" not in imp:
            style = 1
        a, c = f"{(i + 1) * 100 + j * 25}", rnd.choice(['"USD"', '"EUR"', '"JPY"'])
        call = CALLS[style].format(a=a, c=c)
        if "import money" in imp:
            call = call.replace("format_money(", "money.format_money(") if "money." not in call else call
        elif "as fm" in imp:
            call = call.replace("format_money(", "fm(")
        n_calls += 1
        fn = f"line_{j}"
        body.append(f"def {fn}():\n    return {call}\n")
        t3_sites.append((f"report_{i:02d}", fn, a, c.strip('"') if '"' in c else c, style))
    body.append(f"\ndef render():\n    return [{', '.join(f'line_{j}()' for j in range(2 + i % 3))}]\n")
    w(f"{OUT}/t3_rename/core/report_{i:02d}.py", '\n'.join(body))
    # note: style 7 (amount only) has no currency -> default
exp = {}
import importlib.util, sys as _s
_s.path.insert(0, f"{OUT}/t3_rename")
for i in range(20):
    spec = importlib.util.spec_from_file_location(f"report_{i:02d}", f"{OUT}/t3_rename/core/report_{i:02d}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    exp[f"report_{i:02d}"] = mod.render()
_s.path.pop(0)
t3_tests = "import importlib, json\nEXPECTED = " + json.dumps(exp, indent=1) + "\n\n" + textwrap.dedent('''\
    import pytest

    @pytest.mark.parametrize("name", sorted(EXPECTED))
    def test_report(name):
        mod = importlib.import_module(f"core.{name}")
        assert mod.render() == EXPECTED[name]
    ''')
w(f"{OUT}/t3_rename/tests/test_reports.py", t3_tests)
w(f"{OUT}/t3_rename/pytest.ini", "[pytest]\npythonpath = .\n")
t3_truth = {"calls": n_calls}

# ---------------------------------------------------------------- T4: three planted bugs
w(f"{OUT}/t4_fix/shop/__init__.py", "")
w(f"{OUT}/t4_fix/pytest.ini", "[pytest]\npythonpath = .\n")
w(f"{OUT}/t4_fix/shop/pricing.py", textwrap.dedent('''\
    TAX_RATES = {"US": 0.07, "DE": 0.19, "JP": 0.10}


    def line_total(unit_cents, qty, discount_pct=0):
        """Line total in cents after a percentage discount (discount truncated to whole cents)."""
        gross = unit_cents * qty
        return gross - gross * discount_pct // 100


    def tax_for(subtotal_cents, country):
        """Tax in cents, rounded half up to the nearest cent."""
        return int(subtotal_cents * TAX_RATES[country])


    def order_total(lines, country):
        """lines: list of (unit_cents, qty, discount_pct)."""
        subtotal = sum(line_total(*ln) for ln in lines)
        return subtotal + tax_for(subtotal, country)
    '''))
w(f"{OUT}/t4_fix/shop/inventory.py", textwrap.dedent('''\
    class OutOfStock(Exception):
        pass


    class Stock:
        def __init__(self, on_hand):
            self.on_hand = on_hand
            self.reserved = 0

        def available(self):
            return self.on_hand - self.reserved

        def reserve(self, n):
            """Reserve n units; raises OutOfStock if fewer than n are available."""
            if n <= 0:
                raise ValueError("n must be positive")
            if n > self.available() + 1:
                raise OutOfStock(n)
            self.reserved += n

        def release(self, n):
            self.reserved = max(0, self.reserved - n)

        def ship(self, n):
            """Ship n reserved units."""
            if n > self.reserved:
                raise ValueError("not reserved")
            self.reserved -= n
            self.on_hand -= n
    '''))
w(f"{OUT}/t4_fix/shop/shipping.py", textwrap.dedent('''\
    ZONES = {"US": {"1": "east", "9": "west"}, "DE": {"1": "north", "8": "south"}}
    BASE = {"east": 500, "west": 700, "north": 600, "south": 650}
    STEP_G = 500
    STEP_CENTS = 120


    def zone_for(country, postal):
        return ZONES[country][postal[0]]


    def shipping_cents(country, postal, weight_g):
        """Base fee for the zone plus 120 cents per started 500 g."""
        steps = weight_g // STEP_G
        return BASE[zone_for(country, postal)] + steps * STEP_CENTS
    '''))
vis = '''\
from shop.pricing import line_total, tax_for, order_total
from shop.inventory import Stock, OutOfStock
from shop.shipping import zone_for, shipping_cents
import pytest


def test_line_total_plain():
    assert line_total(250, 4) == 1000

def test_line_total_discount():
    assert line_total(1000, 2, 10) == 1800

def test_tax_us():
    assert tax_for(1050, "US") == 74

def test_tax_de_round_up():
    assert tax_for(1000, "DE") == 190

def test_tax_jp_half():
    assert tax_for(25, "JP") == 3

def test_order_total():
    assert order_total([(1000, 1, 0)], "US") == 1070

def test_reserve_ok():
    s = Stock(5); s.reserve(5); assert s.available() == 0

def test_reserve_over():
    s = Stock(5)
    with pytest.raises(OutOfStock):
        s.reserve(6)

def test_reserve_after_partial():
    s = Stock(5); s.reserve(3)
    with pytest.raises(OutOfStock):
        s.reserve(3)

def test_ship():
    s = Stock(5); s.reserve(2); s.ship(2); assert (s.on_hand, s.reserved) == (3, 0)

def test_zone():
    assert zone_for("US", "90210") == "west"

def test_ship_base_only():
    assert shipping_cents("US", "10001", 0) == 500

def test_ship_two_steps():
    assert shipping_cents("DE", "80331", 1000) == 650 + 240

def test_ship_partial_step():
    assert shipping_cents("US", "10001", 1200) == 500 + 3 * 120
'''
w(f"{OUT}/t4_fix/tests/test_shop.py", vis)
hidden = '''\
from shop.pricing import tax_for, order_total
from shop.inventory import Stock, OutOfStock
from shop.shipping import shipping_cents
import pytest


@pytest.mark.parametrize("sub,country,want", [(5, "JP", 1), (15, "JP", 2), (25, "JP", 3), (35, "JP", 4), (150, "DE", 29), (50, "US", 4)])
def test_tax_half_up(sub, country, want):
    assert tax_for(sub, country) == want

def test_order_total_half():
    assert order_total([(5, 1, 0)], "JP") == 6

def test_reserve_exact_boundary():
    s = Stock(1)
    with pytest.raises(OutOfStock):
        s.reserve(2)
    s.reserve(1)
    with pytest.raises(OutOfStock):
        s.reserve(1)

@pytest.mark.parametrize("g,want", [(1, 620), (499, 620), (500, 620), (501, 740), (1000, 740), (1001, 860)])
def test_ship_started_steps(g, want):
    assert shipping_cents("US", "10001", g) == want
'''
w(f"{OUT}/hidden_t4/test_hidden.py", hidden)
w(f"{OUT}/truth.json", json.dumps({"t1": sorted(t1_truth), "t2": t2_truth, "t3": t3_truth}, indent=1))
print("ok", len(t1_truth), "planted;", n_calls, "calls")
