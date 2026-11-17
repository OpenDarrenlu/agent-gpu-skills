#!/usr/bin/env python3
"""T9: implement cron next_fire from SPEC.md; hidden suite from a reference implementation. usage: gen_t9.py FIXTURES_DIR"""
import json, os, random, sys, textwrap
from datetime import datetime, timedelta
OUT = sys.argv[1]; R = random.Random(909)
MONTHS = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(); DOWS = "SUN MON TUE WED THU FRI SAT".split()
RANGES = [(0, 59, None), (0, 23, None), (1, 31, None), (1, 12, MONTHS), (0, 7, DOWS)]

def parse_field(tok, lo, hi, names, is_dow=False):
    out = set(); restricted = tok != '*'
    for part in tok.split(','):
        step = 1
        if '/' in part:
            part, s = part.split('/')
            if not s.isdigit() or int(s) < 1: raise ValueError(tok)
            step = int(s)
        def val(x):
            if names and x.upper() in names: return names.index(x.upper()) + (lo if names is MONTHS else 0)
            if not x.isdigit(): raise ValueError(tok)
            return int(x)
        if part == '*': a, b = lo, (6 if is_dow else hi)
        elif '-' in part:
            x, y = part.split('-'); a, b = val(x), val(y)
            if a > b: raise ValueError(tok)
        else:
            if step != 1 and '/' in tok.split(',')[0] and False: pass
            a = val(part); b = a
            if '/' in tok and part != '*' and '-' not in part: raise ValueError(tok)
        if a < lo or b > hi: raise ValueError(tok)
        out.update(range(a, b + 1, step))
    if is_dow: out = {d % 7 for d in out}
    return out, restricted

def ref_next(expr, after):
    f = expr.split()
    if len(f) != 5: raise ValueError(expr)
    mi, _ = parse_field(f[0], 0, 59, None); hr, _ = parse_field(f[1], 0, 23, None)
    dom, dom_r = parse_field(f[2], 1, 31, None); mon, _ = parse_field(f[3], 1, 12, MONTHS)
    dow, dow_r = parse_field(f[4], 0, 7, DOWS, True)
    t = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    for _ in range(366 * 8 * 24 * 60 // 1):
        if t.month in mon:
            d_ok = t.day in dom; w_ok = (t.isoweekday() % 7) in dow
            day_ok = (d_ok or w_ok) if (dom_r and dow_r) else (d_ok and w_ok)
            if day_ok:
                if t.hour in hr:
                    if t.minute in mi: return t
                    t += timedelta(minutes=1); continue
                t = t.replace(minute=0) + timedelta(hours=1); continue
            t = t.replace(hour=0, minute=0) + timedelta(days=1); continue
        y, m = (t.year + 1, 1) if t.month == 12 else (t.year, t.month + 1)
        t = t.replace(year=y, month=m, day=1, hour=0, minute=0)
    raise ValueError("never fires: " + expr)

SPEC = '''\
# cron.next_fire(expr, after) -> datetime
Return the earliest datetime strictly after `after` (naive, minute resolution, seconds and microseconds zero) matching the 5-field cron expression
`minute hour day-of-month month day-of-week`, fields separated by any run of spaces/tabs.

Field syntax (each field is a comma-separated list of items):
- `*` every value of the field; `n` one value; `a-b` inclusive range (a <= b, no wrap-around); `*/s` or `a-b/s` every s-th value starting at the range start (s >= 1).
- A step on a single number (`5/10`) is invalid. Any other syntax (`L`, `?`, `#`, `W`, empty items, negative numbers) is invalid.
- Ranges: minute 0-59, hour 0-23, day-of-month 1-31, month 1-12, day-of-week 0-7 (0 and 7 both mean Sunday).
- Month names JAN..DEC and weekday names SUN..SAT are accepted case-insensitively wherever a number is (also inside ranges and lists, not with a step on a single name).
Day matching: if both day-of-month and day-of-week are restricted (neither is exactly `*`), a day matches when EITHER matches; otherwise a day must satisfy both fields (an unrestricted field always matches).
`*/s` on day-of-week covers 0-6. A day-of-month that does not exist in a month (e.g. 31 in April, 30 Feb) never matches in that month; `0 0 29 2 *` therefore fires only in leap years.
Invalid expressions (wrong field count, bad syntax, out-of-range values) raise ValueError. An expression that can never fire (e.g. `0 0 31 2 *`) raises ValueError too.
Look ahead at most 8 years. Do not use third-party libraries (standard library only).
'''
def D(x):
    return 'None' if x is None else f'datetime({x.year}, {x.month}, {x.day}, {x.hour}, {x.minute}, {x.second})'

os.makedirs(f'{OUT}/t9_cron/tests', exist_ok=True)
open(f'{OUT}/t9_cron/SPEC.md', 'w').write(SPEC)
open(f'{OUT}/t9_cron/pytest.ini', 'w').write('[pytest]\npythonpath = .\n')
open(f'{OUT}/t9_cron/cron.py', 'w').write('from datetime import datetime\n\n\ndef next_fire(expr: str, after: datetime) -> datetime:\n    raise NotImplementedError\n')
vis = [("*/15 * * * *", datetime(2026, 11, 14, 10, 7)), ("0 9 * * MON-FRI", datetime(2026, 11, 14, 10, 7)), ("30 2 29 2 *", datetime(2026, 11, 14, 10, 7)),
       ("0 0 1,15 * 1", datetime(2026, 11, 14, 10, 7)), ("5 4 * jan-mar */2", datetime(2026, 11, 14, 10, 7))]
body = "from datetime import datetime\nimport pytest\nfrom cron import next_fire\n\nCASES = [\n" + ''.join(f"    ({e!r}, {D(a)}, {D(ref_next(e, a))}),\n" for e, a in vis) + "]\n\n"
body += "@pytest.mark.parametrize('expr,after,want', CASES)\ndef test_examples(expr, after, want):\n    assert next_fire(expr, after) == want\n\ndef test_invalid():\n    with pytest.raises(ValueError):\n        next_fire('61 * * * *', datetime(2026, 1, 1))\n"
open(f'{OUT}/t9_cron/tests/test_cron.py', 'w').write(body)
# hidden cases
atoms = [["*", "*/7", "5", "0,30", "10-20", "10-50/20", "59"], ["*", "*/5", "0", "23", "6-9", "1,13", "2-20/6"], ["*", "1", "15", "29", "30", "31", "1-7", "*/10", "28-31"],
         ["*", "2", "feb", "JAN,DEC", "mar-may", "*/4", "4-12/4", "10"], ["*", "0", "7", "sun", "1-5", "MON,wed", "*/3", "6", "5-7"]]
cases = []; seen = set()
while len(cases) < 70:
    e = ' '.join(R.choice(a) for a in atoms)
    e = e.replace(' ', R.choice([' ', '  ', '\t']), 1) if R.random() < .1 else e
    a = datetime(2026, R.randint(1, 12), R.randint(1, 28), R.randint(0, 23), R.randint(0, 59), R.randint(0, 59))
    if R.random() < .15: a = datetime(2027, 12, 31, 23, 59, 30)
    if R.random() < .1: a = datetime(2027, 2, 28, 23, 59)
    if (e, a) in seen: continue
    seen.add((e, a))
    try: cases.append((e, a, ref_next(e, a)))
    except ValueError: cases.append((e, a, None))
extra = [("0 0 29 2 *", datetime(2026, 3, 1), None), ("0 0 29 2 *", datetime(2027, 3, 1), None), ("59 23 31 12 *", datetime(2026, 12, 31, 23, 59), None),
         ("0 0 31 4 *", datetime(2026, 1, 1), None), ("0 0 30 2 *", datetime(2026, 1, 1), None)]
for e, a, _ in extra:
    try: cases.append((e, a, ref_next(e, a)))
    except ValueError: cases.append((e, a, None))
invalid = ["* * * *", "* * * * * *", "60 * * * *", "* 24 * * *", "* * 0 * *", "* * 32 * *", "* * * 13 *", "* * * * 8", "5/10 * * * *", "*/0 * * * *", "1-5-7 * * * *",
           "20-10 * * * *", "L * * * *", "* * ? * *", "a * * * *", "-5 * * * *", "1,,2 * * * *", "* * * FOO *", "* * * * MON/2", "* * * * 1#2", "", "*/ * * * *", "1- * * * *"]
os.makedirs(f'{OUT}/hidden_t9', exist_ok=True)
h = "from datetime import datetime\nimport pytest\nfrom cron import next_fire\n\nCASES = [\n" + ''.join(f"    ({e!r}, {D(a)}, {D(w)}),\n" for e, a, w in cases) + "]\nINVALID = " + repr(invalid) + "\n\n"
h += "@pytest.mark.parametrize('expr,after,want', CASES)\ndef test_cases(expr, after, want):\n    if want is None:\n        with pytest.raises(ValueError):\n            next_fire(expr, after)\n    else:\n        assert next_fire(expr, after) == want\n\n"
h += "@pytest.mark.parametrize('expr', INVALID)\ndef test_invalid(expr):\n    with pytest.raises(ValueError):\n        next_fire(expr, datetime(2026, 1, 1))\n"
open(f'{OUT}/hidden_t9/test_hidden.py', 'w').write(h)
print('t9 ok', len(cases), 'cases;', sum(1 for c in cases if c[2] is None), 'expect ValueError')
