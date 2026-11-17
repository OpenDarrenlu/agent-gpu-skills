#!/usr/bin/env python3
"""Heavy T1 corpus: 240 modules (~25k lines) with 30 planted docstring violations.  usage: gen_heavy.py OUT_DIR"""
import json, os, random, re, sys, textwrap
OUT = sys.argv[1]
src = open(os.path.join(os.path.dirname(__file__), 'gen_fixtures.py')).read()
ns = {}
exec(src[src.index('T = ['):src.index('def indent')], ns)          # reuse templates T and NOUNS
T, NOUNS = ns['T'], ns['NOUNS']
rnd = random.Random(777)
names = [f"{n}{k}" for k in range(1, 6) for n in NOUNS]            # 240 unique module nouns
plant = set()
for mi in rnd.sample(range(len(names)), 30):
    plant.add((mi, rnd.randrange(len(T))))
truth = []
for mi, noun in enumerate(names):
    chosen = list(dict.fromkeys([(mi * 5 + j * 3) % len(T) for j in range(16)] + [t for (m, t) in plant if m == mi]))
    out = [f'"""{noun.capitalize()} helpers."""\n']
    for ti in chosen:
        base, args, doc, good, bad = T[ti]
        buggy = (mi, ti) in plant
        out.append(f"\ndef {base}_{noun}({args}):\n    \"\"\"{doc}\"\"\"\n{textwrap.indent(bad if buggy else good, '    ')}\n")
        if buggy: truth.append(f"{noun}_utils.{base}_{noun}")
    os.makedirs(f"{OUT}/t1h_audit/orderkit", exist_ok=True)
    open(f"{OUT}/t1h_audit/orderkit/{noun}_utils.py", 'w').write(''.join(out))
open(f"{OUT}/t1h_audit/orderkit/__init__.py", 'w').write('')
json.dump({"t1": sorted(truth)}, open(f"{OUT}/truth.json", 'w'), indent=1)
print(len(truth), 'planted')
