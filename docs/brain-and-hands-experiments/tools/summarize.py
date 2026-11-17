import json, sys, collections
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
by = collections.defaultdict(list)
for r in rows: by[(r['arm'], r['task'])].append(r)
arms = list(dict.fromkeys(r['arm'] for r in rows)); tasks = sorted({r['task'] for r in rows})
print(f"{'arm':12}" + ''.join(f"{t:>20}" for t in tasks) + f"{'| MEAN q':>10}{'paid $':>9}{'Gris tok':>10}{'agents':>7}{'secs':>6}")
for a in arms:
    cells, qs, cs, gt, ag, sc = [], [], [], 0, 0, []
    for t in tasks:
        rs = by.get((a, t), [])
        if not rs: cells.append(f"{'-':>20}"); continue
        q = sum(r['quality'] for r in rs) / len(rs); c = sum(r['paid_usd'] for r in rs) / len(rs)
        cells.append(f"{f'q={q:.2f} ${c:.3f} n={len(rs)}':>20}"); qs.append(q); cs.append(c)
        gt += sum(r['gris_tokens'] for r in rs) / len(rs); ag += sum(len(r['agents']) for r in rs) / len(rs); sc += [r['secs'] for r in rs]
    print(f"{a:12}" + ''.join(cells) + f"{sum(qs)/len(qs):>10.3f}{sum(cs):>9.3f}{int(gt):>10}{ag:>7.1f}{int(sum(sc)/len(sc)):>6}")
