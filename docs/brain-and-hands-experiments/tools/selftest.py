"""Validate fixtures+scorer: broken baseline vs reference solutions."""
import json, os, re, shutil, subprocess, sys
FIX = '/dev/shm/bah-exp/fixtures'; T = '/dev/shm/bah-exp/selftest'
shutil.rmtree(T, ignore_errors=True); os.makedirs(T)
def score(task, work, ans=''):
    open(f'{T}/ans.txt', 'w').write(ans)
    return json.loads(subprocess.check_output([sys.executable, '/dev/shm/bah-exp/tools/score.py', task, work, f'{T}/ans.txt'], text=True))
truth = json.load(open(f'{FIX}/truth.json'))
print('t1 perfect', score('t1', '', '\n'.join(truth['t1'])))
print('t1 half+fp', score('t1', '', '\n'.join(truth['t1'][:5] + ['foo_utils.bar'])))
print('t2 perfect', score('t2', '', 'request: req-c0ffee42\nconfig: cache.ttl_seconds=0\nhost: web-04'))
print('t2 decoy  ', score('t2', '', 'request: req-0badc0de\nconfig: cache.ttl_seconds=600\nhost: web-02'))
shutil.copytree(f'{FIX}/t3_rename', f'{T}/t3'); print('t3 untouched', score('t3', f'{T}/t3'))
# reference t3: regex rewrite of every call style
for f in os.listdir(f'{T}/t3/core'):
    p = f'{T}/t3/core/{f}'
    s = open(p).read()
    if f == 'money.py':
        s = s.replace('def format_money(amount, currency="USD")', 'def fmt_amount(currency, amount)').replace('format_money(1250, "USD")', 'fmt_amount("USD", 1250)')
    else:
        def fix(m):
            name, args = m.group(1), m.group(2)
            parts = [a.strip() for a in re.split(r',\s*(?![^()]*\))', args.strip().rstrip(',')) if a.strip()]
            kw = {a.split('=')[0].strip(): a.split('=', 1)[1].strip() for a in parts if re.match(r'^(amount|currency)\s*=', a)}
            pos = [a for a in parts if not re.match(r'^(amount|currency)\s*=', a)]
            amount = kw.get('amount') or pos[0]
            cur = kw.get('currency') or (pos[1] if len(pos) > 1 and 'amount' not in kw else '"USD"')
            if 'amount' in kw and pos: cur = pos[0]
            return f'{name.replace("format_money", "fmt_amount")}({cur}, {amount})'
        s = re.sub(r'((?:money\.)?(?:format_money|fm))\(([^()]*)\)', lambda m: fix(m) if 'import' not in m.group(0) else m.group(0), s)
        s = s.replace('import format_money as fm', 'import fmt_amount as fm').replace('import format_money', 'import fmt_amount')
    open(p, 'w').write(s)
print('t3 reference', score('t3', f'{T}/t3'))
shutil.copytree(f'{FIX}/t4_fix', f'{T}/t4'); print('t4 untouched', score('t4', f'{T}/t4'))
p = f'{T}/t4/shop/pricing.py'; s = open(p).read().replace('return int(subtotal_cents * TAX_RATES[country])',
 'import math\n    return math.floor(subtotal_cents * TAX_RATES[country] + 0.5)'); open(p, 'w').write(s)
p = f'{T}/t4/shop/inventory.py'; s = open(p).read().replace('self.available() + 1', 'self.available()'); open(p, 'w').write(s)
p = f'{T}/t4/shop/shipping.py'; s = open(p).read().replace('weight_g // STEP_G', '-(-weight_g // STEP_G)'); open(p, 'w').write(s)
print('t4 reference', score('t4', f'{T}/t4'))
