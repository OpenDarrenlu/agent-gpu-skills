#!/usr/bin/env python3
"""Run (arm x task) experiments with headless `claude -p`; append one JSON line per run to RESULTS.

usage: run_exp.py RESULTS ARMS TASKS [--reps N] [--jobs J] [--tag TAG]
  ARMS : comma list from ARMS below     TASKS: comma list of t1,t2,t3,t4
"""
import argparse, concurrent.futures as cf, json, os, shutil, subprocess, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from cleanenv import clean_env

EXP = '/dev/shm/bah-exp'; FIX = f'{EXP}/fixtures'
V01 = f'{EXP}/plugin-v0.1'                                   # pristine upstream copy
V02 = os.environ.get('BAH_PLUGIN_NEW', '/root/code/agent-gpu-skills/skills/other/brain-and-hands')
OPUS, SONNET, HAIKU, GRIS = 'claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-5-5', 'anthropic/a7n-gris[1m]'
ARMS = {                      # name -> (main model, plugin dir or None)
    'opus': (OPUS, None), 'sonnet': (SONNET, None), 'haiku': (HAIKU, None), 'gris': (GRIS, None),
    'opus+v1': (OPUS, V01), 'sonnet+v1': (SONNET, V01),
    'opus+v2': (OPUS, V02), 'sonnet+v2': (SONNET, V02),
    'opus+v2a': (OPUS, f'{EXP}/plugin-v2a'), 'sonnet+v2a': (SONNET, f'{EXP}/plugin-v2a'),   # first 0.2 draft (before the no-polling rules)
}
PROMPTS = {
 't1h': None,
 't1': "The Python package `orderkit/` has many modules of small helper functions; each function's docstring states its contract. "
       "Find every function whose implementation violates its own docstring. Read the code; do not run it and do not modify any files. "
       "Final answer: one `module.function` per line (for example `foo_utils.bar_foo`) for the functions that really violate their docstring, "
       "and nothing else on those lines. Be precise: false positives count against you.",
 't2': "The `logs/` directory holds the logs of a web service during an incident on 2026-11-14 (web-01..04, db-primary, cache-01, lb-01 and the deploy log). "
       "Find: the request id of the earliest user-facing HTTP 5xx (ignore internal health-check probes), the config change that triggered the incident, "
       "and the host that emitted that first 5xx. Do not modify any files. "
       "Final answer in exactly three lines: `request: <id>`, `config: <key>=<new value>`, `host: <hostname>`.",
 't3': "In `core/`, rename `format_money` to `fmt_amount` and change its signature to `fmt_amount(currency, amount)`: `currency` becomes required and comes first. "
       "Update `core/money.py` and every call site in `core/` (positional and keyword calls, aliased imports, calls through the module, multi-line calls); "
       "a call that relied on the default currency must now pass \"USD\" explicitly. Behaviour must not change: `python -m pytest -q` must pass and the tests must not be edited. "
       "Final answer: one line saying it is done, then the pytest summary line.",
 't4': "`python -m pytest -q` shows failing tests in this project. Fix the root causes in `shop/` (do not edit the tests). "
       "Make sure the fixes are right for inputs beyond what the visible tests cover. "
       "Final answer: one line per root cause with file, function and what was wrong.",
}
PROMPTS['t1h'] = PROMPTS['t1'].replace('many modules', 'about 240 modules')
PROMPTS['t5'] = ("The cart pricing code in `cart/engine.py` must follow `SPEC.md`. The visible tests (`python -m pytest -q`) fail. "
       "Fix `cart/` so the implementation conforms to SPEC.md in every respect, not only what the visible tests check: other deviations may exist. "
       "Do not edit the tests or SPEC.md. Final answer: one line per deviation you fixed (spec clause number, what was wrong).")
PROMPTS['t6'] = PROMPTS['t1'].replace('The Python package `orderkit/` has many modules', 'The Python package `lib/` has 40 modules (1000 functions)').replace('foo_utils.bar_foo','mod_00.calc_foo_3')
PROMPTS['t7'] = ("The `logs/` directory holds the logs of a web service during an incident on 2026-11-14 (web-01..04, db-primary, cache-01, lb-01 and the deploy log). "
       "For host web-05: find the request id of its earliest user-facing HTTP 5xx and the config change deployed to web-05 that triggered the incident. Do not modify any files. "
       "Final answer in exactly two lines: `request: <id or 'not found'>` and `config: <key>=<value or 'not found'>`.")
PROMPTS['t8'] = PROMPTS['t2']
PROMPTS['t9'] = ("Implement `next_fire` in `cron.py` exactly as `SPEC.md` specifies (standard library only). The tests in `tests/` are only examples; "
       "the implementation is graded on a much larger hidden suite of edge cases, so follow the spec to the letter. Do not edit SPEC.md or the tests. "
       "Final answer: one line saying it is done, then one line per place where the spec was ambiguous and how you resolved it (or 'none').")
SRC = {'t9': 't9_cron', 't7': 't7_absent', 't8': 't8_inject', 't6': 't6_audit', 't5': 't5_spec', 't1h': 't1h_audit', 't1': 't1_audit', 't2': 't2_logs', 't3': 't3_rename', 't4': 't4_fix'}

FORCE = ' Delegate the bulk of this work to the brain-and-hands subagents as the delegate skill describes, then verify their report yourself before you answer.'

def one(results, arm, task, rep, tag, effort):
    forced = arm.endswith('!'); name = arm
    arm = arm.rstrip('!'); model, plugin = ARMS[arm]
    run = f'{EXP}/runs/{tag}/{name}-{task}-{rep}'
    shutil.rmtree(run, ignore_errors=True); os.makedirs(run)
    base = {'t1h': FIX + '-heavy', 't6': FIX + '-t6'}.get(task, FIX)
    work = f'{run}/work'; shutil.copytree(f'{base}/{SRC[task]}', work)
    cfg = f'{EXP}/cfg/{arm}'; os.makedirs(cfg, exist_ok=True)
    cmd = ['claude', '-p', PROMPTS[task] + (FORCE if forced else ''), '--model', model, '--output-format', 'stream-json', '--verbose',
           '--no-session-persistence', '--permission-mode', 'bypassPermissions', '--max-budget-usd', '6']
    if effort: cmd += ['--effort', effort]
    if plugin: cmd += ['--plugin-dir', plugin]
    t0 = time.time()
    p = subprocess.run(cmd, env=clean_env(cfg), cwd=work, capture_output=True, text=True, timeout=1800)
    open(f'{run}/stream.jsonl', 'w').write(p.stdout); open(f'{run}/stderr.txt', 'w').write(p.stderr)
    res, agents = {}, []
    for l in p.stdout.splitlines():
        try: e = json.loads(l)
        except ValueError: continue
        if e.get('type') == 'result': res = e
        if e.get('type') == 'assistant' and not e.get('parent_tool_use_id'):
            for c in e['message'].get('content', []):
                if c.get('type') == 'tool_use' and c['name'] in ('Agent', 'Task'):
                    i = c['input']; agents.append(f"{i.get('subagent_type', 'general-purpose')}:{i.get('model', '-')}")
    ans = res.get('result') or ''
    open(f'{run}/answer.txt', 'w').write(ans)
    sc = json.loads(subprocess.check_output([sys.executable, f'{EXP}/tools/score.py', task, work, f'{run}/answer.txt'], text=True))
    mu = res.get('modelUsage', {})
    paid = sum(v['costUSD'] for m, v in mu.items() if 'a7n' not in m)
    tok = lambda v: v['inputTokens'] + v['outputTokens'] + v['cacheReadInputTokens'] + v['cacheCreationInputTokens']
    rec = dict(tag=tag, arm=name, task=task, rep=rep, quality=sc['quality'], score=sc, paid_usd=round(paid, 4),
               gris_tokens=sum(tok(v) for m, v in mu.items() if 'a7n' in m),
               tokens_by_model={m: tok(v) for m, v in mu.items()}, cost_by_model={m: round(v['costUSD'], 4) for m, v in mu.items()},
               turns=res.get('num_turns'), secs=round(time.time() - t0), agents=agents, is_error=res.get('is_error'), effort=effort)
    with open(results, 'a') as f: f.write(json.dumps(rec) + '\n')
    print(f"{arm:10} {task} r{rep} q={rec['quality']:<5} paid=${rec['paid_usd']:<7} agents={len(agents)} {rec['secs']}s", flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('results'); ap.add_argument('arms'); ap.add_argument('tasks')
    ap.add_argument('--reps', type=int, default=1); ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--tag', default='p1'); ap.add_argument('--effort', default='medium')
    a = ap.parse_args()
    jobs = [(arm, t, r) for r in range(a.reps) for t in a.tasks.split(',') for arm in a.arms.split(',')]
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        fs = [ex.submit(one, a.results, arm, t, r, a.tag, a.effort) for arm, t, r in jobs]
        for f in fs:
            try: f.result()
            except Exception as e: print('FAILED', e, flush=True)
