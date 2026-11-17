"""Re-score every run in a results file from its saved answer/workdir (scorer changed)."""
import json, subprocess, sys
res = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else None
rows = [json.loads(l) for l in open(res)]
for r in rows:
    d = f"/dev/shm/bah-exp/runs/{r['tag']}/{r['arm']}-{r['task']}-{r['rep']}"
    r['score'] = json.loads(subprocess.check_output([sys.executable, '/dev/shm/bah-exp/tools/score.py', r['task'], f'{d}/work', f'{d}/answer.txt'], text=True))
    r['quality'] = r['score']['quality']
open(res, 'w').write(''.join(json.dumps(r) + '\n' for r in rows))
print('rescored', len(rows))
