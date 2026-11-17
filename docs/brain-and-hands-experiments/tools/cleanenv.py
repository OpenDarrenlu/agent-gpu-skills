"""Clean env for headless `claude -p` runs: gateway creds from user settings (never printed/written), no host-managed vars."""
import json, os

def clean_env(config_dir):
    s = json.load(open('/root/.claude/settings.json'))['env']
    keep = {k: os.environ[k] for k in ('PATH', 'LANG', 'LC_ALL', 'TERM') if k in os.environ}
    keep.update(HOME='/root', CLAUDE_CONFIG_DIR=config_dir,
                ANTHROPIC_API_KEY=s['ANTHROPIC_API_KEY'], ANTHROPIC_BASE_URL=s['ANTHROPIC_BASE_URL'],
                CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS='1', IS_SANDBOX='1',
                CLAUDE_CODE_DISABLE_AUTO_MEMORY='1')
    return keep
