#!/usr/bin/env python3
"""Vendor the pinned plugins in harness/plugins.json so Claude Code cloud sessions load them.

Cloud sessions ignore a repo's enabledPlugins/extraKnownMarketplaces, but they do load the repo's
.claude/skills/ and the hooks in .claude/settings.json. So each plugin's hook scripts are copied to
harness/vendor/<name>/ at its pinned commit, its skills to .claude/skills/, and .claude/settings.json is
written from harness/claude-settings.base.json plus every plugin's hooks, each hook run with
CLAUDE_PLUGIN_ROOT pointing at the vendored copy.

  python3 harness/sync_plugins.py          fetch pinned commits, vendor, rewrite settings and lock
  python3 harness/sync_plugins.py --check  offline: vendored files match the lock, settings match the
                                           registry, every hook script exists and parses
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / 'harness' / 'plugins.json'
BASE_SETTINGS = REPO / 'harness' / 'claude-settings.base.json'
SETTINGS = REPO / '.claude' / 'settings.json'
LOCK = REPO / 'harness' / 'plugins.lock.json'
CACHE = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'harness-plugins'
ROOT_REF = re.compile(r'\$\{CLAUDE_PLUGIN_ROOT\}(/[^"\s)]*)?')


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(p: dict) -> Path:
    """Shallow checkout of the pinned commit, reused across runs; refuses anything but that commit."""
    dest = CACHE / f"{p['name']}-{p['commit'][:12]}"
    if not (dest / '.git').exists():
        dest.mkdir(parents=True, exist_ok=True)
        git('init', '-q', cwd=dest)
        git('fetch', '-q', '--depth', '1', p['repo'], p['commit'], cwd=dest)
        git('checkout', '-q', 'FETCH_HEAD', cwd=dest)
    head = git('rev-parse', 'HEAD', cwd=dest)
    if head != p['commit']:
        sys.exit(f"{p['name']}: checkout is {head}, registry pins {p['commit']}")
    return dest / p.get('root', '')


def vendor(p: dict, src: Path, vendor_dir: Path) -> dict[str, str]:
    out = vendor_dir / p['name']
    shutil.rmtree(out, ignore_errors=True)
    hashes = {}
    for pattern in p['files']:
        matches = [f for f in src.glob(pattern + '/*' if pattern.endswith('**') else pattern) if f.is_file()]
        if not matches:
            sys.exit(f"{p['name']}: '{pattern}' matches nothing at {p['commit'][:12]} (upstream layout changed?)")
        for f in matches:
            rel = f.relative_to(src).as_posix()
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out / rel)
            hashes[rel] = sha256(out / rel)
    return dict(sorted(hashes.items()))


def install_skill(p: dict, src: Path, target: str, skills_dir: Path, vendor_rel: str) -> dict[str, str]:
    """Copy one skill; rename it in its frontmatter and point plugin-root references at the vendored copy."""
    dst = skills_dir / target
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    skill_md = dst / 'SKILL.md'
    text = skill_md.read_text()
    text = re.sub(r'^name:.*$', f'name: {target}', text, count=1, flags=re.M)
    text = text.replace('${CLAUDE_PLUGIN_ROOT}', '${CLAUDE_PROJECT_DIR}/' + vendor_rel)
    skill_md.write_text(text)
    return {f.relative_to(dst).as_posix(): sha256(f) for f in sorted(dst.rglob('*')) if f.is_file()}


def plugin_hooks(p: dict, vendor_dir: Path) -> dict[str, list]:
    """The plugin's hooks, each command prefixed so ${CLAUDE_PLUGIN_ROOT} resolves to the vendored copy."""
    if not p.get('claude_hooks'):
        return {}
    hooks = json.loads((vendor_dir / p['name'] / p['claude_hooks']).read_text())['hooks']
    root = f'$CLAUDE_PROJECT_DIR/{vendor_dir.relative_to(REPO).as_posix()}/{p["name"]}'
    prefix = f'export CLAUDE_PLUGIN_ROOT="{root}" CLAUDE_PLUGIN_DATA="$HOME/.claude/plugin-data/{p["name"]}"; '
    out: dict[str, list] = {}
    for event, groups in hooks.items():
        for group in copy.deepcopy(groups):
            for h in group['hooks']:
                if h.get('type') == 'command':
                    h['command'] = prefix + h['command']
            out.setdefault(event, []).append(group)
    return out


def build_settings(registry: dict, vendor_dir: Path) -> dict:
    settings = json.loads(BASE_SETTINGS.read_text())
    hooks = settings.setdefault('hooks', {})
    for p in registry['plugins']:
        for event, groups in plugin_hooks(p, vendor_dir).items():
            hooks.setdefault(event, []).extend(groups)
    return settings


def dump(obj: dict) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + '\n'


def sync(registry: dict) -> None:
    vendor_dir, skills_dir = REPO / registry['vendor_dir'], REPO / registry['skills_dir']
    old = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    lock = {}
    for p in registry['plugins']:
        src = fetch(p)
        entry = {'repo': p['repo'], 'commit': p['commit'], 'files': vendor(p, src, vendor_dir), 'skills': {}}
        for rel, target in p.get('skills', {}).items():
            vendor_rel = f"{registry['vendor_dir']}/{p['name']}"
            entry['skills'][target] = install_skill(p, src / rel, target, skills_dir, vendor_rel)
        lock[p['name']] = entry
        print(f"vendored {p['name']} @ {p['commit'][:12]}: {len(entry['files'])} files, {len(entry['skills'])} skills")
    kept = {t for e in lock.values() for t in e['skills']}
    for name, e in old.items():  # drop plugins/skills removed from the registry
        for target in set(e['skills']) - kept:
            shutil.rmtree(skills_dir / target, ignore_errors=True)
        if name not in lock:
            shutil.rmtree(vendor_dir / name, ignore_errors=True)
    LOCK.write_text(dump(lock))
    SETTINGS.parent.mkdir(exist_ok=True)
    SETTINGS.write_text(dump(build_settings(registry, vendor_dir)))
    print(f'wrote {SETTINGS.relative_to(REPO)} and {LOCK.relative_to(REPO)}')


def check(registry: dict) -> int:
    vendor_dir, skills_dir = REPO / registry['vendor_dir'], REPO / registry['skills_dir']
    lock = json.loads(LOCK.read_text())
    problems = []
    for p in registry['plugins']:
        e = lock.get(p['name'])
        if not e or e['commit'] != p['commit']:
            problems.append(f"{p['name']}: lock is not at the registry commit; run sync")
            continue
        for rel, digest in e['files'].items():
            f = vendor_dir / p['name'] / rel
            if not f.is_file() or sha256(f) != digest:
                problems.append(f"{p['name']}: {rel} missing or edited after vendoring")
            elif f.suffix in ('.js', '.mjs') and subprocess.run(['node', '--check', str(f)], capture_output=True).returncode:
                problems.append(f"{p['name']}: {rel} does not parse under {subprocess.getoutput('node --version')}")
        for target, files in e['skills'].items():
            for rel, digest in files.items():
                f = skills_dir / target / rel
                if not f.is_file() or sha256(f) != digest:
                    problems.append(f"skill {target}: {rel} missing or edited after vendoring")
        for groups in plugin_hooks(p, vendor_dir).values():
            for h in (h for g in groups for h in g['hooks']):
                for m in ROOT_REF.finditer(h.get('command', '')):
                    if m.group(1) and not (vendor_dir / p['name'] / m.group(1).lstrip('/')).exists():
                        problems.append(f"{p['name']}: hook references {m.group(1)}, not vendored")
    if SETTINGS.read_text() != dump(build_settings(registry, vendor_dir)):
        problems.append('.claude/settings.json differs from base + plugin hooks; run sync (edit the base, not the output)')
    for p in problems:
        print('FAIL', p)
    print(f"{'FAILED' if problems else 'ok'}: {len(registry['plugins'])} plugins, {sum(len(e['skills']) for e in lock.values())} skills")
    return 1 if problems else 0


if __name__ == '__main__':
    reg = json.loads(REGISTRY.read_text())
    sys.exit(check(reg) if '--check' in sys.argv[1:] else sync(reg))
