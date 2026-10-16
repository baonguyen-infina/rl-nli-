"""Build a small, credential-free handoff from existing files; never run ML/network.

Run with an existing Python that has PyYAML. Only standard-library modules,
PyYAML and the configuration-only modules common.io/configs.matrix are imported.
No experiment is launched, no predictions are generated, and no weights copied.
"""
import csv
import datetime
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from configs.matrix import RUNS, build_run_config
from common.io import load_config

NOW = datetime.datetime.now(datetime.timezone.utc).isoformat()
COPIES = []
SKIPPED = []
SECRET_KEYS = {
    'access_token', 'refresh_token', 'client_secret', 'private_key',
    'id_token', 'session_token', 'authorization', 'capabilities',
    'colab-runtime-proxy-token', 'upload_capability', 'download_capability',
}
SECRET_PATTERNS = [
    re.compile(r'4/0A[A-Za-z0-9_-]{20,}'),
    re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}'),
    re.compile(r'github_pat_[A-Za-z0-9_]{20,}'),
    re.compile(r'ya29\.[A-Za-z0-9_-]{20,}'),
    re.compile(r'-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----'),
]


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(128 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def save(path, value):
    path = HERE / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def scrub(value):
    if isinstance(value, dict):
        return {k: '[REDACTED]' if k.casefold() in SECRET_KEYS else scrub(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, str):
        for pattern in SECRET_PATTERNS:
            value = pattern.sub('[REDACTED]', value)
        value = re.sub(
            r'(https?://[^\s"<>]*(?:colab-runtime-proxy-token|access_token|refresh_token|'
            r'X-Goog-Signature|X-Amz-Signature)=[^\s"<>]*)',
            '[REDACTED_SIGNED_URL]', value, flags=re.I)
    return value


def copy_small(source, relative):
    source = pathlib.Path(source)
    if source.is_symlink() or not source.is_file():
        return
    if source.stat().st_size > 1024 * 1024:
        SKIPPED.append({'source': str(source), 'reason': 'over 1MiB metadata/source limit'})
        return
    target = HERE / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = source.read_bytes()
    try:
        text = payload.decode('utf-8')
    except UnicodeDecodeError:
        cleaned = payload
    else:
        if source.suffix == '.json':
            try:
                value = json.loads(text)
                transformed = scrub(value)
                cleaned = (json.dumps(transformed, ensure_ascii=False, indent=2) + '\n').encode()
                changed = transformed != value
            except json.JSONDecodeError:
                cleaned = scrub(text).encode()
                changed = cleaned != payload
        else:
            cleaned = scrub(text).encode()
            changed = cleaned != payload
    target.write_bytes(cleaned)
    COPIES.append({'source': str(source), 'copy': str(target.relative_to(HERE)),
                   'source_sha256': hashlib.sha256(payload).hexdigest(),
                   'copied_sha256': sha(target), 'bytes': len(cleaned),
                   'redacted': locals().get('changed', False)})


def copy_sources_and_evidence():
    for folder in ['common', 'configs', 'data', 'formats', 'rewards', 'train',
                   'eval', 'scripts', 'analysis', 'teacher', 'tests', 'docs', 'paper']:
        for source in sorted((ROOT / folder).rglob('*')):
            if '__pycache__' in source.parts or not source.is_file():
                continue
            allowed = {'.py', '.yaml', '.yml', '.md', '.tex', '.bib', '.sty', '.bst', '.json'}
            if source.suffix in allowed or str(source.relative_to(ROOT)) == 'paper/main.pdf':
                # No data JSONL, model files, generated previews or dependency trees.
                copy_small(source, pathlib.Path('snapshot/repo') / source.relative_to(ROOT))
    for name in ['AGENTS.md', 'CLAUDE.md', 'README.md', 'requirements.txt', '.gitignore']:
        copy_small(ROOT / name, pathlib.Path('snapshot/repo') / name)
    for folder in ['analysis', 'protocols']:
        for source in sorted((ROOT / 'artifacts' / folder).rglob('*')):
            if source.is_file() and source.suffix in {'.json', '.csv', '.md', '.yaml', '.py'}:
                copy_small(source, pathlib.Path('evidence/internal') / folder /
                           source.relative_to(ROOT / 'artifacts' / folder))
    for source in sorted((ROOT / 'docs').rglob('*.md')):
        copy_small(source, pathlib.Path('evidence/internal/docs') / source.relative_to(ROOT / 'docs'))
    for source in sorted((ROOT / 'results').rglob('*')):
        if source.is_file() and source.suffix in {'.json', '.yaml', '.csv'}:
            if any(part in ('best_adapter', 'final_adapter') for part in source.parts):
                continue
            copy_small(source, pathlib.Path('evidence/results_metadata') /
                       source.relative_to(ROOT / 'results'))
    for source in sorted(set(pathlib.Path('/private/tmp').glob('native_r20*.py')) |
                         set(pathlib.Path('/private/tmp').glob('native_github*.py'))):
        copy_small(source, pathlib.Path('operators') / source.name)


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def registry():
    outputs = {}
    for out in sorted((ROOT / 'results').iterdir()):
        if not out.is_dir():
            continue
        match = re.match(r'(.+)_Qwen[^/]*_s(\d+)$', out.name)
        cfg = yaml.safe_load((out / 'config.yaml').read_text()) if (out / 'config.yaml').exists() else {}
        rid = cfg.get('run_id') or (match[1] if match else out.name)
        cost = read_json(out / 'cost.json')
        checkpoints = read_json(out / 'ckpt_scores.json')
        item = {'path': str(out.relative_to(ROOT)), 'seed': int(match[2]) if match else None,
                'cost_status': cost.get('status'), 'selected_checkpoint': checkpoints.get('best'),
                'available_metric_files': [str(x.relative_to(out)) for x in sorted(out.rglob('metrics_*.json'))],
                'test_receipt_status': read_json(out / '.test_claim/receipt.json').get('status')}
        if (out / 'train_log.csv').exists():
            with (out / 'train_log.csv').open() as f:
                logs = list(csv.DictReader(f))
            steps = [int(float(x['step'])) for x in logs if x.get('step')]
            item['last_recorded_train_step'] = max(steps) if steps else None
        outputs.setdefault(rid, []).append(item)
    known_remote = {
        'N10R13': 'attempt1 interrupted; attempt2 train200 observed, final selection requires archive reconciliation',
        'N11R13': 'attempt2 verified dev checkpoint100; no test evidence',
        'N12R15': 'train400 observed; full final selection interrupted; recovered checkpoint200 readout',
        'N13R16': 'interrupted; recovered checkpoint100 readout',
        'N14R17': 'interrupted historical saliency arm; no completed test evidence',
        'N15R18': 'completed dev selection and test; private final archive recorded verified',
        'N16R19': 'completed dev selection and test; private final archive recorded verified',
        'N20R20': 'deliberately interrupted at observed109; checkpoint100 backed up; final logs/release unverified; no test allowed',
        'N21R20': 'checkpoint100 verified/backed up; checkpoint200 observer launched; current remote state unknown after auth loss',
    }
    names = sorted(set(RUNS) | {p.stem for p in (ROOT / 'configs/runs').glob('*.yaml')} | set(outputs))
    records = []
    for name in names:
        cfgpath = ROOT / 'configs/runs' / (name + '.yaml')
        cfg = load_config(name) if cfgpath.exists() else build_run_config(name) if name in RUNS else {}
        filecfg = yaml.safe_load(cfgpath.read_text()) if cfgpath.exists() else None
        fresh = filecfg == build_run_config(name) if name in RUNS and filecfg is not None else None
        role = ('smoke/gate' if name.startswith('G0') else 'frozen inference view' if name.startswith('H')
                else 'repeat/readout validation' if name.startswith('V') else 'diagnostic' if name == 'P0R14'
                else 'training or zero-shot candidate')
        evidence = ('recorded remote evidence; see status note' if name in known_remote
                    else 'local output metadata present; completion scope must be checked' if outputs.get(name)
                    else 'configuration only in this inventory; launch not established')
        records.append({'run_id': name, 'registered_in_matrix': name in RUNS,
                        'yaml_exists': cfgpath.exists(), 'yaml_matches_current_matrix': fresh,
                        'role': role, 'recipe': cfg.get('recipe'), 'format': cfg.get('format'),
                        'training_format': cfg.get('training_format'), 'native_mode': cfg.get('native_mode'),
                        'init_from': cfg.get('init_from'), 'init_checkpoint': cfg.get('init_checkpoint'),
                        'model': cfg.get('model'), 'native_grouping': cfg.get('native_grouping'),
                        'seeds_configured': cfg.get('seeds'), 'native_schedule': cfg.get('native_schedule'),
                        'eval_backend': cfg.get('eval', {}).get('backend', 'vllm'),
                        'effective_native_settings': cfg.get('native_pilot') if cfg.get('recipe') == 'native' else None,
                        'evidence_level': evidence, 'remote_status_note': known_remote.get(name),
                        'local_outputs': outputs.get(name, [])})
    save('catalog/experiment_registry.json', {'generated_at_utc': NOW, 'rows': records,
         'warning': 'Configured does not mean executed. Local metadata can be partial/stale; no current remote queries were made.'})
    fields = ['run_id', 'role', 'recipe', 'format', 'native_mode', 'init_from', 'native_schedule',
              'yaml_matches_current_matrix', 'evidence_level', 'remote_status_note']
    with (HERE / 'catalog/experiment_registry.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(records)
    lines = ['# Complete configuration/evidence registry', '',
             'No development metric in this table is a paper result. Config-only rows are not asserted to have run.', '',
             '| ID | Role | Recipe / mode | Parent | Evidence |', '|---|---|---|---|---|']
    for row in records:
        lines.append('| ' + ' | '.join(str(x or '—').replace('|', '/') for x in [
            row['run_id'], row['role'], str(row['recipe']) + ' / ' + str(row['native_mode']),
            row['init_from'], row['remote_status_note'] or row['evidence_level']]) + ' |')
    (HERE / 'catalog/experiment_registry.md').write_text('\n'.join(lines) + '\n')
    return records


def test_results():
    results = []
    checks = []
    for folder in sorted((ROOT / 'results').iterdir()):
        path = folder / 'metrics_test.json'
        if not path.exists():
            continue
        metrics = read_json(path)
        receipt = read_json(folder / '.test_claim/receipt.json')
        cost = read_json(folder / 'cost_test.json')
        assert receipt['status'] == 'completed' and cost['status'] == 'complete', folder
        for name, expected in receipt['artifacts'].items():
            assert sha(folder / name) == expected, (folder, name)
        prediction_path = folder / 'predictions_test.jsonl'
        with prediction_path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert len(rows) == len({r['id'] for r in rows}) == metrics['n'] == receipt['n'] == 1000
        f1 = []
        for label in ['entailment', 'neutral', 'contradiction']:
            tp = sum(r['gold'] == label and r['pred'] == label for r in rows)
            fp = sum(r['gold'] != label and r['pred'] == label for r in rows)
            fn = sum(r['gold'] == label and r['pred'] != label for r in rows)
            f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
        recalculated = {'macro_f1': sum(f1) / 3,
                        'accuracy': sum(r['gold'] == r['pred'] for r in rows) / len(rows),
                        'parse_rate': sum(bool(r['parse_ok']) for r in rows) / len(rows)}
        assert all(abs(metrics[k] - recalculated[k]) < 1e-12 for k in recalculated), folder
        cfg = yaml.safe_load((folder / 'config.yaml').read_text())
        entry = {'run_id': cfg['run_id'], 'split': 'test', 'seed': 0, 'n': metrics['n'],
                 **{k: metrics[k] for k in recalculated}, 'format': cfg['format'],
                 'native_mode': cfg.get('native_mode'), 'init_from': cfg.get('init_from'),
                 'evidence': 'local predictions recomputed and receipt hashes freshly checked',
                 'source': str(folder.relative_to(ROOT)),
                 'predictions_sha256': sha(prediction_path)}
        results.append(entry); checks.append({'run': cfg['run_id'], 'local_prediction_and_receipt_check': True})
    for run, name in [('N15R18', 'r18_remote_status.json'), ('N16R19', 'r19_remote_status.json')]:
        status = read_json(ROOT / 'artifacts/analysis' / name)
        summary = status['test_state']['summary']
        receipt_path = ROOT / 'artifacts/analysis/github_backup_receipts' / (run + '_final.json')
        receipt = read_json(receipt_path)
        assert summary['split'] == 'test' and summary['n'] == 1000
        assert receipt['repo_private'] and receipt['release_draft']
        assert receipt['github_digest'] == 'sha256:' + receipt['sha256']
        results.append({**summary, 'run_id': run, 'format': 'label',
                        'evidence': 'recorded remote prediction verification plus final private archive receipt; not re-fetched',
                        'source': 'artifacts/analysis/' + name, 'archive_asset_id': receipt['id'],
                        'archive_sha256': receipt['sha256']})
    save('catalog/test_results.json', {'generated_at_utc': NOW, 'target_macro_f1': .58,
         'target_met_by_any_recorded_result': any(r['macro_f1'] >= .58 for r in results), 'rows': results,
         'note': 'All results are seed0 exploratory evidence. Readout/control differences are retained, not pooled.'})
    fields = ['run_id', 'split', 'seed', 'n', 'macro_f1', 'accuracy', 'parse_rate', 'format',
              'native_mode', 'init_from', 'evidence', 'source']
    with (HERE / 'catalog/test_results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(results)
    return checks, results


def archives():
    values = []
    for source in sorted((ROOT / 'artifacts/analysis/github_backup_receipts').glob('*.json')):
        value = read_json(source)
        values.append({'receipt_source': str(source.relative_to(ROOT)), **scrub(value)})
    save('catalog/github_archive_index.json', {'generated_at_utc': NOW, 'fresh_remote_query': False,
         'warning': 'Recorded receipts only. Revalidate ownership/draft/digest through GitHub before restore/release.',
         'assets': values})


def main():
    copy_sources_and_evidence()
    records = registry()
    checks, results = test_results()
    archives()
    for args, dest in [(['git', 'status', '--short'], 'git_status.txt'),
                       (['git', 'diff', '--no-ext-diff'], 'tracked_changes.patch'),
                       (['git', 'rev-parse', 'HEAD'], 'git_head.txt')]:
        command = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True)
        (HERE / 'catalog' / dest).write_text(scrub(command.stdout))
    save('catalog/copy_inventory.json', {'generated_at_utc': NOW, 'files': COPIES, 'skipped': SKIPPED})
    save('CURRENT_STATE.json', {'recorded_at_utc': NOW, 'live_remote_state_verified': False,
         'blocker': 'Google reauthentication required', 'latest_effective_target': 'ViANLI test Macro-F1 >=0.58',
         'model': 'Qwen/Qwen3.5-2B', 'teacher': False, 'max_g4_sessions': 3,
         'n20': 'deliberate SIGINT at observed109; cp100 archived; final logs and session release unverified',
         'n21': 'cp100 archived; observer200 launched; actual current train/test status unknown',
         'goal_status_last_verified': 'blocked', 'no_new_remote_or_ml_operations_for_this_handoff': True})
    assert not any(x in sys.modules for x in ['torch', 'transformers', 'trl', 'vllm', 'numpy'])
    paper = ROOT / 'paper/main.pdf'
    qa = read_json(ROOT / 'paper/visual_qa.json')
    assert sha(paper) == qa['pdf_sha256']
    forbidden = []
    secret_matches = []
    for path in sorted(HERE.rglob('*')):
        if not path.is_file():
            continue
        if path.suffix in {'.safetensors', '.pt', '.pth', '.bin', '.ckpt', '.tar', '.gz', '.pem', '.key'}:
            forbidden.append(str(path.relative_to(HERE)))
        if path.suffix != '.pdf':
            text = path.read_text(errors='replace')
            if any(p.search(text) for p in SECRET_PATTERNS):
                secret_matches.append(str(path.relative_to(HERE)))
    assert not forbidden and not secret_matches, (forbidden, secret_matches)
    broken_links = []
    for doc in HERE.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
            if '://' in target or target.startswith('#'):
                continue
            local = target.split('#', 1)[0]
            if local and not (doc.parent / local).exists():
                broken_links.append({'document': doc.name, 'target': local})
    assert not broken_links, broken_links
    registry_ids = {r['run_id'] for r in records}
    assert set(RUNS).issubset(registry_ids)
    assert {'N15R18', 'N16R19', 'N20R20', 'N21R20'}.issubset(registry_ids)
    save('VALIDATION.json', {'checked_at_utc': NOW, 'local_test_outputs_checked': checks,
         'recorded_remote_test_rows': 2, 'total_test_rows': len(results),
         'registered_or_observed_run_ids': len(records), 'files_copied': len(COPIES),
         'redacted_copies': sum(x['redacted'] for x in COPIES), 'oversize_files_not_copied': len(SKIPPED),
         'paper_pdf_sha256_matches_recorded_visual_qa': True, 'fresh_pdf_render_or_compile': False,
         'forbidden_weight_or_key_files': forbidden, 'literal_secret_pattern_matches': secret_matches,
         'broken_top_level_markdown_links': broken_links, 'all_current_matrix_ids_catalogued': True,
         'ml_libraries_imported': False, 'new_gpu_jobs_started': False,
         'remote_state_revalidated': False, 'source_status': 'uncommitted workspace snapshot; see catalog/git_status.txt'})
    entries = {str(path.relative_to(HERE)): {'bytes': path.stat().st_size, 'sha256': sha(path)}
               for path in sorted(HERE.rglob('*')) if path.is_file() and path.name != 'MANIFEST.json'}
    save('MANIFEST.json', {'created_at_utc': NOW, 'files': entries,
                          'total_payload_bytes': sum(x['bytes'] for x in entries.values()),
                          'note': 'Manifest excludes itself. No checkpoint/model weights or auth credentials included.'})
    print(json.dumps({'folder': str(HERE), 'run_ids': len(records), 'test_rows': len(results),
                      'files': len(entries) + 1, 'payload_bytes': sum(x['bytes'] for x in entries.values()),
                      'weights_copied': False, 'ml_executed': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
