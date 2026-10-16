"""Archive complete experiment logs on Colab, separately from model artifacts."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time


def collect_logs(content, runs, lane):
    content = Path(content)
    files = {}
    suffixes = {'.json', '.jsonl', '.csv', '.yaml', '.log', '.txt'}
    for run in runs:
        if not run.isalnum():
            raise ValueError('Expected an alphanumeric run ID')
        for root in (content / 'rl-nli/results').glob(run + '_*_s0'):
            for path in root.rglob('*'):
                relative = path.relative_to(content / 'rl-nli')
                if (any(part in relative.parts for part in
                        ('grpo_ckpts', 'sft_ckpts', 'best_adapter', 'final_adapter'))
                        and path.name != 'trainer_state.json'):
                    continue
                if (not path.is_file() or path.is_symlink() or path.suffix not in suffixes
                        or any(word in path.name.casefold() for word in
                               ('credential', 'targets', 'restore_manifest', 'oauth', 'token.json'))):
                    continue
                if any(parent.is_symlink() for parent in path.parents if parent != content):
                    continue
                files[str(relative)] = path
    prefixes = tuple(runs) + ('r13_attempt2_', 'r13_full_logs_', 'native_r13_' + lane,
                              'r14_probe_', 'native_r14_', 'r15_run_', 'native_r15_',
                              'r16_run_', 'native_r16_', 'r17_run_', 'native_r17_',
                              'full_logs_' + lane)
    for path in content.glob('*.log'):
        if path.is_file() and not path.is_symlink() and (
                path.name.startswith(prefixes) or path.name == 'native_r9_cache.log'):
            files['runtime/' + path.name] = path
    for name in ('native_r13_full_tests.json', 'native_r9_cache_ready.json',
                 'native_r13_' + lane + '_status.json'):
        path = content / name
        if path.is_file() and not path.is_symlink():
            files['runtime/' + name] = path
    return dict(sorted(files.items()))


def pack_logs(content, runs, lane, output, *, event):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        return output
    manifest = {'scope': 'logs only; no weights', 'event': event, 'lane': lane,
                'runs': runs, 'captured_at': time.time(), 'files': {},
                'live_snapshot': event != 'terminal'}
    temporary = output.with_suffix(output.suffix + '.tmp')
    with tarfile.open(temporary, 'w:gz', compresslevel=1) as archive:
        for name, source in collect_logs(content, runs, lane).items():
            # Read a live file once so payload and manifest describe identical bytes.
            payload = source.read_bytes()
            entry = tarfile.TarInfo(name)
            entry.size = len(payload)
            entry.mtime = int(time.time())
            archive.addfile(entry, io.BytesIO(payload))
            manifest['files'][name] = {'bytes': len(payload),
                                       'sha256': hashlib.sha256(payload).hexdigest()}
        payload = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
        entry = tarfile.TarInfo('logs_manifest.json')
        entry.size = len(payload)
        archive.addfile(entry, io.BytesIO(payload))
    temporary.replace(output)
    return output


def logs_event_ready(content, main, lane, event):
    content = Path(content)
    if event == 'start':
        return True
    if event == 'terminal':
        status = content / ('native_r13_' + lane + '_status.json')
        if not status.exists():
            return False
        return (json.loads(status.read_text()).get('phase') in ('complete', 'failed')
                and time.time() - status.stat().st_mtime > 3)
    if event.startswith('progress-'):
        path = content / 'rl-nli/results' / (main + '_Qwen3.5-2B_s0/train_log.csv')
        if not path.exists():
            return False
        with path.open() as stream:
            steps = [int(float(row['step'])) for row in csv.DictReader(stream) if row.get('step')]
        return bool(steps) and max(steps) >= int(event.split('-')[1])
    raise ValueError('Unknown full-log archive event')


def watch_logs(content, manifest_path):
    from scripts.remote_artifacts import digest, put_resumable
    content = Path(content)
    manifest = json.loads(Path(manifest_path).read_text())
    lane, runs = manifest['lane'], manifest['runs']
    destination = content / 'native_log_archives'
    destination.mkdir(exist_ok=True)
    while True:
        for event, uri in manifest['targets'].items():
            receipt = destination / (lane + '-' + event + '.receipt.json')
            if receipt.exists():
                continue
            try:
                if not logs_event_ready(content, manifest['main'], lane, event):
                    continue
                archive = pack_logs(content, runs, lane, destination / (lane + '-' + event + '.tar.gz'),
                                    event=event)
                result = put_resumable(archive, uri)
                with tarfile.open(archive) as saved:
                    index = json.load(saved.extractfile('logs_manifest.json'))
                result.update(event=event, lane=lane, sha256=digest(archive, 'sha256'),
                              saved_at=time.time(), log_index=index)
                temporary = receipt.with_suffix('.tmp')
                temporary.write_text(json.dumps(result, ensure_ascii=False))
                temporary.replace(receipt)
                archive.unlink()
                print(json.dumps({'saved_logs': lane, 'event': event, 'bytes': result['size']}), flush=True)
            except Exception as exc:
                # Exception text could contain a private upload URL.
                print(json.dumps({'retry_logs': lane, 'event': event,
                                  'error_type': type(exc).__name__}), flush=True)
        if (destination / (lane + '-terminal.receipt.json')).exists():
            return
        time.sleep(20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', default='/content')
    parser.add_argument('--targets', required=True)
    options = parser.parse_args()
    watch_logs(options.content, options.targets)
