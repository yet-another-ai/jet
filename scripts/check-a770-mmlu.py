#!/usr/bin/env python3
"""Check a detached A770 campaign once; import complete results and stop its timer.

Intended for a systemd user timer running every 30 minutes. No model/API calls.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_run(path, size):
    summary = read_json(path / 'summary.json')
    require(summary['completed_runs'] == 1 and summary['attempted_runs'] == 1,
            f'{path}: expected one successful run')
    require(summary['request_count_per_run'] == 14042, f'{path}: incomplete input')
    run = summary['runs'][0]
    require(run['success'] and run['response_count'] == 14042 and run['failures'] == 0,
            f'{path}: incomplete or failed responses')
    report = read_json(path / 'run-01/report.json')
    require(not report['failures'] and report['overall'] == run['overall'],
            f'{path}: scoring report mismatch')
    require(run['overall']['examples'] == 14042, f'{path}: incomplete scoring')
    require(digest(path / 'run-01/responses.jsonl') == run['responses_sha256'],
            f'{path}: response checksum mismatch')
    provenance = read_json(path / 'provenance.json')
    require(provenance['model_id'] == f'qwen/qwen3.5-{size.lower()}-q8_0',
            f'{path}: wrong model')
    require(provenance['backend'] == 'vulkan' and run['stages']['thinking_ms'] == 0,
            f'{path}: wrong backend or thinking mode')
    log = (path / 'run-01/engine.log').read_text()
    placement = re.search(r'offloaded (\d+)/(\d+) layers to GPU', log)
    require('Arc(tm) A770' in log and placement and placement[1] == placement[2],
            f'{path}: expected full placement on A770')
    for key, filename in [('benchmark_input', 'requests.jsonl'),
                          ('benchmark_gold', 'gold.jsonl')]:
        require(digest(path / filename) == provenance[key]['sha256'],
                f'{path}: {filename} checksum mismatch')
    wall = run['wall_seconds']
    require(wall > 0, f'{path}: invalid wall time')
    return {'size': size, 'path': path, 'provenance': provenance,
            'correct': run['overall']['correct'], 'wall': wall,
            'accuracy': 100 * run['overall']['correct'] / 14042,
            'rate': 14042 / wall}


def paper_updates(results):
    """Prepare all edits in memory, preserving unrelated series and prose."""
    paper = ROOT / 'paper'
    data_path = paper / 'figures/mmlu-results.json'
    rows = read_json(data_path)
    for result in results[2:]:
        group = f'JET / Qwen3.5-{result["size"]}'
        rows = [r for r in rows if not (r['group'] == group and
                r.get('deployment') == 'a770' and r['scope'] == 'mmlu')]
        index = next(i for i, row in enumerate(rows) if row['group'] == group)
        rows.insert(index, {
            'group': group, 'label': 'Arc A770, full MMLU, Vulkan',
            'accuracy': round(result['accuracy'], 4), 'correct': result['correct'],
            'req_s': round(result['rate'], 3), 'wall_s': round(result['wall'], 3),
            'requests': 14042, 'scope': 'mmlu', 'deployment': 'a770',
            'backend': 'Vulkan', 'show_in_figure': True, 'figure_label': 'Arc A770',
            'source': str((result['path'] / 'summary.json').relative_to(ROOT)),
        })
    updates = {data_path: json.dumps(rows, indent=2) + '\n'}
    evaluation = paper / 'sections/evaluation.tex'
    text = evaluation.read_text()
    table = '\n'.join(
        f'Qwen3.5-{r["size"]} & Q8\\_0 & {r["correct"]:,} & '
        f'{r["accuracy"]:.2f} & {r["wall"]:,.2f} & {r["rate"]:.2f} \\\\'
        for r in results)
    text, count = re.subn(r'(\\label\{tab:models\}.*?\\midrule\n).*?(\\bottomrule)',
                         lambda m: m[1] + table + '\n' + m[2], text, flags=re.S)
    require(count == 1, 'Cannot locate the A770 model table')
    text = text.replace('compares two models on the Arc', 'compares four models on the Arc')
    text = text.replace('runs below cover two models', 'runs below cover four models')
    text = text.replace('Arc A770. Both models use', 'Arc A770. All four models use')
    updates[evaluation] = text

    evidence = paper / 'evidence.md'
    text = evidence.read_text()
    start = text.index('## Full-MMLU Arc A770 runs')
    end = text.index('## Full-MMLU RTX 4090 runs', start)
    section = ['## Full-MMLU Arc A770 runs', '',
        'Four Qwen3.5 Q8_0 models were evaluated on September 28--29, 2026.',
        'Each retained run contains 14,042 responses and zero failures, with full GPU',
        'placement and reasoning disabled. All four use the same input, gold, evaluator,',
        'benchmark harness, executable, and runtime flags. The executable includes fixes',
        'for duplicate choice texts and small positive Vulkan log-probabilities.', '',
        '| Model | Correct / 14,042 | Accuracy | Wall (s) | Req./s |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in results:
        section.append(f'| Qwen3.5-{r["size"]} Q8_0 | {r["correct"]:,} | '
                       f'{r["accuracy"]:.2f}% | {r["wall"]:,.2f} | {r["rate"]:.2f} |')
    section += ['', 'Retained artifact directories:', '']
    section += [f'- `{r["path"].relative_to(ROOT)}/`' for r in results]
    section += ['', 'Binary SHA-256: `' + results[0]['provenance']['binary']['sha256'] + '`.',
        'The 4B and 9B model revisions and SHA-256 digests are pinned in',
        '`scripts/download-accuracy-models.sh` (`--qwen35-large`).',
        'An interrupted 9B attempt under `a770-qwen35-9b-mmlu-full-20260929/`',
        'and a partial Qwen3.6-35B-A3B A770 attempt are excluded. The 9B result',
        'comes from a new complete process; partial-run timings are not combined.', '',
        'The manuscript uses these complete-set measurements for Arc A770 model quality and throughput.', '', '']
    text = text[:start] + '\n'.join(section) + text[end:]
    text = text.replace('September 28, and the RTX', 'September 28--29, and the RTX')
    text = text.replace('Two Qwen3.5 Q8_0 models', 'Four Qwen3.5 Q8_0 models')
    text = text.replace('two Arc A770', 'four Arc A770')
    text = text.replace('| Local `tests/accuracy/generated/a770-*-mmlu-full-20260928-fixed/` |',
                        '| Retained directories in the Arc A770 section below |')
    updates[evidence] = text

    readme = paper / 'README.md'
    text = readme.read_text().replace('compares two JET models on an Arc A770',
                                      'compares four JET models on an Arc A770')
    if 'Arc A770 runs were performed September 28--29' not in text:
        text += '\nThe Arc A770 runs were performed September 28--29, 2026; the 4B and 9B additions use the same evaluation configuration as 0.8B and 2B.\n'
    updates[readme] = text
    protocol = paper / 'sections/reproducibility.tex'
    text = protocol.read_text().replace(
        'GPU. Qwen3.5-0.8B and Qwen3.5-2B use full GPU placement,',
        'GPU. Qwen3.5-0.8B, 2B, 4B, and 9B use full GPU placement,')
    text = text.replace('zero failures. Both use the same', 'zero failures. All four use the same')
    text = text.replace('full-set study and an initial CUDA check ran on September 28, followed by the',
        'full-set study ran on September 28--29. An initial CUDA check ran on September 28, followed by the')
    updates[protocol] = text
    return updates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-4b', type=Path, required=True)
    parser.add_argument('--run-9b', type=Path, required=True)
    parser.add_argument('--benchmark-unit', required=True)
    parser.add_argument('--timer-unit', required=True)
    parser.add_argument('--status-file', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    state = {'checked_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        old = read_json(args.status_file) if args.status_file.exists() else {}
        if old.get('status') in ('complete', 'failed'):
            state.update(old)
        else:
            paths = [args.run_4b.resolve(), args.run_9b.resolve()]
            missing = [p for p in paths if not (p / 'summary.json').exists()]
            if missing:
                active = subprocess.run(['systemctl', '--user', 'is-active', '--quiet',
                                         args.benchmark_unit]).returncode == 0
                require(active, 'Benchmark service stopped before producing complete summaries')
                state.update(status='running', responses={p.name:
                    (p / 'run-01/responses.jsonl').read_bytes().count(b'\n')
                    if (p / 'run-01/responses.jsonl').exists() else 0 for p in paths})
            else:
                base = ROOT / 'tests/accuracy/generated'
                results = [load_run(base / f'a770-qwen35-{slug}-mmlu-full-20260928-fixed', size)
                           for slug, size in [('08b', '0.8B'), ('2b', '2B')]]
                results += [load_run(p, size) for p, size in zip(paths, ('4B', '9B'))]
                for result in results[1:]:
                    for key in ('binary', 'benchmark_input', 'benchmark_gold', 'harness', 'evaluator'):
                        require(result['provenance'][key]['sha256'] == results[0]['provenance'][key]['sha256'],
                                f'{result["size"]}: mismatched {key}')
                    for key in ('backend', 'batch_requests', 'extra_args'):
                        require(result['provenance'][key] == results[0]['provenance'][key],
                                f'{result["size"]}: mismatched {key}')
                updates = paper_updates(results)
                if not args.dry_run:
                    for path, text in updates.items():
                        path.write_text(text)
                    subprocess.run([sys.executable, 'paper/figures/generate_mmlu_chart.py'], cwd=ROOT, check=True)
                    subprocess.run(['latexmk', 'main.tex'], cwd=ROOT / 'paper', check=True)
                    subprocess.run(['git', 'diff', '--check'], cwd=ROOT, check=True)
                state.update(status='complete', results=[{k: r[k] for k in
                    ('size', 'correct', 'accuracy', 'wall', 'rate')} for r in results])
    except Exception as error:
        state.update(status='failed', error=str(error))
    print(json.dumps(state, ensure_ascii=False), flush=True)
    if not args.dry_run:
        args.status_file.parent.mkdir(parents=True, exist_ok=True)
        args.status_file.write_text(json.dumps(state, indent=2) + '\n')
        if state['status'] in ('complete', 'failed'):
            subprocess.run(['systemctl', '--user', 'stop', args.timer_unit], check=True)
    return 1 if state['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
