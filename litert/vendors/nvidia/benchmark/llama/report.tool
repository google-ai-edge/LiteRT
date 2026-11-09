#!/usr/bin/env python3
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
"""Tables from llama.cpp benchmark runs (llama/bench.sh and llama/bench.tool run directories).

Pure data processing: the numbers are llama-bench's JSON (avg_ts of each
test), llama-server's completion timings (prompt and generation tokens per
second, draft tokens and their acceptance), the server's start-to-ready time,
GNU time's maximum RSS and the nvidia-smi samples. The records have the shape
of the LiteRT benchmark's report.py records (meta, parts.main.summary / time /
gpu), so bench/tables.py reads both the same way.

usage: report.tool [--json] [--md] RUN_DIR...
  RUN_DIR is a bench.sh run directory (holding meta.json) or a directory of
  them. --json prints one JSON document for all runs (also written to each
  run's report.json); --md (the default) prints a markdown table.
"""
import json
import pathlib
import re
import statistics
import sys

GIB = float(1 << 30)


def parse_time(text):
  out = {}
  m = re.search(r'Maximum resident set size \(kbytes\): (\d+)', text)
  if m:
    out['max_rss_gib'] = int(m[1]) * 1024 / GIB
  m = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)', text)
  if m:
    parts = [float(p) for p in m[1].split(':')]
    out['wall_s'] = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
  return out


def parse_gpu(text, baseline_mib):
  values = [int(line.split('\t')[1]) for line in text.splitlines()
            if '\t' in line and line.split('\t')[1].strip().isdigit()]
  if not values:
    return {}
  return {'gpu_peak_gib': max(values) * (1 << 20) / GIB,
          'gpu_baseline_gib': baseline_mib * (1 << 20) / GIB,
          'gpu_samples': len(values)}


def llama_bench_tests(path):
  """llama-bench -o json: one record per test (n_prompt, n_gen, n_depth, avg_ts...)."""
  try:
    data = json.loads(path.read_text() or 'null')
  except json.JSONDecodeError:
    return []
  if isinstance(data, dict):
    data = [data]
  return [d for d in (data or []) if isinstance(d, dict) and 'avg_ts' in d]


def measured(items, meta):
  """The completions after the warm-ups, or the last one when a run has fewer
  completions than warm-ups + 1 (the LiteRT report's rule)."""
  skip = meta['warmups'] if len(items) > meta['warmups'] else max(len(items) - 1, 0)
  return items[skip:]


def load_run(run):
  run = pathlib.Path(run)
  meta = json.loads((run / 'meta.json').read_text())
  summary, detail = {}, {}
  # llama-bench: pp.json / tg.json (latency), mem.json (memory).
  for name in ('pp', 'tg', 'mem'):
    path = run / f'{name}.json'
    if path.exists():
      tests = llama_bench_tests(path)
      detail[name] = tests
      for t in tests:
        speeds = t.get('samples_ts') or [t['avg_ts']]
        # A run that asks llama-bench for warm-up repetitions (the Windows matrix: meta
        # prefill_warmups / decode_warmups) discards them; llama-bench's avg_ts covers them all.
        warmups = meta.get('decode_warmups' if name == 'tg' else 'prefill_warmups', 0) or 0
        if warmups and len(speeds) > warmups:
          speeds = speeds[warmups:]
          mean = statistics.mean(speeds)
        else:
          mean = t['avg_ts']
        entry = {'mean': mean, 'stddev': t.get('stddev_ts'), 'min': min(speeds),
                 'max': max(speeds), 'per_iteration': speeds, 'depth': t.get('n_depth'),
                 'excluded_warmup_repetitions': warmups}
        if t.get('n_prompt', 0) > 0 and t.get('n_gen', 0) == 0:
          summary['prefill_tokens_per_second'] = entry
          summary['prefill_tokens'] = t['n_prompt']
        elif t.get('n_gen', 0) > 0 and t.get('n_prompt', 0) == 0:
          summary['decode_tokens_per_second'] = entry
          summary['decode_tokens'] = t['n_gen']
        else:  # a -pg test: prompt + generation in one measurement
          summary['prompt_and_generation_tokens_per_second'] = entry
  # llama-server completions: client.N.json.
  clients = []
  for path in sorted(run.glob('client.*.json'), key=lambda p: int(p.name.split('.')[1])):
    clients.append(json.loads(path.read_text()))
  clients = [c for c in clients if 'timings' in c]
  if clients:
    detail['completions'] = clients
    use = measured(clients, meta)
    pp = [c['timings']['prompt_per_second'] for c in use if c['timings'].get('prompt_per_second')]
    tg = [c['timings']['predicted_per_second'] for c in use if c['timings'].get('predicted_per_second')]
    if pp:
      summary['prefill_tokens_per_second'] = {'mean': statistics.mean(pp), 'min': min(pp),
                                              'max': max(pp), 'per_iteration': pp}
      summary['prefill_tokens'] = use[0]['timings'].get('prompt_n')
    if tg:
      summary['decode_tokens_per_second'] = {'mean': statistics.mean(tg), 'min': min(tg),
                                             'max': max(tg), 'per_iteration': tg}
      summary['decode_tokens'] = statistics.mean(c['timings']['predicted_n'] for c in use)
      summary['stop_types'] = [c.get('stop_type') for c in use]
    drafted = sum(c['timings'].get('draft_n', 0) or 0 for c in use)
    accepted = sum(c['timings'].get('draft_n_accepted', 0) or 0 for c in use)
    if drafted:
      summary['mtp'] = {'acceptance': accepted / drafted, 'drafted': drafted, 'accepted': accepted}
  # Server start to ready: server/serve.json (text runs), cold/ and warm/ (cold metric).
  init = {}
  for name in ('server', 'cold', 'warm'):
    path = run / name / 'serve.json'
    if path.exists():
      s = json.loads(path.read_text())
      init[name + '_ready_s'] = s.get('ready_s')
  if init:
    summary['init'] = init
  # Keep the first completion of each cold/warm process separate from steady-state results.
  for state in ('cold', 'warm'):
    path = run / state / 'client.1.json'
    if path.exists():
      client = json.loads(path.read_text())
      detail[state + '_completion'] = client
      if 'timings' in client:
        summary[state + '_completion'] = client['timings']
  # GNU time: the largest RSS over the run's processes; the wall of the main one.
  time_info = {}
  for path in sorted(run.glob('time.*.txt')) + sorted(run.glob('*/time.txt')):
    t = parse_time(path.read_text(errors='replace'))
    if 'max_rss_gib' in t:
      time_info['max_rss_gib'] = max(time_info.get('max_rss_gib', 0.0), t['max_rss_gib'])
    if 'wall_s' in t:
      time_info['wall_s'] = time_info.get('wall_s', 0.0) + t['wall_s']
  gpu = {}
  for path in [run / 'gpu.tsv'] + sorted(run.glob('*/gpu.tsv')):
    if path.exists():
      g = parse_gpu(path.read_text(errors='replace'), meta.get('gpu_baseline_mib') or 0)
      if g and g['gpu_peak_gib'] >= gpu.get('gpu_peak_gib', 0.0):
        gpu = g
  part = {'meta': meta, 'summary': summary, 'time': time_info, 'gpu': gpu, 'llama': detail}
  return {'run': str(run), 'meta': meta, 'parts': {'main': part}}


def find_runs(paths):
  for p in paths:
    p = pathlib.Path(p)
    if (p / 'meta.json').exists():
      yield p
    else:
      for child in sorted(p.iterdir()):
        if child.is_dir() and (child / 'meta.json').exists():
          yield child


def fmt(value, digits=1):
  return '—' if value is None else f'{value:,.{digits}f}'


def markdown(records, *, include_rss=True):
  def row(cells):
    if not include_rss:
      cells = cells[:7] + cells[8:]
    return '| ' + ' | '.join(cells) + ' |'

  lines = ['| run | metric | PP tokens | PP tok/s | decode tok/s | MTP acceptance | gen tokens | '
           + ('CPU HWM GiB | ' if include_rss else '')
           + 'GPU peak GiB | ready s (cold/warm) | process wall s |',
           '|---|---|' + '---:|' * (9 if include_rss else 8)]
  for r in records:
    meta = r['meta']
    s = r['parts']['main']['summary']
    t = r['parts']['main']['time']
    g = r['parts']['main']['gpu']
    init = s.get('init', {})
    ready = '—'
    if 'cold_ready_s' in init or 'warm_ready_s' in init:
      ready = f"{fmt(init.get('cold_ready_s'))} / {fmt(init.get('warm_ready_s'))}"
    elif 'server_ready_s' in init:
      ready = fmt(init['server_ready_s'])
    mtp = s.get('mtp', {})
    lines.append(row([
        meta['label'], meta['metric'], fmt(s.get('prefill_tokens'), 0),
        fmt(s.get('prefill_tokens_per_second', {}).get('mean'), 0),
        fmt(s.get('decode_tokens_per_second', {}).get('mean'), 1),
        fmt(mtp['acceptance'] * 100, 0) + '%' if 'acceptance' in mtp else '—',
        fmt(s.get('decode_tokens'), 0),
        fmt(t.get('max_rss_gib'), 2), fmt(g.get('gpu_peak_gib'), 2), ready,
        fmt(t.get('wall_s'), 2)]))
    for state in ('cold', 'warm'):
      c = s.get(state + '_completion')
      if c:
        lines.append(row([
            meta['label'], state + ' first completion', fmt(c.get('prompt_n'), 0), fmt(c.get('prompt_per_second'), 0),
            fmt(c.get('predicted_per_second'), 1), '—', fmt(c.get('predicted_n'), 0),
            '—', '—', fmt(init.get(state + '_ready_s'), 2), '—']))
  return '\n'.join(lines)


def main(argv):
  want_json = '--json' in argv
  paths = [a for a in argv if not a.startswith('--')]
  if not paths:
    print(__doc__)
    return 2
  records = []
  for run in find_runs(paths):
    record = load_run(run)
    (run / 'report.json').write_text(json.dumps(record, indent=1) + '\n')
    records.append(record)
  if want_json:
    print(json.dumps(records, indent=1))
  if '--md' in argv or not want_json:
    print(markdown(records))
  return 0


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
