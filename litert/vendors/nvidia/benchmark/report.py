#!/usr/bin/env python3
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
"""Tables from NVIDIA benchmark runs (bench.sh run directories).

Pure data processing: the numbers are the LiteRT-LM CLI's own benchmark report
(prefill and decode turns, speculative decoding statistics, init phases, peak
RSS), GNU time's maximum RSS, the nvidia-smi samples of the memory metric and
the compiler's and dispatch's log lines (AOT cache events, engine builds,
device memory checkpoints).

usage: report.py [--json] [--md] [--check-ci] RUN_DIR...
  RUN_DIR is a bench.sh run directory (holding meta.json) or a directory of
  them. --json prints one JSON document for all runs (also written to each
  run's report.json); --md (the default) prints a markdown table.
  --check-ci fails on incomplete workloads or unexpected cold/hot cache behavior.
"""
import json
import pathlib
import re
import statistics
import sys

GIB = float(1 << 30)


def duration_seconds(text):
  """absl::Duration text (e.g. "1.2s", "27.3ms", "1m2.5s") in seconds."""
  units = {'h': 3600.0, 'm': 60.0, 's': 1.0, 'ms': 1e-3, 'us': 1e-6,
           'µs': 1e-6, 'ns': 1e-9}
  total = 0.0
  for value, unit in re.findall(r'([0-9.]+)(ms|us|µs|ns|h|m|s)', text):
    total += float(value) * units[unit]
  return total


def parse_cli_log(text):
  """What the CLI printed: one BenchmarkInfo block per iteration."""
  iterations = []
  for block in re.split(r'(?=BenchmarkInfo:\n)', text)[1:]:
    item = {'prefill': [], 'decode': [], 'init_phases': {}}
    for phase, ms in re.findall(r'^\s+- ([^:\n]+): ([0-9.]+) ms$', block, re.M):
      item['init_phases'][phase] = float(ms) / 1e3
    m = re.search(r'Time to first token: ([0-9.]+) s', block)
    if m:
      item['time_to_first_token_s'] = float(m[1])
    for kind in ('Prefill', 'Decode'):
      for tokens, dur, speed in re.findall(
          kind + r' Turn \d+: Processed (\d+) tokens in (\S+) duration\.\s*' +
          kind + r' Speed: ([0-9.]+) tokens/sec', block):
        item[kind.lower()].append({'tokens': int(tokens),
                                   'seconds': duration_seconds(dur),
                                   'tokens_per_second': float(speed)})
    m = re.search(r'Acceptance rate:\s*([0-9.]+)% \((\d+) of (\d+) drafted', block)
    if m:
      item['mtp'] = {'acceptance': float(m[1]) / 100.0, 'accepted': int(m[2]),
                     'drafted': int(m[3])}
      m = re.search(r'Rounds:\s*(\d+) \(([0-9.]+)% of decode steps\)', block)
      if m:
        item['mtp'].update(rounds=int(m[1]), round_share=float(m[2]) / 100.0)
      m = re.search(r'Tokens per round:\s*([0-9.]+) \((\d+) emitted\)', block)
      if m:
        item['mtp'].update(tokens_per_round=float(m[1]), emitted=int(m[2]))
      m = re.search(r'Time per round:\s*([0-9.]+) ms \(draft ([0-9.]+) ms, '
                    r'verify ([0-9.]+) ms\)', block)
      if m:
        item['mtp'].update(round_ms=float(m[1]), draft_ms=float(m[2]),
                           verify_ms=float(m[3]))
      m = re.search(r'Plain decodes:\s*(\d+) \(incl\. re-priming\), ([0-9.]+) ms/step',
                    block)
      if m:
        item['mtp'].update(plain_steps=int(m[1]), plain_step_ms=float(m[2]))
    iterations.append(item)
  result = {'iterations': iterations}
  peaks = re.findall(r'Peak system ram usage: ([0-9.]+) ?MB', text)
  if peaks:
    result['cli_peak_rss_gib'] = max(float(x) for x in peaks) * (1 << 20) / GIB
  return result


def parse_vendor_log(text):
  """The compiler plugin's and the dispatch's lines."""
  starts, compile_s = {}, 0.0
  for phase, context, ns in re.findall(
      r'component=compiler phase=(compile_begin|compile_end) context=(\S+) '
      r'monotonic_ns=(\d+)', text):
    if phase == 'compile_begin':
      starts[context] = int(ns)
    elif context in starts:
      compile_s += (int(ns) - starts.pop(context)) / 1e9
  engines = [float(x) for x in re.findall(
      r'Engine generation completed in ([0-9.]+) seconds', text)]
  checkpoints = [int(x) for x in re.findall(
      r'cuda_available=1 cuda_device_used_bytes=(\d+)', text)]
  result = {
      'aot_hits': text.count('AOT cache hit:'),
      'aot_misses': text.count('AOT cache miss:'),
      'compiled_partitions': len(re.findall(r'compiling partition \d+/\d+', text)),
      'compilation_failures': text.count('failed to compile partition'),
      'compiler_phase_s': compile_s or None,
      'engine_builds_s': sum(engines) or None,
      'device_checkpoint_peak_gib': max(checkpoints) / GIB if checkpoints else None,
  }
  arenas = re.findall(r'arenas now ([0-9.]+) GB device, ([0-9.]+) GB host', text)
  if arenas:
    result['moe_arenas_gb'] = {'device': float(arenas[-1][0]),
                               'host': float(arenas[-1][1])}
  m = re.search(r'Captured model output:\s*(.*)', text)
  if m:
    result['answer'] = m[1].strip()
  return result


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
  values = [int(line.split('\t')[1]) for line in text.splitlines() if '\t' in line]
  if not values:
    return {}
  return {'gpu_peak_gib': max(values) * (1 << 20) / GIB,
          'gpu_baseline_gib': baseline_mib * (1 << 20) / GIB,
          'gpu_samples': len(values)}


def summarize(meta, parsed):
  """Means over the measured iterations: the ones after the warm-ups, or the
  last one when a run has fewer iterations than warm-ups + 1 (a first
  iteration in a fresh process pays page faults and graph capture)."""
  its = parsed.get('iterations', [])
  skip = meta['warmups'] if len(its) > meta['warmups'] else max(len(its) - 1, 0)
  measured = its[skip:]
  out = {}
  for kind in ('prefill', 'decode'):
    speeds = [t['tokens_per_second'] for it in measured for t in it[kind]]
    if speeds:
      out[kind + '_tokens_per_second'] = {
          'mean': statistics.mean(speeds), 'min': min(speeds), 'max': max(speeds),
          'per_iteration': speeds}
      out[kind + '_tokens'] = [t['tokens'] for it in measured for t in it[kind]][0]
  mtp = [it['mtp'] for it in measured if 'mtp' in it]
  if mtp:
    out['mtp'] = {k: statistics.mean(m[k] for m in mtp if k in m)
                  for k in ('acceptance', 'tokens_per_round', 'round_ms',
                            'draft_ms', 'verify_ms', 'plain_step_ms')
                  if any(k in m for m in mtp)}
    out['mtp']['rounds'] = sum(m.get('rounds', 0) for m in mtp)
  phases = [it['init_phases'] for it in its if it.get('init_phases')]
  if phases:
    out['init_phases_s'] = phases[0]
  return out


def load_run(run):
  run = pathlib.Path(run)
  record = {'run': str(run), 'parts': {}}
  for meta_path in sorted(run.glob('*meta.json')):
    prefix = meta_path.name[:-len('meta.json')]
    meta = json.loads(meta_path.read_text())
    name = prefix.rstrip('.') or 'main'
    part = {'meta': meta}
    log = run / (prefix + 'cli.log')
    if log.exists():
      text = log.read_text(errors='replace')
      part['cli'] = parse_cli_log(text)
      part['vendor'] = parse_vendor_log(text)
      part['summary'] = summarize(meta, part['cli'])
      if meta['metric'] == 'verify':
        answer = part['vendor'].get('answer', '')
        part['summary']['answer'] = answer
        part['summary']['passed'] = 'paris' in answer.lower()  # as the CLI checks
    t = run / (prefix + 'time.txt')
    if t.exists():
      part['time'] = parse_time(t.read_text(errors='replace'))
    g = run / (prefix + 'gpu.tsv')
    if g.exists():
      part['gpu'] = parse_gpu(g.read_text(errors='replace'), meta['gpu_baseline_mib'])
    record['parts'][name] = part
  if 'main' in record['parts']:
    record['meta'] = record['parts']['main']['meta']
  elif record['parts']:
    record['meta'] = next(iter(record['parts'].values()))['meta']
  return record


def find_runs(paths):
  for p in paths:
    p = pathlib.Path(p)
    if list(p.glob('*meta.json')):
      yield p
    else:
      for child in sorted(p.iterdir()):
        if child.is_dir() and list(child.glob('*meta.json')):
          yield child


def fmt(value, digits=1):
  return '—' if value is None else f'{value:,.{digits}f}'


def markdown(records):
  lines = ['| run | part | metric | PP tokens | PP tok/s | decode tok/s | MTP acceptance | tokens/round | '
           'CPU HWM GiB | GPU peak GiB | total init s | executor s | wall s | compile s | engines s | AOT hit/miss | partitions | answer |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
  for r in records:
    for name, part in r['parts'].items():
      meta = part.get('meta', r.get('meta', {}))
      s, v = part.get('summary', {}), part.get('vendor', {})
      t, g = part.get('time', {}), part.get('gpu', {})
      phases, mtp = s.get('init_phases_s', {}), s.get('mtp', {})
      lines.append('| ' + ' | '.join([
          meta.get('label', pathlib.Path(r['run']).name), name, meta.get('metric', ''),
          fmt(s.get('prefill_tokens'), 0),
          fmt(s.get('prefill_tokens_per_second', {}).get('mean'), 0),
          fmt(s.get('decode_tokens_per_second', {}).get('mean'), 1),
          fmt(mtp['acceptance'] * 100, 0) + '%' if 'acceptance' in mtp else '—',
          fmt(mtp.get('tokens_per_round'), 2),
          fmt(t.get('max_rss_gib'), 2), fmt(g.get('gpu_peak_gib'), 2),
          fmt(phases.get('Init Total'), 2), fmt(phases.get('Init Executor'), 2),
          fmt(t.get('wall_s'), 2), fmt(v.get('compiler_phase_s'), 2),
          fmt(v.get('engine_builds_s'), 2),
          f"{v.get('aot_hits', 0)}/{v.get('aot_misses', 0)}" if v else '—',
          fmt(v.get('compiled_partitions'), 0), s.get('answer', '—')]) + ' |')
  return '\n'.join(lines)


def check_ci(records):
  """Check execution and cache semantics; timings have no pass/fail threshold."""
  by_label = {r.get('meta', {}).get('label'): r for r in records}
  expected = {'12b_128k_cold': (128001, False, True, 1),
              '12b_30k_hot': (30720, False, False, 3),
              '26b_30k_mtp_cold': (29000, True, True, 1)}
  errors = []
  for label, (tokens, mtp, cold, iterations) in expected.items():
    r = by_label.get(label, {})
    p = r.get('parts', {}).get('main', {})
    m, s, v = p.get('meta', {}), p.get('summary', {}), p.get('vendor', {})
    if not p or m.get('exit_code') != 0:
      errors.append(f'{label}: missing or failed main process')
      continue
    if len(p.get('cli', {}).get('iterations', [])) != iterations:
      errors.append(f'{label}: incomplete iteration count')
    if s.get('prefill_tokens', 0) < tokens or s.get('decode_tokens', 0) <= 0:
      errors.append(f'{label}: missing full prefill or post-prefill decode')
    for kind in ('prefill', 'decode'):
      if s.get(kind + '_tokens_per_second', {}).get('mean', 0) <= 0:
        errors.append(f'{label}: missing {kind} speed')
    if s.get('init_phases_s', {}).get('Init Total', 0) <= 0:
      errors.append(f'{label}: missing total initialization time')
    if mtp and 'acceptance' not in s.get('mtp', {}):
      errors.append(f'{label}: missing MTP statistics')
    if v.get('compilation_failures', 0):
      errors.append(f'{label}: compilation failed')
    if cold:
      if not (v.get('aot_misses', 0) and v.get('compiled_partitions', 0)
              and v.get('engine_builds_s', 0)) or v.get('aot_hits', 0):
        errors.append(f'{label}: expected a fresh compiled-artifact pool')
    elif (not v.get('aot_hits', 0) or v.get('aot_misses', 0)
          or v.get('compiled_partitions', 0) or v.get('engine_builds_s')):
      errors.append(f'{label}: expected cache hits without compilation')
  for error in errors:
    print('CI FAIL:', error, file=sys.stderr)
  if not errors:
    print('CI PASS: all three workloads completed with the expected cache state.')
  return not errors


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
  if '--check-ci' in argv and not check_ci(records):
    return 1
  return 0


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
