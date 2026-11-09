#!/usr/bin/env python3
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
"""The performance tables of one platform from its measured runs.

usage: tables.tool --platform ID --runs DIR... [--out DIR] [--bench-dir DIR] [--sources FILE]

Reads every run under the --runs roots (runs.tool), keeps the newest valid run of every
cell, and writes one Markdown file per table family that has runs on this platform
(models.json: gemma4_dense.md, gemma4_moe.md, qwen3_8_27b.md) into --out/<platform>/
(default: google/ next to this file's parent). The files hold only what the runs say:
a generated provenance line and the tables. --sources writes the run behind every cell.

The tables, as the WSL record defined them: decode and prefill speed by input workload
against llama.cpp, memory by workload (GPU peak, host peak), real-text plain and MTP decode
with draft acceptance, start-up (cold and warm), the answer check.
"""
import argparse
import datetime
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = None  # runs.tool, loaded in main()
CONTEXT_NAME = {'short': '1k', '8k': '8k', '16k': '16k', '30k': '30k', '32k': '32k', '128k': '128k', '256k': '256k'}


def fmt(value, digits=1, unit=''):
  return '—' if value is None else f'{value:,.{digits}f}{unit}'


def ratio(a, b):
  return f'{a / b:.2f}x' if a and b else '—'


def reduction(a, b):
  return f'{(1 - a / b) * 100:.1f}%' if a and b else '—'


def row(cells):
  return '| ' + ' | '.join(cells) + ' |'


def head(cells):
  return [row(cells), '|' + '---|' * len(cells)]


class Tables:
  """The cells of one platform: chosen[(runtime, profile, workload, kind)] -> record."""

  def __init__(self, chosen, models, platform):
    self.chosen, self.models, self.platform = chosen, models, platform
    self.sources = {}

  def record(self, runtime, profile, workload, kind):
    return self.chosen.get((runtime, profile, workload, kind))

  def value(self, runtime, profile, workload, kind, getter):
    record = self.record(runtime, profile, workload, kind)
    if record is None:
      return None
    value = getter(record)
    if value is not None:
      self.sources[f'{runtime} / {profile} / {workload} / {kind}'] = record['run']
    return value

  # The speed getters: synthetic latency first, the real-text plain run where a workload is
  # measured on text only (the 26B's long contexts).
  def speed(self, runtime, profile, workload, kind):
    for source in ('latency', 'text_plain'):
      v = self.value(runtime, profile, workload, source,
                     lambda r: RUNS.summary(r).get(kind + '_tokens_per_second', {}).get('mean'))
      if v is not None:
        return v
    return None

  def gpu(self, runtime, profile, workload):
    for source in ('memory', 'memory_none'):
      v = self.value(runtime, profile, workload, source, lambda r: r['parts']['main'].get('gpu', {}).get('gpu_peak_gib'))
      if v is not None:
        return v
    return None

  def host_memory(self, runtime, profile, workload):
    # llama.cpp's host figure comes from the run without mmap (a mapped model counts its file
    # pages); LiteRT's from its memory run.
    sources = ('memory_none', 'memory') if runtime == 'llama' else ('memory',)
    for source in sources:
      v = self.value(runtime, profile, workload, source, lambda r: r['parts']['main'].get('time', {}).get('max_rss_gib'))
      if v is not None:
        return v
      if runtime == 'llama':
        break
    return None

  def text(self, runtime, profile, workload, kind, key):
    return self.value(runtime, profile, workload, kind, lambda r: RUNS.summary(r).get(key, {}).get('mean') if key != 'acceptance'
                      else RUNS.summary(r).get('mtp', {}).get('acceptance'))

  def inputs(self, runtime, profile, workload):
    values = set()
    for kind in ('text_plain', 'text_mtp'):
      r = self.record(runtime, profile, workload, kind)
      if r is not None and RUNS.summary(r).get('prefill_tokens') is not None:
        values.add(RUNS.summary(r)['prefill_tokens'])
    return ' / '.join(fmt(v, 0) for v in sorted(values)) or '—'

  def init(self, profile, state):
    kind, workload = ('cold', 'short') if state == 'cold' else ('latency', 'short')
    r = self.record('litert', profile, workload, kind)
    if r is None:
      return None
    part = r['parts']['main']
    self.sources[f'litert / {profile} / {state} start-up'] = r['run']
    return {'compile': part.get('vendor', {}).get('compiler_phase_s'),
            'engines': part.get('vendor', {}).get('engine_builds_s'),
            'init': part.get('summary', {}).get('init_phases_s', {}).get('Init Executor'),
            'wall': part.get('time', {}).get('wall_s')}

  def verify(self, profile):
    r = self.record('litert', profile, 'short', 'verify')
    if r is None:
      return None
    self.sources[f'litert / {profile} / verify'] = r['run']
    answers = {name: p.get('summary', {}).get('answer', '') for name, p in r['parts'].items()}
    return {name: 'paris' in answer.lower() for name, answer in answers.items()}

  def workloads(self, profile):
    spec = self.models[profile]
    return [w for w in spec['workloads'] + spec['long_context']
            if any(key[1:3] == (profile, w) for key in self.chosen)]

  def present(self, profiles):
    return [p for p in profiles if any(key[1] == p for key in self.chosen)]

  def render(self, table):
    profiles = self.present(table['models'])
    with_model = len(table['models']) > 1
    name = lambda p: [self.models[p]['name']] if with_model else []
    model_head = ['Model'] if with_model else []
    cpu = self.platform['cpu_memory']
    decode = head(model_head + ['Input workload', 'TensorRT-RTX (tokens/s)', 'llama.cpp (tokens/s)', 'Speedup'])
    prefill = list(decode)
    memory = head(model_head + ['Input workload', 'TensorRT-RTX GPU (GiB)', 'llama.cpp GPU (GiB)', 'GPU reduction',
                                f'TensorRT-RTX CPU {cpu} (GiB)', f'llama.cpp CPU {cpu} (GiB)'])
    mtp = head(model_head + ['Input workload', 'Runtime', 'Actual input tokens', 'Plain (tokens/s)', 'MTP (tokens/s)',
                             'MTP speedup', 'Draft acceptance'])
    init = head(['Model / cache state', 'Compiler phase (s)', 'Engine builds (s)', 'Init executor (s)', 'Whole process (s)'])
    checks = []
    for profile in profiles:
      for workload in self.workloads(profile):
        ctx = CONTEXT_NAME[workload]
        d, ld = self.speed('litert', profile, workload, 'decode'), self.speed('llama', profile, workload, 'decode')
        p, lp = self.speed('litert', profile, workload, 'prefill'), self.speed('llama', profile, workload, 'prefill')
        if d is not None or ld is not None:
          decode.append(row(name(profile) + [ctx, fmt(d, 1), fmt(ld, 1), ratio(d, ld)]))
          prefill.append(row(name(profile) + [ctx, fmt(p, 0), fmt(lp, 0), ratio(p, lp)]))
        g, lg = self.gpu('litert', profile, workload), self.gpu('llama', profile, workload)
        c, lc = self.host_memory('litert', profile, workload), self.host_memory('llama', profile, workload)
        if any(v is not None for v in (g, lg, c, lc)):
          memory.append(row(name(profile) + [ctx, fmt(g, 2), fmt(lg, 2), reduction(g, lg), fmt(c, 2), fmt(lc, 2)]))
        if workload in self.models[profile]['long_context']:
          continue  # measured on text only; shown as decode/prefill rows, not as MTP rows
        for runtime, label in (('litert', 'TensorRT-RTX'), ('llama', 'llama.cpp')):
          plain = self.text(runtime, profile, workload, 'text_plain', 'decode_tokens_per_second')
          spec = self.text(runtime, profile, workload, 'text_mtp', 'decode_tokens_per_second')
          if plain is None and spec is None:
            continue
          acc = self.text(runtime, profile, workload, 'text_mtp', 'acceptance')
          mtp.append(row(name(profile) + [ctx, label, self.inputs(runtime, profile, workload), fmt(plain, 0), fmt(spec, 0),
                                          ratio(spec, plain), fmt(acc * 100 if acc is not None else None, 0, '%')]))
      for state in ('cold', 'warm'):
        i = self.init(profile, state)
        if i:
          skipped = 'Skipped' if state == 'warm' else '—'
          init.append(row([f"{self.models[profile]['name']} {state}", fmt(i['compile'], 1) if i['compile'] else skipped,
                           fmt(i['engines'], 1) if i['engines'] else skipped, fmt(i['init'], 1), fmt(i['wall'], 1)]))
      passed = self.verify(profile)
      if passed is not None:
        parts = ', '.join(f"{'passed' if ok else 'FAILED'} on the {n.upper() if n != 'main' else 'NPU'}" for n, ok in sorted(passed.items()))
        checks.append(f"{self.models[profile]['name']}: {parts}")
    sections = [f"# {table['title']}", '', self.provenance(profiles), '',
                '## 1. Decode, prefill and memory', '',
                '### 1.1 Decode speed by input workload', '', *decode, '',
                '### 1.2 Prefill speed by input workload', '', *prefill, '',
                '### 1.3 Memory by input workload', '', *memory, '']
    if len(mtp) > 2:
      sections += ['## 2. MTP: decode speed and draft acceptance', '', *mtp, '']
    if len(init) > 2:
      sections += ['## 3. Start-up time', '', *init, '']
    if checks:
      sections += ['## 4. Answer check', '', *('- ' + c for c in checks), '']
    return '\n'.join(sections)

  def provenance(self, profiles):
    records = [r for key, r in self.chosen.items() if key[1] in profiles]
    starts = sorted(r['meta']['start'][:10] for r in records if r['meta'].get('start'))
    builds = {}
    for r in records:
      m = r['meta']
      name = 'LiteRT/LiteRT-LM' if r['runtime'] == 'litert' else 'llama.cpp'
      build = m.get('binaries_build_id') or m.get('build_id') or '?'
      if r['runtime'] == 'llama' and m.get('release_tag'):
        build = f"{m['release_tag']} {build}"
      builds.setdefault(name, set()).add(build)
    text = ', '.join(f"{name} {' / '.join(sorted(ids))}" for name, ids in sorted(builds.items()))
    when = starts[0] if not starts or starts[0] == starts[-1] else f'{starts[0]} to {starts[-1]}'
    return (f"{self.platform['name']}. Builds: {text}. Measured {when}; {len(records)} runs. "
            'Columns and their method: README.md.')


def main():
  global RUNS
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--platform', required=True)
  parser.add_argument('--runs', nargs='+', required=True, metavar='DIR')
  parser.add_argument('--out', default=str(ROOT / 'google'))
  parser.add_argument('--bench-dir')
  parser.add_argument('--sources', help='write the run directory behind every cell to this JSON file')
  args = parser.parse_args()
  import importlib.machinery, importlib.util
  loader = importlib.machinery.SourceFileLoader('runs', str(HERE / 'runs.tool'))
  RUNS = importlib.util.module_from_spec(importlib.util.spec_from_loader('runs', loader))
  loader.exec_module(RUNS)
  platforms = json.loads((ROOT / 'platforms.json').read_text())
  models = json.loads((ROOT / 'models.json').read_text())
  if args.platform not in platforms:
    raise SystemExit(f'Unknown platform {args.platform}; platforms.json names ' + ', '.join(k for k in platforms if k != '_'))
  litert_report, llama_report = RUNS.parsers(args.bench_dir)
  records = [RUNS.load(run, litert_report, llama_report) for run in RUNS.find_runs(args.runs)]
  chosen = RUNS.select(records)
  invalid = [(r['run'], r['errors']) for r in records if r['errors']]
  out = pathlib.Path(args.out) / args.platform
  out.mkdir(parents=True, exist_ok=True)
  tables = Tables(chosen, models, platforms[args.platform])
  written = []
  for key, table in models['tables'].items():
    if not tables.present(table['models']):
      continue
    (out / table['file']).write_text(tables.render(table) + '\n', newline='\n')
    written.append(table['file'])
  if args.sources:
    pathlib.Path(args.sources).write_text(json.dumps(
        {'platform': args.platform, 'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'cells': tables.sources, 'invalid_runs': invalid}, indent=1) + '\n')
  print(f"{args.platform}: {len(records)} runs read, {len(chosen)} cells, {len(invalid)} invalid; wrote "
        + ', '.join(str(out / w) for w in written))
  for run, errors in invalid:
    print(f'  invalid: {run}: {"; ".join(errors)}')


if __name__ == '__main__':
  main()
