#!/usr/bin/env python3
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
"""The measured runs of one platform: loading, validity and the newest run per table cell.

A run directory is what bench.sh (LiteRT) or llama/bench.sh (llama.cpp) writes, on Linux or
Windows: meta.json (or <part>.meta.json), cli.log, pp/tg/mem.json or client.N.json, time.txt
(GNU time) or time.json (the Windows runner), gpu.tsv. The LiteRT runs are parsed by LiteRT's
own benchmark/report.py, the llama.cpp runs by llama/report.tool; this file adds the Windows
time.json, the validity rules and the cell selection, and nothing platform-specific.

Cells: (profile, workload, kind) where kind is one of latency, memory, memory_none, text_plain,
text_mtp, cold, verify for LiteRT and the same for llama.cpp. The newest valid run of a cell,
by its start time, is the one the tables show; older runs and diagnostics under other roots
are not read. A run is valid when its process exited 0, it ran every iteration its meta asks
for, and (LiteRT) its log shows the NVIDIA dispatch executing a fully delegated graph.
"""
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import re
import statistics

HERE = pathlib.Path(__file__).resolve().parent
GIB = float(1 << 30)


def load_module(path, name):
  """A module from any path, .py or .tool (importlib needs the loader named for the latter)."""
  loader = importlib.machinery.SourceFileLoader(name, str(path))
  spec = importlib.util.spec_from_loader(name, loader)
  module = importlib.util.module_from_spec(spec)
  loader.exec_module(module)
  return module


def parsers(bench_dir=None):
  """(LiteRT's report module, llama's report module). LITERT_NVIDIA_BENCHMARK or --bench-dir
  names LiteRT's benchmark directory; by default this file's grandparent (the layout inside
  litert/vendors/nvidia/benchmark/), then $LITERT_RT_DIR."""
  candidates = [bench_dir, os.environ.get('LITERT_NVIDIA_BENCHMARK'), HERE.parent,
                os.environ.get('LITERT_RT_DIR') and
                pathlib.Path(os.environ['LITERT_RT_DIR']) / 'litert/vendors/nvidia/benchmark']
  for candidate in candidates:
    if candidate and (pathlib.Path(candidate) / 'report.py').exists():
      litert = load_module(pathlib.Path(candidate) / 'report.py', 'litert_report')
      break
  else:
    raise SystemExit('LiteRT benchmark/report.py not found; set LITERT_NVIDIA_BENCHMARK')
  llama = load_module(HERE.parent / 'llama' / 'report.tool', 'llama_report')
  return litert, llama


def find_runs(roots):
  """Every run directory under the roots (a root may itself be a run)."""
  for root in roots:
    root = pathlib.Path(root)
    if not root.is_dir():
      continue
    seen = set()
    for path in sorted(root.rglob('*meta.json')):
      if path.parent.name != 'prepare' and path.parent not in seen:  # one run per directory, whatever its parts
        seen.add(path.parent)
        yield path.parent


def read_time_json(run, prefix=''):
  path = run / (prefix + 'time.json')
  if not path.exists():
    return {}
  t = json.loads(path.read_text())
  out = {}
  if t.get('wall_s') is not None:
    out['wall_s'] = float(t['wall_s'])
  rss = t.get('peak_working_set_bytes') or t.get('max_rss_bytes')
  if rss:
    out['max_rss_gib'] = rss / GIB
  if t.get('peak_pagefile_bytes'):
    out['peak_commit_gib'] = t['peak_pagefile_bytes'] / GIB
  return out


def nvidia_execution(text):
  """The log of a LiteRT run shows the NVIDIA path executed (a CLI can exit 0 after falling
  back to the CPU): no compiler failure, every op of the Gemma graph delegated, a dispatch
  context initialized."""
  errors = []
  if 'failed to compile partition' in text.lower():
    errors.append('NVIDIA compiler failure')
  selections = re.findall(r'Partitioned subgraph<\d+>, selected (\d+) ops, from a total of (\d+) ops', text)
  if any(selected != total for selected, total in selections):
    errors.append('graph only partially delegated')
  if not re.search(r'NVIDIA dispatch context \S+ (?:uses|shares) ', text):
    errors.append('no NVIDIA dispatch context')
  return errors


def load(run, litert_report, llama_report, models=None):
  """One record: {'run', 'runtime', 'meta', 'parts', 'errors'} in the shape of the parsers'
  records, with the Windows time.json folded into parts[*]['time']. With models (models.json),
  a llama.cpp run of a file other than its profile's comparison model is invalid."""
  run = pathlib.Path(run)
  metas = sorted(run.glob('*meta.json'))
  main_meta = json.loads(next((m for m in metas if m.name == 'meta.json'), metas[0]).read_text())
  if main_meta.get('runtime') == 'llama.cpp':
    record = llama_report.load_run(run)
    record['runtime'] = 'llama'
    part = record['parts']['main']
    part['time'] = part.get('time') or read_time_json(run)
    errors = [] if main_meta.get('exit_code', 0) == 0 else [f"exit {main_meta.get('exit_code')}"]
    expected = (models or {}).get(main_meta.get('profile'), {}).get('llama', {}).get('files')
    if expected:
      name = re.split(r'[\\/]', str(main_meta.get('model') or ''))[-1]
      if name not in expected:
        errors.append(f"llama.cpp model {name} is not the profile's comparison model ({expected[0]})")
  else:
    record = litert_report.load_run(run)
    record['runtime'] = 'litert'
    errors = []
    for name, part in record['parts'].items():
      prefix = '' if name == 'main' else name + '.'
      part['time'] = part.get('time') or read_time_json(run, prefix)
      if name == 'prepare':  # bench.sh's preparation invocation: one iteration, never reported
        continue
      meta = part['meta']
      if meta.get('exit_code', 0) != 0:
        errors.append(f"{name}: exit {meta.get('exit_code')}")
      if meta.get('backend', 'npu') != 'cpu' and name in ('main', 'npu'):
        text = (run / (prefix + 'cli.log')).read_text(errors='replace') if (run / (prefix + 'cli.log')).exists() else ''
        errors += nvidia_execution(text)
      if meta['metric'] == 'latency' and name in ('main', 'npu'):
        done = len(part.get('cli', {}).get('iterations', []))
        if done != meta['iterations']:
          errors.append(f"{name}: {done} of {meta['iterations']} iterations")
      # A prompt beyond the model's context leaves every iteration empty, and the CLI exits 0.
      if (meta['metric'] in ('latency', 'memory') and name in ('main', 'npu')
          and not part.get('summary', {}).get('decode_tokens_per_second')):
        errors.append(f'{name}: no decode turn measured')
  record['errors'] = errors
  return record


def cell(record):
  """(profile, workload, kind) of a record, or None for a run the tables do not show."""
  m = record['meta']
  metric, text, mtp = m.get('metric'), bool(m.get('prompt_file')), bool(m.get('mtp'))
  if metric == 'latency':
    kind = ('text_mtp' if mtp else 'text_plain') if text else 'latency'
  elif metric == 'memory':
    if mtp:  # an MTP memory run (a pool's preparation on Windows) is not a table cell
      return None
    kind = 'memory_none' if m.get('load_mode') == 'none' else 'memory'
  elif metric in ('cold', 'verify'):
    kind = metric
  else:
    return None
  if kind == 'latency' and mtp:
    return None
  return (m['profile'], m['workload'], kind)


def select(records):
  """The newest valid record per (runtime, cell)."""
  chosen = {}
  for record in records:
    if record['errors']:
      continue
    key = cell(record)
    if key is None:
      continue
    key = (record['runtime'],) + key
    if key not in chosen or record['meta'].get('start', '') > chosen[key]['meta'].get('start', ''):
      chosen[key] = record
  return chosen


def summary(record):
  return record['parts'].get('main', {}).get('summary', {})


def mean_of(values):
  values = [v for v in values if v is not None]
  return statistics.mean(values) if values else None
