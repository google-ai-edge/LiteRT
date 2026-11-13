#!/usr/bin/env python3
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
"""The data points of the record on native Windows: matrix.sh's counterpart, each one
bench.tool run (and, with --llama, each llama.cpp counterpart one llama/bench.tool run).

usage: matrix.tool OUT_DIR [MODEL...]          (default models: e2b e4b 12b 26b)
  BENCH_GROUPS="context mtp init verify" selects the groups (environment)
  --llama        also run the llama.cpp counterparts (llama/matrix.tool) into OUT_DIR_llama
  --cooldown C   each run starts with the GPU at most C degrees
OUT_DIR under the state root's measured/ directory is what report/tables.tool reads.

  context  plain decoding on synthetic prefill tokens (latency, memory) at the model's workloads
  mtp      plain and speculative decoding on the real prompts of those lengths
  init     a cold start on an empty cache
  verify   the Paris answer on the NPU and on the CPU
"""
import argparse
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def load(name):
  loader = importlib.machinery.SourceFileLoader(name, str(HERE / f'{name}.tool'))
  module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader))
  loader.exec_module(module)
  return module


common = load('common')


def prefetch(path):
  """The model file into the OS file cache before its group (the CLI's embedder reads it per
  token; a file another model's group evicted costs the 26B a third of its decode speed)."""
  with open(path, 'rb') as stream:
    while stream.read(64 << 20):
      pass


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('out')
  parser.add_argument('models', nargs='*')
  parser.add_argument('--llama', action='store_true')
  parser.add_argument('--cooldown', type=int)
  parser.add_argument('--config')
  args = parser.parse_args()
  config = common.load_config(args.config)
  registry = common.registry('models.json')
  workloads = common.registry('workloads.json')
  models = args.models or ['e2b', 'e4b', '12b', '26b']
  groups = os.environ.get('BENCH_GROUPS', 'context mtp init verify').split()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  snapshot = common.check(config)
  (out / 'host-ready.json').write_text(json.dumps(snapshot, indent=1) + '\n')
  if snapshot['problems']:
    raise SystemExit('The host is not ready: ' + '; '.join(snapshot['problems']))
  cooldown = ['--cooldown', str(args.cooldown)] if args.cooldown else []
  prompts = common.prompts_dir(config)
  failed = []

  def bench(*extra):
    command = [sys.executable, str(HERE / 'bench.tool'), '--out', str(out), *cooldown, *extra]
    if args.config:
      command += ['--config', args.config]
    if subprocess.run(command).returncode:
      failed.append(' '.join(extra))

  for model in models:
    spec = registry[model]
    prefetch(common.model_file(config, model))
    for group in groups:
      if group == 'verify':
        bench('--model', model, '--metric', 'verify')
      elif group == 'init':
        bench('--model', model, '--metric', 'cold')
      elif group == 'context':
        for workload in spec['workloads']:
          extra = spec.get('workload_args', {}).get(workload, [])
          bench('--model', model, '--workload', workload, '--metric', 'latency', *extra)
          bench('--model', model, '--workload', workload, '--metric', 'memory', *extra)
        for workload in spec['long_context']:  # real text, context expansion, host-placed experts
          prompt = prompts / (workloads['text']['prompts'].get(workload, 'prompt_' + workload) + '.txt')
          # --cli-arg=VALUE: argparse reads a separate value that starts with -- as another option.
          long = ['--cli-arg=--allow_context_length_expansion=true', *spec.get('long_context_args', {}).get(workload, [])]
          bench('--model', model, '--workload', workload, '--metric', 'latency', '--prompt-file', str(prompt), *long)
          bench('--model', model, '--workload', workload, '--metric', 'memory', '--prompt-file', str(prompt), *long)
      elif group == 'mtp':
        for workload in spec['workloads']:
          prompt = prompts / (workloads['text']['prompts'][workload] + '.txt')
          extra = spec.get('workload_args', {}).get(workload, [])
          bench('--model', model, '--workload', workload, '--metric', 'latency', '--prompt-file', str(prompt),
                '--iterations', str(workloads['text']['plain_iterations']), '--label', f'{model}_{workload}_plain_text',
                *extra)
          if spec.get('mtp'):
            bench('--model', model, '--workload', workload, '--metric', 'latency', '--prompt-file', str(prompt),
                  '--iterations', str(workloads['text']['mtp_iterations']), '--mtp', '--label', f'{model}_{workload}_mtp_text',
                  *extra)
  if args.llama:
    command = [sys.executable, str(HERE.parent / 'llama' / 'matrix.tool'), str(out) + '_llama', *models, *cooldown]
    if args.config:
      command += ['--config', args.config]
    if subprocess.run(command).returncode:
      failed.append('llama matrix')
  if failed:
    print('Failed runs:\n  ' + '\n  '.join(failed))
    sys.exit(1)
  print(f'Matrix complete: {out}')


if __name__ == '__main__':
  main()
