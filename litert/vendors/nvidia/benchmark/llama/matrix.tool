#!/usr/bin/env python3
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
"""The llama.cpp data points of the record on native Windows: llama/matrix.sh's counterpart,
each one llama/bench.tool run.

usage: matrix.tool OUT_DIR [MODEL...] [--cooldown C] [--config FILE]   (default models: e2b e4b 12b 26b)
  LLAMA_GROUPS="context mtp init" selects the groups (environment)
  context  llama-bench at the model's workloads: prefill of the whole prompt, decode at that
           depth (latency), one prefill+decode pass under GPU sampling (memory: with the model
           mapped, and without mmap for the host-memory column)
  mtp      llama-server completions of the real prompts, plain and with the MTP drafter
  init     the server's start to ready, twice
"""
import argparse
import importlib.machinery
import importlib.util
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent


def load(name, directory):
  loader = importlib.machinery.SourceFileLoader(name, str(directory / f'{name}.tool'))
  module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader))
  loader.exec_module(module)
  return module


common = load('common', ROOT / 'windows')


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('out')
  parser.add_argument('models', nargs='*')
  parser.add_argument('--cooldown', type=int)
  parser.add_argument('--config')
  args = parser.parse_args()
  config = common.load_config(args.config)
  registry = common.registry('models.json')
  workloads = common.registry('workloads.json')
  prompts = common.prompts_dir(config)
  models = args.models or ['e2b', 'e4b', '12b', '26b']
  groups = os.environ.get('LLAMA_GROUPS', 'context mtp init').split()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  failed = []

  def bench(*extra, env=None):
    command = [sys.executable, str(HERE / 'bench.tool'), '--out', str(out), *extra]
    if args.cooldown:
      command += ['--cooldown', str(args.cooldown)]
    if args.config:
      command += ['--config', args.config]
    if subprocess.run(command, env=dict(os.environ, **(env or {}))).returncode:
      failed.append(' '.join(extra))

  for model in models:
    spec = registry[model]
    for group in groups:
      if group == 'init':
        bench('--model', model, '--metric', 'cold')
      elif group == 'context':
        for workload in spec['workloads']:
          bench('--model', model, '--workload', workload, '--metric', 'latency')
          bench('--model', model, '--workload', workload, '--metric', 'memory')
          if model != '26b':  # the 26B already runs without mmap
            bench('--model', model, '--workload', workload, '--metric', 'memory', '--label', f'{model}_{workload}_plain_memory_none',
                  env={'LLAMA_LOAD_MODE': 'none'})
      elif group == 'mtp':
        for workload in spec['workloads']:
          prompt = prompts / (workloads['text']['prompts'][workload] + '.txt')
          bench('--model', model, '--workload', workload, '--metric', 'latency', '--prompt-file', str(prompt),
                '--iterations', str(workloads['text']['plain_iterations'] + 1), '--label', f'{model}_{workload}_plain_text')
          if spec.get('mtp') and config['llama']['models'][model].get('draft'):
            bench('--model', model, '--workload', workload, '--metric', 'latency', '--prompt-file', str(prompt),
                  '--iterations', str(workloads['text']['plain_iterations'] + 1), '--mtp', '--label', f'{model}_{workload}_mtp_text')
  if failed:
    print('Failed runs:\n  ' + '\n  '.join(failed))
    sys.exit(1)
  print(f'llama.cpp matrix complete: {out}')


if __name__ == '__main__':
  main()
