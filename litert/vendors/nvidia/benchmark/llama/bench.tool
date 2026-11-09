#!/usr/bin/env python3
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
"""One llama.cpp measurement on native Windows: llama/bench.sh's counterpart, the same models,
workloads, metrics and flags, the same run directory, so llama/report.tool reads both alike.

usage: bench.tool --model e2b|e4b|12b|26b|qwen27b --out DIR [options]
  --workload short|8k|16k|30k|128k   (default short)
  --metric latency|memory|cold       latency: llama-bench prefill of the whole prompt and decode
                                     at that depth, or with --prompt-file llama-server completions;
                                     memory: one prefill+decode pass (llama-bench -pg) under GPU
                                     sampling; cold: server start to ready twice (the model's file
                                     pages stay as the OS keeps them: Windows evicts nothing)
  --mtp  --prompt-file FILE  --iterations N  --label NAME  --sampling greedy|model  --arg ARG
  --output-tokens N  --cooldown C  --config FILE   (windows/config.json's "llama" section)
Output: DIR/LABEL/{meta.json, pp.json+tg.json | mem.json | client.N.{json,txt}+server/ |
  cold/+warm/, time.json, gpu.tsv}.
"""
import argparse
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent


def load(name, directory):
  loader = importlib.machinery.SourceFileLoader(name, str(directory / f'{name}.tool'))
  module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader))
  loader.exec_module(module)
  return module


common = load('common', ROOT / 'windows')
PRESETS = {'short': (1024, 256, 2048, 5), '8k': (8192, 256, 9216, 3), '16k': (16384, 256, 17408, 3),
           '30k': (30720, 256, 31744, 3), '128k': (128001, 128, 130048, 2)}


class Server:
  """A llama-server under GPU sampling: start, wait for /health, stop (its own shutdown)."""

  def __init__(self, command, handle, env, cwd, port, sample):
    handle.mkdir(parents=True, exist_ok=True)
    self.handle, self.port = handle, port
    self.sampler = common.Sampler(handle / 'gpu.tsv', 0.1) if sample else None
    self.log = (handle / 'server.log').open('wb')
    self.started = time.perf_counter()
    self.process = subprocess.Popen(command, cwd=cwd, env=env, stdout=self.log, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    self.ready_s = None
    deadline = self.started + 900
    while time.perf_counter() < deadline and self.process.poll() is None:
      try:
        urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2).read()
        self.ready_s = time.perf_counter() - self.started
        break
      except OSError:
        time.sleep(0.2)
    (handle / 'serve.json').write_text(json.dumps({'command': command, 'pid': self.process.pid, 'ready_s': self.ready_s,
                                                   'elapsed_clock': 'QueryPerformanceCounter'}, indent=1) + '\n')

  def stop(self):
    if self.process.poll() is None:
      try:
        self.process.send_signal(signal.CTRL_BREAK_EVENT)
        self.process.wait(timeout=30)
      except (OSError, subprocess.TimeoutExpired):
        self.process.terminate()
        self.process.wait()
    memory = common.process_memory(self.process)
    self.log.close()
    if self.sampler:
      self.sampler.stop()
    (self.handle / 'time.json').write_text(json.dumps(dict(
        {'wall_s': time.perf_counter() - self.started, 'exit_code': 0, 'stopped_by_runner': True}, **memory), indent=1) + '\n')
    return memory


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--model', required=True)
  parser.add_argument('--out', required=True)
  parser.add_argument('--workload', default='short', choices=list(PRESETS))
  parser.add_argument('--metric', default='latency', choices=['latency', 'memory', 'cold'])
  parser.add_argument('--mtp', action='store_true')
  parser.add_argument('--prompt-file')
  parser.add_argument('--iterations', type=int)
  parser.add_argument('--output-tokens', type=int)
  parser.add_argument('--label')
  parser.add_argument('--sampling', choices=['greedy', 'model'])
  parser.add_argument('--arg', action='append', default=[])
  parser.add_argument('--cooldown', type=int)
  parser.add_argument('--port', type=int, default=8089)
  parser.add_argument('--config')
  args = parser.parse_args()
  if os.name != 'nt':
    raise SystemExit('bench.tool runs on native Windows; on Linux use bench.sh')
  config = common.load_config(args.config)
  llama = config['llama']
  registry = common.registry('models.json')
  spec = llama['models'][args.model]
  model = pathlib.Path(spec['main'])
  draft = pathlib.Path(spec['draft']) if spec.get('draft') else None
  if args.mtp and not draft:
    raise SystemExit(f'{args.model} has no MTP drafter GGUF')
  if args.mtp and not args.prompt_file and args.metric != 'cold':
    raise SystemExit('--mtp needs --prompt-file (llama-bench has no speculative mode)')
  binary = pathlib.Path(llama['bin'])
  inputs, output, context, default_iterations = PRESETS[args.workload]
  if args.output_tokens:
    output = args.output_tokens
  iterations, warmups = (args.iterations or default_iterations), 1
  if args.metric != 'latency':
    iterations, warmups = 1, 0
  profile = args.model
  sampling = args.sampling or registry.get(profile, {}).get('sampler', 'model')
  mode = 'mtp' if args.mtp else 'plain'
  label = args.label or f'{profile}_{args.workload}_{mode}_{args.metric}'
  run = pathlib.Path(args.out) / label
  run.mkdir(parents=True, exist_ok=True)
  ubatch = int(os.environ.get('LLAMA_UBATCH', 512 if profile == '26b' else 1024))
  load_mode = os.environ.get('LLAMA_LOAD_MODE', 'none' if profile == '26b' else 'mmap')
  threads = str(llama.get('threads', 20))
  model_args = ['-fa', 'on', '-ngl', '99', '-t', threads, '-b', '2048', '-ub', str(ubatch)]
  if profile == '26b':
    model_args += ['-ncmoe', '0']
  state = pathlib.Path(config['state_root'])
  env = dict(os.environ)
  env['PATH'] = os.pathsep.join([str(binary), *config.get('sdk_bin_dirs', []), env.get('PATH', '')])
  env['CUDA_CACHE_DISABLE'] = '0'
  env['CUDA_CACHE_PATH'] = str(state / 'cache' / 'llama' / 'cuda')
  pathlib.Path(env['CUDA_CACHE_PATH']).mkdir(parents=True, exist_ok=True)
  busy = common.busy_processes()
  if busy:
    raise SystemExit('Another benchmark process is running: ' + '; '.join(busy))
  cooldown = common.cooldown(args.cooldown) if args.cooldown else None
  prompt_file = pathlib.Path(args.prompt_file).resolve() if args.prompt_file else None
  baseline = common.gpu_memory_used_mib()
  host = common.host_snapshot()
  start = common.utc()
  commands, timings, rc = [], {}, 0

  def llama_bench(name, sample, flags):
    command = [str(binary / 'llama-bench.exe'), '-m', str(model), *model_args, '-lm', load_mode, '-lzm', 'off', '-o', 'json',
               *flags, *args.arg]
    commands.append(' '.join(command))
    print(f'{label} ({name}): {" ".join(command)[:240]}', flush=True)
    timing = common.run_process(command, run / f'{name}.log', env, str(binary),
                                sample_path=run / 'gpu.tsv' if sample else None, sample_interval=0.1, stdin_text=None)
    # llama-bench writes its JSON to stdout, which run_process merged into the log: split it out.
    text = (run / f'{name}.log').read_text(encoding='utf-8', errors='replace')
    begin, end = text.find('['), text.rfind(']')
    (run / f'{name}.json').write_text(text[begin:end + 1] if begin >= 0 and end > begin else '')
    timings[name] = timing
    return timing['exit_code']

  def completions(n, prefix):
    status = 0
    for i in range(1, n + 1):
      command = [sys.executable, str(HERE / 'client.tool'), '--prompt-file', str(prompt_file), '--out', f'{prefix}.{i}',
                 '--n-predict', str(output), '--port', str(args.port), '--sampling', sampling, '--seed', str(i)]
      status = subprocess.run(command, env=env).returncode or status
    return status

  def server_command():
    command = [str(binary / 'llama-server.exe'), '-m', str(model), *model_args, '-lm', load_mode, '-c', str(context),
               '-np', '1', '--port', str(args.port), '--no-warmup']
    if args.mtp:
      command += ['--spec-type', 'draft-mtp', '-md', str(draft), '--spec-draft-n-max', '3', '-ngld', '99']
    return command + args.arg

  def serve(state_name, n_completions, sample):
    nonlocal rc
    server = Server(server_command(), run / state_name, env, str(binary), args.port, sample)
    commands.append(' '.join(server_command()))
    try:
      if server.ready_s is None:
        rc = 1
        print(f'{label}: the server did not become ready; see {run / state_name / "server.log"}', flush=True)
      elif n_completions:
        prefix = run / 'client' if state_name == 'server' else run / state_name / 'client'
        rc = completions(n_completions, prefix) or rc
    finally:
      timings[state_name] = server.stop()

  if args.metric == 'latency':
    if prompt_file is None:
      rc = llama_bench('pp', False, ['-p', str(inputs), '-n', '0', '-d', '0', '-r', str(iterations)]) or rc
      rc = llama_bench('tg', False, ['-p', '0', '-n', str(output), '-d', str(inputs), '-r', str(iterations)]) or rc
    else:
      serve('server', iterations, True)
  elif args.metric == 'memory':
    if prompt_file is None:
      rc = llama_bench('mem', True, ['-pg', f'{inputs},{output}', '-p', '0', '-n', '0', '-d', '0', '-r', '1']) or rc
    else:
      serve('server', 1, True)
  else:
    for state_name in ('cold', 'warm'):
      serve(state_name, 1 if prompt_file else 0, False)
  end = common.utc()
  peak = max((t.get('peak_working_set_bytes', 0) for t in timings.values()), default=0)
  (run / 'time.json').write_text(json.dumps({'wall_s': sum(t.get('wall_s', 0) for t in timings.values()),
                                             'peak_working_set_bytes': peak,
                                             'peak_pagefile_bytes': max((t.get('peak_pagefile_bytes', 0) for t in timings.values()), default=0),
                                             'processes': timings}, indent=1) + '\n')
  meta = {'label': label, 'runtime': 'llama.cpp', 'profile': profile, 'model': str(model), 'draft': str(draft) if draft else None,
          'workload': args.workload, 'metric': args.metric, 'mtp': args.mtp, 'prompt_file': str(prompt_file) if prompt_file else None,
          'input_tokens': inputs, 'output_tokens': output, 'context': context, 'iterations': iterations, 'warmups': warmups,
          'build_id': (llama.get('commit') or '?')[:9], 'release_tag': llama.get('release_tag'), 'load_mode': load_mode,
          'physical_batch_tokens': ubatch, 'sampling': sampling, 'commands': commands, 'exit_code': rc, 'start': start, 'end': end,
          'gpu_baseline_mib': baseline, 'host_mode': os.environ.get('CI_HOST_MODE', config.get('host_mode', 'unspecified')),
          'platform': config.get('platform'), 'host': host, 'cooldown': cooldown, 'runner': 'llama/bench.tool',
          'binaries': {name: common.sha256(binary / name) for name in ('llama-bench.exe', 'llama-server.exe') if (binary / name).exists()}}
  (run / 'meta.json').write_text(json.dumps(meta, indent=1) + '\n')
  print(f'Run: {run} (exit {rc})')
  sys.exit(rc)


if __name__ == '__main__':
  main()
