#!/usr/bin/env python3
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
"""One LiteRT measurement on native Windows: bench.sh's counterpart, the same flags, the
same CLI invocation, the same run directory (meta.json, cli.log, gpu.tsv; time.json in place
of GNU time), so report/tables.tool reads both alike.

usage: bench.tool --model e2b|e4b|12b|26b|qwen27b|FILE --out DIR [options]
  --workload short|8k|16k|30k|32k|128k|256k   (default short; workloads.json)
  --metric latency|memory|verify|cold         (default latency)
  --mtp                                       speculative decoding with the model's drafter
  --prompt-file FILE                          real text instead of synthetic tokens
  --context N  --iterations N  --label NAME  --pool DIR  --output-tokens N
  --cli-arg ARG  --env NAME=VALUE             (repeatable; after the workload's)
  --cooldown C                                wait until the GPU is at most C degrees first
  --config FILE                               windows/config.json by default
Output: DIR/LABEL/{cli.log,time.json,gpu.tsv,meta.json} (+ prepare.* and npu./cpu. parts).
"""
import argparse
import datetime
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent


def load(name):
  loader = importlib.machinery.SourceFileLoader(name, str(HERE / f'{name}.tool'))
  module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader))
  loader.exec_module(module)
  return module


common = load('common')


def runtime_environment(config, pool, model, policy):
  """env.sh's nb_runtime_env: the TensorRT-RTX settings of one model cache pool, the SDK and
  runtime directories on PATH."""
  env = {k: v for k, v in os.environ.items() if not k.startswith('LITERT_NVIDIA_')}
  for name in ('aot', 'runtime', 'compiler/cuda', 'cpu'):
    (pool / name).mkdir(parents=True, exist_ok=True)
  env.update({
      'LITERT_NVIDIA_TENSORRT_PARTITION_POLICY': policy,
      'LITERT_NVIDIA_TENSORRT_FP16_ACTIVATIONS': 'bf16',
      'LITERT_NVIDIA_TENSORRT_PREDEQUANTIZE_FC_WEIGHTS': 'cuda_gemv',
      'LITERT_NVIDIA_TENSORRT_SHARED_WEIGHTS': '1',
      'LITERT_NVIDIA_TENSORRT_WEIGHT_STORE': '1',
      'LITERT_NVIDIA_TENSORRT_JIT_HANDLE': '0',
      'LITERT_NVIDIA_TENSORRT_AOT_MODEL_PATH': str(model),
      'LITERT_NVIDIA_TENSORRT_AOT_CACHE_DIR': str(pool / 'aot'),
      'LITERT_NVIDIA_DISPATCH_RUNTIME_CACHE_DIR': str(pool / 'runtime'),
      'LITERT_NVIDIA_DISPATCH_SHARED_WEIGHT_SEGMENTS': '1',
      'LITERT_NVIDIA_DISPATCH_LAZY_AOT_ENGINES': '0',
      'CUDA_CACHE_DISABLE': '0', 'CUDA_CACHE_PATH': str(pool / 'compiler/cuda'),
  })
  env.pop('NVIDIA_TF32_OVERRIDE', None)
  env['PATH'] = os.pathsep.join([config['runtime_dir'], *config.get('sdk_bin_dirs', []), env.get('PATH', '')])
  return env


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--model', required=True)
  parser.add_argument('--out', required=True)
  parser.add_argument('--workload', default='short')
  parser.add_argument('--metric', default='latency', choices=['latency', 'memory', 'verify', 'cold'])
  parser.add_argument('--mtp', action='store_true')
  parser.add_argument('--prompt-file')
  parser.add_argument('--context', type=int)
  parser.add_argument('--iterations', type=int)
  parser.add_argument('--output-tokens', type=int)
  parser.add_argument('--label')
  parser.add_argument('--pool')
  parser.add_argument('--cli-arg', action='append', default=[])
  parser.add_argument('--env', action='append', default=[])
  parser.add_argument('--cooldown', type=int)
  parser.add_argument('--config')
  args = parser.parse_args()
  if os.name != 'nt':
    raise SystemExit('bench.tool runs on native Windows; on Linux use bench.sh')

  config = common.load_config(args.config)
  models = common.registry('models.json')
  workloads = common.registry('workloads.json')
  if args.workload not in workloads or args.workload == 'text':
    raise SystemExit(f'Unknown workload {args.workload}')
  preset = workloads[args.workload]
  model = common.model_file(config, args.model).resolve()
  profile = args.model if args.model in models else model.stem
  spec = models.get(profile, {})
  if args.mtp and spec and not spec.get('mtp', True):
    raise SystemExit(f'{profile} has no MTP drafter')
  cli, libs = common.runtime_files(config)
  state = pathlib.Path(config['state_root'])

  inputs, output, context = preset['input'], preset['output'], args.context or preset['context']
  iterations, warmups, prefill = args.iterations or preset['iterations'], preset['warmups'], preset['prefill']
  if args.output_tokens:
    output = args.output_tokens
  if args.mtp and warmups < 2:
    warmups = 2  # the decode engine's CUDA-graph capture falls in the second conversation
  if args.metric in ('memory', 'cold'):
    iterations, warmups = 1, 0
  mode = 'mtp' if args.mtp else 'plain'
  label = args.label or f'{profile}_{args.workload}_{mode}_{args.metric}'
  run = pathlib.Path(args.out) / label
  run.mkdir(parents=True, exist_ok=True)
  stamp = common.runtime_stamp(config, cli, libs)  # the binaries' build: the pools' key
  build = common.checkout_build_id(config) or stamp  # the checkouts' heads, as bench.sh records
  if args.metric == 'cold':
    pool = pathlib.Path(args.pool) if args.pool else state / 'cache' / 'scratch' / f"{label}-{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%S}"
    if pool.exists():
      raise SystemExit(f'A cold run needs an empty pool: {pool}')
  else:
    pool = pathlib.Path(args.pool) if args.pool else state / 'cache' / 'models' / profile / stamp
  env = runtime_environment(config, pool, model, spec.get('partition_policy', 'gemma4'))
  for assignment in args.env:
    key, value = assignment.split('=', 1)
    env[key] = value
  busy = common.busy_processes()
  if busy:
    raise SystemExit('Another benchmark process is running: ' + '; '.join(busy))
  cooldown = common.cooldown(args.cooldown) if args.cooldown else None
  prompt_file = pathlib.Path(args.prompt_file).resolve() if args.prompt_file else None

  selected = []
  if not args.mtp and (spec.get('plain_selected_signatures') or prefill == 128):
    selected = [f'--selected_signatures=prefill_{prefill},decode']
  base = [str(cli), f'--model_path={model}', f'--prefill_batch_sizes={prefill}',
          f"--litert_dispatch_lib_dir={config['runtime_dir']}", '--min_log_severity=0']

  def benchmark_args(n):
    out = ['--benchmark=true', f'--num_iterations={n}', f'--max_output_tokens={output}']
    if prompt_file:
      out += ['--benchmark_prefill_tokens=0', '--benchmark_decode_tokens=0', f'--input_prompt_file={prompt_file}']
    else:
      out += [f'--benchmark_prefill_tokens={inputs}', f'--benchmark_decode_tokens={output}',
              '--input_prompt=Write one sentence explaining why CUDA is useful for neural network inference:']
    return out

  def invoke(name, backend, sample, extra):
    prefix = str(run) + os.sep + (name + '.' if name else '')  # DIR/cli.log, DIR/npu.cli.log, ...
    command = base + [f'--backend={backend}', f'--max_num_tokens={context}',
                      f"--cache_dir={pool / ('cpu' if backend == 'cpu' else 'compiler')}"]
    if backend != 'cpu':
      command += selected + [f'--enable_speculative_decoding={str(args.mtp).lower()}']
    command += extra + args.cli_arg  # --cli-arg flags last, so they override the workload's
    env['LITERT_NVIDIA_MEMORY_PROFILE'] = '1' if sample else '0'
    baseline = common.gpu_memory_used_mib()
    host = common.host_snapshot()
    start = common.utc()
    print(f"{label}{' (' + name + ')' if name else ''}: {' '.join(command)[:240]}", flush=True)
    timing = common.run_process(command, prefix + 'cli.log', env, config['runtime_dir'],
                                sample_path=prefix + 'gpu.tsv' if sample else None, sample_interval=0.1)
    end = common.utc()
    meta = {
        'label': label, 'name': name or None, 'profile': profile, 'model': str(model), 'workload': args.workload,
        'metric': args.metric, 'mtp': args.mtp, 'prompt_file': str(prompt_file) if prompt_file else None,
        'input_tokens': inputs, 'output_tokens': output, 'context': context, 'iterations': iterations,
        'warmups': warmups, 'build_id': build, 'binaries_build_id': stamp, 'pool': str(pool), 'command': command,
        'exit_code': timing['exit_code'], 'host_mode': os.environ.get('CI_HOST_MODE', config.get('host_mode', 'unspecified')),
        'start': start, 'end': end, 'gpu_baseline_mib': baseline, 'backend': backend,
        'environment': {k: v for k, v in env.items() if k.startswith('LITERT_NVIDIA_')},
        'binaries': {p.name: common.sha256(p) for p in [cli, *libs.values()]},
        'platform': config.get('platform'), 'host': host, 'cooldown': cooldown, 'runner': 'windows/bench.tool'}
    pathlib.Path(prefix + 'meta.json').write_text(json.dumps(meta, indent=1) + '\n')
    pathlib.Path(prefix + 'time.json').write_text(json.dumps(timing, indent=1) + '\n')
    return timing['exit_code']

  # A pool without AOT artifacts compiles in its first run; that run is kept apart as "prepare"
  # so the measured runs read warm caches (the cold metric is that first run itself).
  if args.metric != 'cold' and not list((pool / 'aot').glob('tensorrt_aot*.bin')):
    if invoke('prepare', 'npu', False, benchmark_args(1)):
      raise SystemExit(f'Preparation failed; see {run}/prepare.cli.log')
  rc = 0
  if args.metric == 'latency':
    rc = invoke('', 'npu', False, benchmark_args(iterations))
  elif args.metric == 'cold':
    rc = invoke('', 'npu', True, benchmark_args(1))
  elif args.metric == 'memory':
    rc = invoke('', 'npu', True, benchmark_args(1) + ['--report_peak_memory_footprint=true'])
  else:
    answer = ['--benchmark=false', '--max_output_tokens=16',
              '--input_prompt=Answer with only the capital city: What is the capital of France?', '--expected_output=Paris']
    rc = invoke('npu', 'npu', False, answer) or rc
    rc = invoke('cpu', 'cpu', False, answer + ['--enable_speculative_decoding=false']) or rc
  print(f'Run: {run} (exit {rc})')
  sys.exit(rc)


if __name__ == '__main__':
  main()
