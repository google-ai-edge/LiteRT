#!/usr/bin/env python3
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
"""Tests of the processing: run validity, cell selection and the tables, on synthetic run
directories in the common layout; and that workloads.json states bench.sh's presets.

  python3 test_tables.tool   (LITERT_NVIDIA_BENCHMARK or LITERT_RT_DIR names LiteRT's benchmark/)
"""
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent


def load(path, name):
  loader = importlib.machinery.SourceFileLoader(name, str(path))
  module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader))
  loader.exec_module(module)
  return module


RUNS = load(ROOT / 'report/runs.tool', 'runs')
LITERT, LLAMA = RUNS.parsers()
MODELS = json.loads((ROOT / 'models.json').read_text())

BENCHMARK = """BenchmarkInfo:
  Init Phases (7):
    - Init Executor: 1500.00 ms
    - Init Total: 1600.00 ms
--------------------------------------------------
  Prefill Turns (Total 1 turns):
    Prefill Turn 1: Processed 1024 tokens in 30ms duration.
      Prefill Speed: {prefill} tokens/sec.
--------------------------------------------------
  Decode Turns (Total 1 turns):
    Decode Turn 1: Processed 256 tokens in 1s duration.
      Decode Speed: {decode} tokens/sec.
"""
DISPATCH = ('Partitioned subgraph<0>, selected 10 ops, from a total of 10 ops\n'
            'NVIDIA dispatch context prefill_1024 uses CUDA graph capture (device_memory_bytes=1)\n')


def litert_run(root, label, speeds, *, profile='e2b', workload='short', metric='latency', mtp=False,
               prompt=None, start='2026-11-01T00:00:00Z', warmups=1, delegated=True, exit_code=0, iterations=None):
  run = root / label
  run.mkdir(parents=True)
  log = (DISPATCH if delegated else '') + ''.join(BENCHMARK.format(prefill=p * 100, decode=p) for p in speeds)
  (run / 'cli.log').write_text(log)
  (run / 'meta.json').write_text(json.dumps({
      'label': label, 'profile': profile, 'workload': workload, 'metric': metric, 'mtp': mtp,
      'prompt_file': prompt, 'input_tokens': 1024, 'output_tokens': 256, 'context': 2048,
      'iterations': iterations or len(speeds), 'warmups': warmups, 'exit_code': exit_code, 'start': start,
      'build_id': 'abc-def', 'gpu_baseline_mib': 100}))
  (run / 'time.json').write_text(json.dumps({'wall_s': 5.0, 'peak_working_set_bytes': 2 << 30}))
  (run / 'gpu.tsv').write_text('0.0\t1100\n0.1\t2148\n')
  return run


def llama_run(root, label, pp, tg, *, start='2026-11-01T00:00:00Z', load_mode='mmap',
              model='/models/gemma-4-E2B-it-qat-UD-Q2_K_XL-cuda-Q2_0-repacked.gguf', warmups=1):
  run = root / label
  run.mkdir(parents=True)
  (run / 'meta.json').write_text(json.dumps({
      'label': label, 'runtime': 'llama.cpp', 'profile': 'e2b', 'workload': 'short', 'metric': 'latency',
      'mtp': False, 'prompt_file': None, 'iterations': 3, 'warmups': warmups, 'exit_code': 0, 'start': start,
      'build_id': 'b1', 'release_tag': 'v0.8.0', 'load_mode': load_mode, 'gpu_baseline_mib': 100, 'model': model}))
  # llama-bench's first repetition is cold; the run's warm-up count leaves it out of the mean.
  cold = [x / 4 for x in (pp, tg)] if warmups else [pp, tg]
  (run / 'pp.json').write_text(json.dumps([{'n_prompt': 1024, 'n_gen': 0, 'n_depth': 0, 'avg_ts': (cold[0] + 2 * pp) / 3, 'samples_ts': [cold[0], pp, pp]}]))
  (run / 'tg.json').write_text(json.dumps([{'n_prompt': 0, 'n_gen': 256, 'n_depth': 1024, 'avg_ts': (cold[1] + 2 * tg) / 3, 'samples_ts': [cold[1], tg, tg]}]))
  return run


class SelectionTest(unittest.TestCase):

  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.root = pathlib.Path(self.tmp.name)

  def tearDown(self):
    self.tmp.cleanup()

  def records(self):
    return [RUNS.load(run, LITERT, LLAMA, MODELS) for run in RUNS.find_runs([self.root])]

  def test_llama_run_of_another_model_is_not_a_cell(self):
    llama_run(self.root / 'q4', 'e2b_short_plain_latency', 20000.0, 250.0, start='2026-11-02T00:00:00Z',
              model='C:\\models\\gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf')
    llama_run(self.root / 'mobile', 'e2b_short_plain_latency', 21000.0, 260.0, start='2026-11-01T00:00:00Z')
    records = self.records()
    self.assertEqual([r['errors'] for r in records if r['run'].endswith('q4' + os.sep + 'e2b_short_plain_latency')],
                     [["llama.cpp model gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf is not the profile's comparison model "
                       "(gemma-4-E2B-it-qat-UD-Q2_K_XL-cuda-Q2_0-repacked.gguf)"]])
    chosen = RUNS.select(records)
    self.assertTrue(chosen[('llama', 'e2b', 'short', 'latency')]['run'].endswith('mobile' + os.sep + 'e2b_short_plain_latency'))

  def test_newest_valid_run_wins(self):
    litert_run(self.root / 'a', 'e2b_short_plain_latency', [300, 310], start='2026-11-01T00:00:00Z')
    litert_run(self.root / 'b', 'e2b_short_plain_latency', [320, 330], start='2026-11-02T00:00:00Z')
    chosen = RUNS.select(self.records())
    record = chosen[('litert', 'e2b', 'short', 'latency')]
    self.assertTrue(record['run'].endswith(os.sep + 'b' + os.sep + 'e2b_short_plain_latency'))
    self.assertAlmostEqual(RUNS.summary(record)['decode_tokens_per_second']['mean'], 330.0)  # one warm-up discarded

  def test_invalid_runs_are_not_selected(self):
    litert_run(self.root / 'a', 'ok', [300, 310], start='2026-11-01T00:00:00Z')
    litert_run(self.root / 'b', 'cpu_fallback', [999, 999], start='2026-11-03T00:00:00Z', delegated=False)
    litert_run(self.root / 'c', 'failed', [999, 999], start='2026-11-04T00:00:00Z', exit_code=1)
    litert_run(self.root / 'd', 'incomplete', [999], start='2026-11-05T00:00:00Z', iterations=2)
    records = self.records()
    self.assertEqual(sorted(r['run'].split(os.sep)[-1] for r in records if r['errors']), ['cpu_fallback', 'failed', 'incomplete'])
    chosen = RUNS.select(records)
    self.assertTrue(chosen[('litert', 'e2b', 'short', 'latency')]['run'].endswith('ok'))

  def test_text_and_mtp_cells_and_prepare_part(self):
    run = litert_run(self.root, 'e2b_short_mtp_text', [100, 200, 210, 220], mtp=True, prompt='/p.txt', warmups=2)
    (run / 'prepare.cli.log').write_text(DISPATCH + BENCHMARK.format(prefill=1, decode=1))
    (run / 'prepare.meta.json').write_text(json.dumps({**json.loads((run / 'meta.json').read_text()), 'name': 'prepare', 'iterations': 1}))
    litert_run(self.root, 'e2b_short_plain_text', [100, 150, 150], prompt='/p.txt')
    litert_run(self.root, 'e2b_short_plain_memory', [100], metric='memory', warmups=0)
    self.assertEqual(len(list(RUNS.find_runs([self.root]))), 3)  # a run with several parts is one run
    chosen = RUNS.select(self.records())
    self.assertEqual({k[3] for k in chosen}, {'text_mtp', 'text_plain', 'memory'})
    self.assertAlmostEqual(RUNS.summary(chosen[('litert', 'e2b', 'short', 'text_mtp')])['decode_tokens_per_second']['mean'], 215.0)
    self.assertAlmostEqual(chosen[('litert', 'e2b', 'short', 'memory')]['parts']['main']['gpu']['gpu_peak_gib'], 2148 / 1024)
    self.assertAlmostEqual(chosen[('litert', 'e2b', 'short', 'memory')]['parts']['main']['time']['max_rss_gib'], 2.0)

  def test_llama_runs_and_tables(self):
    litert_run(self.root / 'lt', 'e2b_short_plain_latency', [300, 310])
    llama_run(self.root / 'll', 'e2b_short_plain_latency', 20000.0, 250.0)
    chosen = RUNS.select(self.records())
    self.assertIn(('llama', 'e2b', 'short', 'latency'), chosen)
    out = subprocess.run([sys.executable, str(ROOT / 'report/tables.tool'), '--platform', '5080_wsl', '--runs', str(self.root),
                          '--out', str(self.root / 'google')], capture_output=True, text=True)
    self.assertEqual(out.returncode, 0, out.stderr)
    text = (self.root / 'google/5080_wsl/gemma4_dense.md').read_text()
    self.assertIn('| E2B | 1k | 310.0 | 250.0 | 1.24x |', text)
    self.assertIn('| E2B | 1k | 31,000 | 20,000 | 1.55x |', text)
    self.assertNotIn('26B', text)
    self.assertFalse((self.root / 'google/5080_wsl/gemma4_moe.md').exists())


class WorkloadsTest(unittest.TestCase):

  def test_presets_match_bench_sh(self):
    bench = pathlib.Path(LITERT.__file__).parent / 'bench.sh'
    presets = json.loads((ROOT / 'workloads.json').read_text())
    found = dict(re.findall(r'^\s+(\w+)\) (input=\d+ output=\d+ context=\d+ default_iterations=\d+ warmups=\d+ prefill=\d+) ;;',
                            bench.read_text(), re.M))
    self.assertTrue(found, 'bench.sh presets not found')
    for name, line in found.items():
      values = dict(pair.split('=') for pair in line.split())
      self.assertEqual({k: presets[name][k] for k in ('input', 'output', 'context', 'iterations', 'warmups', 'prefill')},
                       {'input': int(values['input']), 'output': int(values['output']), 'context': int(values['context']),
                        'iterations': int(values['default_iterations']), 'warmups': int(values['warmups']),
                        'prefill': int(values['prefill'])}, name)


if __name__ == '__main__':
  unittest.main()
