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
from unittest import mock

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


def llama_text_run(root, label, decode, *, mtp=False, workload='128k', prompt_tokens=128533, acceptance=0.57):
  """llama-server completions of the 26B (four, the first a warm-up) as llama/bench.tool writes them."""
  run = root / label
  run.mkdir(parents=True)
  (run / 'meta.json').write_text(json.dumps({
      'label': label, 'runtime': 'llama.cpp', 'profile': '26b', 'workload': workload, 'metric': 'latency',
      'mtp': mtp, 'prompt_file': f'/prompt_{workload}.txt', 'iterations': 4, 'warmups': 1, 'exit_code': 0,
      'start': '2026-11-01T00:00:00Z', 'build_id': 'b1', 'release_tag': 'v1.0.0', 'load_mode': 'none',
      'gpu_baseline_mib': 100, 'model': '/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf'}))
  for i in range(1, 5):
    timings = {'prompt_n': prompt_tokens, 'prompt_per_second': 1500.0, 'predicted_n': 128,
               'predicted_per_second': decode / 2 if i == 1 else decode}
    if mtp:
      timings.update({'draft_n': 100, 'draft_n_accepted': round(acceptance * 100)})
    (run / f'client.{i}.json').write_text(json.dumps({'timings': timings}))
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
    # A prompt beyond the model's context: the CLI prints its iterations without a turn and exits 0.
    empty = litert_run(self.root / 'e', 'empty', [999, 999], start='2026-11-06T00:00:00Z')
    (empty / 'cli.log').write_text(DISPATCH + 2 * ('BenchmarkInfo:\n  Init Phases (7):\n    - Init Executor: 1500.00 ms\n'
                                                   '  Decode Turns (Total 0 turns):\n    No decode turns recorded.\n'))
    records = self.records()
    self.assertEqual(sorted(r['run'].split(os.sep)[-1] for r in records if r['errors']),
                     ['cpu_fallback', 'empty', 'failed', 'incomplete'])
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

  def test_long_context_mtp_rows_follow_the_platform(self):
    # The 26B's 128k: TensorRT-RTX's long-context run (real text, plain), llama.cpp's plain and MTP completions.
    litert_run(self.root, '26b_128k_plain_latency', [40, 50, 50, 50], profile='26b', workload='128k', prompt='/prompt_128k.txt')
    llama_text_run(self.root, '26b_128k_plain_text', 44.0)
    llama_text_run(self.root, '26b_128k_mtp_text', 58.0, mtp=True)
    for platform, mtp_rows in (('rtx_spark', ['| 128k | TensorRT-RTX | 1,024 | 50 | — | — | — |',
                                              '| 128k | llama.cpp | 128,533 | 44 | 58 | 1.32x | 57% |']),
                               ('5080', [])):
      out = subprocess.run([sys.executable, str(ROOT / 'report/tables.tool'), '--platform', platform, '--runs', str(self.root),
                            '--out', str(self.root / 'google')], capture_output=True, text=True)
      self.assertEqual(out.returncode, 0, out.stderr)
      text = (self.root / f'google/{platform}/gemma4_moe.md').read_text(encoding='utf-8')
      self.assertIn('| 128k | 50.0 | 44.0 | 1.14x |', text)  # the decode table, on both platforms
      self.assertEqual([line for line in text.split('## 2. MTP')[1:2] for line in line.splitlines() if line.startswith('| 128k')],
                       mtp_rows, platform)


class MatrixTest(unittest.TestCase):
  """The cells both matrix tools run for the 26B, by platform and by whether its file has an MTP drafter."""

  def labels(self, tool, platform, drafter):
    with tempfile.TemporaryDirectory() as tmp:
      tmp = pathlib.Path(tmp)
      header = b'tf_lite_prefill_decode' + (b'tf_lite_mtp_drafter' if drafter else b'')
      (tmp / '26b.litertlm').write_bytes(b'LITERTLM' + bytes(16) + (32 + len(header)).to_bytes(8, 'little') + header)
      (tmp / 'prompts').mkdir()
      (tmp / 'config.json').write_text(json.dumps({
          'platform': platform, 'models': {'26b': str(tmp / '26b.litertlm')}, 'prompts_dir': str(tmp / 'prompts'),
          'llama': {'models': {'26b': {'main': 'm.gguf', 'draft': 'd.gguf'}}}}))
      matrix = load(ROOT / tool, 'matrix')
      commands = []
      with mock.patch.object(matrix.common, 'check', return_value={'problems': []}), \
           mock.patch.object(matrix.subprocess, 'run', side_effect=lambda c, **_: commands.append(c) or mock.Mock(returncode=0)), \
           mock.patch.object(sys, 'argv', [tool, str(tmp / 'out'), '26b', '--config', str(tmp / 'config.json')]), \
           mock.patch.dict(os.environ, {'BENCH_GROUPS': 'mtp', 'LLAMA_GROUPS': 'mtp'}), \
           mock.patch('builtins.print'):
        matrix.main()
      prompt = lambda c: pathlib.Path(c[c.index('--prompt-file') + 1]).name
      return {(c[c.index('--label') + 1], prompt(c), *c[c.index('--context'):c.index('--context') + 4]) if '--context' in c
              else (c[c.index('--label') + 1], prompt(c)) for c in commands}

  def test_windows_matrix(self):
    standard = {(f'26b_{w}_{m}_text', f'prompt_{p}.txt') for w, p in (('short', '1024'), ('8k', '8k'), ('16k', '16k'), ('30k', '30k'))
                for m in ('plain', 'mtp')}
    long = {('26b_128k_mtp_text', 'prompt_128k.txt', '--context', '131072', '--env', 'LITERT_NVIDIA_MOE_DEVICE_RESERVE_MB=5120'),
            ('26b_256k_mtp_text', 'prompt_256k.txt', '--context', '262144', '--env', 'LITERT_NVIDIA_MOE_DEVICE_RESERVE_MB=10240')}
    self.assertEqual(self.labels('windows/matrix.tool', 'rtx_spark', True), standard | long)
    self.assertEqual(self.labels('windows/matrix.tool', '5080', True), standard)
    self.assertEqual(self.labels('windows/matrix.tool', 'rtx_spark', False), {c for c in standard if '_plain_' in c[0]})

  def test_llama_matrix(self):
    cells = lambda workloads: {(f'26b_{w}_{m}_text', f'prompt_{p}.txt') for w, p in workloads for m in ('plain', 'mtp')}
    standard = (('short', '1024'), ('8k', '8k'), ('16k', '16k'), ('30k', '30k'))
    self.assertEqual(self.labels('llama/matrix.tool', 'rtx_spark', True), cells(standard + (('128k', '128k'), ('256k', '256k'))))
    self.assertEqual(self.labels('llama/matrix.tool', '5080', True), cells(standard))


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
