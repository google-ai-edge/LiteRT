#!/usr/bin/env python3
"""CPU-only checks for initialization reporting and CI cache failures."""
import contextlib
import copy
import io
import unittest

import report


class ReportTest(unittest.TestCase):
  def test_preparation_does_not_supply_measured_compile_time(self):
    log = ('BenchmarkInfo:\n  Init Phases (2):\n'
           '    - Init Executor: 18000.00 ms\n    - Init Total: 18318.97 ms\n')
    summary = report.summarize({'warmups': 0}, report.parse_cli_log(log))
    record = {'run': '/runs/example', 'parts': {
        'prepare': {'summary': summary, 'vendor': {'engine_builds_s': 6.17}},
        'main': {'summary': summary, 'vendor': {'aot_hits': 2, 'aot_misses': 0}}}}
    rows = report.markdown([record]).splitlines()
    self.assertIn('18.32', rows[2])
    self.assertIn('6.17', rows[2])
    self.assertNotIn('6.17', rows[3])
    self.assertIn('2/0', rows[3])

  def records(self):
    records = []
    for label, tokens, cold, count in [('12b_128k_cold', 128001, True, 1),
                                      ('12b_30k_hot', 30720, False, 3),
                                      ('26b_30k_mtp_cold', 30024, True, 1)]:
      meta = {'label': label, 'exit_code': 0}
      part = {'meta': meta, 'cli': {'iterations': [{}] * count},
              'summary': {'prefill_tokens': tokens, 'decode_tokens': 128,
                          'prefill_tokens_per_second': {'mean': 1000},
                          'decode_tokens_per_second': {'mean': 100},
                          'init_phases_s': {'Init Total': 18}, 'mtp': {'acceptance': 0.7}},
              'vendor': {'aot_hits': 0 if cold else 1, 'aot_misses': int(cold),
                         'compiled_partitions': int(cold), 'engine_builds_s': 6 if cold else None}}
      records.append({'meta': meta, 'parts': {'main': part}})
    return records

  def test_cache_miss_and_incomplete_prefill_fail_ci(self):
    records = self.records()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
      self.assertTrue(report.check_ci(records))
      bad = copy.deepcopy(records)
      bad[1]['parts']['main']['vendor']['aot_misses'] = 1
      self.assertFalse(report.check_ci(bad))
      bad = copy.deepcopy(records)
      bad[0]['parts']['main']['summary']['prefill_tokens'] = 1024
      self.assertFalse(report.check_ci(bad))
      self.assertFalse(report.check_ci(records[:2]))


if __name__ == '__main__':
  unittest.main()
