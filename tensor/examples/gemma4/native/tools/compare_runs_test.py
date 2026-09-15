#!/usr/bin/env python3
# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parent / 'compare_runs.py'
spec = importlib.util.spec_from_file_location('compare_runs', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ComparatorIntegrity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.a, self.b = self.root / 'baseline', self.root / 'candidate'
        self.fixtures = self.root / 'fixtures.json'
        self.fixture = {'cases': [{'id': 'test', 'prompt_token_ids': [2],
                                  'forced_token_ids': [4], 'score_target_ids': [4, 7]}]}
        self.fixtures.write_text(json.dumps(self.fixture))
        for directory in [self.a, self.b]:
            directory.mkdir()
            (directory / 'run.json').write_text(json.dumps({'status': 'completed', 'cases': 1}))
            passes = []
            for i in range(2):
                filename = f'test.{i}.f32'
                np.zeros(262144, dtype='<f4').tofile(directory / filename)
                passes.append({'kind': 'prefill' if i == 0 else 'decode',
                               'decode_index': i, 'input_ids': [2] if i == 0 else [4],
                               'context_length_after': i + 1, 'logits_position': i,
                               'argmax_id': 0, 'logits_file': filename})
            report = {'case_id': 'test', 'run_index': 0, 'warmup': False,
                      'vocab_size': 262144, 'num_threads': 2,
                      'logits_dtype': 'float32-little-endian',
                      'prompt_token_ids': [2], 'forced_decode_token_ids': [4], 'passes': passes}
            (directory / 'test.run_000.json').write_text(json.dumps(report))

    def compare(self):
        return module.compare(self.a, self.b, self.fixtures)

    def test_zero_vectors_have_defined_cosine_and_correct_shift(self):
        result = self.compare()
        self.assertEqual([r['target_id'] for r in result['rows']], [4, 7])
        self.assertEqual(result['overall']['prediction_rows'], 2)
        self.assertEqual(result['overall']['minimum_cosine'], 1)
        self.assertAlmostEqual(result['overall']['baseline_mean_nll'], math.log(262144))
        json.dumps(result, allow_nan=False)

    def test_divergent_argmax_does_not_skip_forced_later_pass(self):
        values = np.zeros(262144, dtype='<f4'); values[8] = 1
        values.tofile(self.b / 'test.0.f32')
        path = self.b / 'test.run_000.json'; report = json.loads(path.read_text())
        report['passes'][0]['argmax_id'] = 8; path.write_text(json.dumps(report))
        result = self.compare()
        self.assertFalse(result['rows'][0]['argmax_match'])
        self.assertEqual(len(result['rows']), 2)
        self.assertEqual(result['rows'][1]['decode_index'], 1)

    def test_negative_target_rejected(self):
        self.fixture['cases'][0]['score_target_ids'][-1] = -1
        self.fixtures.write_text(json.dumps(self.fixture))
        with self.assertRaisesRegex(ValueError, 'target ID'): self.compare()

    def test_target_history_misalignment_rejected(self):
        self.fixture['cases'][0]['score_target_ids'][0] = 5
        self.fixtures.write_text(json.dumps(self.fixture))
        with self.assertRaisesRegex(ValueError, 'follow forced history'): self.compare()

    def test_explicitly_invalidated_control_rejected(self):
        (self.b / 'INVALID_CONTROL.md').write_text('Known dangling input buffers')
        with self.assertRaisesRegex(ValueError, 'explicitly invalidated'): self.compare()

    def test_incomplete_run_rejected(self):
        (self.a / 'run.json').write_text(json.dumps({'status': 'running', 'cases': 1}))
        with self.assertRaisesRegex(ValueError, 'incomplete'): self.compare()

    def test_matching_partial_case_sets_rejected(self):
        for directory in [self.a, self.b]:
            (directory / 'run.json').write_text(json.dumps({'status': 'completed', 'cases': 2}))
        with self.assertRaisesRegex(ValueError, 'case count'): self.compare()

    def test_duplicate_fixture_rejected(self):
        self.fixture['cases'].append(self.fixture['cases'][0])
        self.fixtures.write_text(json.dumps(self.fixture))
        with self.assertRaisesRegex(ValueError, 'Duplicate fixture'): self.compare()

    def test_raw_truncation_rejected(self):
        (self.b / 'test.1.f32').write_bytes(b'\x00' * 4)
        with self.assertRaisesRegex(ValueError, 'raw length'): self.compare()


if __name__ == '__main__':
    unittest.main()
