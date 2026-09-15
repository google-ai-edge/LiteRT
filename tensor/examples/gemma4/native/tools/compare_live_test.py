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

"""Synthetic integrity/layout tests; no model inference or device access."""
import json
from pathlib import Path
import tempfile
import unittest

import compare_live as compare
import numpy as np


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='comparator-test-')
        self.root = Path(self.temp.name)
        self.a, self.b = self.root / 'a', self.root / 'b'
        self.a.mkdir()
        self.b.mkdir()
        self.fixtures = self.root / 'fixtures.json'
        self.fixtures.write_text(json.dumps(dict(cases=[
            dict(id='probe', prompt_token_ids=[2], forced_token_ids=[]),
            dict(id='other', prompt_token_ids=[2], forced_token_ids=[])])))
        self.values = np.full(262144, -1, dtype='<f4')
        self.values[42] = 2
        for directory in [self.a, self.b]:
            self.write_run(directory)

    def tearDown(self):
        self.temp.cleanup()

    def write_run(self, directory, extra=False):
        keys = ['probe', 'other'] if extra else ['probe']
        (directory / 'run.json').write_text(json.dumps(dict(status='completed', cases=len(keys))))
        for key in keys:
            report = dict(case_id=key, prompt_token_ids=[2], forced_decode_token_ids=[],
                          vocab_size=262144, logits_dtype='float32-little-endian',
                          run_index=0, warmup=False, num_threads=2,
                          passes=[dict(kind='prefill', decode_index=0, input_ids=[2],
                                       context_length_after=1, logits_position=0,
                                       argmax_id=42, logits_file=key + '.prefill.f32')])
            (directory / (key + '.run_000.json')).write_text(json.dumps(report))
            self.values.tofile(directory / (key + '.prefill.f32'))

    def update_report(self, directory, fn):
        path = directory / 'probe.run_000.json'
        data = json.loads(path.read_text())
        fn(data)
        path.write_text(json.dumps(data))

    def test_subset_preserves_source_and_bitwise(self):
        self.write_run(self.a, extra=True)
        before = {p.name: p.read_bytes() for p in self.a.iterdir()}
        result = compare.compare(self.a, self.b, self.fixtures, cache=False)
        self.assertEqual(result['selected_cases'], ['probe'])
        self.assertEqual(result['overall']['bitwise_equal_rows'], 1)
        self.assertNotIn('strict_diagnostic_rows', result['overall'])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.a.iterdir()})

    def test_default_cache_mode_loads_published_owner_specs(self):
        owner_path = Path(compare.__file__).parent / 'owner_specs.json'
        metadata = json.loads(owner_path.read_text())
        self.assertFalse(Path(metadata['source']).is_absolute())
        source = owner_path.parent / metadata['source']
        self.assertEqual(compare.sha(source), metadata['source_sha256'])
        specs = metadata['specs']
        self.assertEqual([spec['owner'] for spec in specs], list(range(15)))
        expected_codes = 0
        for spec in specs:
            dim = 512 if spec['owner'] in (4, 9, 14) else 256
            self.assertEqual(spec['head_dim'], dim)
            values = (np.arange(dim, dtype=np.int16) % 256 - 128).astype(np.int8)
            for kind in ['k', 'v']:
                name = f"probe.prefill.f32.owner{spec['owner']}.{kind}.i8"
                for directory in [self.a, self.b]:
                    values.tofile(directory / name)
                expected_codes += dim
        for directory in [self.a, self.b]:
            self.update_report(directory, lambda report: report.update(
                kv_dtype='int8', kv_layout='owner_position_channel', kv_capacity=1))

        # Exercise the public entry point's default cache=True path, including
        # the packaged metadata dependency rather than injecting owner specs.
        result = compare.compare(self.a, self.b, self.fixtures)
        self.assertEqual(result['owner_specs_sha256'], compare.sha(owner_path))
        self.assertEqual(result['cache']['status'], 'complete')
        self.assertEqual(result['cache']['equal_owner_tensors'], 30)
        self.assertEqual(result['cache']['compared_codes'], expected_codes)
        self.assertEqual(result['cache']['different_codes'], 0)

        path = self.b / 'probe.prefill.f32.owner14.v.i8'
        values = np.fromfile(path, dtype=np.int8)
        values[-1] -= 1
        values.tofile(path)
        changed = compare.compare(self.a, self.b, self.fixtures)['cache']
        self.assertEqual(changed['different_codes'], 1)
        self.assertEqual(changed['equal_owner_tensors'], 29)

    def test_numerical_difference_without_argmax_change(self):
        altered = self.values.copy()
        altered[7] = 1
        altered.tofile(self.b / 'probe.prefill.f32')
        result = compare.compare(self.a, self.b, self.fixtures, cache=False)['overall']
        self.assertEqual(result['bitwise_equal_rows'], 0)
        self.assertEqual(result['argmax_matches'], 1)
        self.assertGreater(result['max_rmse'], 0)
        self.assertGreater(result['max_kl_baseline_to_candidate'], 0)

    def test_reject_history_and_thread_mismatch(self):
        self.update_report(self.b, lambda d: d.update(prompt_token_ids=[2, 3]))
        with self.assertRaisesRegex(ValueError, 'prompt mismatch'):
            compare.compare(self.a, self.b, self.fixtures, cache=False)
        self.write_run(self.b)
        self.update_report(self.b, lambda d: d.update(num_threads=4))
        with self.assertRaisesRegex(ValueError, 'thread count mismatch'):
            compare.compare(self.a, self.b, self.fixtures, cache=False)

    def test_reject_missing_kind_and_nonfinite_raw(self):
        self.update_report(self.b, lambda d: d['passes'][0].pop('kind'))
        with self.assertRaises(KeyError):
            compare.compare(self.a, self.b, self.fixtures, cache=False)
        self.write_run(self.b)
        altered = self.values.copy()
        altered[7] = np.nan
        altered.tofile(self.b / 'probe.prefill.f32')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            compare.compare(self.a, self.b, self.fixtures, cache=False)

    def test_reject_wrong_raw_size_and_argmax(self):
        (self.b / 'probe.prefill.f32').write_bytes(b'1234')
        with self.assertRaisesRegex(ValueError, 'Wrong raw length'):
            compare.compare(self.a, self.b, self.fixtures, cache=False)
        self.write_run(self.b)
        self.update_report(self.b, lambda d: d['passes'][0].update(argmax_id=43))
        with self.assertRaisesRegex(ValueError, 'argmax disagreement'):
            compare.compare(self.a, self.b, self.fixtures, cache=False)

    def cache_pair(self):
        specs = {4: dict(owner=4, head_dim=3, key_scale=0.5, value_scale=0.25)}
        report = dict(case_id='probe', prompt_token_ids=[2, 11],
                      forced_decode_token_ids=[], kv_dtype='int8',
                      kv_layout='owner_position_channel', kv_capacity=4)
        row = dict(decode_index=0, context_length_after=2, logits_file='probe.prefill.f32')
        report['passes'] = [row]
        keys = np.array([[-128, 0, 127], [-1, 2, 3]], dtype=np.int8)
        values = np.array([[1, 10, 100], [2, 20, 110]], dtype=np.int8)
        metadata = dict(status='completed', active_tokens=2, consumed_token_ids=[2, 11],
                        capacity=4, tensors=[])
        for kind, source in [('k', keys), ('v', values)]:
            full = np.full((4, 3), 77, dtype=np.int8)
            full[:2] = source
            data = full if kind == 'k' else full.T.copy()
            name = 'old_' + kind + '.i8'
            data.tofile(self.a / name)
            metadata['tensors'].append(dict(owner=4, kind='key' if kind == 'k' else 'value',
                dtype='int8', active_begin=0, active_end=2,
                shape=[1, 1] + list(data.shape), file=name))
            source.tofile(self.b / f'probe.prefill.f32.owner4.{kind}.i8')
        (self.a / 'probe.prefill.kv.json').write_text(json.dumps(metadata))
        return specs, report, row

    def test_cache_transpose_and_exclude_unused_capacity(self):
        specs, report, row = self.cache_pair()
        result = compare.compare_caches(self.a, self.b, {'probe': report}, {'probe': report}, ['probe'], specs)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['compared_codes'], 12)
        self.assertEqual(result['different_codes'], 0)
        self.assertEqual(result['equal_owner_tensors'], 2)

    def test_cache_pinpoints_value_layout_or_code_bug(self):
        specs, report, row = self.cache_pair()
        path = self.b / 'probe.prefill.f32.owner4.v.i8'
        values = np.fromfile(path, dtype=np.int8)
        values[4] += 1
        values.tofile(path)
        result = compare.compare_caches(self.a, self.b, {'probe': report}, {'probe': report}, ['probe'], specs)
        self.assertEqual(result['different_codes'], 1)
        mismatch = next(r for r in result['rows'] if not r['bitwise_equal'])
        self.assertEqual(mismatch['first_different_position_channel'], [1, 1])

    def test_cache_reject_missing_owner_and_wrong_history(self):
        specs, report, row = self.cache_pair()
        (self.b / 'probe.prefill.f32.owner4.v.i8').unlink()
        with self.assertRaisesRegex(ValueError, 'Live cache missing'):
            compare.cache_for_pass(self.b, report, row, specs)
        path = self.a / 'probe.prefill.kv.json'
        data = json.loads(path.read_text())
        data['consumed_token_ids'] = [2, 12]
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'token history mismatch'):
            compare.cache_for_pass(self.a, report, row, specs)

    def test_cache_unavailable_is_explicit(self):
        specs, report, row = self.cache_pair()
        empty = self.root / 'empty'
        empty.mkdir()
        result = compare.compare_caches(self.a, empty, {'probe': report}, {'probe': report}, ['probe'], specs)
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(result['compared_codes'], 0)
        self.assertTrue(result['missing_passes'][0]['candidate_missing'])


if __name__ == '__main__':
    unittest.main()
