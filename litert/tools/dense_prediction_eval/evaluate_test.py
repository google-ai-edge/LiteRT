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
"""Scoring, provenance and interrupted-run regression tests."""

import argparse
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

from litert.tools.dense_prediction_eval import evaluate
from litert.tools.dense_prediction_eval import run


class EvaluationTest(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.root = Path(self.temporary.name)
    (self.root / 'input.raw').write_bytes(struct.pack('<f', 1))
    np.save(self.root / 'target.npy', np.ones((1, 1)))
    self.manifest = self.root / 'manifest.json'
    evaluate.write_json(
        self.manifest,
        dict(version=1,
             outputs=dict(
                 depth=dict(shape=[1, 1, 1, 1], dtype='float32', task='depth')),
             samples=[
                 dict(id=f'sample_{i}',
                      inputs=dict(pixel_values='input.raw'),
                      input_sha256=evaluate.digest(self.root / 'input.raw'),
                      targets=dict(depth='target.npy')) for i in range(2)
             ]))

  def test_invalid_manifest_ids(self):
    value = json.loads(self.manifest.read_text())
    value['samples'][0]['id'] = '../escape'
    evaluate.write_json(self.manifest, value)
    with self.assertRaisesRegex(ValueError, 'Sample IDs'):
      evaluate.load_manifest(self.manifest)

  def test_outputs_must_be_complete_and_finite(self):
    contract = dict(depth=dict(shape=[1, 1, 1, 1]))
    path = self.root / 'depth.raw'
    path.write_bytes(b'')
    with self.assertRaisesRegex(ValueError, 'size mismatch'):
      evaluate.read_outputs(self.root, contract)
    path.write_bytes(struct.pack('<f', float('nan')))
    with self.assertRaisesRegex(ValueError, 'NaN'):
      evaluate.read_outputs(self.root, contract)

  def test_explicit_sample_selection(self):
    folder = self.root / 'sample_1'
    folder.mkdir()
    (folder / 'depth.raw').write_bytes(struct.pack('<f', 2))
    result = evaluate.score(self.manifest, self.root, sample_ids=['sample_1'])
    self.assertEqual(result['sample_ids'], ['sample_1'])
    self.assertEqual(result['metrics']['depth']['abs_rel'], 1)
    with self.assertRaisesRegex(ValueError, 'unknown'):
      evaluate.score(self.manifest, self.root, sample_ids=['missing'])

  def test_resume_after_partial_failure_and_reject_changed_model(self):
    # The first invocation completes one sample and then fails. The second
    # must receive only the remaining input, not silently repeat the first.
    runner = self.root / 'runner'
    runner.write_text(f'#!{sys.executable}\n' + '''
from pathlib import Path
import json, struct, sys
args = dict(a[2:].split('=', 1) for a in sys.argv[1:])
lines = Path(args['input_list']).read_text().splitlines()
marker = Path(__file__).with_suffix('.failed')
for i, line in enumerate(lines):
  if i == 1 and not marker.exists():
    marker.touch()
    sys.exit(7)
  out = Path(args['output_dir']) / f'Result_{i}'
  out.mkdir(parents=True)
  (out/'depth.raw').write_bytes(struct.pack('<f', 1))
  (out/'metrics.json').write_text(json.dumps(dict(inference_us=1, run_and_output_us=2)))
''')
    runner.chmod(0o755)
    model = self.root / 'model.tflite'
    model.write_bytes(b'fake model')
    args = argparse.Namespace(manifest=self.manifest,
                              model=model,
                              runner=runner,
                              output=self.root / 'results',
                              runtime_version='test',
                              accelerator='cpu',
                              device=None,
                              signature_index=0,
                              library_path='',
                              env=[],
                              runner_arg=[],
                              chunk_size=10,
                              chunk_timeout=30,
                              limit=None)
    with self.assertRaisesRegex(RuntimeError, 'Runner failed'):
      run.run(args)
    complete = args.output / 'sample_0/complete.json'
    self.assertTrue(complete.exists())
    modified = complete.stat().st_mtime_ns
    self.assertFalse((args.output / 'sample_1/complete.json').exists())
    run.run(args)
    self.assertEqual(complete.stat().st_mtime_ns, modified)
    self.assertTrue((args.output / 'sample_1/complete.json').exists())
    model.write_bytes(b'changed model')
    with self.assertRaisesRegex(ValueError, 'provenance changed'):
      run.run(args)

  def test_environment_validation(self):
    self.assertEqual(run.environment_options(['ADSP_LIBRARY_PATH=/a;/b']),
                     dict(ADSP_LIBRARY_PATH='/a;/b'))
    for entries in (['bad name=x'], ['A=x',
                                     'A=y'], ['LD_LIBRARY_PATH=x'], ['A']):
      with self.assertRaises(ValueError):
        run.environment_options(entries)


if __name__ == '__main__':
  unittest.main()
