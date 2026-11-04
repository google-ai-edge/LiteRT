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
"""Exercise real runner output collection and timing on a small ADD model."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
from python.runfiles import runfiles


class RunnerTest(unittest.TestCase):

  def setUp(self):
    files = runfiles.Create()
    workspace = os.environ.get('TEST_WORKSPACE', 'litert')
    self.runner = files.Rlocation(workspace + '/litert/tools/run_model')
    self.model = files.Rlocation(workspace +
                                 '/litert/test/testdata/simple_add_op.tflite')
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.root = Path(self.temporary.name)

  def invoke(self, *flags):
    return subprocess.run(
        [self.runner, '--graph=' + self.model, '--accelerator=cpu', *flags],
        capture_output=True,
        text=True,
        timeout=60,
        check=False)

  def test_input_list_writes_all_samples_before_completion_marker(self):
    inputs = []
    for index in range(3):
      path = self.root / f'input_{index}.raw'
      np.full((1, 128, 1), index, dtype='<f4').tofile(path)
      inputs.append(path)
    manifest = self.root / 'inputs.txt'
    manifest.write_text(f'{inputs[0]} {inputs[1]}\n{inputs[1]} {inputs[2]}\n')
    output = self.root / 'outputs'
    result = self.invoke('--input_list=' + str(manifest),
                         '--output_dir=' + str(output),
                         '--write_sample_metrics=true')
    self.assertEqual(result.returncode, 0, result.stderr)
    for index, expected in enumerate((1, 3)):
      directory = output / f'Result_{index}'
      tensors = list(directory.glob('*.raw'))
      self.assertEqual(len(tensors), 1)
      np.testing.assert_array_equal(np.fromfile(tensors[0], '<f4'),
                                    np.full(128, expected, dtype=np.float32))
      metrics = json.loads((directory / 'metrics.json').read_text())
      self.assertGreaterEqual(metrics['inference_us'], 0)
      self.assertGreaterEqual(metrics['run_and_output_us'],
                              metrics['inference_us'])

  def test_warmup_is_excluded_from_saved_timings(self):
    output = self.root / 'timings.json'
    result = self.invoke('--warmup_iterations=2', '--iterations=3',
                         '--timing_output=' + str(output))
    self.assertEqual(result.returncode, 0, result.stderr)
    values = json.loads(output.read_text())
    self.assertEqual(len(values), 3)
    self.assertTrue(all(isinstance(x, int) and x >= 0 for x in values))

  def test_conflicting_modes_and_cpu_fallback_are_rejected(self):
    for flags in (('--write_sample_metrics=true',), ('--input_list=unused',
                                                     '--warmup_iterations=1'),
                  ('--require_full_acceleration=true',)):
      with self.subTest(flags=flags):
        result = self.invoke(*flags)
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
  unittest.main()
