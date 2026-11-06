# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Check the optional host adapter with local labels; no HF downloads."""

import io
import pathlib

from absl.testing import absltest
import evaluation_pb2 as pb
import numpy as np
from PIL import Image
import prepare_tipsv2 as prepare


def png(values):
  stream = io.BytesIO()
  Image.fromarray(values).save(stream, format='PNG')
  return {'bytes': stream.getvalue()}


class PrepareTest(absltest.TestCase):

  def test_signed_axes_are_explicit_and_validated(self):
    self.assertIsNone(prepare.normal_axis_order(None))
    self.assertEqual(prepare.normal_axis_order('1,-3,2'), [1, -3, 2])
    for text in ['1,1,3', '0,2,3', '1,2', '1,2,4']:
      with self.assertRaises(ValueError):
        prepare.normal_axis_order(text)

  def test_ade_labels_and_raw_input_have_typed_metadata(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    manifest = prepare.new_manifest('ade20k', 'repo', 'commit', 'validation', 3, None, 'pixel_values')
    row = {'image': png(np.full((3, 4, 3), 128, dtype=np.uint8)),
           'annotation': png(np.array([[0, 1, 150, 2]] * 3, dtype=np.uint8))}
    prepare.add_sample(manifest, row, 'ade20k', root / 'sample', 'sample', None)
    sample = manifest.samples[0]
    self.assertEqual(sample.targets[0].tensor.type, pb.UINT8)
    self.assertEqual(list(sample.targets[0].tensor.shape), [3, 4])
    self.assertEqual((root / sample.targets[0].file.path).read_bytes(), bytes([255, 0, 149, 1] * 3))
    raw = (root / sample.inputs[0].file.path).read_bytes()
    self.assertEqual(sample.inputs[0].file.path, 'sample/input.raw')
    values = np.frombuffer(raw, dtype='<f4').reshape(1, 3, 448, 448)
    np.testing.assert_allclose(values, 128 / 255, atol=1e-7)
    self.assertFalse(manifest.source.normal_axis_order)

  def test_nyuv2_does_not_guess_normal_convention(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    row = {'image': np.zeros((3, 288, 384)), 'depth': np.ones((1, 288, 384)),
           'normal': np.broadcast_to(np.array([1, 2, 3])[:, None, None], (3, 288, 384))}
    manifest = prepare.new_manifest('nyuv2', 'repo', 'commit', 'test', 7, None, 'pixel_values')
    prepare.add_sample(manifest, row, 'nyuv2', root / 'depth', 'depth', None)
    self.assertEqual([target.name for target in manifest.samples[0].targets], ['depth'])
    axes = prepare.normal_axis_order('1,-3,2')
    prepare.add_sample(manifest, row, 'nyuv2', root / 'normals', 'normals', axes)
    normal_file = manifest.samples[1].targets[1].file.path
    normal = np.fromfile(root / normal_file, dtype='<f4').reshape(3, 288, 384)
    np.testing.assert_array_equal(normal[:, 0, 0], [1, -3, 2])


if __name__ == '__main__':
  absltest.main()
