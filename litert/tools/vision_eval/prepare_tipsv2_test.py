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

from google.protobuf import text_format
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from absl.testing import absltest as googletest
from litert.tools.vision_eval import evaluation_pb2 as pb
from litert.tools.vision_eval import prepare_tipsv2 as prepare


def png(values: np.ndarray) -> dict[str, bytes]:
  """Encodes an image array as PNG bytes inside a Parquet cell dictionary."""
  stream = io.BytesIO()
  Image.fromarray(values).save(stream, format='PNG')
  return {'bytes': stream.getvalue()}


class PrepareTest(googletest.TestCase):

  def test_signed_axes_are_explicit_and_validated(self):
    self.assertIsNone(prepare.normal_axis_order(None))
    self.assertIsNone(prepare.normal_axis_order(''))
    self.assertEqual(prepare.normal_axis_order('1,-3,2'), [1, -3, 2])
    for text in ['1,1,3', '0,2,3', '1,2', '1,2,4']:
      with self.assertRaises(ValueError):
        prepare.normal_axis_order(text)

  def test_ade_labels_and_raw_input_have_typed_metadata(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    manifest = prepare.new_manifest(
        'ade20k', 'repo', 'commit', 'validation', 3, None, 'pixel_values'
    )
    row = {
        'image': png(np.full((3, 4, 3), 128, dtype=np.uint8)),
        'annotation': png(np.array([[0, 1, 150, 2]] * 3, dtype=np.uint8)),
    }
    prepare.add_sample(manifest, row, 'ade20k', root / 'sample', 'sample', None)
    sample = manifest.samples[0]
    self.assertEqual(sample.targets[0].tensor.type, pb.UINT8)
    self.assertEqual(list(sample.targets[0].tensor.shape), [3, 4])
    self.assertEqual(
        (root / sample.targets[0].file.path).read_bytes(),
        bytes([255, 0, 149, 1] * 3),
    )
    raw = (root / sample.inputs[0].file.path).read_bytes()
    self.assertEqual(sample.inputs[0].file.path, 'sample/input.raw')
    values = np.frombuffer(raw, dtype='<f4').reshape(1, 3, 448, 448)
    np.testing.assert_allclose(values, 128 / 255, atol=1e-7)
    self.assertFalse(manifest.source.normal_axis_order)

  def test_nyuv2_does_not_guess_normal_convention(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    row = {
        'image': np.zeros((3, 288, 384)),
        'depth': np.ones((1, 288, 384)),
        'normal': np.broadcast_to(
            np.array([1, 2, 3])[:, None, None], (3, 288, 384)
        ),
    }
    manifest = prepare.new_manifest(
        'nyuv2', 'repo', 'commit', 'test', 7, None, 'pixel_values'
    )
    prepare.add_sample(manifest, row, 'nyuv2', root / 'depth', 'depth', None)
    self.assertEqual(
        [target.name for target in manifest.samples[0].targets], ['depth']
    )
    axes = prepare.normal_axis_order('1,-3,2')
    prepare.add_sample(
        manifest, row, 'nyuv2', root / 'normals', 'normals', axes
    )
    normal_file = manifest.samples[1].targets[1].file.path
    normal = np.fromfile(root / normal_file, dtype='<f4').reshape(3, 288, 384)
    np.testing.assert_array_equal(normal[:, 0, 0], [1, -3, 2])

  def test_prepare_shards_samples_across_multiple_shards_and_writes_manifests(
      self,
  ):
    root = pathlib.Path(self.create_tempdir().full_path)
    shards_dir = root / 'shards'
    shards_dir.mkdir()
    ade_row = {
        'image': png(np.full((3, 4, 3), 128, dtype=np.uint8)),
        'annotation': png(np.array([[0, 1, 150, 2]] * 3, dtype=np.uint8)),
    }
    shard_0 = shards_dir / 'validation-00000.parquet'
    shard_1 = shards_dir / 'validation-00001.parquet'
    pq.write_table(pa.Table.from_pylist([ade_row, ade_row]), shard_0)
    pq.write_table(pa.Table.from_pylist([ade_row]), shard_1)
    ade_shards = [
        ('scene_parsing/validation/00000.parquet', shard_0),
        ('scene_parsing/validation/00001.parquet', shard_1),
    ]

    output_ade = root / 'ade_prepared'
    manifest_path = prepare.prepare_shards(
        dataset='ade20k',
        repo='ade/repo',
        revision='ade-sha',
        split='validation',
        shards=ade_shards,
        count=1,
        seed=2026,
        axes=None,
        output=output_ade,
        input_name='pixel_values',
        manifest_format='text',
    )
    self.assertEqual(manifest_path, output_ade / 'manifest.txtpb')
    manifest = pb.DatasetManifest()
    text_format.Parse(manifest_path.read_text(encoding='utf-8'), manifest)
    self.assertLen(manifest.samples, 1)
    self.assertEqual(manifest.source.revision, 'ade-sha')

    nyu_row = {
        'image': np.zeros((3, 288, 384), dtype=np.float32).tolist(),
        'depth': np.ones((1, 288, 384), dtype=np.float32).tolist(),
        'normal': (
            np.broadcast_to(
                np.array([1.0, 2.0, 3.0], dtype=np.float32)[:, None, None],
                (3, 288, 384),
            ).tolist()
        ),
    }
    nyu_shard_0 = shards_dir / 'nyu-00000.parquet'
    nyu_shard_1 = shards_dir / 'nyu-00001.parquet'
    pq.write_table(pa.Table.from_pylist([nyu_row]), nyu_shard_0)
    pq.write_table(pa.Table.from_pylist([nyu_row]), nyu_shard_1)
    nyu_shards = [
        ('data/test-00000.parquet', nyu_shard_0),
        ('data/test-00001.parquet', nyu_shard_1),
    ]
    output_nyu = root / 'nyu_prepared'
    bin_path = prepare.prepare_shards(
        dataset='nyuv2',
        repo='nyu/repo',
        revision='nyu-sha',
        split='test',
        shards=nyu_shards,
        count=0,
        seed=2026,
        axes=[1, -3, 2],
        output=output_nyu,
        input_name='image_input',
        manifest_format='binary',
    )
    self.assertEqual(bin_path, output_nyu / 'manifest.binpb')
    bin_manifest = pb.DatasetManifest()
    bin_manifest.ParseFromString(bin_path.read_bytes())
    self.assertLen(bin_manifest.samples, 2)
    self.assertEqual(
        [s.id for s in bin_manifest.samples],
        ['nyuv2_test_000000', 'nyuv2_test_000001'],
    )
    self.assertEqual(
        [s.source_file for s in bin_manifest.samples],
        ['data/test-00000.parquet', 'data/test-00001.parquet'],
    )
    self.assertEqual([s.source_row for s in bin_manifest.samples], [0, 0])
    self.assertEqual(list(bin_manifest.source.normal_axis_order), [1, -3, 2])

    with self.assertRaises(FileExistsError):
      prepare.prepare_shards(
          dataset='ade20k',
          repo='ade/repo',
          revision='ade-sha',
          split='validation',
          shards=ade_shards,
          count=1,
          seed=2026,
          axes=None,
          output=output_ade,
      )
    with self.assertRaises(ValueError):
      prepare.prepare_shards(
          dataset='ade20k',
          repo='ade/repo',
          revision='ade-sha',
          split='validation',
          shards=ade_shards,
          count=-1,
          seed=2026,
          axes=None,
          output=root / 'unused',
      )
    with self.assertRaises(ValueError):
      prepare.prepare_shards(
          dataset='ade20k',
          repo='ade/repo',
          revision='ade-sha',
          split='validation',
          shards=[],
          count=1,
          seed=2026,
          axes=None,
          output=root / 'unused',
      )
    too_many_output = root / 'too_many'
    with self.assertRaises(ValueError):
      prepare.prepare_shards(
          dataset='ade20k',
          repo='ade/repo',
          revision='ade-sha',
          split='validation',
          shards=ade_shards,
          count=10,
          seed=2026,
          axes=None,
          output=too_many_output,
      )
    self.assertFalse(too_many_output.exists())


if __name__ == '__main__':
  googletest.main()
