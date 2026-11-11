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
"""Check VOC label preservation and the raw-tensor preparation contract."""

import io
import pathlib

from google.protobuf import text_format
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from absl.testing import absltest as googletest
from litert.tools.vision_eval import evaluation_pb2 as pb
from litert.tools.vision_eval import prepare_voc2012 as prepare


def encoded(image: Image.Image) -> dict[str, bytes]:
  """Encodes a PIL Image as PNG bytes inside a Parquet cell dictionary."""
  stream = io.BytesIO()
  if image.mode == 'P':
    palette = []
    for index in range(256):
      palette.extend(prepare.voc_color(index))
    image.putpalette(palette)
  image.save(stream, format='PNG')
  return {'bytes': stream.getvalue()}


class PrepareVocTest(googletest.TestCase):

  def test_palette_indices_preserve_background_and_void(self):
    mask = Image.frombytes('P', (2, 2), bytes([0, 1, 20, 255]))
    np.testing.assert_array_equal(
        prepare.label_array(mask), [[0, 1], [20, 255]]
    )

  def test_rgb_palette_decodes_to_class_indices(self):
    colors = np.asarray(
        [[[0, 0, 0], [128, 0, 0]], [[0, 64, 128], [224, 224, 192]]],
        dtype=np.uint8,
    )
    labels = prepare.label_array(Image.fromarray(colors))
    np.testing.assert_array_equal(labels, [[0, 1], [20, 255]])
    self.assertEqual(prepare.voc_color(20), (0, 64, 128))
    self.assertEqual(prepare.voc_color(255), (224, 224, 192))

  def test_unknown_colors_and_labels_are_rejected(self):
    with self.assertRaises(ValueError):
      prepare.label_array(Image.fromarray(np.asarray([[21]], dtype=np.uint8)))
    with self.assertRaises(ValueError):
      prepare.label_array(
          Image.fromarray(np.asarray([[[1, 2, 3]]], dtype=np.uint8))
      )

  def test_bounded_preparation_records_shapes_paths_and_normalization(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    source = root / 'source'
    source.mkdir()
    image = encoded(Image.fromarray(np.full((2, 2, 3), 128, dtype=np.uint8)))
    mask = encoded(Image.frombytes('P', (2, 2), bytes([0, 1, 20, 255])))
    pq.write_table(
        pa.Table.from_pylist([{'image': image, 'mask': mask}] * 3),
        source / 'validation-00000.parquet',
    )
    output = root / 'prepared'
    self.assertEqual(
        prepare.prepare(
            source,
            output,
            max_samples=2,
            size=4,
            output_name='upsample_bilinear2d',
            repo='source/repo',
            revision='pinned-commit',
        ),
        2,
    )
    manifest = pb.DatasetManifest()
    text_format.Parse((output / 'manifest.txtpb').read_text(), manifest)
    self.assertLen(manifest.samples, 2)
    self.assertEqual(manifest.outputs[0].scoring.classes, 21)
    self.assertEqual(manifest.outputs[0].scoring.ignore_label, 255)
    self.assertEqual(
        manifest.outputs[0].tensor.binding_name, 'upsample_bilinear2d'
    )
    self.assertEqual(list(manifest.inputs[0].shape), [1, 3, 4, 4])
    self.assertEqual(manifest.source.revision, 'pinned-commit')
    self.assertEqual([sample.source_row for sample in manifest.samples], [0, 1])
    sample = manifest.samples[0]
    raw = (output / sample.inputs[0].file.path).read_bytes()
    self.assertEqual(
        sample.inputs[0].file.path, 'voc_validation_000000/input.raw'
    )
    values = np.frombuffer(raw, dtype='<f4').reshape(1, 3, 4, 4)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    stddev = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    expected = (np.float32(128 / 255) - mean) / stddev
    np.testing.assert_allclose(values[0, :, 0, 0], expected, atol=1e-7)
    labels = (output / sample.targets[0].file.path).read_bytes()
    self.assertEqual(labels, bytes([0, 1, 20, 255]))
    self.assertEqual(list(sample.targets[0].tensor.shape), [2, 2])
    with self.assertRaises(FileExistsError):
      prepare.prepare(source, output)

  def test_missing_shards_and_invalid_ranges_fail(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    with self.assertRaises(ValueError):
      prepare.prepare(root, root / 'out')
    with self.assertRaises(ValueError):
      prepare.prepare(root, root / 'out', max_samples=-1)
    with self.assertRaises(ValueError):
      prepare.prepare(root, root / 'out', size=0)


if __name__ == '__main__':
  googletest.main()
