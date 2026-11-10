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
"""Check COCO category, overlap, resize, and manifest protocol rules."""

from collections.abc import Sequence
import hashlib
import json
import pathlib
from typing import Any

from google.protobuf import text_format
import numpy as np
from PIL import Image

from absl.testing import absltest as googletest
from litert.tools.vision_eval import evaluation_pb2 as pb
from litert.tools.vision_eval import prepare_coco2017 as prepare


def instance(category: int, polygon: Sequence[int]) -> dict[str, Any]:
  """Builds a non-crowd COCO polygon annotation dictionary."""
  return {'category_id': category, 'segmentation': [polygon], 'iscrowd': 0}


class PrepareCocoTest(googletest.TestCase):

  def test_selected_instances_overlap_even_with_the_same_class(self):
    annotations = [
        instance(1, [0, 0, 2, 0, 2, 2, 0, 2]),
        instance(1, [1, 0, 3, 0, 3, 2, 1, 2]),
        instance(16, [0, 2, 2, 2, 2, 4, 0, 4]),
        # An unselected COCO class neither creates a foreground label nor
        # invalidates selected objects it overlaps.
        instance(8, [0, 0, 4, 0, 4, 4, 0, 4]),
    ]
    labels = prepare.semantic_mask(annotations, 4, 4)
    np.testing.assert_array_equal(
        labels, [[15, 255, 15, 0], [15, 255, 15, 0], [3, 3, 0, 0], [3, 3, 0, 0]]
    )

  def test_crowd_rle_is_included_and_empty_images_keep_background(self):
    annotation = {
        'category_id': 1,
        'iscrowd': 1,
        'segmentation': {'size': [4, 4], 'counts': [4, 4, 4, 4]},
    }
    labels = prepare.semantic_mask([annotation], 4, 4)
    np.testing.assert_array_equal(labels, [[0, 15, 0, 15]] * 4)
    np.testing.assert_array_equal(
        prepare.semantic_mask([], 4, 4), np.zeros((4, 4))
    )

  def test_shorter_side_resize_preserves_aspect_ratio_and_integer_rounding(
      self,
  ):
    self.assertEqual(prepare.resized_shape(480, 640), (520, 693))
    self.assertEqual(prepare.resized_shape(640, 480), (693, 520))
    self.assertEqual(prepare.resized_shape(639, 640), (520, 520))
    self.assertIsNone(prepare.parse_shape(None))
    self.assertIsNone(prepare.parse_shape(''))
    self.assertEqual(prepare.parse_shape('520,693'), (520, 693))
    with self.assertRaises(ValueError):
      prepare.resized_shape(0, 3)
    with self.assertRaises(ValueError):
      prepare.parse_shape('520')

  def test_target_is_nearest_resized_to_the_inference_grid(self):
    image = Image.fromarray(np.full((2, 4, 3), 128, dtype=np.uint8))
    target = np.asarray([[0, 1, 20, 255], [1, 0, 255, 20]], dtype=np.uint8)
    pixels, labels = prepare.transform(image, target, size=4)
    self.assertEqual(pixels.shape, (1, 3, 4, 8))
    self.assertEqual(pixels.dtype, np.dtype('<f4'))
    np.testing.assert_array_equal(
        labels, np.repeat(np.repeat(target, 2, axis=0), 2, axis=1)
    )
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    stddev = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    expected = (np.float32(128 / 255) - mean) / stddev
    np.testing.assert_array_equal(pixels[0, :, 0, 0], expected)

  def test_selection_is_uniform_reproducible_and_filters_resized_shapes(self):
    images = [{'id': index, 'height': 639, 'width': 640} for index in range(20)]
    images.append({'id': 20, 'height': 480, 'width': 640})
    selected, population = prepare.select_images(
        images, 7, 2026, 520, (520, 520)
    )
    again, _ = prepare.select_images(
        list(reversed(images)), 7, 2026, 520, (520, 520)
    )
    self.assertEqual(selected, again)
    self.assertLen(selected, 7)
    self.assertEqual(population, 20)
    self.assertNotIn(20, [image['id'] for image in selected])
    with self.assertRaises(ValueError):
      prepare.select_images(images, 21, 2026, 520, (520, 520))

  def test_preparation_keeps_background_only_images_and_records_shape_groups(
      self,
  ):
    root = pathlib.Path(self.create_tempdir().full_path)
    images = root / 'val2017'
    images.mkdir()
    metadata = {'images': [], 'annotations': []}
    for identifier, height, width in [(9, 2, 4), (2, 2, 2), (7, 2, 4)]:
      name = f'{identifier:012d}.jpg'
      Image.new('RGB', (width, height), (128, 64, 0)).save(images / name)
      metadata['images'].append({
          'id': identifier,
          'file_name': name,
          'height': height,
          'width': width,
      })
    annotations = root / 'instances_val2017.json'
    annotations.write_text(json.dumps(metadata))
    digest = hashlib.sha256(annotations.read_bytes()).hexdigest()
    output = root / 'prepared'
    counts = prepare.prepare(
        annotations,
        images,
        output,
        size=4,
        annotation_sha256=digest,
        output_name='upsample_bilinear2d',
    )
    self.assertEqual(counts, {'4x4': 1, '4x8': 2})
    self.assertEqual((output / 'image_ids.txt').read_text(), '2\n7\n9\n')
    manifest = pb.DatasetManifest()
    text_format.Parse((output / '4x8/manifest.txtpb').read_text(), manifest)
    self.assertEqual(list(manifest.inputs[0].shape), [1, 3, 4, 8])
    self.assertEqual(
        manifest.outputs[0].tensor.binding_name, 'upsample_bilinear2d'
    )
    self.assertEqual(list(manifest.outputs[0].tensor.shape), [1, 21, 4, 8])
    self.assertEqual(manifest.source.revision, digest)
    self.assertIn(prepare.REFERENCE_REVISION, manifest.source.protocol)
    self.assertIn('selected=3/3', manifest.source.protocol)
    self.assertEqual([sample.source_row for sample in manifest.samples], [1, 2])
    for sample in manifest.samples:
      for identity in [sample.inputs[0].file, sample.targets[0].file]:
        self.assertTrue((output / '4x8' / identity.path).is_file())
      labels = np.fromfile(
          output / '4x8' / sample.targets[0].file.path, dtype=np.uint8
      )
      np.testing.assert_array_equal(labels, np.zeros(32))
    with self.assertRaises(FileExistsError):
      prepare.prepare(annotations, images, output, size=4)
    with self.assertRaisesRegex(ValueError, 'SHA256'):
      prepare.prepare(
          annotations, images, root / 'bad', annotation_sha256='bad'
      )


if __name__ == '__main__':
  googletest.main()
