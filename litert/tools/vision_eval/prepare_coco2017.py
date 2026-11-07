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
"""Prepare COCO val2017 with TorchVision's published VOC-label protocol.

Preserves image aspect ratios and writes one static tensor manifest per resized
height/width. Each manifest requires a model accepting that exact shape. This
adapter never warps non-square images into a square model's input contract.
"""

import hashlib
import json
import pathlib
import random

from absl import app
from absl import flags
import evaluation_pb2 as pb
from google.protobuf import text_format
import numpy as np
import PIL
from PIL import Image

_ANNOTATIONS = flags.DEFINE_string('annotations', None, 'COCO instances_val2017.json')
_IMAGES = flags.DEFINE_string('images', None, 'Directory of original val2017 JPEGs')
_OUTPUT = flags.DEFINE_string('output', None, 'New prepared dataset directory')
_COUNT = flags.DEFINE_integer('count', 0, 'Uniform sample count; 0 selects all', lower_bound=0)
_SEED = flags.DEFINE_integer('seed', 2026, 'Subset selection seed')
_SIZE = flags.DEFINE_integer('size', 520, 'Resized shorter side', lower_bound=1, upper_bound=2048)
_SHAPE = flags.DEFINE_string('input_shape', None, 'Optional compatible height,width filter')
_INPUT_NAME = flags.DEFINE_string('input_name', 'x', 'Model signature input name')
_OUTPUT_NAME = flags.DEFINE_string(
    'output_name', 'upsample_bilinear2d_1', 'Model signature logits name')
_ANNOTATION_SHA = flags.DEFINE_string('annotation_sha256', None, 'Required annotation digest')

# COCO category IDs in the published weights' VOC class order, including the
# implicit background class. These are not the contiguous COCO category IDs.
COCO_CATEGORIES = (0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19,
                   4, 1, 64, 20, 63, 7, 72)
REFERENCE_REVISION = '1f1a920d685d46dfbe187934b785b7ecb7a182f2'
REFERENCE_URL = (
    f'https://github.com/pytorch/vision/tree/{REFERENCE_REVISION}'
    '/references/segmentation')
ANNOTATION_URL = (
    'https://s3.amazonaws.com/images.cocodataset.org/'
    'annotations/annotations_trainval2017.zip')


def resized_shape(height, width, size=520):
  """Match TorchVision's integer shorter-side resize, returning height,width."""
  if height <= 0 or width <= 0 or not 1 <= size <= 2048:
    raise ValueError('Image dimensions and resize size must be positive')
  if height <= width:
    return size, int(size * width / height)
  return int(size * height / width), size


def parse_shape(value):
  if value is None:
    return None
  shape = tuple(int(part) for part in value.split(','))
  if len(shape) != 2 or min(shape) <= 0:
    raise ValueError('input_shape must be two positive integers: height,width')
  return shape


def semantic_mask(annotations, height, width):
  """Map selected instances to VOC labels; ignore their overlapping pixels."""
  from pycocotools import mask as coco_mask

  labels = np.zeros((height, width), dtype=np.uint8)
  occupied = np.zeros((height, width), dtype=bool)
  overlap = np.zeros((height, width), dtype=bool)
  for annotation in annotations:
    category = annotation['category_id']
    if category not in COCO_CATEGORIES:
      continue
    # Include crowd annotations, just as the reference's val dataset does.
    # Both polygons and uncompressed crowd RLE use frPyObjects in the reference.
    rle = coco_mask.frPyObjects(annotation['segmentation'], height, width)
    mask = coco_mask.decode(rle)
    if mask.ndim == 3:
      mask = mask.any(axis=2)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != labels.shape:
      raise ValueError('COCO instance mask does not match its image dimensions')
    overlap |= occupied & mask
    occupied |= mask
    labels[mask] = COCO_CATEGORIES.index(category)
  # Even two instances of the same class produce ignored overlap pixels.
  labels[overlap] = 255
  return labels


def transform(image, labels, size=520):
  """Match the weights' PIL transform and reference nearest mask resize."""
  if labels.shape != (image.height, image.width):
    raise ValueError('Image and target dimensions differ')
  height, width = resized_shape(image.height, image.width, size)
  image = image.convert('RGB').resize((width, height), Image.Resampling.BILINEAR)
  pixels = np.asarray(image, dtype=np.float32) / np.float32(255)
  mean = np.asarray([.485, .456, .406], dtype=np.float32)
  stddev = np.asarray([.229, .224, .225], dtype=np.float32)
  pixels = np.ascontiguousarray(((pixels - mean) / stddev).transpose(2, 0, 1)[None], dtype='<f4')
  labels = np.asarray(Image.fromarray(labels).resize(
      (width, height), Image.Resampling.NEAREST), dtype=np.uint8)
  return pixels, labels


def select_images(images, count, seed, size, input_shape):
  if count < 0:
    raise ValueError('count must be nonnegative')
  candidates = sorted(images, key=lambda image: image['id'])
  if len({image['id'] for image in candidates}) != len(candidates):
    raise ValueError('COCO image IDs must be unique')
  if input_shape is not None:
    candidates = [image for image in candidates
                  if resized_shape(image['height'], image['width'], size) == input_shape]
  if not candidates or count > len(candidates):
    raise ValueError('Not enough eligible images for the requested selection')
  population = len(candidates)
  if count:
    candidates = sorted(random.Random(seed).sample(candidates, count),
                        key=lambda image: image['id'])
  return candidates, population


def prepare(annotations, images, output, count=0, seed=2026, size=520,
            input_shape=None, input_name='x', output_name='upsample_bilinear2d_1',
            annotation_sha256=None):
  """Write shape-group manifests and return their sample counts."""
  raw_annotations = pathlib.Path(annotations).read_bytes()
  digest = hashlib.sha256(raw_annotations).hexdigest()
  if annotation_sha256 is not None and digest != annotation_sha256:
    raise ValueError('Annotation SHA256 mismatch')
  metadata = json.loads(raw_annotations)
  selected, population = select_images(metadata['images'], count, seed, size, input_shape)
  by_image = {}
  for annotation in metadata['annotations']:
    by_image.setdefault(annotation['image_id'], []).append(annotation)
  source_rows = {image['id']: index for index, image in
                 enumerate(sorted(metadata['images'], key=lambda image: image['id']))}
  output = pathlib.Path(output)
  output.mkdir(parents=True, exist_ok=False)
  selection = ''.join(f'{image["id"]}\n' for image in selected)
  selection_digest = hashlib.sha256(selection.encode('ascii')).hexdigest()
  (output / 'image_ids.txt').write_text(selection, encoding='ascii')
  protocol = (
      f'COCO val2017; reference {REFERENCE_URL}; annotation_sha256={digest}; '
      f'selected={len(selected)}/{population} eligible, total={len(metadata["images"])}; '
      f'input_shape_filter={input_shape}; seed={seed}; image_ids_sha256={selection_digest}; '
      f'COCO category IDs {COCO_CATEGORIES} remapped by index to VOC 0..20; '
      'unselected categories are background 0; include crowd annotations and '
      'images with no selected objects; overlaps of selected instances ignored '
      '(255), including same-class instances; target nearest-resized to model '
      'input grid; argmax logits without further resizing; global 21-class '
      'confusion matrix including background; report present-class count with mIoU')
  manifests = {}
  source_digests = []
  for image_metadata in selected:
    height, width = image_metadata['height'], image_metadata['width']
    shape = resized_shape(height, width, size)
    if 21 * shape[0] * shape[1] * 4 > 512 * 1024 * 1024:
      raise ValueError('Resized logits exceed the evaluator tensor size limit')
    name = image_metadata['file_name']
    if pathlib.Path(name).name != name:
      raise ValueError('Expected COCO image file_name without directories')
    image_path = pathlib.Path(images) / name
    source_bytes = image_path.read_bytes()
    source_digests.append(f'{hashlib.sha256(source_bytes).hexdigest()}  {name}\n')
    with Image.open(image_path) as image:
      if image.size != (width, height):
        raise ValueError('COCO image dimensions differ from annotation metadata')
      labels = semantic_mask(by_image.get(image_metadata['id'], []), height, width)
      pixels, labels = transform(image, labels, size)
    key = f'{shape[0]}x{shape[1]}'
    if key not in manifests:
      manifest = pb.DatasetManifest(version=1)
      manifest.inputs.add(name='image', binding_name=input_name,
                          shape=[1, 3, *shape], type=pb.FLOAT32, layout=pb.NCHW)
      prediction = manifest.outputs.add()
      prediction.tensor.CopyFrom(pb.TensorSpec(
          name='segmentation', binding_name=output_name,
          shape=[1, 21, *shape], type=pb.FLOAT32, layout=pb.NCHW))
      prediction.scoring.CopyFrom(pb.TaskOptions(
          task=pb.SEGMENTATION, classes=21, ignore_label=255))
      manifest.source.repository = ANNOTATION_URL
      manifest.source.revision = digest
      manifest.source.split = 'val2017'
      manifest.source.seed = seed
      manifest.source.preprocessing = (
          f'RGB; Pillow {PIL.__version__} bilinear antialiased shorter-side resize '
          f'to {size}, aspect ratio preserved with integer output dimensions; '
          'float32 /255; mean [0.485,0.456,0.406], std [0.229,0.224,0.225]; NCHW')
      manifest.source.protocol = protocol + f'; this manifest contains only shape {key}'
      manifests[key] = manifest
    manifest = manifests[key]
    identifier = f'coco_val2017_{image_metadata["id"]:012d}'
    directory = output / key / identifier
    directory.mkdir(parents=True)
    sample = manifest.samples.add(id=identifier, source_file=name,
                                  source_row=source_rows[image_metadata['id']])
    data = sample.inputs.add(name='image')
    target = sample.targets.add(name='segmentation')
    target.tensor.CopyFrom(pb.TensorSpec(shape=labels.shape, type=pb.UINT8))
    for name, array, identity in [
        ('input.raw', pixels, data.file),
        ('labels.raw', labels, target.file),
    ]:
      raw = array.tobytes()
      (directory / name).write_bytes(raw)
      identity.path = f'{identifier}/{name}'
  for key, manifest in manifests.items():
    path = output / key / 'manifest.txtpb'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(text_format.MessageToString(manifest), encoding='utf-8')
    temporary.replace(path)
  (output / 'source_images.sha256').write_text(''.join(source_digests), encoding='utf-8')
  (output / 'protocol.txt').write_text(protocol + '\n', encoding='utf-8')
  return {key: len(manifest.samples) for key, manifest in sorted(manifests.items())}


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Unexpected positional arguments')
  groups = prepare(_ANNOTATIONS.value, _IMAGES.value, _OUTPUT.value,
                   _COUNT.value, _SEED.value, _SIZE.value, parse_shape(_SHAPE.value),
                   _INPUT_NAME.value, _OUTPUT_NAME.value, _ANNOTATION_SHA.value)
  print(f'Prepared {sum(groups.values())} samples in {len(groups)} shape groups: {_OUTPUT.value}')


if __name__ == '__main__':
  flags.mark_flags_as_required(['annotations', 'images', 'output', 'annotation_sha256'])
  app.run(main)
