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
"""Prepares COCO val2017 with TorchVision's published VOC-label protocol.

Preserves image aspect ratios and writes one static tensor manifest per resized
height/width. Each manifest requires a model accepting that exact shape. This
adapter never warps non-square images into a square model's input contract.
"""

import collections
from collections.abc import Mapping, Sequence
import hashlib
import json
import pathlib
import random
from typing import Any

from absl import app
from absl import flags
from google.protobuf import text_format
import numpy as np
import PIL
from PIL import Image
from pycocotools import mask as coco_mask

from litert.tools.vision_eval import evaluation_pb2 as pb

# Default shorter-side resize resolution in pixels for TorchVision VOC-subset
# semantic segmentation evaluation.
_DEFAULT_IMAGE_SIZE = 520
# Upper bound on resized shorter-side resolution in pixels.
_MAX_IMAGE_SIZE = 2048
# Pascal VOC semantic segmentation ignore/void label value (uint8).
_IGNORE_LABEL = 255
# Byte width of a single float32 element.
_FLOAT32_BYTES = 4
# Maximum tensor byte size accepted by the C++ vision_eval runner (512 MiB).
_MAX_TENSOR_BYTES = 512 * 1024 * 1024

_ANNOTATIONS = flags.DEFINE_string(
    'annotations', None, 'COCO instances_val2017.json'
)
_IMAGES = flags.DEFINE_string(
    'images', None, 'Directory of original val2017 JPEGs'
)
_OUTPUT = flags.DEFINE_string('output', None, 'New prepared dataset directory')
_COUNT = flags.DEFINE_integer(
    'count', 0, 'Uniform sample count; 0 selects all', lower_bound=0
)
_SEED = flags.DEFINE_integer('seed', 2026, 'Subset selection seed')
_SIZE = flags.DEFINE_integer(
    'size',
    _DEFAULT_IMAGE_SIZE,
    'Resized shorter side',
    lower_bound=1,
    upper_bound=_MAX_IMAGE_SIZE,
)
_INPUT_SHAPE = flags.DEFINE_string(
    'input_shape', '', 'Optional compatible height,width filter'
)
_INPUT_NAME = flags.DEFINE_string(
    'input_name', 'x', 'Model signature input name'
)
_OUTPUT_NAME = flags.DEFINE_string(
    'output_name', 'upsample_bilinear2d_1', 'Model signature logits name'
)
_ANNOTATION_SHA256 = flags.DEFINE_string(
    'annotation_sha256', None, 'Required annotation digest'
)

# COCO category IDs in the published weights' VOC class order, including the
# implicit background class. These are not the contiguous COCO category IDs.
COCO_CATEGORIES: tuple[int, ...] = (
    0,
    5,
    2,
    16,
    9,
    44,
    6,
    3,
    17,
    62,
    21,
    67,
    18,
    19,
    4,
    1,
    64,
    20,
    63,
    7,
    72,
)
# Number of Pascal VOC semantic segmentation classes (background + 20 classes).
_NUM_VOC_CLASSES = len(COCO_CATEGORIES)
REFERENCE_REVISION = '1f1a920d685d46dfbe187934b785b7ecb7a182f2'
REFERENCE_URL = (
    f'https://github.com/pytorch/vision/tree/{REFERENCE_REVISION}'
    '/references/segmentation'
)
ANNOTATION_URL = (
    'https://s3.amazonaws.com/images.cocodataset.org/'
    'annotations/annotations_trainval2017.zip'
)

_IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STDDEV = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def resized_shape(
    height: int, width: int, size: int = _DEFAULT_IMAGE_SIZE
) -> tuple[int, int]:
  """Matches TorchVision's integer shorter-side resize, returning height,width.

  Args:
    height: Original image height in pixels.
    width: Original image width in pixels.
    size: Target length for the shorter image side.

  Returns:
    A `(resized_height, resized_width)` tuple preserving aspect ratio.

  Raises:
    ValueError: If `height` or `width` is not positive, or `size` is outside
      `1..2048`.
  """
  if height <= 0 or width <= 0 or not 1 <= size <= _MAX_IMAGE_SIZE:
    raise ValueError(
        'Image dimensions must be positive and resize size must be in'
        f' 1..{_MAX_IMAGE_SIZE}; got height={height}, width={width},'
        f' size={size}'
    )
  if height <= width:
    return size, int(size * width / height)
  return int(size * height / width), size


def parse_shape(value: str | None) -> tuple[int, int] | None:
  """Parses an optional 'height,width' filter string into a 2-tuple of ints.

  Args:
    value: Comma-separated `'height,width'` filter string, or `None`/`''` when
      no shape filter is configured.

  Returns:
    A `(height, width)` tuple of positive integers, or `None` if `value` is
    empty or `None`.

  Raises:
    ValueError: If `value` is non-empty and does not contain two positive ints.
  """
  if not value:
    return None
  shape = tuple(int(part) for part in value.split(','))
  if len(shape) != 2 or min(shape) <= 0:
    raise ValueError('input_shape must be two positive integers: height,width')
  return shape[0], shape[1]


def semantic_mask(
    annotations: Sequence[Mapping[str, Any]], height: int, width: int
) -> np.ndarray:
  """Maps selected instances to VOC labels and ignores overlapping pixels.

  Args:
    annotations: Sequence of COCO instance annotation dictionaries for one
      image.
    height: Image height in pixels.
    width: Image width in pixels.

  Returns:
    A 2D `uint8` array of shape `(height, width)` with VOC labels `0..20` and
    ignore value `255` on overlapping selected instances.

  Raises:
    ValueError: If a decoded COCO instance mask does not match `(height,
    width)`.
  """
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
  labels[overlap] = _IGNORE_LABEL
  return labels


def transform(
    image: Image.Image, labels: np.ndarray, size: int = _DEFAULT_IMAGE_SIZE
) -> tuple[np.ndarray, np.ndarray]:
  """Matches the weights' PIL transform and reference nearest mask resize.

  Args:
    image: Input PIL Image.
    labels: 2D `uint8` array of original-resolution semantic labels.
    size: Target shorter-side resize dimension.

  Returns:
    A `(pixels, resized_labels)` tuple where `pixels` is an NCHW float32 array
    and `resized_labels` is a 2D `uint8` array at the resized spatial grid.

  Raises:
    ValueError: If `labels.shape` differs from `(image.height, image.width)`.
  """
  if labels.shape != (image.height, image.width):
    raise ValueError('Image and target dimensions differ')
  height, width = resized_shape(image.height, image.width, size)
  image = image.convert('RGB').resize(
      (width, height), Image.Resampling.BILINEAR
  )
  pixels = np.asarray(image, dtype=np.float32) / np.float32(255)
  normalized = ((pixels - _IMAGENET_MEAN) / _IMAGENET_STDDEV).transpose(
      2, 0, 1
  )[None]
  pixels = np.ascontiguousarray(normalized, dtype='<f4')
  labels = np.asarray(
      Image.fromarray(labels).resize((width, height), Image.Resampling.NEAREST),
      dtype=np.uint8,
  )
  return pixels, labels


def select_images(
    images: Sequence[Mapping[str, Any]],
    count: int,
    seed: int,
    size: int,
    input_shape: tuple[int, int] | None,
) -> tuple[list[Mapping[str, Any]], int]:
  """Selects a deterministic subset of COCO image records matching the shape.

  Args:
    images: Sequence of COCO image metadata dictionaries.
    count: Number of images to sample (`0` selects all eligible images).
    seed: Random seed for uniform subset selection without replacement.
    size: Shorter-side resize dimension used to compute each image's shape.
    input_shape: Optional `(height, width)` filter for resized dimensions, or
      `None` to accept all resized shapes.

  Returns:
    A `(selected_images, eligible_population)` tuple sorted by image ID.

  Raises:
    ValueError: If `count` is negative, image IDs are not unique, or fewer than
      `count` (or zero) eligible images match `input_shape`.
  """
  if count < 0:
    raise ValueError('count must be nonnegative')
  candidates = sorted(images, key=lambda image: image['id'])
  if len({image['id'] for image in candidates}) != len(candidates):
    raise ValueError('COCO image IDs must be unique')
  if input_shape is not None:
    candidates = [
        image
        for image in candidates
        if resized_shape(image['height'], image['width'], size) == input_shape
    ]
  if not candidates or count > len(candidates):
    raise ValueError('Not enough eligible images for the requested selection')
  population = len(candidates)
  if count:
    candidates = sorted(
        random.Random(seed).sample(candidates, count),
        key=lambda image: image['id'],
    )
  return candidates, population


def prepare(
    annotations: pathlib.Path | str,
    images: pathlib.Path | str,
    output: pathlib.Path | str,
    count: int = 0,
    seed: int = 2026,
    size: int = _DEFAULT_IMAGE_SIZE,
    input_shape: tuple[int, int] | None = None,
    input_name: str = 'x',
    output_name: str = 'upsample_bilinear2d_1',
    annotation_sha256: str | None = None,
) -> dict[str, int]:
  """Writes shape-group manifests and returns their sample counts.

  Args:
    annotations: Path to COCO `instances_val2017.json`.
    images: Directory containing original `val2017` JPEG files.
    output: Destination directory for shape-group subdirectories and manifests.
    count: Number of images to sample (`0` selects all eligible images).
    seed: Random seed for deterministic subset selection.
    size: Target shorter-side resize dimension.
    input_shape: Optional `(height, width)` filter for resized dimensions, or
      `None` to include all resized shapes.
    input_name: Model signature input tensor binding name.
    output_name: Model signature output logits tensor binding name.
    annotation_sha256: Expected SHA-256 hex digest of `annotations`, or `None`
      to skip digest verification.

  Returns:
    A dictionary mapping each `'HEIGHTxWIDTH'` shape group key to its prepared
    sample count.

  Raises:
    ValueError: If the annotation digest, image metadata, or tensor dimensions
      are invalid.
    FileExistsError: If `output` already exists.
  """
  raw_annotations = pathlib.Path(annotations).read_bytes()
  digest = hashlib.sha256(raw_annotations).hexdigest()
  if annotation_sha256 is not None and digest != annotation_sha256:
    raise ValueError('Annotation SHA256 mismatch')
  metadata = json.loads(raw_annotations)
  selected, population = select_images(
      metadata['images'], count, seed, size, input_shape
  )
  by_image: collections.defaultdict[int, list[dict[str, Any]]] = (
      collections.defaultdict(list)
  )
  for annotation in metadata['annotations']:
    by_image[annotation['image_id']].append(annotation)
  sorted_images = sorted(metadata['images'], key=lambda image: image['id'])
  source_rows = {
      image['id']: index for index, image in enumerate(sorted_images)
  }
  output = pathlib.Path(output)
  output.mkdir(parents=True, exist_ok=False)
  selection = ''.join(f'{image["id"]}\n' for image in selected)
  selection_digest = hashlib.sha256(selection.encode('ascii')).hexdigest()
  (output / 'image_ids.txt').write_text(selection, encoding='ascii')
  total_images = len(metadata['images'])
  protocol = (
      f'COCO val2017; reference {REFERENCE_URL}; annotation_sha256={digest}; '
      f'selected={len(selected)}/{population} eligible, total={total_images}; '
      f'input_shape_filter={input_shape}; seed={seed}; '
      f'image_ids_sha256={selection_digest}; '
      f'COCO category IDs {COCO_CATEGORIES} remapped by index to VOC 0..20; '
      'unselected categories are background 0; include crowd annotations and '
      'images with no selected objects; overlaps of selected instances ignored '
      '(255), including same-class instances; target nearest-resized to model '
      'input grid; argmax logits without further resizing; global 21-class '
      'confusion matrix including background; report present-class count with '
      'mIoU'
  )
  manifests: dict[str, pb.DatasetManifest] = {}
  source_digests: list[str] = []
  for image_metadata in selected:
    height, width = image_metadata['height'], image_metadata['width']
    shape = resized_shape(height, width, size)
    if (
        _NUM_VOC_CLASSES * shape[0] * shape[1] * _FLOAT32_BYTES
        > _MAX_TENSOR_BYTES
    ):
      raise ValueError('Resized logits exceed the evaluator tensor size limit')
    name = image_metadata['file_name']
    if pathlib.Path(name).name != name:
      raise ValueError('Expected COCO image file_name without directories')
    image_path = pathlib.Path(images) / name
    source_bytes = image_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    source_digests.append(f'{source_hash}  {name}\n')
    with Image.open(image_path) as image:
      if image.size != (width, height):
        raise ValueError(
            'COCO image dimensions differ from annotation metadata'
        )
      image_annotations = by_image.get(image_metadata['id'], [])
      labels = semantic_mask(image_annotations, height, width)
      pixels, labels = transform(image, labels, size)
    key = f'{shape[0]}x{shape[1]}'
    manifest = manifests.get(key)
    if manifest is None:
      manifest = pb.DatasetManifest(version=1)
      manifest.inputs.add(
          name='image',
          binding_name=input_name,
          shape=[1, 3, *shape],
          type=pb.FLOAT32,
          layout=pb.NCHW,
      )
      prediction = manifest.outputs.add()
      prediction.tensor.CopyFrom(
          pb.TensorSpec(
              name='segmentation',
              binding_name=output_name,
              shape=[1, _NUM_VOC_CLASSES, *shape],
              type=pb.FLOAT32,
              layout=pb.NCHW,
          )
      )
      prediction.scoring.CopyFrom(
          pb.TaskOptions(
              task=pb.SEGMENTATION,
              classes=_NUM_VOC_CLASSES,
              ignore_label=_IGNORE_LABEL,
          )
      )
      manifest.source.repository = ANNOTATION_URL
      manifest.source.revision = digest
      manifest.source.split = 'val2017'
      manifest.source.seed = seed
      manifest.source.preprocessing = (
          f'RGB; Pillow {PIL.__version__} bilinear antialiased shorter-side '
          f'resize to {size}, aspect ratio preserved with integer output '
          'dimensions; float32 /255; mean [0.485,0.456,0.406], '
          'std [0.229,0.224,0.225]; NCHW'
      )
      manifest.source.protocol = (
          protocol + f'; this manifest contains only shape {key}'
      )
      manifests[key] = manifest
    identifier = f'coco_val2017_{image_metadata["id"]:012d}'
    directory = output / key / identifier
    directory.mkdir(parents=True)
    sample = manifest.samples.add(
        id=identifier,
        source_file=name,
        source_row=source_rows[image_metadata['id']],
    )
    data = sample.inputs.add(name='image')
    target = sample.targets.add(name='segmentation')
    target.tensor.CopyFrom(pb.TensorSpec(shape=labels.shape, type=pb.UINT8))
    for filename, array, identity in [
        ('input.raw', pixels, data.file),
        ('labels.raw', labels, target.file),
    ]:
      (directory / filename).write_bytes(array.tobytes())
      identity.path = f'{identifier}/{filename}'
  for key, manifest in manifests.items():
    path = output / key / 'manifest.txtpb'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(
        text_format.MessageToString(manifest), encoding='utf-8'
    )
    temporary.replace(path)
  (output / 'source_images.sha256').write_text(
      ''.join(source_digests), encoding='utf-8'
  )
  (output / 'protocol.txt').write_text(protocol + '\n', encoding='utf-8')
  return {
      key: len(manifest.samples) for key, manifest in sorted(manifests.items())
  }


def main(argv: Sequence[str]) -> None:
  """Runs COCO val2017 dataset preparation from command-line flags."""
  if len(argv) > 1:
    raise app.UsageError('Unexpected positional arguments')
  groups = prepare(
      annotations=_ANNOTATIONS.value,
      images=_IMAGES.value,
      output=_OUTPUT.value,
      count=_COUNT.value,
      seed=_SEED.value,
      size=_SIZE.value,
      input_shape=parse_shape(_INPUT_SHAPE.value),
      input_name=_INPUT_NAME.value,
      output_name=_OUTPUT_NAME.value,
      annotation_sha256=_ANNOTATION_SHA256.value,
  )
  total = sum(groups.values())
  print(
      f'Prepared {total} samples in {len(groups)} shape groups: {_OUTPUT.value}'
  )


if __name__ == '__main__':
  flags.mark_flags_as_required(
      ['annotations', 'images', 'output', 'annotation_sha256']
  )
  app.run(main)
