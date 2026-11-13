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
"""Prepares VOC validation Parquet rows for static TorchVision segmentation.

Writes FP32 NCHW inputs, original-resolution UINT8 labels, and a protobuf
manifest. Images are resized directly to the exported square input size.
"""

from collections.abc import Sequence
import io
import pathlib

from absl import app
from absl import flags
from google.protobuf import text_format
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from litert.tools.vision_eval import evaluation_pb2 as pb

# Default square spatial resolution in pixels for exported Pascal VOC inputs.
_DEFAULT_IMAGE_SIZE = 520
# Upper bound on square input spatial resolution in pixels.
_MAX_IMAGE_SIZE = 2048
# Number of Pascal VOC semantic segmentation classes (background + 20 classes).
_NUM_VOC_CLASSES = 21
# Pascal VOC semantic segmentation ignore/void label value (uint8).
_IGNORE_LABEL = 255

_INPUT_DIR = flags.DEFINE_string(
    'input_dir', None, 'Directory of validation-*.parquet shards'
)
_OUTPUT = flags.DEFINE_string('output', None, 'New prepared dataset directory')
_MAX_SAMPLES = flags.DEFINE_integer(
    'max_samples', 0, 'Number of rows; 0 selects all', lower_bound=0
)
_SIZE = flags.DEFINE_integer(
    'size',
    _DEFAULT_IMAGE_SIZE,
    'Square exported input size',
    lower_bound=1,
    upper_bound=_MAX_IMAGE_SIZE,
)
_INPUT_NAME = flags.DEFINE_string(
    'input_name', 'x', 'Model signature input name'
)
_OUTPUT_NAME = flags.DEFINE_string(
    'output_name', 'upsample_bilinear2d_1', 'Model signature logits name'
)
_REPO = flags.DEFINE_string(
    'repo', 'shijli/voc2012', 'Source dataset repository'
)
_REVISION = flags.DEFINE_string(
    'revision', None, 'Pinned source dataset commit'
)

_ALLOWED_VOC_LABELS: tuple[int, ...] = (
    *range(_NUM_VOC_CLASSES),
    _IGNORE_LABEL,
)
_IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STDDEV = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def voc_color(index: int) -> tuple[int, int, int]:
  """Computes the standard Pascal VOC RGB palette color for a label index.

  Args:
    index: Integer VOC palette index in `0..255`.

  Returns:
    An `(r, g, b)` tuple of integer channel values in `0..255`.
  """
  color = [0, 0, 0]
  for bit in range(8):
    for channel in range(3):
      color[channel] |= ((index >> channel) & 1) << (7 - bit)
    index >>= 3
  return color[0], color[1], color[2]


_VOC_COLOR_TO_LABEL: dict[tuple[int, int, int], int] = {
    voc_color(index): index for index in _ALLOWED_VOC_LABELS
}


def label_array(image: Image.Image) -> np.ndarray:
  """Preserves palette indices, or inverts an exact VOC RGB palette.

  Args:
    image: A PIL Image containing either single-channel VOC label indices or an
      RGB VOC palette mask.

  Returns:
    A contiguous 2D `uint8` array of VOC class indices (`0..20` or `255`).

  Raises:
    ValueError: If the image contains unknown RGB palette colors or labels
      outside `0..20` and `255`.
  """
  labels = np.asarray(image)
  if labels.ndim == 3 and labels.shape[2] == 3:
    decoded = np.empty(labels.shape[:2], dtype=np.uint8)
    for color in np.unique(labels.reshape(-1, 3), axis=0):
      key = (int(color[0]), int(color[1]), int(color[2]))
      label_index = _VOC_COLOR_TO_LABEL.get(key)
      if label_index is None:
        raise ValueError(f'Unknown VOC mask color: {key}')
      decoded[np.all(labels == color, axis=2)] = label_index
    labels = decoded
  if labels.ndim != 2 or not np.isin(labels, _ALLOWED_VOC_LABELS).all():
    raise ValueError(
        'VOC masks must contain labels 0..20 and optional void 255'
    )
  return np.ascontiguousarray(labels, dtype=np.uint8)


def prepare(
    input_dir: pathlib.Path | str,
    output: pathlib.Path | str,
    max_samples: int = 0,
    size: int = _DEFAULT_IMAGE_SIZE,
    input_name: str = 'x',
    output_name: str = 'upsample_bilinear2d_1',
    repo: str = '',
    revision: str = '',
) -> int:
  """Exports VOC validation Parquet rows to raw tensors and a manifest.

  Args:
    input_dir: Directory containing `validation-*.parquet` shards.
    output: Destination directory for the prepared dataset and `manifest.txtpb`.
    max_samples: Maximum number of rows to export (`0` selects all rows).
    size: Square spatial resolution for exported float32 inputs.
    input_name: Model signature input tensor binding name.
    output_name: Model signature output logits tensor binding name.
    repo: Source dataset repository recorded in the manifest.
    revision: Pinned source dataset commit recorded in the manifest.

  Returns:
    The number of prepared validation samples written to `output`.

  Raises:
    ValueError: If `max_samples` or `size` is out of range, no Parquet shards or
      rows are found, or image and mask dimensions do not match.
    FileExistsError: If `output` already exists.
  """
  if max_samples < 0 or not 1 <= size <= _MAX_IMAGE_SIZE:
    raise ValueError(
        'max_samples must be nonnegative and size must be in'
        f' 1..{_MAX_IMAGE_SIZE}; got max_samples={max_samples}, size={size}'
    )
  shards = sorted(pathlib.Path(input_dir).glob('validation-*.parquet'))
  if not shards:
    raise ValueError('No validation-*.parquet shards found')
  output = pathlib.Path(output)
  output.mkdir(parents=True, exist_ok=False)
  manifest = pb.DatasetManifest(version=1)
  manifest.inputs.add(
      name='image',
      binding_name=input_name,
      shape=[1, 3, size, size],
      type=pb.FLOAT32,
      layout=pb.NCHW,
  )
  prediction = manifest.outputs.add()
  prediction.tensor.CopyFrom(
      pb.TensorSpec(
          name='segmentation',
          binding_name=output_name,
          shape=[1, _NUM_VOC_CLASSES, size, size],
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
  manifest.source.repository = repo
  manifest.source.revision = revision
  manifest.source.split = 'validation'
  manifest.source.preprocessing = (
      f'RGB; Pillow bilinear direct resize to {size}x{size}; float32 /255; '
      'mean [0.485,0.456,0.406], std [0.229,0.224,0.225]; NCHW'
  )
  manifest.source.protocol = (
      'First validation rows in shard order; fixed-square model input; '
      'original-resolution VOC labels: background 0, foreground 1..20, '
      'void 255; half-pixel bilinear logit resize before argmax'
  )
  count = 0
  for shard in shards:
    source_row = 0
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=3):
      for row in batch.to_pylist():
        if max_samples and count >= max_samples:
          break
        image = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')
        labels = label_array(Image.open(io.BytesIO(row['mask']['bytes'])))
        if labels.shape != (image.height, image.width):
          raise ValueError('Image and label mask must have the same size')
        resized = image.resize((size, size), Image.Resampling.BILINEAR)
        pixels = np.asarray(resized, dtype=np.float32)
        normalized = (
            (pixels / 255 - _IMAGENET_MEAN) / _IMAGENET_STDDEV
        ).transpose(2, 0, 1)[None]
        pixels = np.ascontiguousarray(normalized, dtype='<f4')
        identifier = f'voc_validation_{count:06d}'
        directory = output / identifier
        directory.mkdir()
        sample = manifest.samples.add(
            id=identifier, source_file=shard.name, source_row=source_row
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
        count += 1
        source_row += 1
      if max_samples and count >= max_samples:
        break
    if max_samples and count >= max_samples:
      break
  if not count:
    raise ValueError('No validation rows found')
  path = output / 'manifest.txtpb'
  temporary = output / 'manifest.tmp'
  temporary.write_text(text_format.MessageToString(manifest), encoding='utf-8')
  temporary.replace(path)
  return count


def main(argv: Sequence[str]) -> None:
  """Runs VOC 2012 dataset preparation from command-line flags."""
  if len(argv) > 1:
    raise app.UsageError('Unexpected positional arguments')
  count = prepare(
      input_dir=_INPUT_DIR.value,
      output=_OUTPUT.value,
      max_samples=_MAX_SAMPLES.value,
      size=_SIZE.value,
      input_name=_INPUT_NAME.value,
      output_name=_OUTPUT_NAME.value,
      repo=_REPO.value,
      revision=_REVISION.value,
  )
  print(
      f'Prepared {count} samples:'
      f' {pathlib.Path(_OUTPUT.value) / "manifest.txtpb"}'
  )


if __name__ == '__main__':
  flags.mark_flags_as_required(['input_dir', 'output', 'revision'])
  app.run(main)
