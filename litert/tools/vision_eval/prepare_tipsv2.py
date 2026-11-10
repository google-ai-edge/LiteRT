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
"""Prepares a reproducible TIPSv2 subset from pinned ADE20K/NYUv2 Parquet.

This optional host adapter writes raw tensors and a typed protobuf manifest.
Inference and scoring require only litert_vision_eval. See README.md for setup.
"""

from collections.abc import Mapping, Sequence
import io
import pathlib
from typing import Any

from absl import app
from absl import flags
from google.protobuf import text_format
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from litert.tools.vision_eval import evaluation_pb2 as pb

# Square input spatial resolution in pixels expected by TIPSv2 dense heads.
_INPUT_SIZE = 448
# Spatial resolution in pixels of TIPSv2 semantic segmentation output logits.
_SEGMENTATION_LOGITS_SIZE = 256
# Number of ADE20K semantic segmentation foreground classes (1..150 -> 0..149).
_ADE20K_CLASSES = 150
# Semantic segmentation ignore/void label value (uint8).
_IGNORE_LABEL = 255

_DATASET = flags.DEFINE_enum(
    'dataset', None, ['nyuv2', 'ade20k'], 'Dataset layout'
)
_REPO = flags.DEFINE_string('repo', None, 'Hugging Face dataset repository')
_REVISION = flags.DEFINE_string(
    'revision', None, 'Dataset revision; resolved to a commit'
)
_SPLIT = flags.DEFINE_string('split', None, 'Dataset split')
_COUNT = flags.DEFINE_integer(
    'count', 0, 'Number of samples; 0 selects all', lower_bound=0
)
_SEED = flags.DEFINE_integer('seed', 2026, 'Uniform subset selection seed')
_NORMAL_AXIS_ORDER = flags.DEFINE_string(
    'normal_axis_order',
    '',
    'Enable normals with a signed one-based permutation, for example 1,-2,3',
)
_OUTPUT = flags.DEFINE_string('output', None, 'New prepared dataset directory')
_INPUT_NAME = flags.DEFINE_string(
    'input_name', 'pixel_values', 'Model signature input name'
)
_MANIFEST_FORMAT = flags.DEFINE_enum(
    'manifest_format', 'text', ['text', 'binary'], 'Manifest encoding'
)


def normal_axis_order(text: str | None) -> list[int] | None:
  """Parses an explicit mapping from label axes to the model's normal axes.

  Args:
    text: Comma-separated signed 1-based permutation of `1,2,3` (e.g.
      `'1,-2,3'`), or `None`/`''` to disable surface normals.

  Returns:
    A 3-element list of signed 1-based axis indices, or `None` if `text` is
    empty or `None`.

  Raises:
    ValueError: If `text` is non-empty and not a signed permutation of `1,2,3`.
  """
  if not text:
    return None
  axes = [int(axis) for axis in text.split(',')]
  if sorted(abs(axis) for axis in axes) != [1, 2, 3]:
    raise ValueError('Expected a signed permutation of 1,2,3')
  return axes


def tensor(
    name: str, shape: Sequence[int], element_type: int = pb.FLOAT32
) -> pb.TensorSpec:
  """Constructs an NCHW TensorSpec protobuf with the given name and shape."""
  return pb.TensorSpec(
      name=name, shape=shape, type=element_type, layout=pb.NCHW
  )


def write_tensor(path: pathlib.Path, array: np.ndarray) -> pb.FileIdentity:
  """Writes contiguous little-endian bytes and returns their identity."""
  array = np.ascontiguousarray(array)
  data = array.tobytes()
  path.write_bytes(data)
  return pb.FileIdentity(path=path.name)


def new_manifest(
    dataset: str,
    repo: str,
    revision: str,
    split: str,
    seed: int,
    axes: Sequence[int] | None,
    input_name: str,
) -> pb.DatasetManifest:
  """Creates a TIPSv2 DatasetManifest initialized with tensor and task specs.

  Args:
    dataset: Dataset identifier (`'nyuv2'` or `'ade20k'`).
    repo: Source Hugging Face dataset repository.
    revision: Resolved source dataset commit hash.
    split: Source dataset split name.
    seed: Random seed used for uniform subset selection.
    axes: Optional signed 1-based normal axis permutation, or `None` when
      surface normal targets are omitted.
    input_name: Model signature input tensor binding name.

  Returns:
    An initialized `pb.DatasetManifest` message.
  """
  manifest = pb.DatasetManifest(version=1)
  manifest.inputs.add().CopyFrom(
      tensor('image', [1, 3, _INPUT_SIZE, _INPUT_SIZE])
  )
  manifest.inputs[0].binding_name = input_name
  for name, channels, size, task in [
      ('depth', 1, _INPUT_SIZE, pb.DEPTH),
      ('normals', 3, _INPUT_SIZE, pb.NORMALS),
      (
          'segmentation_logits',
          _ADE20K_CLASSES,
          _SEGMENTATION_LOGITS_SIZE,
          pb.SEGMENTATION,
      ),
  ]:
    output = manifest.outputs.add()
    output.tensor.CopyFrom(tensor(name, [1, channels, size, size]))
    output.scoring.task = task
    if task == pb.DEPTH:
      output.scoring.min_depth = 0.001
      output.scoring.max_depth = 10.0
    if task == pb.SEGMENTATION:
      output.scoring.classes = _ADE20K_CLASSES
      output.scoring.ignore_label = _IGNORE_LABEL
  manifest.source.repository = repo
  manifest.source.revision = revision
  manifest.source.split = split
  manifest.source.seed = seed
  manifest.source.preprocessing = (
      'RGB, Pillow bilinear resize to 448x448, float32 [0,1], NCHW; '
      'uniform sample without replacement across the complete split'
  )
  if dataset == 'nyuv2':
    manifest.source.protocol = (
        'MTAN NYUv2 CHW 288x384 labels, depth in meters, no additional crop; '
        'normals require an explicit label-to-model axis mapping'
    )
  else:
    manifest.source.protocol = (
        'ADE20K original-size labels; 0 ignored and 1..150 mapped to 0..149; '
        'half-pixel bilinear logit resize before argmax'
    )
  if axes is not None:
    manifest.source.normal_axis_order.extend(axes)
  return manifest


def add_sample(
    manifest: pb.DatasetManifest,
    row: Mapping[str, Any],
    dataset: str,
    directory: pathlib.Path,
    identifier: str,
    axes: Sequence[int] | None,
    source_file: str = '',
    source_row: int = 0,
) -> None:
  """Writes one sample's input and target tensors and appends it to manifest.

  Args:
    manifest: DatasetManifest protobuf to append the prepared sample to.
    row: Decoded Parquet row dictionary.
    dataset: Dataset layout (`'nyuv2'` or `'ade20k'`).
    directory: Sample output directory to create and populate with `.raw` files.
    identifier: Unique sample identifier and relative subdirectory name.
    axes: Optional signed 1-based normal axis permutation for NYUv2, or `None`
      to omit normal targets.
    source_file: Relative Parquet shard filename recorded in the sample entry.
    source_row: Zero-based row index within `source_file`.

  Raises:
    ValueError: If row arrays or label values do not match the dataset contract.
  """
  directory.mkdir(parents=True)
  if dataset == 'nyuv2':
    rgb = np.asarray(row['image'], dtype=np.float32)
    if (
        rgb.shape != (3, 288, 384)
        or not np.isfinite(rgb).all()
        or rgb.min() < 0
        or rgb.max() > 1
    ):
      raise ValueError('Expected MTAN-format NYUv2 RGB CHW 288x384 in [0,1]')
    quantized = np.rint(rgb.transpose(1, 2, 0) * 255).astype(np.uint8)
    image = Image.fromarray(quantized)
    depth = np.asarray(row['depth'], dtype='<f4')
    if depth.shape != (1, 288, 384):
      raise ValueError('Expected NYUv2 depth [1,288,384] in meters')
    targets = {'depth': depth[0]}
    if axes is not None:
      labels = np.asarray(row['normal'], dtype='<f4')
      if labels.shape != (3, 288, 384):
        raise ValueError('Expected NYUv2 normal labels [3,288,384]')
      targets['normals'] = np.stack(
          [np.sign(axis) * labels[abs(axis) - 1] for axis in axes]
      ).astype('<f4')
  else:
    image = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')
    annotation_image = Image.open(io.BytesIO(row['annotation']['bytes']))
    labels = np.asarray(annotation_image, dtype=np.int32)
    if labels.ndim != 2 or labels.min() < 0 or labels.max() > _ADE20K_CLASSES:
      raise ValueError('Expected original ADE20K integer HW labels 0..150')
    remapped = np.where(labels == 0, _IGNORE_LABEL, labels - 1).astype(np.uint8)
    targets = {'segmentation_logits': remapped}
  resized = image.resize((_INPUT_SIZE, _INPUT_SIZE), Image.Resampling.BILINEAR)
  pixels = np.asarray(resized, dtype=np.float32)
  pixels = np.ascontiguousarray(
      (pixels / 255).transpose(2, 0, 1)[None], dtype='<f4'
  )
  sample = manifest.samples.add(
      id=identifier, source_file=source_file, source_row=source_row
  )
  source = sample.inputs.add(name='image')
  source.file.CopyFrom(write_tensor(directory / 'input.raw', pixels))
  source.file.path = f'{identifier}/input.raw'
  for name, values in targets.items():
    target = sample.targets.add(name=name)
    element_type = pb.UINT8 if values.dtype == np.uint8 else pb.FLOAT32
    target.tensor.CopyFrom(tensor(name, values.shape, element_type))
    target.file.CopyFrom(write_tensor(directory / f'{name}.raw', values))
    target.file.path = f'{identifier}/{name}.raw'


def prepare_shards(
    dataset: str,
    repo: str,
    revision: str,
    split: str,
    shards: Sequence[tuple[str, pathlib.Path | str]],
    count: int,
    seed: int,
    axes: Sequence[int] | None,
    output: pathlib.Path | str,
    input_name: str = 'pixel_values',
    manifest_format: str = 'text',
) -> pathlib.Path:
  """Prepares a reproducible TIPSv2 evaluation subset from local Parquet shards.

  Args:
    dataset: Dataset layout (`'nyuv2'` or `'ade20k'`).
    repo: Source dataset repository ID recorded in the manifest.
    revision: Resolved source dataset commit hash recorded in the manifest.
    split: Dataset split name recorded in the manifest.
    shards: Ordered sequence of `(source_filename, local_parquet_path)` pairs
      comprising the split.
    count: Number of samples to select without replacement (`0` selects all).
    seed: Random seed for uniform subset selection.
    axes: Optional signed 1-based normal axis permutation for NYUv2, or `None`
      to omit normal targets.
    output: Destination directory for the prepared dataset and manifest.
    input_name: Model signature input tensor binding name.
    manifest_format: Manifest serialization format (`'text'` or `'binary'`).

  Returns:
    The path to the written `manifest.txtpb` or `manifest.binpb` file.

  Raises:
    ValueError: If `count` is negative, `shards` is empty, or `count` exceeds
      the total number of rows across `shards`.
    FileExistsError: If `output` already exists.
  """
  if count < 0:
    raise ValueError(f'count must be nonnegative; got count={count}')
  if not shards:
    raise ValueError('No matching Parquet shards in the selected snapshot')
  files = [filename for filename, _ in shards]
  paths = [pathlib.Path(path) for _, path in shards]
  sizes = [pq.ParquetFile(path).metadata.num_rows for path in paths]
  total_rows = sum(sizes)
  if count > total_rows:
    raise ValueError(
        'Requested more samples than the complete split contains; got'
        f' count={count}, total_rows={total_rows}'
    )
  output = pathlib.Path(output)
  # Partial preparations are not silently mixed with another selection.
  output.mkdir(parents=True, exist_ok=False)
  if count:
    rng = np.random.default_rng(seed)
    selected = set(rng.choice(total_rows, count, replace=False).tolist())
  else:
    selected = set(range(total_rows))
  manifest = new_manifest(
      dataset, repo, revision, split, seed, axes, input_name
  )
  offset = 0
  for filename, path, size in zip(files, paths, sizes, strict=True):
    rows = sorted(
        index - offset for index in selected if offset <= index < offset + size
    )
    offset += size
    if not rows:
      continue
    if dataset == 'nyuv2':
      columns = ['image', 'depth', 'normal']
    else:
      columns = ['image', 'annotation']
    table = pq.read_table(path, columns=columns).take(rows)
    for index, row in zip(rows, table.to_pylist(), strict=True):
      identifier = f'{dataset}_{split}_{offset-size+index:06d}'
      add_sample(
          manifest,
          row,
          dataset,
          output / identifier,
          identifier,
          axes,
          filename,
          index,
      )
  manifest_name = (
      'manifest.binpb' if manifest_format == 'binary' else 'manifest.txtpb'
  )
  path = output / manifest_name
  if manifest_format == 'binary':
    data = manifest.SerializeToString()
  else:
    data = text_format.MessageToString(manifest).encode()
  temporary = path.with_suffix('.tmp')
  temporary.write_bytes(data)
  temporary.replace(path)
  return path


def prepare(
    dataset: str,
    repo: str,
    revision: str,
    split: str,
    count: int,
    seed: int,
    axes: Sequence[int] | None,
    output: pathlib.Path | str,
    input_name: str = 'pixel_values',
    manifest_format: str = 'text',
) -> pathlib.Path:
  """Downloads and prepares a reproducible TIPSv2 evaluation subset.

  Args:
    dataset: Dataset layout (`'nyuv2'` or `'ade20k'`).
    repo: Hugging Face dataset repository ID.
    revision: Pinned dataset revision or commit hash.
    split: Dataset split to sample from.
    count: Number of samples to select without replacement (`0` selects all).
    seed: Random seed for uniform subset selection.
    axes: Optional signed 1-based normal axis permutation for NYUv2, or `None`
      to omit normal targets.
    output: Destination directory for the prepared dataset and manifest.
    input_name: Model signature input tensor binding name.
    manifest_format: Manifest serialization format (`'text'` or `'binary'`).

  Returns:
    The path to the written `manifest.txtpb` or `manifest.binpb` file.

  Raises:
    ValueError: If `count` is negative, no matching Parquet shards exist, or
      `count` exceeds the total split size.
    FileExistsError: If `output` already exists.
  """
  import huggingface_hub  # pylint: disable=g-import-not-at-top

  if count < 0:
    raise ValueError(f'count must be nonnegative; got count={count}')
  output = pathlib.Path(output)
  if output.exists():
    raise FileExistsError(f'Output directory already exists: {output}')
  info = huggingface_hub.HfApi().dataset_info(repo, revision=revision)
  prefix = f'data/{split}-' if dataset == 'nyuv2' else f'scene_parsing/{split}/'
  files = sorted(
      s.rfilename
      for s in info.siblings
      if s.rfilename.startswith(prefix) and s.rfilename.endswith('.parquet')
  )
  if not files:
    raise ValueError('No matching Parquet shards in the selected snapshot')
  shards = [
      (
          filename,
          pathlib.Path(
              huggingface_hub.hf_hub_download(
                  repo, filename, revision=info.sha, repo_type='dataset'
              )
          ),
      )
      for filename in files
  ]
  return prepare_shards(
      dataset=dataset,
      repo=repo,
      revision=info.sha,
      split=split,
      shards=shards,
      count=count,
      seed=seed,
      axes=axes,
      output=output,
      input_name=input_name,
      manifest_format=manifest_format,
  )


def main(argv: Sequence[str]) -> None:
  """Runs TIPSv2 dataset preparation from command-line flags."""
  if len(argv) > 1:
    raise app.UsageError('Unexpected positional arguments')
  try:
    axes = normal_axis_order(_NORMAL_AXIS_ORDER.value)
    path = prepare(
        dataset=_DATASET.value,
        repo=_REPO.value,
        revision=_REVISION.value,
        split=_SPLIT.value,
        count=_COUNT.value,
        seed=_SEED.value,
        axes=axes,
        output=_OUTPUT.value,
        input_name=_INPUT_NAME.value,
        manifest_format=_MANIFEST_FORMAT.value,
    )
  except ValueError as error:
    raise app.UsageError(str(error)) from error
  print(path)


if __name__ == '__main__':
  flags.mark_flags_as_required(
      ['dataset', 'repo', 'revision', 'split', 'output']
  )
  app.run(main)
