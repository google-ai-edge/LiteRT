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
"""Prepare a reproducible TIPSv2 subset from pinned ADE20K/NYUv2 Parquet.

This optional host adapter writes raw tensors and a typed protobuf manifest.
Inference and scoring require only litert_vision_eval. See README.md for setup.
"""

import io
import pathlib

from absl import app
from absl import flags
import evaluation_pb2 as pb
from google.protobuf import text_format
import numpy as np

_DATASET = flags.DEFINE_enum('dataset', None, ['nyuv2', 'ade20k'], 'Dataset layout')
_REPO = flags.DEFINE_string('repo', None, 'Hugging Face dataset repository')
_REVISION = flags.DEFINE_string('revision', None, 'Dataset revision; resolved to a commit')
_SPLIT = flags.DEFINE_string('split', None, 'Dataset split')
_COUNT = flags.DEFINE_integer('count', 3, 'Number of samples', lower_bound=1)
_SEED = flags.DEFINE_integer('seed', 2026, 'Uniform subset selection seed')
_NORMAL_AXES = flags.DEFINE_string(
    'normal_axis_order', None,
    'Enable normals with a signed one-based permutation, for example 1,-2,3')
_OUTPUT = flags.DEFINE_string('output', None, 'New prepared dataset directory')
_INPUT_NAME = flags.DEFINE_string('input_name', 'pixel_values', 'Model signature input name')
_FORMAT = flags.DEFINE_enum('manifest_format', 'text', ['text', 'binary'], 'Manifest encoding')


def normal_axis_order(text):
  """Parse an explicit mapping from label axes to the model's normal axes."""
  if text is None:
    return None
  axes = [int(axis) for axis in text.split(',')]
  if sorted(abs(axis) for axis in axes) != [1, 2, 3]:
    raise ValueError('Expected a signed permutation of 1,2,3')
  return axes


def tensor(name, shape, element_type=pb.FLOAT32):
  return pb.TensorSpec(name=name, shape=shape, type=element_type, layout=pb.NCHW)


def write_tensor(path, array):
  """Write contiguous little-endian bytes and return their identity."""
  array = np.ascontiguousarray(array)
  data = array.tobytes()
  path.write_bytes(data)
  return pb.FileIdentity(path=path.name)


def new_manifest(dataset, repo, revision, split, seed, axes, input_name):
  manifest = pb.DatasetManifest(version=1)
  manifest.inputs.add().CopyFrom(tensor('image', [1, 3, 448, 448]))
  manifest.inputs[0].binding_name = input_name
  for name, channels, size, task in [
      ('depth', 1, 448, pb.DEPTH),
      ('normals', 3, 448, pb.NORMALS),
      ('segmentation_logits', 150, 256, pb.SEGMENTATION),
  ]:
    output = manifest.outputs.add()
    output.tensor.CopyFrom(tensor(name, [1, channels, size, size]))
    output.scoring.task = task
    if task == pb.DEPTH:
      output.scoring.min_depth = .001
      output.scoring.max_depth = 10.
    if task == pb.SEGMENTATION:
      output.scoring.classes = 150
      output.scoring.ignore_label = 255
  manifest.source.repository = repo
  manifest.source.revision = revision
  manifest.source.split = split
  manifest.source.seed = seed
  manifest.source.preprocessing = (
      'RGB, Pillow bilinear resize to 448x448, float32 [0,1], NCHW; '
      'uniform sample without replacement across the complete split')
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


def add_sample(manifest, row, dataset, directory, identifier, axes,
               source_file='', source_row=0):
  # Optional dependencies stay on the preparation path.
  from PIL import Image

  directory.mkdir(parents=True)
  if dataset == 'nyuv2':
    rgb = np.asarray(row['image'], dtype=np.float32)
    if (rgb.shape != (3, 288, 384) or not np.isfinite(rgb).all()
        or rgb.min() < 0 or rgb.max() > 1):
      raise ValueError('Expected MTAN-format NYUv2 RGB CHW 288x384 in [0,1]')
    image = Image.fromarray(np.rint(rgb.transpose(1, 2, 0) * 255).astype(np.uint8))
    depth = np.asarray(row['depth'], dtype='<f4')
    if depth.shape != (1, 288, 384):
      raise ValueError('Expected NYUv2 depth [1,288,384] in meters')
    targets = {'depth': depth[0]}
    if axes is not None:
      labels = np.asarray(row['normal'], dtype='<f4')
      if labels.shape != (3, 288, 384):
        raise ValueError('Expected NYUv2 normal labels [3,288,384]')
      targets['normals'] = np.stack([
          np.sign(axis) * labels[abs(axis) - 1] for axis in axes
      ]).astype('<f4')
  else:
    image = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')
    labels = np.asarray(Image.open(io.BytesIO(row['annotation']['bytes'])), dtype=np.int32)
    if labels.ndim != 2 or labels.min() < 0 or labels.max() > 150:
      raise ValueError('Expected original ADE20K integer HW labels 0..150')
    targets = {'segmentation_logits': np.where(labels == 0, 255, labels - 1).astype(np.uint8)}
  pixels = np.asarray(image.resize((448, 448), Image.Resampling.BILINEAR), dtype=np.float32)
  pixels = np.ascontiguousarray((pixels / 255).transpose(2, 0, 1)[None], dtype='<f4')
  sample = manifest.samples.add(id=identifier, source_file=source_file, source_row=source_row)
  source = sample.inputs.add(name='image')
  source.file.CopyFrom(write_tensor(directory / 'input.raw', pixels))
  source.file.path = f'{identifier}/input.raw'
  for name, values in targets.items():
    target = sample.targets.add(name=name)
    target.tensor.CopyFrom(tensor(name, values.shape, pb.UINT8 if values.dtype == np.uint8 else pb.FLOAT32))
    target.file.CopyFrom(write_tensor(directory / f'{name}.raw', values))
    target.file.path = f'{identifier}/{name}.raw'


def prepare(dataset, repo, revision, split, count, seed, axes, output,
            input_name='pixel_values', manifest_format='text'):
  import huggingface_hub
  import pyarrow.parquet as pq

  if count <= 0:
    raise ValueError('count must be positive')
  output = pathlib.Path(output)
  # Partial preparations are not silently mixed with another selection.
  output.mkdir(parents=True, exist_ok=False)
  info = huggingface_hub.HfApi().dataset_info(repo, revision=revision)
  prefix = f'data/{split}-' if dataset == 'nyuv2' else f'scene_parsing/{split}/'
  files = sorted(s.rfilename for s in info.siblings
                 if s.rfilename.startswith(prefix) and s.rfilename.endswith('.parquet'))
  if not files:
    raise ValueError('No matching Parquet shards in the selected snapshot')
  paths = [
      pathlib.Path(
          huggingface_hub.hf_hub_download(
              repo, filename, revision=info.sha, repo_type='dataset'
          )
      )
      for filename in files
  ]
  sizes = [pq.ParquetFile(path).metadata.num_rows for path in paths]
  if count > sum(sizes):
    raise ValueError('Requested more samples than the complete split contains')
  selected = set(np.random.default_rng(seed).choice(sum(sizes), count, replace=False).tolist())
  manifest = new_manifest(dataset, repo, info.sha, split, seed, axes, input_name)
  offset = 0
  for filename, path, size in zip(files, paths, sizes):
    rows = sorted(index - offset for index in selected if offset <= index < offset + size)
    offset += size
    if not rows:
      continue
    columns = ['image', 'depth', 'normal'] if dataset == 'nyuv2' else ['image', 'annotation']
    table = pq.read_table(path, columns=columns).take(rows)
    for index, row in zip(rows, table.to_pylist()):
      identifier = f'{dataset}_{split}_{offset-size+index:06d}'
      add_sample(manifest, row, dataset, output / identifier, identifier, axes, filename, index)
  path = output / ('manifest.binpb' if manifest_format == 'binary' else 'manifest.txtpb')
  data = manifest.SerializeToString() if manifest_format == 'binary' else text_format.MessageToString(manifest).encode()
  temporary = path.with_suffix('.tmp')
  temporary.write_bytes(data)
  temporary.replace(path)
  return path


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Unexpected positional arguments')
  try:
    axes = normal_axis_order(_NORMAL_AXES.value)
    path = prepare(_DATASET.value, _REPO.value, _REVISION.value, _SPLIT.value,
                   _COUNT.value, _SEED.value, axes, _OUTPUT.value,
                   _INPUT_NAME.value, _FORMAT.value)
  except ValueError as error:
    raise app.UsageError(str(error)) from error
  print(path)


if __name__ == '__main__':
  flags.mark_flags_as_required(['dataset', 'repo', 'revision', 'split', 'output'])
  app.run(main)
