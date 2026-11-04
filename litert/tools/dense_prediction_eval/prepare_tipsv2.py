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
"""Example adapter for pinned ADE20K/NYUv2 HF Parquet snapshots and TIPSv2."""
import argparse
import io
from pathlib import Path

import numpy as np

from litert.tools.dense_prediction_eval.evaluate import digest, write_json

OUTPUTS = dict(depth=dict(shape=[1, 1, 448, 448],
                          dtype='float32',
                          task='depth',
                          min_depth=.001,
                          max_depth=10.),
               normals=dict(shape=[1, 3, 448, 448],
                            dtype='float32',
                            task='normals'),
               segmentation_logits=dict(shape=[1, 150, 256, 256],
                                        dtype='float32',
                                        task='segmentation',
                                        classes=150,
                                        ignore_label=255))


def normal_axis_order(text):
  """Parse a signed, one-based permutation from label axes to model axes."""
  axes = [int(x) for x in text.split(',')]
  if sorted(abs(x) for x in axes) != [1, 2, 3]:
    raise argparse.ArgumentTypeError('Expected a signed permutation of 1,2,3')
  return axes


def prepare(args):
  # Optional dependencies: inference and scoring do not import the HF client.
  from huggingface_hub import HfApi, hf_hub_download
  from PIL import Image
  import pyarrow.parquet as pq

  if args.count <= 0:
    raise ValueError('count must be positive')
  if (args.output / 'manifest.json').exists():
    raise ValueError('Use a new directory; prepared datasets are immutable')
  info = HfApi().dataset_info(args.repo, revision=args.revision)
  prefix = f'data/{args.split}-' if args.dataset == 'nyuv2' else f'scene_parsing/{args.split}/'
  files = sorted(
      s.rfilename
      for s in info.siblings
      if s.rfilename.startswith(prefix) and s.rfilename.endswith('.parquet'))
  if not files:
    raise ValueError(
        'No matching Parquet files; select the dataset snapshot containing Parquet data'
    )
  paths = [
      Path(hf_hub_download(args.repo, f, revision=info.sha,
                           repo_type='dataset')) for f in files
  ]
  sizes = [pq.ParquetFile(p).metadata.num_rows for p in paths]
  if args.count > sum(sizes):
    raise ValueError('Requested more samples than the split contains')
  selected = set(
      np.random.default_rng(args.seed).choice(sum(sizes),
                                              args.count,
                                              replace=False).tolist())
  samples = []
  offset = 0
  for filename, path, count in zip(files, paths, sizes):
    rows = sorted(i - offset for i in selected if offset <= i < offset + count)
    offset += count
    if not rows:
      continue
    columns = ['image', 'depth', 'normal'
              ] if args.dataset == 'nyuv2' else ['image', 'annotation']
    table = pq.read_table(path, columns=columns).take(rows)
    for row_index, row in zip(rows, table.to_pylist()):
      identifier = f'{args.dataset}_{args.split}_{offset-count+row_index:06d}'
      directory = args.output / identifier
      directory.mkdir(parents=True, exist_ok=True)
      targets = {}
      if args.dataset == 'nyuv2':
        rgb = np.asarray(row['image'], dtype=np.float32)
        if rgb.shape != (3, 288, 384) or not np.isfinite(rgb).all() or rgb.min(
        ) < 0 or rgb.max() > 1:
          raise ValueError(
              'Expected MTAN-format NYUv2 RGB in [0,1], CHW 288x384')
        image = Image.fromarray(
            np.rint(rgb.transpose(1, 2, 0) * 255).astype(np.uint8))
        values = dict(depth=np.asarray(row['depth'], dtype=np.float32)[0])
        if args.normal_axis_order is not None:
          labels = np.asarray(row['normal'], dtype=np.float32)
          values['normals'] = np.stack([
              np.sign(axis) * labels[abs(axis) - 1]
              for axis in args.normal_axis_order
          ])
      else:
        image = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')
        labels = np.asarray(Image.open(io.BytesIO(row['annotation']['bytes'])),
                            dtype=np.int32)
        if labels.min() < 0 or labels.max() > 150:
          raise ValueError('Expected original ADE20K labels 0..150')
        values = dict(segmentation_logits=np.where(labels == 0, 255, labels -
                                                   1).astype(np.uint8))
      for name, value in values.items():
        np.save(directory / (name + '.npy'), value, allow_pickle=False)
        targets[name] = f'{identifier}/{name}.npy'
      pixels = np.asarray(image.resize((448, 448), Image.Resampling.BILINEAR),
                          dtype=np.float32)
      pixels = np.ascontiguousarray((pixels / 255).transpose(2, 0, 1)[None],
                                    dtype='<f4')
      pixels.tofile(directory / 'input.raw')
      samples.append(
          dict(id=identifier,
               inputs={'pixel_values': f'{identifier}/input.raw'},
               input_sha256=digest(directory / 'input.raw'),
               targets=targets,
               source_file=filename,
               source_row=row_index))
  manifest = dict(
      version=1,
      outputs=OUTPUTS,
      samples=samples,
      input=dict(shape=[1, 3, 448, 448], dtype='float32'),
      source=dict(
          repo=args.repo,
          revision=info.sha,
          split=args.split,
          seed=args.seed,
          selection='Uniform random without replacement over the complete split'
      ),
      preprocessing=dict(resize=[448, 448],
                         interpolation='PIL bilinear',
                         rgb=True,
                         scale=1 / 255,
                         mean_std_normalization=False),
      normal_axis_order=args.normal_axis_order,
      protocol=
      'MTAN processed NYUv2, 288x384 labels, meters, explicit normal axis mapping when enabled, no additional crop'
      if args.dataset == 'nyuv2' else
      'ADE20K original-size labels; background ignored; bilinear resize logits before argmax'
  )
  write_json(args.output / 'manifest.json', manifest)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--dataset', choices=('nyuv2', 'ade20k'), required=True)
  parser.add_argument('--repo', required=True)
  parser.add_argument('--revision', required=True)
  parser.add_argument('--split', required=True)
  parser.add_argument('--count', type=int, required=True)
  parser.add_argument('--seed', type=int, default=2026)
  parser.add_argument(
      '--normal-axis-order',
      type=normal_axis_order,
      help='Enable normal targets with an explicit signed axis permutation; '
      'use 1,2,3 only when source and model conventions match')
  parser.add_argument('--output', type=Path, required=True)
  prepare(parser.parse_args())


if __name__ == '__main__':
  main()
