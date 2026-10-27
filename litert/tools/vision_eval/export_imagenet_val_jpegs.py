#!/usr/bin/env python3
# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extracts original validation image bytes from downloaded HF Parquet shards."""

from collections.abc import Sequence
import pathlib

from absl import app
from absl import flags
import pyarrow.parquet as pq

_INPUT_DIR = flags.DEFINE_string(
    'input_dir', None, 'Downloaded validation Parquet shards'
)
_OUT_DIR = flags.DEFINE_string('out_dir', None, 'Prepared dataset directory')
_MAX_SAMPLES = flags.DEFINE_integer(
    'max_samples', 0, 'Number of images; 0 selects all', lower_bound=0
)


def export(
    shards: pathlib.Path | str,
    out_dir: pathlib.Path | str,
    max_samples: int = 0,
) -> int:
  """Writes encoded images and a TSV manifest; zero selects all rows.

  Args:
    shards: Directory containing `validation-*.parquet` shards.
    out_dir: Destination directory for `images/*.JPEG` and `manifest.tsv`.
    max_samples: Maximum number of images to export (`0` exports all rows).

  Returns:
    The number of exported images written to `out_dir`.

  Raises:
    ValueError: If no `validation-*.parquet` shards exist, `max_samples` is
      negative, or a row has empty image bytes or a label outside `0..999`.
  """
  shard_paths = sorted(pathlib.Path(shards).glob('validation-*.parquet'))
  if not shard_paths:
    raise ValueError('No validation-*.parquet shards found')
  if max_samples < 0:
    raise ValueError('max_samples must be nonnegative')
  out_dir = pathlib.Path(out_dir)
  images_dir = out_dir / 'images'
  images_dir.mkdir(parents=True, exist_ok=True)
  count = 0
  with (out_dir / 'manifest.tsv').open('w', encoding='utf-8') as manifest:
    manifest.write('id\tlabel\tpath\n')
    for shard in shard_paths:
      for batch in pq.ParquetFile(shard).iter_batches(
          batch_size=32, columns=['image', 'label']
      ):
        for row in batch.to_pylist():
          encoded = row['image']['bytes']
          label = int(row['label'])
          if not encoded or not 0 <= label < 1000:
            raise ValueError('Expected encoded image and label 0..999')
          relative = f'images/{count:08d}.JPEG'
          (out_dir / relative).write_bytes(encoded)
          manifest.write(f'{count}\t{label}\t{relative}\n')
          count += 1
          if max_samples and count >= max_samples:
            return count
  return count


def main(argv: Sequence[str]) -> None:
  """Runs ImageNet validation JPEG export from command-line flags."""
  if len(argv) > 1:
    raise app.UsageError('Unexpected positional arguments')
  count = export(
      shards=_INPUT_DIR.value,
      out_dir=_OUT_DIR.value,
      max_samples=_MAX_SAMPLES.value,
  )
  print(f'Wrote {count} images')
  print(pathlib.Path(_OUT_DIR.value) / 'manifest.tsv')


if __name__ == '__main__':
  flags.mark_flags_as_required(['input_dir', 'out_dir'])
  app.run(main)
