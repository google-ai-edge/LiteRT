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
"""Score named dense outputs saved by run_model or a reference implementation."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from litert.tools.dense_prediction_eval import metrics


def digest(path):
  with open(path, 'rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
  temporary.replace(path)


def load_manifest(path):
  path = Path(path).resolve()
  data = json.loads(path.read_text())
  if data.get(
      'version') != 1 or not data.get('samples') or not data.get('outputs'):
    raise ValueError(
        'Expected a version-1 manifest with samples and output contracts')
  identifiers = [s['id'] for s in data['samples']]
  if len(set(identifiers)) != len(identifiers) or any(
      not re.fullmatch(r'[A-Za-z0-9_-]+', s) for s in identifiers):
    raise ValueError(
        'Sample IDs must be unique and contain only letters, digits, _ or -')
  for name, contract in data['outputs'].items():
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
      raise ValueError('Output names must be safe filenames')
    if contract['dtype'] != 'float32' or any(d <= 0 for d in contract['shape']):
      raise ValueError(
          'This evaluator currently requires static FLOAT32 outputs')
  return data


def read_outputs(directory, contracts):
  values = {}
  for name, contract in contracts.items():
    path = Path(directory) / (name + '.raw')
    count = int(np.prod(contract['shape']))
    if path.stat().st_size != count * 4:
      raise ValueError(f'Output size mismatch: {path}')
    value = np.fromfile(path, dtype='<f4').reshape(contract['shape'])
    metrics.require_finite(value)
    values[name] = value
  return values


def score(manifest_path, results, *, sample_ids=None):
  manifest_path = Path(manifest_path)
  manifest = load_manifest(manifest_path)
  selected = set(sample_ids) if sample_ids is not None else {
      s['id'] for s in manifest['samples']
  }
  known = {s['id'] for s in manifest['samples']}
  if not selected or not selected <= known:
    raise ValueError('Empty or unknown sample selection')
  accumulators = {}
  evaluated = []
  for sample in manifest['samples']:
    if sample['id'] not in selected:
      continue
    contracts = {name: manifest['outputs'][name] for name in sample['targets']}
    outputs = read_outputs(Path(results) / sample['id'], contracts)
    for name, target_path in sample['targets'].items():
      contract = contracts[name]
      task = contract['task']
      if name not in accumulators:
        if task == 'segmentation':
          accumulators[name] = metrics.SegmentationMetrics(
              contract['classes'], contract.get('ignore_label', 255))
        elif task == 'depth':
          accumulators[name] = metrics.DepthMetrics(
              contract.get('min_depth', .001), contract.get('max_depth', 10.))
        elif task == 'normals':
          accumulators[name] = metrics.NormalMetrics()
        else:
          raise ValueError(f'Unknown task: {task}')
      target = np.load(manifest_path.parent / target_path, allow_pickle=False)
      prediction = outputs[name]
      if prediction.shape[0] != 1:
        raise ValueError('Scoring currently supports batch one')
      prediction = prediction[0]
      if task == 'depth':
        prediction = prediction[0]
      accumulators[name].update(prediction, target)
    evaluated.append(sample['id'])
  return dict(manifest_sha256=digest(manifest_path),
              sample_ids=evaluated,
              metrics={
                  name: metric.result() for name, metric in accumulators.items()
              })


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest', type=Path, required=True)
  parser.add_argument('--results', type=Path, required=True)
  parser.add_argument('--reference', type=Path)
  parser.add_argument('--sample-ids',
                      type=Path,
                      help='JSON list; identical subset used for both runs')
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  ids = json.loads(args.sample_ids.read_text()) if args.sample_ids else None
  result = score(args.manifest, args.results, sample_ids=ids)
  if args.reference:
    result['reference'] = score(args.manifest,
                                args.reference,
                                sample_ids=result['sample_ids'])
  write_json(args.output, result)


if __name__ == '__main__':
  main()
