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
"""Compare C++ scoring against the NumPy protocols from PR #11544.

The Python implementation is a test oracle, never an evaluation dependency.
"""

import os
import pathlib
import subprocess

from google.protobuf import text_format
import numpy as np

from absl.testing import absltest as googletest
from litert.tools.vision_eval import evaluation_pb2 as pb
from litert.tools.vision_eval import metrics_reference as reference


class MetricParityTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.root = pathlib.Path(self.create_tempdir().full_path)
    self.binary = (
        pathlib.Path(os.environ['TEST_SRCDIR'])
        / os.environ['TEST_WORKSPACE']
        / os.environ['LITERT_VISION_EVAL_BIN']
    )
    self.manifest = pb.DatasetManifest(version=1)
    self.manifest.inputs.add(name='image', shape=[1])
    (self.root / 'input.raw').write_bytes(np.zeros(1, dtype='<f4').tobytes())
    self.oracles = {
        'seg': reference.SegmentationMetrics(classes=3),
        'depth': reference.DepthMetrics(),
        'normals': reference.NormalMetrics(),
    }
    for name, channels, task in [
        ('seg', 3, pb.SEGMENTATION),
        ('depth', 1, pb.DEPTH),
        ('normals', 3, pb.NORMALS),
    ]:
      output = self.manifest.outputs.add()
      output.tensor.CopyFrom(
          pb.TensorSpec(name=name, shape=[1, channels, 2, 2])
      )
      output.scoring.task = task
      output.scoring.classes = 3
    rng = np.random.default_rng(31)
    for i in range(3):
      sample = self.manifest.samples.add(id=f'image_{i}')
      source = sample.inputs.add(name='image')
      source.file.path = 'input.raw'
      directory = self.root / 'predictions' / sample.id
      directory.mkdir(parents=True)
      for name in self.oracles:
        channels = 1 if name == 'depth' else 3
        prediction = rng.normal(size=(channels, 2, 2)).astype('<f4')
        if name == 'depth':
          prediction = np.abs(prediction) * 3
          target = rng.uniform(0, 10, size=(3, 4)).astype('<f4')
          target[0, :3] = [0, np.nan, 0.001]
        elif name == 'seg':
          target = rng.integers(0, 3, size=(3, 4), dtype=np.uint8)
          target[0, 0] = 255
        else:
          target = rng.normal(size=(3, 3, 4)).astype('<f4')
          target[:, 0, 0] = 0
          target[0, 0, 1] = np.nan
          prediction[:, 0, 0] = 0
        prediction.tofile(directory / f'{name}.raw')
        target.tofile(self.root / f'{sample.id}_{name}.raw')
        label = sample.targets.add(name=name)
        label.file.path = f'{sample.id}_{name}.raw'
        label.tensor.CopyFrom(
            pb.TensorSpec(
                shape=target.shape,
                type=pb.UINT8 if name == 'seg' else pb.FLOAT32,
            )
        )
        oracle_pred = prediction[0] if name == 'depth' else prediction
        self.oracles[name].update(oracle_pred, target)
    self.manifest_path = self.root / 'manifest.txtpb'
    self.manifest_path.write_text(text_format.MessageToString(self.manifest))

  def command(self, output: str = 'results', binary: bool = False) -> list[str]:
    """Builds the score-only litert_vision_eval CLI invocation."""
    summary_format = 'binaryproto' if binary else 'textproto'
    return [
        str(self.binary),
        '--score_only=true',
        f'--manifest={self.manifest_path}',
        f'--predictions_dir={self.root / "predictions"}',
        f'--output_dir={self.root / output}',
        f'--summary_format={summary_format}',
    ]

  def summary(
      self, directory: str = 'results', binary: bool = False
  ) -> pb.RunSummary:
    """Reads and parses the RunSummary protobuf from `directory`."""
    summary = pb.RunSummary()
    if binary:
      summary.ParseFromString(
          (self.root / directory / 'summary.binpb').read_bytes()
      )
    else:
      text_format.Parse(
          (self.root / directory / 'summary.txtpb').read_text(encoding='utf-8'),
          summary,
      )
    return summary

  def _assert_metrics_match(self, row, expected):
    if row.HasField('segmentation'):
      actual = row.segmentation
      scalars = {
          'images': actual.images,
          'valid_pixels': actual.valid_pixels,
          'classes': actual.classes,
          'evaluated_classes': actual.evaluated_classes,
          'miou': actual.miou,
      }
    elif row.HasField('depth'):
      actual = row.depth
      scalars = {
          'images': actual.images,
          'empty_images': actual.empty_images,
          'valid_pixels': actual.valid_pixels,
          'abs_rel': actual.abs_rel,
          'rmse': actual.rmse,
          'delta1': actual.delta1,
      }
    elif row.HasField('normals'):
      actual = row.normals
      scalars = {
          'images': actual.images,
          'empty_images': actual.empty_images,
          'valid_pixels': actual.valid_pixels,
          'mean_degrees': actual.mean_degrees,
          'median_degrees': actual.median_degrees,
          'within_11.25_degrees': actual.within_11_25_degrees,
          'within_22.5_degrees': actual.within_22_5_degrees,
          'within_30_degrees': actual.within_30_degrees,
      }
    else:
      self.fail(f'Unexpected metrics oneof in {row}')
    for name, value in expected.items():
      if name == 'confusion':
        self.assertEqual(
            list(actual.confusion), np.asarray(value).ravel().tolist()
        )
      elif value is None:
        self.assertFalse(actual.HasField(name.replace('.', '_')))
      else:
        np.testing.assert_allclose(scalars[name], value, atol=2e-5, rtol=2e-6)

  def test_cpp_metrics_match_numpy_and_resume(self):
    subprocess.run(self.command(), check=True, capture_output=True)
    summary = self.summary()
    self.assertEqual(summary.status, pb.PASSED)
    self.assertEqual(summary.completed, 3)
    for row in summary.metrics:
      self._assert_metrics_match(row, self.oracles[row.name].result())
    subprocess.run(self.command(), check=True, capture_output=True)
    resumed = self.summary()
    self.assertEqual(resumed.resumed, 3)
    self.assertEqual(resumed.metrics, summary.metrics)
    subprocess.run(
        self.command('binary', binary=True), check=True, capture_output=True
    )
    self.assertEqual(
        self.summary('binary', binary=True).metrics, summary.metrics
    )

  def test_truncated_prediction_and_corrupt_checkpoint_are_rejected(self):
    subprocess.run(self.command(), check=True, capture_output=True)
    path = self.root / 'predictions/image_0/depth.raw'
    original = path.read_bytes()
    np.zeros((1,), dtype='<f4').tofile(path)
    status = subprocess.run(self.command(), check=False, capture_output=True)
    self.assertNotEqual(status.returncode, 0)
    self.assertEqual(self.summary().status, pb.FAILED)
    path.write_bytes(original)
    subprocess.run(self.command('fresh'), check=True, capture_output=True)
    checkpoint_path = self.root / 'fresh/samples/image_0/result.binpb'
    checkpoint_path.write_bytes(b'broken checkpoint')
    status = subprocess.run(
        self.command('fresh'), check=False, capture_output=True
    )
    self.assertNotEqual(status.returncode, 0)


if __name__ == '__main__':
  googletest.main()
