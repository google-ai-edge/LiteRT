# Copyright 2024 Google LLC
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
# ==============================================================================

import io
import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import tqdm

import os
import io
from litert_quantizer.utils import progress_utils
from litert_quantizer.utils import test_utils


class ProgressBarTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_tqdm = self.enter_context(
        mock.patch.object(tqdm, 'tqdm', autospec=True, spec_set=True)
    )

  def test_progress_bar_update(self):
    mock_progress_bar_instance = self.mock_tqdm.return_value
    with progress_utils.ProgressBar(total_steps=10, enable=True) as pb:
      pb.update_single_step()
      pb.update_single_step()

    self.mock_tqdm.assert_called_once_with(
        total=10, desc='', leave=True, disable=False
    )
    self.assertEqual(mock_progress_bar_instance.update.call_count, 2)
    mock_progress_bar_instance.update.assert_called_with(1)
    mock_progress_bar_instance.close.assert_called_once()

  def test_progress_bar_disable(self):
    mock_progress_bar_instance = self.mock_tqdm.return_value
    with progress_utils.ProgressBar(total_steps=10, enable=False):
      pass
    self.mock_tqdm.assert_called_once_with(
        total=10, desc='', leave=True, disable=True
    )
    mock_progress_bar_instance.close.assert_called_once()

  def test_progress_bar_disappear_on_finish(self):
    mock_progress_bar_instance = self.mock_tqdm.return_value
    with progress_utils.ProgressBar(
        total_steps=10, disappear_on_finish=True, enable=True
    ):
      pass
    self.mock_tqdm.assert_called_once_with(
        total=10, desc='', leave=False, disable=False
    )
    mock_progress_bar_instance.close.assert_called_once()


class ProgressReportTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_time = self.enter_context(
        mock.patch.object(progress_utils, 'time', autospec=True, spec_set=True)
    )
    self.mock_tracemalloc = self.enter_context(
        mock.patch.object(
            progress_utils, 'tracemalloc', autospec=True, spec_set=True
        )
    )

  @parameterized.named_parameters(
      ('trace_memory_enabled', True), ('trace_memory_disabled', False)
  )
  def test_generate_progress_report(self, trace_memory: bool):
    self.mock_time.time.side_effect = [100.0, 105.5]  # Start time, end time.

    if trace_memory:
      self.mock_tracemalloc.is_tracing.return_value = False
      self.mock_tracemalloc.start.side_effect = None
      self.mock_tracemalloc.stop.return_value = None
      self.mock_tracemalloc.get_traced_memory.return_value = (
          1 * 1024 * 1024,
          2 * 1024 * 1024,
      )

    progress_report = progress_utils.ProgressReport(trace_memory=trace_memory)
    progress_report.capture_progess_start()

    original_model_path = test_utils.get_path_to_datafile(
        '../tests/models/conv_fc_mnist.tflite'
    )
    quantized_model_path = test_utils.get_path_to_datafile(
        '../tests/models/mnist_quantized.tflite'
    )
    with open(original_model_path, 'rb') as f:
      original_model_data = f.read()
    with open(quantized_model_path, 'rb') as f:
      quantized_model = f.read()

    mock_stdout = io.StringIO()
    with mock.patch.object(sys, 'stdout', mock_stdout):
      progress_report.generate_progress_report(
          len(original_model_data), len(quantized_model)
      )

    output = mock_stdout.getvalue()
    self.assertIn('Original model size: 200.23 KiB', output)
    self.assertIn('Quantized model size: 53.08 KiB', output)
    self.assertIn('Quantization Ratio: 0.27', output)
    self.assertIn('Total time: 5.50 s', output)

    if trace_memory:
      self.mock_tracemalloc.is_tracing.assert_called_once_with()
      self.mock_tracemalloc.start.assert_called_once_with()
      self.mock_tracemalloc.stop.assert_called_once_with()
      self.assertIn('Memory peak: 2.00 MiB', output)


if __name__ == '__main__':
  absltest.main()
