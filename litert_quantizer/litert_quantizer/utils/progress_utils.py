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

"""Utility functions to display a progress bar and progress report."""

from collections.abc import Sequence
import logging
import time
import tracemalloc

import tqdm


def _format_base(
    value: int | float, multiple: int | float, units: Sequence[str]
) -> str:
  """Converts `value` to a string using the given `units` per `multiple`.

  Creates a `str` representation of `value / (multiple ** k)` with `units[k]`
  appended, where `k` is the largest integer in the range `[0, len(units))`
  such that `multiple ** k` is less than `value`, rounded to two decimal digits.

  For all `value` in the range `[1, multiple**len(units))`, the represented
  value will be in the range `[1, multiple)`.

  E.g. if `multiple=10` and `units=['mm', 'cm', 'dm', 'm']`, the following
  `value`s will generate the following outputs:
  * `1`: `'1.00 mm'`,
  * `9`: `'9.00 mm'`,
  * `10`: `'1.00 cm'`,
  * `110`: `'1.10 dm'`,
  * `1002`: `'1.00 m'`.

  Args:
    value: The numerical value to convert to a `str`.
    multiple: The multiple used for the `units`.
    units: A sequence of unit names.

  Returns:
    A `str` representation of an appropriately scaled `value` with the given
    `units` appended.
  """
  for unit in units[:-1]:
    if abs(value) < multiple:
      return f'{value:.2f} {unit}'
    value /= multiple
  return f'{value:.2f} {units[-1]}'


def _format_bytes(value: int | float) -> str:
  return _format_base(value, 1024, ['B', 'KiB', 'MiB', 'GiB'])


def _format_ns(value: int | float) -> str:
  return _format_base(value, 1000, ['ns', 'us', 'ms', 's'])


class ProgressBar:
  """A Progress Bar that can be used to track the progress of a process."""

  def __init__(
      self,
      total_steps: int,
      description: str = '',
      disappear_on_finish: bool = False,
      enable: bool | None = None,
  ):

    if enable is None:
      # Progress bar will be skipped for smaller models.
      disable = total_steps < 100
    else:
      disable = not enable
    self._progress_bar = tqdm.tqdm(
        total=total_steps,
        desc=description,
        leave=not disappear_on_finish,
        disable=disable,
    )

  def __enter__(self):
    """Enters the context manager."""
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Exits the context manager and closes the progress bar."""
    self.close()

  def update_single_step(self):
    """Updates the progress bar by a single step."""
    self._progress_bar.update(1)

  def close(self):
    """Closes the progress bar."""
    self._progress_bar.close()


class ProgressReport:
  """A class to generate a progress report for the quantization process.

  If initialized with `trace_memory=True`, it will also track the peak memory
  use using `tracemalloc`, which may hurt performance and interfere with other
  code using `tracemalloc` concurrently.
  """

  _model_name: str | None
  _trace_memory: bool
  _start_time: float | None = None
  _tracemalloc_started_by_me: bool = False

  def __init__(self, model_name: str | None = None, trace_memory: bool = False):
    self._model_name = model_name
    self._trace_memory = trace_memory

  def capture_progess_start(self):
    self._start_time = time.time()
    if self._trace_memory:
      logging.warning(
          'Progress bar reporting with `trace_memory=True` is enabled which may'
          ' significantly slow down your computations!'
      )
      self._tracemalloc_started_by_me = not tracemalloc.is_tracing()
      if self._tracemalloc_started_by_me:
        tracemalloc.start()
      else:
        tracemalloc.reset_peak()

  def _capture_progress_end(self) -> int | None:
    if self._trace_memory:
      _, mem_peak_bytes = tracemalloc.get_traced_memory()
      if self._tracemalloc_started_by_me:
        tracemalloc.stop()
        self._tracemalloc_started_by_me = False
      return mem_peak_bytes

  def generate_progress_report(
      self, original_model_size: int, quantized_model_size: int
  ):
    """Prints out the progress report."""
    # Collect the metrics.
    quantization_ratio = quantized_model_size / original_model_size
    total_time = time.time() - self._start_time  # pyrefly: ignore[unsupported-operation]
    mem_peak_bytes = self._capture_progress_end()

    # Print out the progress report.
    if self._model_name:
      print(f'Model name: {self._model_name}')
    print(f'Original model size: {_format_bytes(original_model_size)}')
    print(f'Quantized model size: {_format_bytes(quantized_model_size)}')
    print(
        f'Quantization Ratio: {quantization_ratio:.2f}'
        f' ({1/quantization_ratio:.1f}x smaller)'
    )
    print(f'Total time: {_format_ns(total_time * 1e9)}')
    if mem_peak_bytes is not None:
      print(f'Memory peak: {_format_bytes(mem_peak_bytes)}')
