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

"""Utilities for QSV update."""

from typing import Any, Union

import numpy as np

from litert_quantizer import qtyping


def _update_moving_average(
    smoothing_factor: Union[np.ndarray, float],
    w: np.ndarray,
    update: np.ndarray,
) -> np.ndarray:
  """Updates weight w with moving average.

  Args:
    smoothing_factor: Smoothing factor used to update w.
    w: Weights to be updated.
    update: Value used for update.

  Returns:
    Weighted sum of w and update.
  """
  return smoothing_factor * w + (1.0 - smoothing_factor) * update


def moving_average_update(
    qsv: qtyping.QSV, new_qsv: qtyping.QSV, smoothing_factor: float = 0.95
) -> qtyping.QSV:
  """Update the QSV (i.e., min/max) using moving average.

  Args:
    qsv: The quantization statistical value of the tensor (min/max) that need to
      be updated.
    new_qsv: The new QSVs (e.g., from new round of calibration).
    smoothing_factor: The weight of moving average.

  Returns:
    The updated QSV for the tensor.
  """
  if not qsv:
    return new_qsv

  updated_qsv = {}
  updated_qsv["min"] = _update_moving_average(
      smoothing_factor, qsv["min"], new_qsv["min"]
  )

  updated_qsv["max"] = _update_moving_average(
      smoothing_factor, qsv["max"], new_qsv["max"]
  )
  return updated_qsv


def _gptq_merge_hessian(
    qsv: qtyping.QSV, new_qsv: qtyping.QSV
) -> tuple[Any, int]:
  """Merges hessian and num_samples from qsv and new_qsv for GPTQ."""
  curr_avg_hessian = qsv["hessian"]
  new_avg_hessian = new_qsv["hessian"]
  curr_samples = qsv["num_samples"]
  new_samples = new_qsv["num_samples"]
  total_samples = curr_samples + new_samples

  if total_samples == 0:
    return new_avg_hessian, 0

  merged_hessian = (
      curr_avg_hessian * curr_samples + new_avg_hessian * new_samples
  ) / total_samples

  return merged_hessian, total_samples


def gptq_and_moving_average_update(
    qsv: qtyping.QSV, new_qsv: qtyping.QSV
) -> qtyping.QSV:
  """Update the QSV with GPTQ logic and moving average logic."""
  if not qsv:
    return new_qsv
  updated_qsv = moving_average_update(qsv, new_qsv)

  merged_hessian, total_samples = _gptq_merge_hessian(qsv, new_qsv)
  updated_qsv["hessian"] = merged_hessian
  updated_qsv["num_samples"] = total_samples
  return updated_qsv


def min_max_update(qsv: qtyping.QSV, new_qsv: qtyping.QSV) -> qtyping.QSV:
  """Update the QSV with minimum min values and maximum max values.

  Args:
    qsv: The quantization statistical value of the tensor (min/max) that need to
      be updated.
    new_qsv: The new QSVs (e.g., from new round of calibration).

  Returns:
    The updated QSV for the tensor.
  """
  if not qsv:
    return new_qsv

  updated_qsv = {}
  updated_qsv["min"] = np.minimum(qsv["min"], new_qsv["min"])
  updated_qsv["max"] = np.maximum(qsv["max"], new_qsv["max"])
  return updated_qsv


def _oscar_merge_mu2(qsv1: qtyping.QSV, qsv2: qtyping.QSV) -> tuple[Any, int]:
  """Merges mu2 and num_samples from qsv1 and qsv2 for OSCAR.

  Args:
    qsv1: The first quantization state variable dict.
    qsv2: The second quantization state variable dict.

  Returns:
    A tuple of (merged_mu2, total_samples), where merged_mu2 is the weighted
    average of the second moment vectors and total_samples is the sum of sample
    counts from both QSV dicts.
  """
  if "mu2" not in qsv1 and "mu2" not in qsv2:
    return None, 0
  if "mu2" not in qsv1:
    return qsv2.get("mu2"), qsv2.get("num_samples", 0)
  if "mu2" not in qsv2:
    return qsv1.get("mu2"), qsv1.get("num_samples", 0)

  curr_avg_mu2 = qsv1["mu2"]
  new_avg_mu2 = qsv2["mu2"]
  curr_samples = qsv1.get("num_samples", 0)
  new_samples = qsv2.get("num_samples", 0)
  total_samples = curr_samples + new_samples

  if total_samples == 0:
    return new_avg_mu2, 0

  merged_mu2 = (
      curr_avg_mu2 * curr_samples + new_avg_mu2 * new_samples
  ) / total_samples

  return merged_mu2, total_samples


def oscar_and_moving_average_update(
    qsv: qtyping.QSV, new_qsv: qtyping.QSV
) -> qtyping.QSV:
  """Update the QSV with OSCAR logic and moving average logic."""
  if not qsv:
    return new_qsv
  updated_qsv = moving_average_update(qsv, new_qsv)

  merged_mu2, total_samples = _oscar_merge_mu2(qsv, new_qsv)
  updated_qsv["mu2"] = merged_mu2
  updated_qsv["num_samples"] = total_samples
  return updated_qsv
