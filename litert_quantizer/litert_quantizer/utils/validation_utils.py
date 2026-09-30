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

"""A library of comparator function to be used by validation function."""


import enum
from typing import Protocol, Tuple

import numpy as np


class ValidationFuncType(Protocol):
  """Type hint and documentation for validation functions."""

  def __call__(
      self,
      data1: np.typing.ArrayLike,
      data2: np.typing.ArrayLike,
  ) -> float:
    ...


class ValidationErrorMetric(enum.Enum):
  MSE = "mse"
  MEDIAN_DIFF_RATIO = "median_diff_ratio"
  COSINE_SIMILARITY = "cosine_similarity"
  KL_DIVERGENCE = "kl_divergence"
  SNR = "snr"


VALIDATION_FUNCS: dict[ValidationErrorMetric, ValidationFuncType] = {}


def _register_validation_func(metric_name: ValidationErrorMetric):
  def decorator(func: ValidationFuncType):
    VALIDATION_FUNCS[metric_name] = func
    return func

  return decorator


def get_validation_func(
    func_name: ValidationErrorMetric,
) -> ValidationFuncType:
  """Returns a validation function based on the metric type."""
  if func_name not in VALIDATION_FUNCS:
    raise ValueError(f"Validation function {func_name} not supported.")
  return VALIDATION_FUNCS[func_name]


@_register_validation_func(ValidationErrorMetric.MSE)
def mean_squared_difference(
    data1: np.typing.ArrayLike,
    data2: np.typing.ArrayLike,
) -> float:
  """Calculates the mean squared difference between data1 & data2.

  ref: https://en.wikipedia.org/wiki/Mean_squared_error

  Args:
    data1: input data to be used for comparison
    data2: input data to be used for comparison, data1 & 2 must be of the same
      shape

  Returns:
    a float value representing the MSD between data1 & 2

  Raises:
    ValueError: if the two inputs don't have the same number of elements
  """
  data1, data2 = _preprocess_same_size_arrays(data1, data2)
  # special handling for tensor of size 0
  if data1.size == 0:
    return float(0)
  return float(np.square(np.subtract(data1, data2)).mean())


@_register_validation_func(ValidationErrorMetric.MEDIAN_DIFF_RATIO)
def median_diff_ratio(
    data1: np.typing.ArrayLike,
    data2: np.typing.ArrayLike,
    tolerance_threshold=1e-6,
) -> float:
  """Calculates the median absolute diff ratio between data1 & data2.

  mdr = median(abs(data1 - data2) / data2)

  Args:
    data1: input data to be used for comparison
    data2: input data to be used for comparison, data1 & 2 must be of the same
      shape
    tolerance_threshold: a float value to be used as a threshold to avoid
      division by zero

  Returns:
    a float value representing the median diff ratio between data1 & 2

  Raises:
    ValueError: if the two inputs don't have the same number of elements
  """
  data1, data2 = _preprocess_same_size_arrays(data1, data2)
  # special handling for tensor of size 0
  if data1.size == 0:
    return float(0)
  diff = abs(data1 - data2)
  demoninator = abs(data2) + tolerance_threshold
  median_ratio = np.median(diff / demoninator)
  return median_ratio


@_register_validation_func(ValidationErrorMetric.COSINE_SIMILARITY)
def cosine_similarity(
    data1: np.typing.ArrayLike,
    data2: np.typing.ArrayLike,
) -> float:
  """Calculates the cosine similarity between data1 & data2.

  ref: https://en.wikipedia.org/wiki/Cosine_similarity

  Args:
    data1: input data to be used for comparison
    data2: input data to be used for comparison, data1 & 2 must be of the same
      shape

  Returns:
    a float value representing the cosine similarity between data1 & 2

  Raises:
    ValueError: if the two inputs don't have the same number of elements
  """
  data1, data2 = _preprocess_same_size_arrays(data1, data2)
  # special handling for tensor of size 0
  if data1.size == 0:
    return float(0)
  norm_data1 = np.linalg.norm(data1)
  norm_data2 = np.linalg.norm(data2)
  # special handling for tensor of length 0
  if norm_data1 == 0 and norm_data2 == 0:
    return 1.0
  if norm_data1 == 0 or norm_data2 == 0:
    return 0.0
  return np.dot(data1, data2) / (norm_data1 * norm_data2)


@_register_validation_func(ValidationErrorMetric.KL_DIVERGENCE)
def kl_divergence(
    data1: np.typing.ArrayLike,
    data2: np.typing.ArrayLike,
    epsilon: float = 1e-9,
) -> float:
  """Calculates the KL divergence between data1 & data2.

  KL(data2 || data1) = sum(data2 * log(data2 / data1)).
  data2 is treated as the true distribution P, and data1 as the
  approximated distribution Q.
  Non-positive values in data1 and data2 are clipped to 0 before
  KL divergence calculation. Epsilon is added to avoid log(0) and
  division by zero.

  Args:
    data1: input data to be used for comparison (distribution Q)
    data2: input data to be used for comparison (distribution P), data1 & 2 must
      be of the same shape
    epsilon: small value to avoid log(0) and division by zero.

  Returns:
    A float value representing the KL divergence between data1 & 2.

  Raises:
    ValueError: if the two inputs don't have the same number of elements.
  """
  data1, data2 = _preprocess_same_size_arrays(data1, data2)
  # special handling for tensor of size 0
  if data1.size == 0:
    return float(0)

  p = np.maximum(0, data2)
  q = np.maximum(0, data1)

  return float(np.sum(p * np.log((p + epsilon) / (q + epsilon))))


@_register_validation_func(ValidationErrorMetric.SNR)
def signal_to_noise_ratio(
    noisy_signal: np.typing.ArrayLike,
    signal: np.typing.ArrayLike,
    epsilon: float = 1e-9,
) -> float:
  """Calculates the signal to noise ratio between noisy_signal & signal.

  SNR = P_signal / P_noise, where signal is treated as the clean signal and
  noisy_signal-signal is treated as the noise samples.
  P_signal = mean(signal^2)
  P_noise = mean((noisy_signal-signal)^2) = mse(noisy_signal, signal)

  Args:
    noisy_signal: Input data to be used for comparison (e.g. noisy signal).
    signal: Input data to be used for comparison (e.g. clean signal),
      noisy_signal & signal must be of the same shape.
    epsilon: Small value to avoid division by zero.

  Returns:
    A float value representing the SNR between noisy_signal & signal.

  Raises:
    ValueError: If the two inputs don't have the same number of elements.
  """
  noisy_signal, signal = _preprocess_same_size_arrays(noisy_signal, signal)
  if signal.size == 0:
    return float(0)

  mse = mean_squared_difference(noisy_signal, signal)
  signal_power = float(np.square(signal).mean())
  snr = signal_power / (mse + epsilon)
  return snr


def _preprocess_same_size_arrays(
    data1: np.typing.ArrayLike,
    data2: np.typing.ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
  """Flattens and removes the nan, inf, and -inf values from the input data.

  Args:
    data1: input data to be used for comparison
    data2: input data to be used for comparison, data1 & 2 must be of the same
      shape

  Returns:
    a tuple of the preprocessed data1 & 2

  Raises:
    ValueError: if the two inputs don't have the same number of elements
  """
  data1 = np.ravel(np.asarray(data1, dtype=np.float32))
  data2 = np.ravel(np.asarray(data2, dtype=np.float32))
  if np.shape(data1) != np.shape(data2):
    raise ValueError("data1 & data2 must be of the same size")
  data1 = np.nan_to_num(data1, nan=1e-9, neginf=-1e9, posinf=1e9, copy=False)
  data2 = np.nan_to_num(data2, nan=1e-9, neginf=-1e9, posinf=1e9, copy=False)

  return data1, data2
