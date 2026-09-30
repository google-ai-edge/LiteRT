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

"""Utilities for dynamic histogram collection."""

from typing import Any, Mapping, Sequence

from absl import logging
import numpy as np


class _DynamicHistogram1D:
  """A 1D histogram that dynamically grows and adapts to stay bounded.

  This class implements a dynamic histogram that automatically expands its range
  to accommodate new data points. To prevent the number of bins from growing
  infinitely and exceeding a memory budget (max_bins), the histogram will
  dynamically double its bin width and compact existing bins (merging adjacent
  bins) when it needs to expand beyond its budget.

  This ensures that the memory footprint remains constant and bounded, while
  still capturing the full distribution of the data over time, albeit at a
  coarser resolution if the range is very wide.

  Attributes:
    max_bins: The maximum number of bins allowed.
    bin_width: The width of each bin.
    counts: A numpy array storing the bin counts.
    lower_bound: The lower bound of the histogram range.
    initialized: Whether the histogram has been initialized with data.
    global_min: The minimum value seen so far.
    global_max: The maximum value seen so far.
  """

  max_bins: int
  bin_width: float | None = None
  counts: np.ndarray
  lower_bound: float = 0.0
  initialized: bool = False
  global_min: float = float('inf')
  global_max: float = float('-inf')

  def __init__(
      self,
      max_bins: int = 2048,
      initial_bin_width: float | None = None,
  ):
    """Initializes the 1D dynamic histogram.

    Args:
      max_bins: The maximum number of bins allowed.
      initial_bin_width: Optional initial bin width. If None, it will be
        inferred from the first added data.
    """
    self.bin_width = initial_bin_width
    self.max_bins = max_bins
    self.counts = np.zeros(1, dtype=np.int64)

  @classmethod
  def from_dict(
      cls, d: Mapping[str, Any], max_bins: int = 2048
  ) -> '_DynamicHistogram1D':
    """Reconstructs DynamicHistogram from a dictionary."""
    obj = cls(max_bins=max_bins)
    if 'hist_counts' in d:
      obj.counts = np.array(d['hist_counts'], dtype=np.int64)
      obj.lower_bound = d['lower_bound']
      obj.bin_width = d['bin_width']
      obj.initialized = True
      # Handle both scalar and array min/max.
      g_min = d['min']
      g_max = d['max']
      obj.global_min = (
          g_min[0] if isinstance(g_min, (np.ndarray, list, tuple)) else g_min
      )
      obj.global_max = (
          g_max[0] if isinstance(g_max, (np.ndarray, list, tuple)) else g_max
      )
    return obj

  def to_dict(self) -> dict[str, Any]:
    """Exports DynamicHistogram to a dictionary."""
    if not self.initialized:
      return {}
    return {
        'hist_counts': self.counts,
        'bin_edges': self.bin_edges,
        'bin_width': self.bin_width,
        'lower_bound': self.lower_bound,
        'min': np.array([self.global_min]),
        'max': np.array([self.global_max]),
    }

  @property
  def bin_edges(self) -> np.ndarray:
    """Returns the bin edges of the histogram."""
    if not self.initialized:
      return np.array([self.lower_bound])
    return self.lower_bound + np.arange(len(self.counts) + 1) * self.bin_width  # pyrefly: ignore[unsupported-operation]

  def _initialize(self, d_min: float, d_max: float) -> None:
    """Initializes the bin width and lower bound based on the first data."""
    if self.bin_width is None:
      # Infer bin width from the first data range with a 10% padding buffer
      range_val = d_max - d_min
      if range_val > 0:
        pad = range_val * 0.1
        d_min_padded = d_min - pad
        d_max_padded = d_max + pad
      else:
        # Fallback for zero range (all values identical)
        d_min_padded = d_min - 1e-4
        d_max_padded = d_max + 1e-4
      self.lower_bound = d_min_padded
      self.bin_width = (d_max_padded - d_min_padded) / self.max_bins
      self.bin_width = max(self.bin_width, 1e-5)  # Avoid zero width
      num_bins = self.max_bins
    else:
      self.lower_bound = d_min
      # Ensure we have at least 1 bin when initial_bin_width is provided
      num_bins = int(np.ceil((d_max - d_min) / self.bin_width))
      num_bins = max(num_bins, 1)

    self.counts = np.zeros(num_bins, dtype=np.int64)
    self.initialized = True

  def add(self, data: np.ndarray) -> None:
    """Adds data to the histogram, expanding and compacting if needed."""
    if data.size == 0:
      return

    data = data.ravel()
    d_min = np.min(data)
    d_max = np.max(data)

    self.global_min = min(self.global_min, d_min)
    self.global_max = max(self.global_max, d_max)

    if not self.initialized:
      self._initialize(d_min, d_max)

    self._expand_to_fit(d_min, d_max)

    # Calculate bin indices
    indices = np.floor((data - self.lower_bound) / self.bin_width).astype(  # pyrefly: ignore[unsupported-operation]
        np.int32
    )
    indices = np.clip(indices, 0, len(self.counts) - 1)

    # Accumulate
    bin_counts = np.bincount(indices, minlength=len(self.counts))
    self.counts += bin_counts

  def _accumulate_resampled(self, other: '_DynamicHistogram1D') -> None:
    """Accumulates counts from another histogram using resampling."""
    accum_counts = self.counts.astype(np.float64)

    for i, c in enumerate(other.counts):
      if c == 0:
        continue
      l = other.lower_bound + i * other.bin_width  # pyrefly: ignore[unsupported-operation]
      r = l + other.bin_width  # pyrefly: ignore[unsupported-operation]

      start_idx = int(np.floor((l - self.lower_bound) / self.bin_width))  # pyrefly: ignore[unsupported-operation]
      end_idx = int(np.ceil((r - self.lower_bound) / self.bin_width))  # pyrefly: ignore[unsupported-operation]

      start_idx = max(0, start_idx)
      end_idx = min(len(self.counts), end_idx)

      for j in range(start_idx, end_idx):
        sl = self.lower_bound + j * self.bin_width  # pyrefly: ignore[unsupported-operation]
        sr = sl + self.bin_width  # pyrefly: ignore[unsupported-operation]

        overlap_l = max(l, sl)
        overlap_r = min(r, sr)

        if overlap_l < overlap_r:
          overlap_w = overlap_r - overlap_l
          frac = overlap_w / other.bin_width  # pyrefly: ignore[unsupported-operation]
          accum_counts[j] += c * frac

    self.counts = np.round(accum_counts).astype(np.int64)

  def merge(self, other: '_DynamicHistogram1D') -> None:
    """Merges another histogram into this one, resampling if needed."""
    self.global_min = min(self.global_min, other.global_min)
    self.global_max = max(self.global_max, other.global_max)

    if not other.initialized:
      return
    if not self.initialized:
      # Copy other to self
      self.bin_width = other.bin_width
      self.counts = np.copy(other.counts)
      self.lower_bound = other.lower_bound
      self.initialized = True
      return

    assert self.bin_width is not None
    assert other.bin_width is not None
    # Ensure self.bin_width is at least other.bin_width
    while self.bin_width < other.bin_width:
      self._double_bin_width_and_compact()

    # Expand self to fit other's range
    other_upper_bound = other.lower_bound + len(other.counts) * other.bin_width
    self._expand_to_fit(other.lower_bound, other_upper_bound)

    # Accumulate other's counts into self using resampling
    self._accumulate_resampled(other)

  def _expand_to_fit(self, d_min: float, d_max: float) -> None:
    """Expands the histogram range to fit new data, doubling width if needed."""
    assert self.bin_width is not None

    # 1. Expand Left
    if d_min < self.lower_bound:
      num_new_left = int(np.ceil((self.lower_bound - d_min) / self.bin_width))
      if len(self.counts) + num_new_left > self.max_bins:
        # Doubling needed
        while len(self.counts) + num_new_left > self.max_bins:
          self._double_bin_width_and_compact()
          # Recalculate based on new bin_width
          num_new_left = int(
              np.ceil((self.lower_bound - d_min) / self.bin_width)
          )
      # Pad counts on left
      self.counts = np.pad(
          self.counts, (num_new_left, 0), 'constant', constant_values=0
      )
      self.lower_bound -= num_new_left * self.bin_width

    # 2. Expand Right
    upper_bound = self.lower_bound + len(self.counts) * self.bin_width
    if d_max > upper_bound:
      num_new_right = int(np.ceil((d_max - upper_bound) / self.bin_width))
      if len(self.counts) + num_new_right > self.max_bins:
        # Doubling needed
        while len(self.counts) + num_new_right > self.max_bins:
          self._double_bin_width_and_compact()
          # Recalculate based on new bin_width and lower_bound
          upper_bound = self.lower_bound + len(self.counts) * self.bin_width
          num_new_right = int(np.ceil((d_max - upper_bound) / self.bin_width))
      # Pad counts on right
      self.counts = np.pad(
          self.counts, (0, num_new_right), 'constant', constant_values=0
      )

  def _double_bin_width_and_compact(self) -> None:
    """Doubles the bin width and merges adjacent bins to halve the bin count."""
    assert self.bin_width is not None
    # Merge adjacent bins: counts[0] = counts[0] + counts[1], etc.
    # If len(counts) is odd, we pad with a zero at the end before merging
    if len(self.counts) % 2 != 0:
      self.counts = np.pad(self.counts, (0, 1), 'constant', constant_values=0)

    # Reshape and sum adjacent pairs
    self.counts = self.counts.reshape(-1, 2).sum(axis=1)
    self.bin_width *= 2.0


class DynamicHistogram:
  """A wrapper around _DynamicHistogram1D that supports both per-tensor and per-channel tracking.

  It manages the total bin budget by allocating bins to each channel
  individually when running in per-channel mode.

  Attributes:
    initial_bin_width: Optional initial bin width.
    max_tensor_bins: The maximum number of bins allocated for the entire tensor.
    axis: The axis along which to track independent histograms.
  """

  initial_bin_width: float | None
  max_tensor_bins: int
  axis: int | None
  _impls: Sequence[_DynamicHistogram1D] | None = None

  def __init__(
      self,
      max_tensor_bins: int = 2048,
      initial_bin_width: float | None = None,
      axis: int | None = None,
  ):
    """Initializes the DynamicHistogram.

    Args:
      max_tensor_bins: The maximum number of bins allocated for the entire
        tensor. If per-channel tracking is enabled, this budget is divided
        equally among all channels.
      initial_bin_width: Optional initial bin width. If None, it will be
        inferred from the first added data.
      axis: The axis along which to track independent histograms (per-channel).
        If None, tracks a single histogram for the entire tensor (per-tensor).
    """
    self.initial_bin_width = initial_bin_width
    self.max_tensor_bins = max_tensor_bins
    self.axis = axis
    self._impls = None

  @property
  def initialized(self) -> bool:
    """Returns whether the histogram has been initialized."""
    if self._impls is None:
      return False
    if self.axis is None:
      return self._impls[0].initialized
    return True

  @property
  def global_min(self) -> np.ndarray:
    """Returns the minimum value seen so far for each channel."""
    if self._impls is None:
      return np.array([float('inf')]) if self.axis is None else np.array([])
    return np.array([h.global_min for h in self._impls])

  @property
  def global_max(self) -> np.ndarray:
    """Returns the maximum value seen so far for each channel."""
    if self._impls is None:
      return np.array([float('-inf')]) if self.axis is None else np.array([])
    return np.array([h.global_max for h in self._impls])

  @property
  def counts(self) -> np.ndarray:
    """Returns the bin counts (only supported for per-tensor)."""
    if self.axis is None:
      if self._impls is None:
        return np.zeros(1, dtype=np.int64)
      return self._impls[0].counts
    raise AttributeError(
        'counts is not supported for per-channel histogram, use'
        ' _impls[i].counts'
    )

  @property
  def bin_width(self) -> float | None:
    """Returns the bin width (only supported for per-tensor)."""
    if self.axis is None:
      if self._impls is None:
        return None
      return self._impls[0].bin_width
    raise AttributeError(
        'bin_width is not supported for per-channel histogram, use'
        ' _impls[i].bin_width'
    )

  @property
  def lower_bound(self) -> float:
    """Returns the lower bound (only supported for per-tensor)."""
    if self.axis is None:
      if self._impls is None:
        return 0.0
      return self._impls[0].lower_bound
    raise AttributeError(
        'lower_bound is not supported for per-channel histogram, use'
        ' _impls[i].lower_bound'
    )

  def _initialize_impls(self, data: np.ndarray) -> None:
    """Initializes the underlying 1D histograms on first add."""
    if self.axis is None:
      num_channels = 1
      max_bins_per_channel = self.max_tensor_bins
    else:
      num_channels = data.shape[self.axis]
      max_bins_per_channel = max(self.max_tensor_bins // num_channels, 1)
      logging.info(
          'Initializing per-channel histograms with %d bins per channel'
          ' (total budget: %d, channels: %d)',
          max_bins_per_channel,
          self.max_tensor_bins,
          num_channels,
      )

    self._impls = [
        _DynamicHistogram1D(
            max_bins=max_bins_per_channel,
            initial_bin_width=self.initial_bin_width,
        )
        for _ in range(num_channels)
    ]

  def add(self, data: np.ndarray) -> None:
    """Adds data to the histogram."""
    if data.size == 0:
      return

    if self._impls is None:
      self._initialize_impls(data)

    if self.axis is None:
      data_flat = data.ravel()
      data_finite = data_flat[np.isfinite(data_flat)]
      if data_finite.size > 0:
        self._impls[0].add(data_finite)  # pyrefly: ignore[unsupported-operation]
    else:
      num_channels = len(self._impls)  # pyrefly: ignore[bad-argument-type]
      data_swapped = np.moveaxis(data, self.axis, 0)
      for i in range(num_channels):
        channel_data = data_swapped[i].ravel()
        channel_data_finite = channel_data[np.isfinite(channel_data)]
        if channel_data_finite.size > 0:
          self._impls[i].add(channel_data_finite)  # pyrefly: ignore[unsupported-operation]

  def merge(self, other: 'DynamicHistogram') -> None:
    """Merges another DynamicHistogram into this one."""
    # pylint: disable=protected-access
    if self.axis != other.axis:
      raise ValueError(
          f'Cannot merge histograms with different axis: {self.axis} vs'
          f' {other.axis}'
      )

    if self._impls is None and other._impls is not None:
      # Initialize self with same structure as other
      max_bins_per_channel = other._impls[0].max_bins
      self._impls = [
          _DynamicHistogram1D(
              max_bins=max_bins_per_channel,
              initial_bin_width=self.initial_bin_width,
          )
          for _ in range(len(other._impls))
      ]

    if self._impls is None or other._impls is None:
      return

    if len(self._impls) != len(other._impls):
      raise ValueError(
          'Cannot merge: different number of channels:'
          f' {len(self._impls)} vs {len(other._impls)}'
      )

    for self_impl, other_impl in zip(self._impls, other._impls):
      self_impl.merge(other_impl)

  def to_dict(self) -> dict[str, Any]:
    """Exports DynamicHistogram to a dictionary."""
    if not self.initialized:
      return {}
    return {
        'min': self.global_min,
        'max': self.global_max,
        'axis': self.axis,
        'channels': [h.to_dict() for h in self._impls],  # pyrefly: ignore[not-iterable]
    }

  @classmethod
  def from_dict(
      cls, d: Mapping[str, Any], max_tensor_bins: int = 2048
  ) -> 'DynamicHistogram':
    """Reconstructs DynamicHistogram from a dictionary."""
    if not d:
      return cls(max_tensor_bins=max_tensor_bins)

    if 'channels' not in d:
      raise ValueError(f'Invalid dictionary format for DynamicHistogram: {d}')

    axis = d['axis']
    obj = cls(max_tensor_bins=max_tensor_bins, axis=axis)
    num_channels = len(d['channels'])
    max_bins_per_channel = max(max_tensor_bins // num_channels, 1)
    obj._impls = [
        _DynamicHistogram1D.from_dict(h_dict, max_bins=max_bins_per_channel)
        for h_dict in d['channels']
    ]
    return obj
