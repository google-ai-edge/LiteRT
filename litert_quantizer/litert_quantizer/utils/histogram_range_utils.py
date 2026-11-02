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

"""Range selection utilities from dynamic histograms for activation calibration."""

import dataclasses
from typing import Literal, Sequence

from absl import logging
import numpy as np

from litert_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from litert_quantizer.utils import histogram_utils

MseSearchMethod = Literal['symmetric', 'max_frequency', 'bruteforce']

DEFAULT_MIN_PERCENTILE = 0.001
DEFAULT_MAX_PERCENTILE = 99.999
DEFAULT_MSE_SEARCH: MseSearchMethod = 'symmetric'
_BRUTEFORCE_BIN_WARNING_THRESHOLD = 512


def get_hist_mids(
    num_bins: int, lower_bound: float, bin_width: float
) -> np.ndarray:
  """Computes the center values of each histogram bin.

  The i-th bin covers [lower_bound + i * bin_width,
  lower_bound + (i + 1) * bin_width), so its midpoint is
  (lower_bound + bin_width / 2) + i * bin_width.

  Args:
    num_bins: Number of bins in the histogram.
    lower_bound: Lower bound of the first bin.
    bin_width: Width of each bin.

  Returns:
    1D numpy array of bin midpoints with shape (num_bins,).
  """
  if num_bins <= 0:
    return np.array([], dtype=np.float64)
  first_mid = lower_bound + bin_width / 2.0
  last_mid = first_mid + (num_bins - 1) * bin_width
  return np.linspace(first_mid, last_mid, num_bins, dtype=np.float64)


def clamp_and_snap_zero_bin(
    mids: np.ndarray,
    bin_width: float,
    data_min: float | None,
    data_max: float | None,
) -> np.ndarray:
  """Clamps bin midpoints to [data_min, data_max] and snaps the 0-straddling bin to 0.0."""
  if data_min is None or data_max is None or data_min > data_max:
    return mids
  clamped = np.clip(mids, float(data_min), float(data_max))
  if data_min <= 0.0 <= data_max and bin_width > 0.0:
    zero_bin_mask = np.abs(mids) <= 0.5 * bin_width + 1e-12
    clamped = np.where(zero_bin_mask, 0.0, clamped)
  return clamped


def percentile_range(
    counts: np.ndarray | Sequence[int],
    lower_bound: float,
    bin_width: float | None,
    *,
    min_percentile: float = DEFAULT_MIN_PERCENTILE,
    max_percentile: float = DEFAULT_MAX_PERCENTILE,
    fallback_min: float | None = None,
    fallback_max: float | None = None,
    data_min: float | None = None,
    data_max: float | None = None,
) -> tuple[float, float]:
  """Selects (min_value, max_value) from a 1D histogram using percentiles.

  Args:
    counts: 1D array of bin counts.
    lower_bound: Lower bound of the histogram.
    bin_width: Width of each bin.
    min_percentile: Lower percentile in [0, 100]. Defaults to 0.001.
    max_percentile: Upper percentile in [0, 100]. Defaults to 99.999.
    fallback_min: Optional fallback min value when the histogram is empty.
    fallback_max: Optional fallback max value when the histogram is empty.
    data_min: Optional exact minimum observed value used to clamp bin midpoints
      so boundary bins do not extend below the true data range.
    data_max: Optional exact maximum observed value used to clamp bin midpoints
      so boundary bins do not extend above the true data range.

  Returns:
    Tuple of (min_value, max_value) corresponding to the requested percentiles.

  Raises:
    ValueError: If min_percentile or max_percentile are outside [0, 100] or
      min_percentile > max_percentile.
  """
  if (
      min_percentile < 0.0
      or max_percentile > 100.0
      or min_percentile > max_percentile
  ):
    raise ValueError(
        'Percentiles must satisfy 0.0 <= min_percentile <= max_percentile <='
        f' 100.0, got min_percentile={min_percentile},'
        f' max_percentile={max_percentile}.'
    )

  hist_freq = np.asarray(counts, dtype=np.float64).ravel()
  num_bins = len(hist_freq)
  total_freq = float(np.sum(hist_freq))
  if (
      num_bins == 0
      or total_freq <= 0.0
      or bin_width is None
      or bin_width <= 0.0
  ):
    if fallback_min is not None and fallback_max is not None:
      return float(fallback_min), float(fallback_max)
    return float(lower_bound), float(lower_bound)

  hist_mids = clamp_and_snap_zero_bin(
      get_hist_mids(num_bins, lower_bound, bin_width),
      bin_width,
      data_min,
      data_max,
  )
  hist_freq_cumsum = np.cumsum(hist_freq) / total_freq

  min_quantile = min_percentile / 100.0
  max_quantile = max_percentile / 100.0

  min_quantile_idx = int(
      np.clip(
          np.searchsorted(hist_freq_cumsum, min_quantile, side='right'),
          0,
          num_bins - 1,
      )
  )
  max_quantile_idx = int(
      np.clip(
          np.searchsorted(hist_freq_cumsum, max_quantile, side='left'),
          0,
          num_bins - 1,
      )
  )

  return float(hist_mids[min_quantile_idx]), float(hist_mids[max_quantile_idx])


@dataclasses.dataclass(frozen=True)
class _HistogramMoments:
  """Precomputed prefix and suffix moments for O(log M) MSE evaluation."""

  mids: np.ndarray
  left_f0: np.ndarray
  left_f1: np.ndarray
  left_f2: np.ndarray
  right_f0: np.ndarray
  right_f1: np.ndarray
  right_f2: np.ndarray


def _precompute_histogram_moments(
    hist_mids: np.ndarray, hist_freq: np.ndarray
) -> _HistogramMoments:
  """Precomputes shifted prefix and suffix moments for sorted `hist_mids`."""
  mids = np.asarray(hist_mids, dtype=np.float64).ravel()
  freq = np.asarray(hist_freq, dtype=np.float64).ravel()
  m0 = float(mids[0])
  m_end = float(mids[-1])

  dl = mids - m0
  left_f0 = np.concatenate(([0.0], np.cumsum(freq)))
  left_f1 = np.concatenate(([0.0], np.cumsum(freq * dl)))
  left_f2 = np.concatenate(([0.0], np.cumsum(freq * np.square(dl))))

  dr = mids - m_end
  rev_f0 = np.cumsum(freq[::-1])[::-1]
  rev_f1 = np.cumsum((freq * dr)[::-1])[::-1]
  rev_f2 = np.cumsum((freq * np.square(dr))[::-1])[::-1]
  right_f0 = np.concatenate((rev_f0, [0.0]))
  right_f1 = np.concatenate((rev_f1, [0.0]))
  right_f2 = np.concatenate((rev_f2, [0.0]))

  return _HistogramMoments(
      mids=mids,
      left_f0=left_f0,
      left_f1=left_f1,
      left_f2=left_f2,
      right_f0=right_f0,
      right_f1=right_f1,
      right_f2=right_f2,
  )


def _compute_weighted_mse_from_moments(
    moments: _HistogramMoments,
    quant_mins: np.ndarray,
    quant_maxs: np.ndarray,
    num_bits: int,
    symmetric: bool,
    signed: bool,
) -> np.ndarray:
  """Computes weighted quantization MSE in O(N log M) using precomputed moments."""
  q_mins = np.asarray(quant_mins, dtype=np.float64).ravel()
  q_maxs = np.asarray(quant_maxs, dtype=np.float64).ravel()
  mids = moments.mids

  if symmetric:
    qtype = uniform_quantize_tensor.IntType(num_bits, signed=signed)
    minbound, maxbound = uniform_quantize_tensor.get_quantized_range(qtype)
    bound = np.maximum(np.abs(q_mins), np.abs(q_maxs))
    bound = np.maximum(bound, 1e-9)
    if not signed:
      half_q = (maxbound - 1) / 2.0
      scale = bound / half_q
      zero_point = np.full_like(scale, half_q + 1.0)
    else:
      scale = bound / maxbound
      zero_point = np.zeros_like(scale)
  else:
    maxbound = (1 << num_bits) - 1
    minbound = 0
    scale = (q_maxs - q_mins) / maxbound
    scale = np.where(np.abs(scale) < 1e-9, 1e-9, scale)
    zero_point = -q_mins / scale
    zero_point = np.where(np.abs(zero_point) > 9e9, 9e9, zero_point)
    zero_point = np.round(zero_point)

  dequant_min = scale * (minbound - zero_point)
  dequant_max = scale * (maxbound - zero_point)

  i_left = np.searchsorted(mids, dequant_min, side='left')
  i_right = np.searchsorted(mids, dequant_max, side='right')
  i_eff_right = np.maximum(i_left, i_right)

  d_left = dequant_min - float(mids[0])
  left_err = (
      moments.left_f2[i_left]
      - 2.0 * d_left * moments.left_f1[i_left]
      + np.square(d_left) * moments.left_f0[i_left]
  )

  d_right = dequant_max - float(mids[-1])
  right_err = (
      moments.right_f2[i_right]
      - 2.0 * d_right * moments.right_f1[i_right]
      + np.square(d_right) * moments.right_f0[i_right]
  )

  in_range_count = moments.left_f0[i_eff_right] - moments.left_f0[i_left]
  gran_err = (np.square(scale) / 12.0) * in_range_count
  return np.maximum(0.0, left_err) + np.maximum(0.0, right_err) + gran_err


def _generate_expanding_candidates(
    num_bins: int, start_idx: int
) -> tuple[np.ndarray, np.ndarray]:
  """Generates (left_idx, right_idx) pairs by expanding alternately from start_idx."""
  left, right = start_idx, start_idx
  move_left = True
  left_indices = []
  right_indices = []
  while not (left == 0 and right == num_bins - 1):
    if (move_left and left > 0) or (right == num_bins - 1):
      left = max(left - 1, 0)
    else:
      right = min(right + 1, num_bins - 1)
    move_left = not move_left
    left_indices.append(left)
    right_indices.append(right)
  return np.asarray(left_indices, dtype=np.int64), np.asarray(
      right_indices, dtype=np.int64
  )


def _select_best_candidate(
    moments: _HistogramMoments,
    quant_mins: np.ndarray,
    quant_maxs: np.ndarray,
    num_bits: int,
    symmetric: bool,
    signed: bool,
    current_best: tuple[float, float, float] = (
        float('inf'),
        float('inf'),
        float('inf'),
    ),
) -> tuple[float, float, float]:
  """Evaluates candidate (min, max) pairs and returns the best (mse, min, max)."""
  errors = _compute_weighted_mse_from_moments(
      moments,
      quant_mins,
      quant_maxs,
      num_bits=num_bits,
      symmetric=symmetric,
      signed=signed,
  )
  min_idx = int(np.argmin(errors))
  min_err = float(errors[min_idx])
  if min_err > current_best[0]:
    return current_best
  mse_min = current_best
  # Check ties within float64 precision.
  tie_indices = np.flatnonzero(errors <= min_err + 1e-12)
  for idx in tie_indices.tolist():
    candidate = (
        float(errors[idx]),
        float(quant_mins[idx]),
        float(quant_maxs[idx]),
    )
    if candidate < mse_min:
      mse_min = candidate
  return mse_min


def mse_range(
    counts: np.ndarray | Sequence[int],
    lower_bound: float,
    bin_width: float | None,
    *,
    num_bits: int = 8,
    symmetric: bool = False,
    signed: bool = True,
    search: MseSearchMethod | str = DEFAULT_MSE_SEARCH,
    fallback_min: float | None = None,
    fallback_max: float | None = None,
    data_min: float | None = None,
    data_max: float | None = None,
) -> tuple[float, float]:
  """Selects (min_value, max_value) from a 1D histogram by minimizing MSE.

  Args:
    counts: 1D array of bin counts.
    lower_bound: Lower bound of the histogram.
    bin_width: Width of each bin.
    num_bits: Number of quantization bits. Defaults to 8.
    symmetric: Whether the target quantization config is symmetric.
    signed: Whether the target quantization config is signed.
    search: Candidate search strategy: 'symmetric' (expand from center/median
      bin with coordinate refinement), 'max_frequency' (expand from peak bin
      with coordinate refinement), or 'bruteforce' (all bin pairs).
    fallback_min: Optional fallback min value when the histogram is empty or
      degenerate.
    fallback_max: Optional fallback max value when the histogram is empty or
      degenerate.
    data_min: Optional exact minimum observed value used to clamp bin midpoints
      so boundary bins do not extend below the true data range.
    data_max: Optional exact maximum observed value used to clamp bin midpoints
      so boundary bins do not extend above the true data range.

  Returns:
    Tuple of (min_value, max_value) that minimizes weighted quantization MSE.

  Raises:
    ValueError: If `search` is not a supported search strategy or `num_bits` is
      less than 1.
  """
  if num_bits < 1:
    raise ValueError(f'num_bits must be >= 1, got {num_bits}.')
  if search not in ('symmetric', 'max_frequency', 'bruteforce'):
    raise ValueError(
        f'Unsupported MSE search strategy: {search!r}. '
        "Expected 'symmetric', 'max_frequency', or 'bruteforce'."
    )

  hist_freq = np.asarray(counts, dtype=np.float64).ravel()
  num_bins = len(hist_freq)
  total_freq = float(np.sum(hist_freq))
  if (
      num_bins == 0
      or total_freq <= 0.0
      or bin_width is None
      or bin_width <= 0.0
  ):
    if fallback_min is not None and fallback_max is not None:
      return float(fallback_min), float(fallback_max)
    return float(lower_bound), float(lower_bound)

  nonzero_indices = np.flatnonzero(hist_freq > 0)
  first_nz = int(nonzero_indices[0])
  last_nz = int(nonzero_indices[-1])
  if first_nz == last_nz:
    if fallback_min is not None and fallback_max is not None:
      return float(fallback_min), float(fallback_max)
    mid = float(lower_bound + (first_nz + 0.5) * bin_width)
    return mid, mid

  all_mids = clamp_and_snap_zero_bin(
      get_hist_mids(num_bins, lower_bound, bin_width),
      bin_width,
      data_min,
      data_max,
  )
  active_mids = all_mids[first_nz : last_nz + 1]
  active_freq = hist_freq[first_nz : last_nz + 1]
  n_active = len(active_freq)

  nz_mask = active_freq > 0
  eval_mids = active_mids[nz_mask]
  eval_freq = active_freq[nz_mask]
  moments = _precompute_histogram_moments(eval_mids, eval_freq)

  if search == 'bruteforce':
    if n_active > _BRUTEFORCE_BIN_WARNING_THRESHOLD:
      logging.warning(
          'num_bins=%d is large. The bruteforce MSE search tests all histogram'
          ' mid value pairs and may be slow.',
          n_active,
      )
    left_indices, right_indices = np.triu_indices(n_active, k=1)
    mse_min = _select_best_candidate(
        moments,
        active_mids[left_indices],
        active_mids[right_indices],
        num_bits=num_bits,
        symmetric=symmetric,
        signed=signed,
    )
    best_min, best_max = mse_min[1], mse_min[2]
  else:
    mode_idx = int(np.argmax(active_freq))
    cumsum = np.cumsum(active_freq) / total_freq
    median_idx = int(
        np.clip(
            np.searchsorted(cumsum, 0.5, side='left'), 0, n_active - 1
        )
    )
    primary_idx = n_active // 2 if search == 'symmetric' else mode_idx

    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    # Expand from each distinct center, preserving priority order.
    for center_idx in dict.fromkeys((primary_idx, median_idx, mode_idx)):
      l_idx, r_idx = _generate_expanding_candidates(n_active, center_idx)
      left_parts.append(l_idx)
      right_parts.append(r_idx)

    # Include one-sided tail sweeps: (0, r) and (l, n_active - 1).
    left_parts.append(np.zeros(n_active - 1, dtype=np.int64))
    right_parts.append(np.arange(1, n_active, dtype=np.int64))
    left_parts.append(np.arange(0, n_active - 1, dtype=np.int64))
    right_parts.append(np.full(n_active - 1, n_active - 1, dtype=np.int64))

    left_indices = np.concatenate(left_parts)
    right_indices = np.concatenate(right_parts)

    mse_min = _select_best_candidate(
        moments,
        active_mids[left_indices],
        active_mids[right_indices],
        num_bits=num_bits,
        symmetric=symmetric,
        signed=signed,
    )

    # Alternating coordinate refinement around the best initial candidate to
    # handle skewed distributions where optimal left and right clipping depths
    # differ.
    best_l = int(np.searchsorted(active_mids, mse_min[1]))
    best_r = int(np.searchsorted(active_mids, mse_min[2]))
    best_l = max(0, min(best_l, n_active - 2))
    best_r = max(best_l + 1, min(best_r, n_active - 1))

    for pass_idx in range(2):
      prev_l, prev_r = best_l, best_r
      if pass_idx > 0 or best_l > 0:
        r_cands = np.arange(best_l + 1, n_active, dtype=np.int64)
        mse_min = _select_best_candidate(
            moments,
            active_mids[np.full_like(r_cands, best_l)],
            active_mids[r_cands],
            num_bits=num_bits,
            symmetric=symmetric,
            signed=signed,
            current_best=mse_min,
        )
        best_r = int(np.searchsorted(active_mids, mse_min[2]))
        best_r = max(best_l + 1, min(best_r, n_active - 1))

      if pass_idx > 0 or best_r < n_active - 1:
        l_cands = np.arange(0, best_r, dtype=np.int64)
        mse_min = _select_best_candidate(
            moments,
            active_mids[l_cands],
            active_mids[np.full_like(l_cands, best_r)],
            num_bits=num_bits,
            symmetric=symmetric,
            signed=signed,
            current_best=mse_min,
        )
        best_l = int(np.searchsorted(active_mids, mse_min[1]))
        best_l = max(0, min(best_l, best_r - 1))

      if best_l == prev_l and best_r == prev_r:
        break

    best_min, best_max = mse_min[1], mse_min[2]

  if symmetric:
    bound = max(abs(best_min), abs(best_max))
    return -float(bound), float(bound)
  return float(best_min), float(best_max)


def percentile_range_from_histogram(
    hist: histogram_utils.DynamicHistogram,
    *,
    min_percentile: float = DEFAULT_MIN_PERCENTILE,
    max_percentile: float = DEFAULT_MAX_PERCENTILE,
    fallback_min: float | None = None,
    fallback_max: float | None = None,
) -> tuple[float, float]:
  """Selects (min_value, max_value) from a per-tensor DynamicHistogram using percentiles."""
  if not hist.initialized:
    if fallback_min is not None and fallback_max is not None:
      return float(fallback_min), float(fallback_max)
    return 0.0, 0.0
  g_min = float(hist.global_min[0])
  g_max = float(hist.global_max[0])
  default_min = g_min if fallback_min is None else float(fallback_min)
  default_max = g_max if fallback_max is None else float(fallback_max)
  return percentile_range(
      hist.counts,
      hist.lower_bound,
      hist.bin_width,
      min_percentile=min_percentile,
      max_percentile=max_percentile,
      fallback_min=default_min,
      fallback_max=default_max,
      data_min=g_min,
      data_max=g_max,
  )


def mse_range_from_histogram(
    hist: histogram_utils.DynamicHistogram,
    *,
    num_bits: int = 8,
    symmetric: bool = False,
    signed: bool = True,
    search: MseSearchMethod = DEFAULT_MSE_SEARCH,
    fallback_min: float | None = None,
    fallback_max: float | None = None,
) -> tuple[float, float]:
  """Selects (min_value, max_value) from a per-tensor DynamicHistogram by minimizing MSE."""
  if not hist.initialized:
    if fallback_min is not None and fallback_max is not None:
      return float(fallback_min), float(fallback_max)
    return 0.0, 0.0
  g_min = float(hist.global_min[0])
  g_max = float(hist.global_max[0])
  default_min = g_min if fallback_min is None else float(fallback_min)
  default_max = g_max if fallback_max is None else float(fallback_max)
  return mse_range(
      hist.counts,
      hist.lower_bound,
      hist.bin_width,
      num_bits=num_bits,
      symmetric=symmetric,
      signed=signed,
      search=search,
      fallback_min=default_min,
      fallback_max=default_max,
      data_min=g_min,
      data_max=g_max,
  )
