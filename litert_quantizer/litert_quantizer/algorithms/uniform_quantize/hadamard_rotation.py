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

"""Implements the Hadamard Rotation quantization.

Hadamard rotation mitigates the quantization loss caused by the large
eigenvalues of the weight matrix (outliers) by rotating the weights with a
Hadamard matrix (H). The rotation distributes the larger eigenvalues more evenly
across the Hadamard transformed weight matrix. In other words, it "smears" the
outliers across the entire tensor, making the distribution more uniform.

Matrix H is a square matrix with entries +1 or -1 and orthogonal rows and cols
(H * H^T = n * I, where n is the dimension). The rotation is often performed
using a normalized Hadamard matrix (1/sqrt(n) * H), which is orthogonal.

A fully connected linear layer with weight W and input X can be written as:
Y = X * W^T = (X * H) * (W * H)^T
- The input X is rotated by H (updated online during inference)
- The weight W is rotated by H (updated offline during quantization, with OCTAV)
"""

import dataclasses
import enum
from typing import Any, Optional
import numpy as np
from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import octav
from litert_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from litert_quantizer.algorithms.utils import common_utils
from litert_quantizer.utils import tfl_flatbuffer_utils


CUSTOM_OP_ALGORITHM_KEY = "HADAMARD_ROTATION"
DECOMPOSED_ALGORITHM_KEY = "DECOMPOSED_HADAMARD_ROTATION"

# Theoretical crossover dimension `m` (`D = m * d`) when `m` is not a power of 2
# between skew-symmetric Rademacher Cayley (`m <= 48`) and Polar SVD (`m > 48`):
#   * Rademacher Cayley (`Q = (I - A)(I + A)^{-1}`, `A^T = -A`,
#     `|a_ij| = 1 / (sqrt(m) + 1)`): `(I + A)^T (I + A) = I + A^T A >= I`
#     bounds all singular values below by `1` and keeps off-diagonal entries
#     concentrated at `O(1 / sqrt(m))` without Gaussian tail spikes. However,
#     by Wigner's semicircle Stieltjes transform on `2(I + A)^{-1} - I`, the
#     diagonal entries retain a non-vanishing `O(1)` floor as `m -> inf`:
#       `E[Q_ii] = (sqrt(1 + 4 * alpha_m^2) - 1 - alpha_m^2) / alpha_m^2`
#       `-> sqrt(5) - 2 ~= 0.236` (`alpha_m = sqrt(m) / (sqrt(m) + 1)`).
#   * Polar SVD (`Q = U @ V^T` of `R in {-1, +1}^{m x m}`): zero diagonal bias
#     (`E[Q_ii^2] = 1 / m`), so expected column peak magnitude decays to zero
#     as `sqrt(2 * ln(m) / m)`. Below `m ~= 48`, sub-Gaussian tail spikes
#     (`sqrt(2 * ln(m) / m) > 0.40`) and small-`m` singular value dispersion
#     exceed Cayley's entries; above `m ~= 48`, `sqrt(2 * ln(m) / m)` drops
#     into and below Cayley's non-vanishing diagonal floor (`~0.33 -> 0.236`),
#     giving Polar SVD lower worst-case outlier coherence for all larger `m`.
_HYBRID_MAX_DIM = 48


class _AutoLowerNameEnum(enum.Enum):
  """Enum base whose `enum.auto()` values are lowercase member names."""

  @staticmethod
  def _generate_next_value_(
      name: str, start: int, count: int, last_values: list[str]
  ) -> str:
    del start, count, last_values
    return name.lower()


@enum.unique
class QInitMode(_AutoLowerNameEnum):
  """Cross-block orthogonal matrix initializer mode (`Q_1` in `(Q_1, H_d)`)."""

  NONE = enum.auto()
  AUTO = enum.auto()


@enum.unique
class SizingMode(_AutoLowerNameEnum):
  """Hadamard block size (`d`) strategy for `QInitMode.AUTO`."""

  MIN_COMPUTE = enum.auto()
  MAX_POWER_OF_TWO = enum.auto()


VALID_SCALE_REFINEMENTS = ("none", "mse_grid")

# `mse_grid` scale refinement tries `_MSE_GRID_NUM_ALPHAS` evenly spaced
# multipliers in [0.80, 1.20] of each channel's OCTAV scale. Tensors with more
# than `_MSE_GRID_MAX_FINE_ROWS` rows (for example, `[262144, 2560]` embedding
# tables) use the coarser `_MSE_GRID_COARSE_NUM_ALPHAS` grid to bound
# quantization time.
_MSE_GRID_NUM_ALPHAS = 25
_MSE_GRID_COARSE_NUM_ALPHAS = 15
_MSE_GRID_MAX_FINE_ROWS = 16384

_HADAMARD_MATRIX_CACHE = {}
_CROSS_BLOCK_MATRIX_CACHE = {}
_IntLike = int | np.integer


def _as_non_bool_int(value: _IntLike, name: str) -> int:
  """Validates that `value` is a non-boolean integer and returns `int`."""
  if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
    raise ValueError(f"{name} must be a non-boolean integer; got {value!r}.")
  return int(value)


def _get_max_hadamard_size(
    dim: _IntLike, max_size: Optional[_IntLike] = None
) -> int:
  """Returns the largest power of 2 that divides `dim`, capped at `max_size`.

  Args:
    dim: The size of the dimension to rotate.
    max_size: The maximum size of the Hadamard matrix (`None` or `0` means
      uncapped).

  Returns:
    The Hadamard matrix size.

  Raises:
    ValueError: If `dim` or `max_size` is not a non-boolean integer, `dim` is
      odd or less than 2, or `max_size` is negative or 1.
  """
  dim = _as_non_bool_int(dim, "dim")
  max_size = (
      _as_non_bool_int(max_size, "max_hadamard_size")
      if max_size is not None
      else None
  )
  if dim < 2 or dim % 2 or (max_size and max_size < 2):
    raise ValueError(
        "Hadamard rotation requires an even reduction dimension and"
        f" max_hadamard_size >= 2; got dim={dim},"
        f" max_hadamard_size={max_size}."
    )
  # 2**30 is just a large power of 2 to calculate this factor.
  hadamard_size = int(np.gcd(dim, 2**30))
  if max_size:
    hadamard_size = min(hadamard_size, 1 << (max_size.bit_length() - 1))
  return hadamard_size


def _select_kronecker_hadamard_size(
    dim: _IntLike,
    max_size: Optional[_IntLike] = None,
    sizing_mode: SizingMode | str = SizingMode.MIN_COMPUTE,
) -> int:
  """Selects power-of-2 divisor `d | dim` using the specified `sizing_mode`."""
  sizing_mode = SizingMode(sizing_mode)
  dim = _as_non_bool_int(dim, "dim")
  max_pow2 = _get_max_hadamard_size(dim, max_size)
  if sizing_mode == SizingMode.MAX_POWER_OF_TWO:
    return max_pow2

  best_d = 2
  best_cost = (dim // 2) + 2
  d = 2
  while d <= max_pow2:
    cost = (dim // d) + d
    if cost <= best_cost:
      best_cost = cost
      best_d = d
    d *= 2
  return best_d


def get_hadamard_block_size(
    dim: _IntLike,
    max_hadamard_size: Optional[_IntLike] = None,
    q_init_mode: QInitMode | str = QInitMode.NONE,
    sizing_mode: SizingMode | str = SizingMode.MIN_COMPUTE,
) -> int:
  """Returns the Hadamard block size `d` selected for reduction dimension `dim`.

  Exposes the exact `(m, d)` Kronecker block-factor selection used by
  `DECOMPOSED_HADAMARD_ROTATION` so external offline calibration/QAT pipelines
  (e.g., PyTorch rotation-aware calibration) and cross-framework parity tests
  can match LiteRT's block factorization bit-for-bit prior to FlatBuffer export.

  Args:
    dim: Size of the reduction dimension to rotate.
    max_hadamard_size: Optional upper cap on the Hadamard block size (`None` or
      `0` means uncapped).
    q_init_mode: Cross-block orthogonal matrix initializer (`QInitMode.NONE` /
      `"none"` or `QInitMode.AUTO` / `"auto"`). When `"none"`, `sizing_mode` is
      ignored and the largest power-of-2 divisor (capped by `max_hadamard_size`)
      is returned.
    sizing_mode: Block sizing strategy when `q_init_mode != "none"`
      (`SizingMode.MIN_COMPUTE` / `"min_compute"` or
      `SizingMode.MAX_POWER_OF_TWO` / `"max_power_of_two"`).

  Returns:
    Power-of-2 Hadamard block size `d >= 2` that evenly divides `dim`.

  Raises:
    ValueError: If `q_init_mode` or `sizing_mode` is unsupported, `dim` or
      `max_hadamard_size` is not a non-boolean integer, `dim` is odd or less
      than 2, or `max_hadamard_size` is invalid.
  """
  q_init_mode = QInitMode(q_init_mode)
  sizing_mode = SizingMode(sizing_mode)
  if q_init_mode == QInitMode.NONE:
    return _get_max_hadamard_size(dim, max_hadamard_size)
  return _select_kronecker_hadamard_size(
      dim, max_hadamard_size, sizing_mode=sizing_mode
  )


def _make_rademacher_cayley_matrix(
    m: int, rng: np.random.Generator
) -> np.ndarray:
  """Generates orthogonal `Q_1 = (I - A)(I + A)^{-1}` with Rademacher skew `A`."""
  scale = 1.0 / (np.sqrt(float(m)) + 1.0)
  num_upper = (m * (m - 1)) // 2
  signs = rng.choice([-scale, scale], size=num_upper).astype(np.float64)
  skew = np.zeros((m, m), dtype=np.float64)
  rows, cols = np.triu_indices(m, k=1)
  skew[rows, cols] = signs
  skew -= skew.T
  eye = np.eye(m, dtype=np.float64)
  q1 = np.linalg.solve(eye + skew.T, eye - skew.T).T
  return q1.astype(np.float32)


def _make_polar_svd_matrix(m: int, rng: np.random.Generator) -> np.ndarray:
  """Generates orthogonal `Q_1 = U @ V^T` via SVD of a Rademacher matrix."""
  signs = rng.choice([-1.0, 1.0], size=(m, m)).astype(np.float64)
  u, _, vh = np.linalg.svd(signs, full_matrices=False)
  return (u @ vh).astype(np.float32)


def _make_hadamard_matrix(size: int) -> np.ndarray:
  """Generates a Hadamard matrix of the given size.

  Hadamard matrices are square matrices with entries +1 or -1 and orthogonal
  rows and columns. They are defined recursively:

    H_1 = [[1]]
    H_{2n} = [[H_n, H_n], [H_n, -H_n]]

  Args:
    size: The size of the Hadamard matrix. Must be a power of 2 that is at least
      2. This represents a single dimension. E.g. if size is 4, then the
      Hadamard matrix is a 4x4 matrix.

  Returns:
    The Hadamard matrix.

  Raises:
    ValueError: If the size is not a power of 2 that is at least 2.
  """
  if not isinstance(size, int):
    size = int(size)

  if size < 2 or (size & (size - 1)) != 0:
    raise ValueError(
        "Hadamard matrix size must be a power of 2 that is at least 2, got"
        f" {size}."
    )

  # Check whether we already have a Hadamard matrix for this size.
  if size in _HADAMARD_MATRIX_CACHE:
    return _HADAMARD_MATRIX_CACHE[size]

  # Otherwise, create one.
  h_int8 = h2 = np.array([[1, 1], [1, -1]], dtype=np.int8)
  h_norm = h_int8 / np.sqrt(2, dtype=np.float32)
  current_size = 2
  while current_size < size:
    h_int8 = np.kron(h_int8, h2)
    current_size *= 2
    if current_size not in _HADAMARD_MATRIX_CACHE:
      h_norm = h_int8 / np.sqrt(current_size, dtype=np.float32)
      _HADAMARD_MATRIX_CACHE[current_size] = h_norm
    else:
      h_norm = _HADAMARD_MATRIX_CACHE[current_size]
  return h_norm


def make_cross_block_matrix(
    m: _IntLike,
    q_init_mode: QInitMode | str,
    seed: _IntLike = 0,
) -> Optional[np.ndarray]:
  """Constructs the cross-block orthogonal matrix `Q_1` of shape `(m, m)`.

  Exposes the deterministic cross-block orthogonal factor `Q_1` used by 2-stage
  Kronecker rotation (`R = np.kron(Q_1, H_d)`) so external offline
  calibration/QAT pipelines and cross-framework parity tests can construct the
  identical orthogonal matrix without duplicating the RNG/Cayley/SVD logic or
  reaching into private symbols.

  Args:
    m: Number of Hadamard blocks (`dim // d`). Must be at least 1.
    q_init_mode: Cross-block orthogonal matrix initializer (`QInitMode.NONE` /
      `"none"` or `QInitMode.AUTO` / `"auto"`).
    seed: Deterministic RNG seed used when `m` is not a power of 2.

  Returns:
    `None` when `q_init_mode == "none"` or `m == 1`; otherwise a read-only
    `float32` orthogonal matrix of shape `(m, m)` (Sylvester Hadamard when `m`
    is a power of 2, Rademacher Cayley when `m <= 48`, or Polar SVD when
    `m > 48`).

  Raises:
    ValueError: If `q_init_mode` is unsupported, `m` or `seed` is not a
      non-boolean integer, or `m < 1`.
  """
  q_init_mode = QInitMode(q_init_mode)
  m = _as_non_bool_int(m, "m")
  seed = _as_non_bool_int(seed, "seed")
  if m < 1:
    raise ValueError(f"Cross-block dimension m must be >= 1, got m={m}.")
  if q_init_mode == QInitMode.NONE or m == 1:
    return None

  if (m & (m - 1)) == 0:
    q_had = _make_hadamard_matrix(m).astype(np.float32)
    q_had.setflags(write=False)
    return q_had

  resolved_mode = "rademacher" if m <= _HYBRID_MAX_DIM else "polar_svd"
  cache_key = (m, resolved_mode, seed)
  if cache_key in _CROSS_BLOCK_MATRIX_CACHE:
    return _CROSS_BLOCK_MATRIX_CACHE[cache_key]

  rng = np.random.default_rng(seed)
  if resolved_mode == "rademacher":
    q1 = _make_rademacher_cayley_matrix(m, rng)
  else:
    q1 = _make_polar_svd_matrix(m, rng)

  q1.setflags(write=False)
  _CROSS_BLOCK_MATRIX_CACHE[cache_key] = q1
  return q1


def _make_random_binary_vector(size: int, seed: _IntLike = 0) -> np.ndarray:
  """Generates a deterministic Rademacher (+/-1) sign vector of length `size`."""
  rng = np.random.default_rng([int(size), int(seed), 1])
  return rng.choice(np.array([-1, 1], dtype=np.int8), size=int(size))


def _rotate_with_diagonal_hadamard(
    tensor_content: np.ndarray,
    axis: int,
    max_size: _IntLike | None = None,
    q_init_mode: QInitMode | str = QInitMode.NONE,
    q_seed: _IntLike = 0,
    randomize: bool = False,
    sizing_mode: SizingMode | str = SizingMode.MIN_COMPUTE,
):
  """Rotates the float array using single Hadamard or Kronecker `(Q_1, H_d)`.

  Args:
    tensor_content: The float array to rotate before quantization.
    axis: The axis of the tensor to rotate.
    max_size: The maximum size of the Hadamard matrix.
    q_init_mode: Cross-block orthogonal matrix initializer (`"none"` or
      `"auto"`).
    q_seed: Random seed for constructing the cross-block orthogonal matrix and
      random sign flips.
    randomize: Whether to apply deterministic Rademacher `+/-1` sign flips
      (`diag(random_vector) @ H_d`) before the Hadamard transform.
    sizing_mode: Strategy for choosing Hadamard block size `d` when `q_init_mode
      != "none"` (`"min_compute"` or `"max_power_of_two"`).

  Returns:
    A tuple `(w_rotated, hadamard_size, random_vector, cross_block_matrix)`.

  Raises:
    ValueError: If the axis is not the last axis of tensor_content. To support
      other axes, please add support to the matrix multiplication.
    ValueError: If the rotated dimension is odd or `max_size` is less than 2.
  """
  if axis != tensor_content.ndim - 1:
    raise ValueError(
        "Hadamard rotation is only supported for tensors with quantized"
        " dimension 0 (rotate last dimension)."
    )

  dim = tensor_content.shape[axis]
  hadamard_size = get_hadamard_block_size(
      dim,
      max_hadamard_size=max_size,
      q_init_mode=q_init_mode,
      sizing_mode=sizing_mode,
  )
  num_blocks = dim // hadamard_size
  cross_block_matrix = make_cross_block_matrix(
      num_blocks, q_init_mode=q_init_mode, seed=q_seed
  )

  if randomize:
    random_vector = _make_random_binary_vector(hadamard_size, seed=q_seed)
  else:
    random_vector = np.ones(hadamard_size, dtype=np.int8)
  # Use a canonical Hadamard matrix, with optional random sign flips.
  hadamard = random_vector.astype(np.float32)[
      :, np.newaxis
  ] * _make_hadamard_matrix(hadamard_size)

  reshaped_tensor = tensor_content.reshape((-1, hadamard_size), copy=False)
  w_rotated = np.matmul(reshaped_tensor, hadamard)
  if cross_block_matrix is not None:
    num_blocks = cross_block_matrix.shape[0]
    # (m, m) @ (R, m, d) -> (R, m, d): mixes the m blocks of every row.
    w_rotated = np.matmul(
        cross_block_matrix,
        w_rotated.reshape((-1, num_blocks, hadamard_size), copy=False),
    )

  return (
      w_rotated.reshape(tensor_content.shape, copy=False),
      hadamard_size,
      random_vector,
      cross_block_matrix,
  )


# Centered 2-bit quantization (`num_bits=2, centered=True` by default) vs.
# `int2_in_int4` (`num_bits=4` emulation):
# Mid-tread 2-bit quantization (`zero_point=0`) places a wide deadzone bin at
# zero and uses only 3 levels `{-1, 0, +1} * scale` inside `[-clip, clip]`,
# incurring ~30% higher Relative RMSE (~2x higher MSE) on rotated distributions
# than the 4-level symmetric mid-riser grid `{-1.5, -0.5, +0.5, +1.5} * delta`.
#
# - Native centered `INT2` (`num_bits=2`, `centered=True` by default) stores
#   codes `q in {-2, -1, 0, +1}` packed 4 to a byte (`TensorType.INT2`) with
#   `BlockwiseQuantization` (`zero_points = -0.5` and `scale = delta = clip / 2`
#   so `(q + 0.5) * scale` reconstructs `{-1.5, -0.5, +0.5, +1.5} * delta`).
# - `int2_in_int4=True` (`num_bits=4`) encodes the same 4 levels as odd integers
#   `{-3, -1, +1, +3}` (`zero_point=0`, `scale = delta / 2 = clip / 4`) inside
#   a signed `INT4` container for backends that only implement `INT4` kernels.


def _quantize_mse_chunk(
    w_chunk: np.ndarray,
    s_chunk: np.ndarray,
    qmin: float,
    qmax: float,
    int2_in_int4: bool,
    centered_int2: bool = False,
) -> np.ndarray:
  """Quantizes a weight chunk during MSE scale refinement.

  Args:
    w_chunk: Float32 rotated weight chunk `[rows, cols]`.
    s_chunk: Candidate per-row scale `[rows, 1]`.
    qmin: Minimum integer code for standard uniform rounding.
    qmax: Maximum integer code for standard uniform rounding.
    int2_in_int4: Whether to restrict `INT4` codes to `{-3, -1, 1, 3}`.
    centered_int2: Whether to round onto centered `INT2` codes `{-2, -1, 0, 1}`
      with `zero_point = -0.5`.

  Returns:
    Quantized float32 codes matching `w_chunk` shape.
  """
  if int2_in_int4:
    return octav.to_odd_int4_codes(w_chunk, s_chunk).astype(np.float32)
  if centered_int2:
    return octav.to_centered_int2_codes(w_chunk, s_chunk).astype(np.float32)
  return np.clip(np.rint(w_chunk / s_chunk), qmin, qmax)


def _refine_quant_params_mse(
    w_rotated: np.ndarray,
    qparams: qtyping.UniformQuantParams,
    int2_in_int4: bool = False,
) -> qtyping.UniformQuantParams:
  """Refines per-channel symmetric scale and quantized_data for L2 MSE.

  Preserves both the centered `INT2` grid (`(q + 0.5) * scale` for `q in
  {-2, -1, 0, 1}`) and the odd `INT4` mid-riser grid (`q * scale` for `q in
  {-3, -1, 1, 3}` when `int2_in_int4=True`) across both the multiplier grid
  search and the 2-step alternating least-squares (ALS) update.

  Args:
    w_rotated: Rotated weight tensor `[m_rows, d_cols]`.
    qparams: Initial quantization parameters from OCTAV.
    int2_in_int4: Whether to preserve the 4-level `{-3, -1, 1, 3}` codebook.

  Returns:
    Refined `UniformQuantParams`.
  """
  if (
      not qparams.symmetric
      or qparams.block_size != 0
      or qparams.quantized_dimension != 0
      or w_rotated.ndim != 2
      or qparams.quantized_data is None
  ):
    return qparams

  m_rows, d_cols = w_rotated.shape
  orig_scale_shape = qparams.scale.shape
  orig_scale_dtype = qparams.scale.dtype
  orig_q_dtype = qparams.quantized_data.dtype

  centered_int2 = uniform_quantize_tensor.is_centered_int2_zero_point(
      uniform_quantize_tensor.IntType(qparams.num_bits, signed=qparams.signed),
      qparams.zero_point,
  )
  code_offset = np.float32(0.5) if centered_int2 else np.float32(0.0)

  qmax = float((1 << (qparams.num_bits - 1)) - 1)
  narrow = qparams.symmetric and qparams.num_bits >= 8
  qmin = float(-(1 << (qparams.num_bits - 1)) + (1 if narrow else 0))

  base_scale = qparams.scale.reshape(m_rows, 1).astype(np.float32)
  best_scale = base_scale.copy()
  best_q = qparams.quantized_data.reshape(m_rows, d_cols).copy()
  if narrow:
    best_q = np.clip(best_q, int(qmin), int(qmax))

  num_alphas = (
      _MSE_GRID_NUM_ALPHAS
      if m_rows <= _MSE_GRID_MAX_FINE_ROWS
      else _MSE_GRID_COARSE_NUM_ALPHAS
  )
  alphas = np.linspace(0.80, 1.20, num_alphas, dtype=np.float32)
  chunk_size = 4096
  for start in range(0, m_rows, chunk_size):
    end = min(start + chunk_size, m_rows)
    w_chunk = w_rotated[start:end].astype(np.float32, copy=False)
    s_cur = best_scale[start:end].copy()
    s_base_chunk = base_scale[start:end]
    q_cur = best_q[start:end].astype(np.float32)
    err_cur = np.sum(
        (w_chunk - (q_cur + code_offset) * s_cur) ** 2, axis=1, keepdims=True
    )

    for alpha in alphas:
      s_cand = np.maximum(s_base_chunk * alpha, 1e-9)
      q_cand = _quantize_mse_chunk(
          w_chunk, s_cand, qmin, qmax, int2_in_int4, centered_int2
      )
      err_cand = np.sum(
          (w_chunk - (q_cand + code_offset) * s_cand) ** 2,
          axis=1,
          keepdims=True,
      )
      improved = err_cand < err_cur
      err_cur = np.where(improved, err_cand, err_cur)
      s_cur = np.where(improved, s_cand, s_cur)
      q_cur = np.where(improved, q_cand, q_cur)

    # Perform 2 final alternating least-squares steps with exact re-rounding
    s_cand = s_cur
    for _ in range(2):
      q_eff = q_cur + code_offset
      qq = np.sum(q_eff * q_eff, axis=1, keepdims=True)
      wq = np.sum(w_chunk * q_eff, axis=1, keepdims=True)
      s_cand = np.where(qq > 0.0, wq / np.maximum(qq, 1e-12), s_cand)
      s_cand = np.maximum(s_cand, 1e-9)
      q_cand = _quantize_mse_chunk(
          w_chunk, s_cand, qmin, qmax, int2_in_int4, centered_int2
      )
      err_cand = np.sum(
          (w_chunk - (q_cand + code_offset) * s_cand) ** 2,
          axis=1,
          keepdims=True,
      )
      improved = err_cand < err_cur
      err_cur = np.where(improved, err_cand, err_cur)
      s_cur = np.where(improved, s_cand, s_cur)
      q_cur = np.where(improved, q_cand, q_cur)

    best_scale[start:end] = s_cur
    best_q[start:end] = q_cur.astype(orig_q_dtype)

  return dataclasses.replace(
      qparams,
      scale=best_scale.reshape(orig_scale_shape).astype(orig_scale_dtype),
      quantized_data=best_q.reshape(w_rotated.shape),
  )


def get_tensor_quant_params(
    op_info: qtyping.OpInfo,
    tensor_quant_config: qtyping.TensorQuantizationConfig,
    tensor_content: Optional[np.ndarray] = None,
    tensor_qsv: Optional[dict[str, Any]] = None,
) -> qtyping.UniformQuantParams:
  """Returns the quantization parameters for a tensor.

  This function will rotate the tensor with a Hadamard matrix and then
  quantize it with OCTAV.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    tensor_quant_config: The quantization config for the tensor.
    tensor_content: The content of the tensor. When None, it means the tensor is
      not a weight tensor (e.g. static quantization).
    tensor_qsv: A dictionary containing the min/max of the tensor.

  Raises:
    ValueError: If the blockwise quantization is requested.
    ValueError: If the asymmetric quantization is requested.
    ValueError: `tensor_qsv` must contain min/max values, or `tensor_content`
      must be provided so that they can be inferred.
    ValueError: If the reduction dimension is odd or `max_hadamard_size` is
      less than 2.
    ValueError: If `int2_in_int4` is not a boolean or is enabled without
      symmetric 4-bit quantization.
    ValueError: If `centered` is not a boolean or is enabled without symmetric
      2-bit quantization.
  """
  if tensor_content is None:
    raise ValueError("Hadamard rotation is only supported for weight tensors.")

  if tensor_qsv is not None:
    raise ValueError(
        "Hadamard rotation is not supported for static quantization."
    )

  if tensor_content.ndim < 2:
    raise ValueError(
        "Hadamard rotation is only supported for tensors with rank >= 2."
    )

  # Reduction axis is the last non-quantized dimension. Since we only support
  # quantized_dim of 0 (or 1 for blockwise), the reduction axis is the last
  # axis.
  reduce_axis = tensor_content.ndim - 1
  alg_params = tensor_quant_config.algorithm_params or {}

  # Rotate the tensor with a Hadamard (or Kronecker) orthogonal transformation.
  (
      w_rotated,
      hadamard_size,
      random_vector,
      cross_block_matrix,
  ) = _rotate_with_diagonal_hadamard(
      tensor_content,
      axis=reduce_axis,
      max_size=alg_params.get("max_hadamard_size"),
      q_init_mode=alg_params.get("q_init_mode", QInitMode.NONE),
      q_seed=alg_params.get("q_seed", 0),
      randomize=alg_params.get("randomize", False),
      sizing_mode=alg_params.get("sizing_mode", SizingMode.MIN_COMPUTE),
  )

  # Get the quantized values of the rotated tensor (validates int2_in_int4).
  qparams = octav.get_tensor_quant_params(
      op_info, tensor_quant_config, w_rotated, tensor_qsv
  )

  scale_refinement = alg_params.get("scale_refinement", "none")
  if scale_refinement not in VALID_SCALE_REFINEMENTS:
    raise ValueError(
        f"Unsupported scale_refinement: {scale_refinement!r}. Expected one of"
        f" {VALID_SCALE_REFINEMENTS}."
    )
  if scale_refinement == "mse_grid":
    qparams = _refine_quant_params_mse(
        w_rotated,
        qparams,
        int2_in_int4=bool(alg_params.get("int2_in_int4", False)),
    )

  return qtyping.UniformQuantParams(
      quantized_dimension=qparams.quantized_dimension,
      num_bits=qparams.num_bits,
      scale=qparams.scale,
      zero_point=qparams.zero_point,
      symmetric=qparams.symmetric,
      quantized_data=qparams.quantized_data,
      block_size=qparams.block_size,
      hadamard=qtyping.UniformQuantParams.HadamardRotationParams(
          random_binary_vector=random_vector,
          hadamard_size=hadamard_size,
          cross_block_matrix=cross_block_matrix,
      ),
  )


def _materialize_fully_connected(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_quant_params_cache: common_utils.TensorQuantParamsCache,
    is_decomposed: bool = False,
    tensor_name_to_qsv: Optional[dict[str, Any]] = None,  # pylint: disable=unused-argument
) -> list[qtyping.TensorTransformationParams]:
  """Materialize the fully_connected op.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_quant_params_cache: Cache of already computed
      `UniformQuantParams|NonLinearQuantParams` objects keyed on a tuple of the
      buffer ID and the `TensorQuantizationConfig` used to compute it.
    is_decomposed: Whether to use decomposed Hadamard rotation ops or a custom
      op.
    tensor_name_to_qsv: A map of tensor name to quantization parameters.

  Returns:
    Quantization configuration for the tensors associated with the op (e.g.,
    weights, bias).
  """
  if op_info.op_quant_config.weight_tensor_config is None:
    raise ValueError(
        "Weight tensor quantization config is not provided for Hadamard"
        " Rotation quantization."
    )

  op_tensor_params = []

  # Materialize weight.
  weight_tensor_index = 1
  weight_tensor = graph_info.subgraph_tensors[
      op_info.op.inputs[weight_tensor_index]
  ]
  tensor_data = tfl_flatbuffer_utils.get_tensor_data(
      weight_tensor, graph_info.buffers
  )

  if not (
      quant_params := tensor_quant_params_cache.lookup(
          weight_tensor.buffer, op_info.op_quant_config.weight_tensor_config
      )
  ):
    # quant_params contains the rotated and quantized weights done by
    # get_tensor_quant_params().
    quant_params = get_tensor_quant_params(
        op_info,
        op_info.op_quant_config.weight_tensor_config,
        tensor_data,
        None,
    )
    tensor_quant_params_cache.insert(
        weight_tensor.buffer,
        op_info.op_quant_config.weight_tensor_config,
        quant_params,
    )

  transformations = [qtyping.QuantTransformation.QUANTIZE_TENSOR]
  op2tensor_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      parameters=quant_params,
      transformations=transformations,
  )
  weight_transformation_params = qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(weight_tensor),
      consumers=[op2tensor_params],
  )

  # Materialize input. A hadamard rotation op should be inserted on the input
  # tensor to do the inverse of the weight's transformation.
  input_tensor_index = 0
  input_tensor = graph_info.subgraph_tensors[
      op_info.op.inputs[input_tensor_index]
  ]
  transformations = [
      qtyping.QuantTransformation.INSERT_DECOMPOSED_HADAMARD_ROTATION
      if is_decomposed
      else qtyping.QuantTransformation.INSERT_HADAMARD_ROTATION,
  ]
  op2tensor_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      parameters=quant_params,
      transformations=transformations,
  )
  input_transformation_params = qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(input_tensor),
      consumers=[op2tensor_params],
  )
  op_tensor_params.append(input_transformation_params)
  op_tensor_params.append(weight_transformation_params)

  # Materialize bias. Since static quantization is not supported, we do not
  # quantize the bias tensor.
  bias_tensor_index = 2
  bias_tensor = graph_info.subgraph_tensors[
      op_info.op.inputs[bias_tensor_index]
  ]
  no_quant_tensor_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      transformations=[qtyping.QuantTransformation.NO_QUANTIZE],
  )
  bias_transformation_params = qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(bias_tensor),
      consumers=[no_quant_tensor_params],
  )
  op_tensor_params.append(bias_transformation_params)

  # Materialize output. Since static quantization is not supported, we do not
  # quantize the output tensor.
  output_tensor_index = 0
  output_tensor = graph_info.subgraph_tensors[
      op_info.op.outputs[output_tensor_index]
  ]
  no_quant_tensor_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      transformations=[qtyping.QuantTransformation.NO_QUANTIZE],
  )
  output_transformation_params = qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(output_tensor),
      producer=no_quant_tensor_params,
  )
  op_tensor_params.append(output_transformation_params)

  return op_tensor_params


def materialize_fully_connected_custom_op(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_quant_params_cache: common_utils.TensorQuantParamsCache,
    tensor_name_to_qsv: Optional[dict[str, Any]] = None,
) -> list[qtyping.TensorTransformationParams]:
  return _materialize_fully_connected(
      op_info,
      graph_info,
      is_decomposed=False,
      tensor_name_to_qsv=tensor_name_to_qsv,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def materialize_fully_connected_decomposed(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_quant_params_cache: common_utils.TensorQuantParamsCache,
    tensor_name_to_qsv: Optional[dict[str, Any]] = None,
) -> list[qtyping.TensorTransformationParams]:
  return _materialize_fully_connected(
      op_info,
      graph_info,
      is_decomposed=True,
      tensor_name_to_qsv=tensor_name_to_qsv,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def _materialize_embedding_lookup(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_quant_params_cache: common_utils.TensorQuantParamsCache,
    is_decomposed: bool = False,
    tensor_name_to_qsv: Optional[dict[str, Any]] = None,  # pylint: disable=unused-argument
) -> list[qtyping.TensorTransformationParams]:
  """Materialize the embedding_lookup op.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_quant_params_cache: Cache of already computed
      `UniformQuantParams|NonLinearQuantParams` objects keyed on a tuple of the
      buffer ID and the `TensorQuantizationConfig` used to compute it.
    is_decomposed: Whether to use decomposed Hadamard rotation ops or a custom
      op.
    tensor_name_to_qsv: A map of tensor name to quantization parameters.

  Returns:
    Quantization configuration for the tensors associated with the op (e.g.,
    weights, bias).
  """
  op_tensor_params = []

  # Materialize lookup.
  lookup_tensor_index = 0
  lookup_tensor = graph_info.subgraph_tensors[
      op_info.op.inputs[lookup_tensor_index]
  ]
  transformations = [
      qtyping.QuantTransformation.NO_QUANTIZE,
  ]
  op2tensor_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      parameters=None,
      transformations=transformations,
  )
  lookup_transformation_params = qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(lookup_tensor),
      consumers=[op2tensor_params],
  )
  op_tensor_params.append(lookup_transformation_params)

  # Materialize embedding. The embedding table should be rotated and then
  # quantized.
  embedding_tensor_index = 1
  embedding_tensor = graph_info.subgraph_tensors[
      op_info.op.inputs[embedding_tensor_index]
  ]
  tensor_data = tfl_flatbuffer_utils.get_tensor_data(
      embedding_tensor, graph_info.buffers
  )

  if tensor_quant_params_cache is None or not (
      quant_params := tensor_quant_params_cache.lookup(
          embedding_tensor.buffer, op_info.op_quant_config.weight_tensor_config  # pyrefly: ignore[bad-argument-type]
      )
  ):
    quant_params = get_tensor_quant_params(
        op_info,
        op_info.op_quant_config.weight_tensor_config,  # pyrefly: ignore[bad-argument-type]
        tensor_data,
        None,
    )
    if tensor_quant_params_cache:
      tensor_quant_params_cache.insert(
          embedding_tensor.buffer,
          op_info.op_quant_config.weight_tensor_config,  # pyrefly: ignore[bad-argument-type]
          quant_params,
      )

  transformations = [qtyping.QuantTransformation.QUANTIZE_TENSOR]
  op2tensor_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      parameters=quant_params,
      transformations=transformations,
  )
  weight_transformation_params = qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(embedding_tensor),
      consumers=[op2tensor_params],
  )
  op_tensor_params.append(weight_transformation_params)

  # Materialize output. A hadamard rotation op should be inserted on the output
  # tensor to do the inverse of the embedding's transformation.
  output_tensor_index = 0
  output_tensor = graph_info.subgraph_tensors[
      op_info.op.outputs[output_tensor_index]
  ]
  transformations = [
      qtyping.QuantTransformation.INSERT_DECOMPOSED_HADAMARD_ROTATION
      if is_decomposed
      else qtyping.QuantTransformation.INSERT_HADAMARD_ROTATION,
  ]
  op2tensor_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      parameters=quant_params,
      transformations=transformations,
  )
  output_transformation_params = qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(output_tensor),
      producer=op2tensor_params,
  )
  op_tensor_params.append(output_transformation_params)

  return op_tensor_params


def materialize_embedding_lookup_custom_op(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_quant_params_cache: common_utils.TensorQuantParamsCache,
    tensor_name_to_qsv: Optional[dict[str, Any]] = None,
) -> list[qtyping.TensorTransformationParams]:
  return _materialize_embedding_lookup(
      op_info,
      graph_info,
      is_decomposed=False,
      tensor_name_to_qsv=tensor_name_to_qsv,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def materialize_embedding_lookup_decomposed(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_quant_params_cache: common_utils.TensorQuantParamsCache,
    tensor_name_to_qsv: Optional[dict[str, Any]] = None,
) -> list[qtyping.TensorTransformationParams]:
  return _materialize_embedding_lookup(
      op_info,
      graph_info,
      is_decomposed=True,
      tensor_name_to_qsv=tensor_name_to_qsv,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )
