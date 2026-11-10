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

"""Implements the GPTQ quantization.

GPTQ (Generalized Post-Training Quantization) is a quantization method that
minimizes the quantization error by considering the Hessian of the loss
function with respect to the weights. More specifically, GPTQ quantizes weights
one by one (or in columns). When a weight is quantized (introducing quantization
error), GPTQ uses the Hessian to propagate the error to the remaining
unquantized weights through the Optimal Brain Surgeon (OBS) update rule.

The algorithm:
1. Hessian calculation: during calibration, the quantizer collects input
activation samples X and computes the Hessian matrix H
2. Damping and Inversion of H: damping factor is added to the diagonal of H to
avoid ill-conditioned matrix, then the inverse of H is computed by its Cholesky
decomposition: H = L @ L.T
3. Lazy Block-wise execution: partitioning the weight columns into blocks for
more efficient computation. There are 2 ways:
  3.1 Intra-block update (Iterative): for each column i in the current block,
  - Calculate the raw quantization error
  - Normalize it by the diagonal of H^{-1}
  - Update only the remaining unquantized columns within the current block.
  3.2 Inter-block update: once all columns in the current block are quantized,
  the accumulated normalized errors are used to update all remaining columns
  in the subsequent blocks simultaneously via matrix multiplication.
"""

from collections.abc import Mapping, MutableMapping, Sequence
import dataclasses
from typing import Any
import numpy as np
import scipy.linalg
from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import common_quantize
from litert_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from litert_quantizer.algorithms.utils import common_utils

ALGORITHM_KEY = "GPTQ"


def calibrate(
    tfl_op: qtyping.OperatorT,
    graph_info: qtyping.GraphInfo,
    tensor_content_map: MutableMapping[str, np.ndarray],
    inputs_to_ignore: Sequence[int] | None = None,
    outputs_to_ignore: Sequence[int] | None = None,
    valid_range: tuple[float, float] = (-3e38, 3e38),
) -> dict[str, qtyping.QSV]:
  """Collect quantization statistics variable (QSV, e.g., min/max) for the op.

  Args:
    tfl_op: The tfl operation.
    graph_info: Graph information needed to perform quantization for the op.
    tensor_content_map: A map of tensor name to tensor content.
    inputs_to_ignore: Input tensor indices to ignore.
    outputs_to_ignore: Output tensor indices to ignore.
    valid_range: The valid range for tensor content, excluding the boundaries.
      Tensor values outside this range are ignored during calibration. Defaults
      to an approximate bfloat16 range. This range is chosen to address issues
      with `padv2` where a bfloat16 -inf padding constant can cause problems.
      Values exceeding this range can lead to quantization issues and are
      therefore excluded from min/max calibration.

  Returns:
    A dictionary mapping tensor names to the collected QSVs.
  """

  op_qsvs = {}
  min_val, max_val = valid_range

  tensor_ids = common_quantize.get_tensor_indices_requiring_calibration(
      tfl_op, graph_info, inputs_to_ignore, outputs_to_ignore
  )
  for tensor_idx in tensor_ids:
    result = common_quantize.collect_activation_tensor_statistics(
        tensor_idx,
        graph_info,
        tensor_content_map,
        valid_float_range_min=min_val,
        valid_float_range_max=max_val,
    )
    if result is None:
      continue
    tensor_name, tensor_content, tensor_qsvs = result

    x = tensor_content.reshape([-1, tensor_content.shape[-1]])

    # Currently, GPTQ is only supported for Fully Connected layer, which means
    # the last dimension of the tensor is the channel dimension.
    # Hessian calculation : 2 / num_samples * (X.T @ X)
    num_samples = tensor_qsvs["num_samples"]
    tensor_qsvs["hessian"] = (2.0 / num_samples) * x.T.dot(x)
    op_qsvs[tensor_name] = tensor_qsvs
  return op_qsvs


def _prepare_hessian_inverse(
    hessian: np.ndarray, damp_factor: float = 0.01
) -> np.ndarray:
  """Conditions and inverts the Hessian matrix via Cholesky decomposition."""
  orig_diag = np.diag(hessian)
  new_diag = np.where(orig_diag, orig_diag, 1.0)
  new_diag += damp_factor * np.mean(new_diag)
  np.fill_diagonal(hessian, new_diag)

  # Compute Cholesky decomposition: hessian = l_matrix @ l_matrix.T
  l_matrix = np.linalg.cholesky(hessian)
  # Restore the original diagonal of hessian.
  np.fill_diagonal(hessian, orig_diag)
  l_matrix, err = scipy.linalg.lapack.strtri(
      l_matrix, lower=True, overwrite_c=True
  )
  assert err == 0
  return np.einsum("ji,jk->ik", l_matrix, l_matrix, out=l_matrix)


def _apply_gptq(
    tensor_content: np.ndarray,
    quant_params: qtyping.UniformQuantParams,
    activation_tensor_qsv: Mapping[str, Any],
    tensor_quant_config: qtyping.TensorQuantizationConfig,
    blocksize: int = 64,
) -> qtyping.UniformQuantParams:
  """Applies GPTQ to adjust weight values based on Hessian information."""
  fp_weights = tensor_content.copy()

  def _get_quantized_dtype(num_bits: int) -> np.dtype:
    """Returns the appropriate NumPy dtype for the given number of bits."""
    if num_bits <= 8:
      return np.dtype(np.int8)
    elif num_bits <= 16:
      return np.dtype(np.int16)
    elif num_bits <= 32:
      return np.dtype(np.int32)
    else:
      raise ValueError(f"Unsupported num_bits for quantization: {num_bits}")

  target_dtype = _get_quantized_dtype(tensor_quant_config.num_bits)

  # target_dtype = quant_params.quantized_data.dtype
  quantized_weights = np.zeros_like(fp_weights, dtype=target_dtype)
  hessian_inv = _prepare_hessian_inverse(activation_tensor_qsv["hessian"])
  num_cols = hessian_inv.shape[0]
  is_blockwise = uniform_quantize_tensor.is_blockwise(
      tensor_quant_config.granularity
  )

  for block_start in range(0, num_cols, blocksize):
    block_end = min(block_start + blocksize, num_cols)
    cols_in_block = block_end - block_start

    weights_block = fp_weights[:, block_start:block_end]
    q_block = np.zeros_like(weights_block, dtype=target_dtype)
    err_block = np.zeros_like(weights_block)

    for i in range(cols_in_block):
      col_idx = block_start + i
      weight_col = weights_block[:, i]
      h_inv_diag = hessian_inv[col_idx, col_idx]

      col_quant_params = quant_params
      is_col_blockwise = is_blockwise
      if is_blockwise:
        # If blockwise, select the scale and zero point for the current column.
        # It assumes block quantization is applied along axis 1 (columns).
        block_idx = col_idx // quant_params.block_size
        col_quant_params = dataclasses.replace(
            quant_params,
            scale=quant_params.scale[:, block_idx],
            zero_point=quant_params.zero_point[:, block_idx],
            # weight_col is 1D, so per-channel is on axis 0
            quantized_dimension=0,
            block_size=0,  # Scales are sliced, so not blockwise anymore.
        )
        is_col_blockwise = False

      q = uniform_quantize_tensor.uniform_quantize(
          np.expand_dims(weight_col, axis=-1),
          col_quant_params,
          is_col_blockwise,
      ).reshape(-1, 1)
      dq = uniform_quantize_tensor.uniform_dequantize(
          q,
          col_quant_params,
      ).reshape(-1)

      q_block[:, i] = q.reshape(-1)
      np.subtract(weight_col, dq, out=err_block[:, i])
      err_block[:, i] /= h_inv_diag

      # Intra-block update: update remaining columns in the current block.
      if i < cols_in_block - 1:
        h_inv_row = hessian_inv[col_idx, (col_idx + 1) : block_end]
        weights_block[:, (i + 1) :] -= np.outer(err_block[:, i], h_inv_row)

    quantized_weights[:, block_start:block_end] = q_block

    # Inter-block update: update columns in subsequent blocks.
    hessian_inv_off_diag = hessian_inv[block_start:block_end, block_end:]
    fp_weights[:, block_end:] -= np.matmul(err_block, hessian_inv_off_diag)
  del hessian_inv
  return dataclasses.replace(quant_params, quantized_data=quantized_weights)


def get_tensor_quant_params(
    op_info: qtyping.OpInfo,
    tensor_quant_config: qtyping.TensorQuantizationConfig,
    tensor_content: np.ndarray | None = None,
    tensor_qsv: Mapping[str, Any] | None = None,
) -> qtyping.UniformQuantParams:
  """Get the quantization parameters for a tensor.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    tensor_quant_config: The quantization config for the tensor.
    tensor_content: The content of the tensor.
    tensor_qsv: A dictionary containingthe min/max of the tensor. It may
      contain 'activation_tensor_qsv' which contains min/max and hessian of the
      activation tensor. This will be only used for GPTQ when tensor_content is
      weight tensor.

  Returns:
    The quantization parameters for the tensor.
  """
  if tensor_quant_config.dtype == qtyping.TensorDataType.UINT:
    raise ValueError("Unsigned quantization is not supported by GPTQ.")

  activation_tensor_qsv = (
      tensor_qsv.get("activation_tensor_qsv") if tensor_qsv else None
  )
  # Get quant params.
  if tensor_qsv is None or "min" not in tensor_qsv:
    if tensor_content is not None:
      # We need min/max to calculate quantization parameters, which
      # should be collected during the calibration process. However,
      # weight-only and DRQ do not require calibration, thus it is
      # possible that this information is missing here. In that case we
      # collect min/max on the spot.
      tensor_min_max = common_quantize.init_tensor_min_max(
          tensor_content,
          op_info,
      )
    else:
      raise ValueError(
          f"{op_info.op_name}(index: {op_info.subgraph_op_index}) not found in"
          " tensor_name_to_qsv. Check if the correct calibration results are"
          " passed into the ParamsGenerator."
      )
  else:
    tensor_min_max = tensor_qsv

  if "min" not in tensor_min_max or "max" not in tensor_min_max:
    raise ValueError(
        "min and max must be provided to produce tensor quantization"
        " parameters. Check if the correct calibration results are passed into"
        " the ParamsGenerator."
    )
  zp, scale = uniform_quantize_tensor.tensor_zp_scale_from_min_max(
      tensor_min_max["min"],
      tensor_min_max["max"],
      tensor_quant_config.num_bits,
      tensor_quant_config.symmetric,
      tensor_quant_config.granularity,
      None,
  )
  quantized_dim = common_utils.get_weight_quantized_dim(
      op_info, tensor_content, tensor_quant_config.granularity  # pyrefly: ignore[bad-argument-type]
  )
  quant_params = qtyping.UniformQuantParams(
      scale=scale,
      zero_point=zp,
      num_bits=tensor_quant_config.num_bits,
      symmetric=tensor_quant_config.symmetric,
      quantized_dimension=quantized_dim,
      block_size=uniform_quantize_tensor.extract_block_size_from_granularity(
          tensor_quant_config.granularity
      ),
  )
  if tensor_content is None:
    return quant_params

  if activation_tensor_qsv is None or "hessian" not in activation_tensor_qsv:
    return quant_params
  return _apply_gptq(
      tensor_content,
      quant_params,
      activation_tensor_qsv,
      tensor_quant_config,
  )
