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

"""Implements the MSE quantization."""

import dataclasses
from typing import Any, Optional
import numpy as np
from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import common_quantize
from litert_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from litert_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from litert_quantizer.algorithms.utils import common_utils

ALGORITHM_KEY = "MSE"

# Coefficients from offline numeric analysis.
_MSE_QUANT_MULS = {
    8: 0.05408,
    4: 0.37755,
}


def get_tensor_quant_params(
    op_info: qtyping.OpInfo,
    tensor_quant_config: qtyping.TensorQuantizationConfig,
    tensor_content: Optional[np.ndarray] = None,
    tensor_qsv: Optional[dict[str, Any]] = None,
) -> qtyping.UniformQuantParams:
  """Returns the quantization parameters for a tensor.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    tensor_quant_config: The quantization config for the tensor.
    tensor_content: The content of the tensor. When None, it means the tensor is
      not a weight tensor (e.g. static quantization) so we fallback to using
      naive_min_max_quantize.
    tensor_qsv: A dictionary containing the min/max of the tensor.

  Raises:
    ValueError: If the blockwise quantization is requested.
    ValueError: If the asymmetric quantization is requested.
    ValueError: `tensor_qsv` must contain min/max values, or `tensor_content`
      must be provided so that they can be inferred.
  """
  if uniform_quantize_tensor.is_blockwise(tensor_quant_config.granularity):
    raise ValueError(
        "Blockwise quantization is not supported for MSE quantization."
    )

  # Fallback to naive_min_max_quantize.py for non-weight tensors.
  if tensor_content is None:
    return naive_min_max_quantize.get_tensor_quant_params(
        op_info, tensor_quant_config, tensor_content, tensor_qsv
    )

  if not tensor_quant_config.symmetric:
    raise ValueError(
        f"Unsupported symmetry: {tensor_quant_config.symmetric}. MSE"
        " supports symmetric quantization only for now."
    )

  # Use tensor_content if QSV lacks min/max (e.g., including GPTQ case).
  if not tensor_qsv or "min" not in tensor_qsv:
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
    tensor_min_max = tensor_qsv

  if "min" not in tensor_min_max or "max" not in tensor_min_max:
    raise ValueError(
        "min and max must be provided to produce tensor quantization"
        " parameters. Check if the correct calibration results are passed into"
        " the ParamsGenerator."
    )

  quantized_dim = common_utils.get_weight_quantized_dim(
      op_info, tensor_content, tensor_quant_config.granularity
  )

  reshaped_data = tensor_content
  reduce_dims = common_utils.get_reduce_dims(
      quantized_dim, tensor_content.shape
  )

  multiplier = _MSE_QUANT_MULS[tensor_quant_config.num_bits]
  scale = multiplier * np.sqrt(
      np.mean(reshaped_data**2, axis=reduce_dims, keepdims=True)
  )
  zp = np.zeros_like(scale, dtype=np.int32)

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

  quantized_vars = uniform_quantize_tensor.uniform_quantize(
      tensor_content,
      quant_params,
      uniform_quantize_tensor.is_blockwise(tensor_quant_config.granularity),
  )

  return dataclasses.replace(quant_params, quantized_data=quantized_vars)
