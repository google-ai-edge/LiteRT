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

"""Quantizer Algorithm Manager Interface."""

import enum
import functools
import immutabledict
from litert_quantizer import algorithm_manager_api
from litert_quantizer import default_policy
from litert_quantizer import qtyping
from litert_quantizer.algorithms.nonlinear_quantize import float_casting
from litert_quantizer.algorithms.uniform_quantize import common_quantize
from litert_quantizer.algorithms.uniform_quantize import dequantized_weight_recovery
from litert_quantizer.algorithms.uniform_quantize import gptq
from litert_quantizer.algorithms.uniform_quantize import hadamard_rotation
from litert_quantizer.algorithms.uniform_quantize import mse
from litert_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from litert_quantizer.algorithms.uniform_quantize import octav
from litert_quantizer.algorithms.uniform_quantize import oscar
from litert_quantizer.utils import qsv_utils


immutabledict = immutabledict.immutabledict


# TODO: b/399775701 - Clean up this file.

_TFLOpName = qtyping.TFLOperationName

_alg_manager_instance = algorithm_manager_api.AlgorithmManagerApi()

# Expose instance functions.
get_quantization_func = _alg_manager_instance.get_quantization_func
get_supported_ops = _alg_manager_instance.get_supported_ops
get_update_qsv_func = _alg_manager_instance.get_update_qsv_func
get_init_qsv_func = _alg_manager_instance.get_init_qsv_func
register_op_quant_config_validation_func = (
    _alg_manager_instance.register_op_quant_config_validation_func
)
register_config_check_policy_func = (
    _alg_manager_instance.register_config_check_policy
)
register_quantized_op = _alg_manager_instance.register_quantized_op
is_op_registered = _alg_manager_instance.is_op_registered
is_algorithm_registered = _alg_manager_instance.is_algorithm_registered
check_op_quantization_config = (
    _alg_manager_instance.check_op_quantization_config
)


# Quantization algorithms.
class AlgorithmName(str, enum.Enum):
  NO_QUANTIZE = "no_quantize"
  MIN_MAX_UNIFORM_QUANT = naive_min_max_quantize.ALGORITHM_KEY
  FLOAT_CASTING = float_casting.ALGORITHM_KEY
  DEQUANTIZED_WEIGHT_RECOVERY = dequantized_weight_recovery.ALGORITHM_KEY
  OCTAV = octav.ALGORITHM_KEY
  HADAMARD_ROTATION = hadamard_rotation.CUSTOM_OP_ALGORITHM_KEY
  DECOMPOSED_HADAMARD_ROTATION = hadamard_rotation.DECOMPOSED_ALGORITHM_KEY
  MSE = mse.ALGORITHM_KEY
  GPTQ = gptq.ALGORITHM_KEY
  OSCAR = oscar.ALGORITHM_KEY


### MIN/MAX_UNIFORM_QUANT ###

# Register MIN_MAX_UNIFORM_QUANT algorithm.
register_op_quant_config_validation_func(
    AlgorithmName.MIN_MAX_UNIFORM_QUANT,
    common_quantize.check_op_quantization_config,
)

# Register a config check policy for MIN_MAX_UNIFORM_QUANT algorithm.
register_config_check_policy_func(
    AlgorithmName.MIN_MAX_UNIFORM_QUANT,
    default_policy.DEFAULT_CONFIG_CHECK_POLICY,
)

MIN_MAX_OP_NAME_MATERIALIZE_FUNC_DICT = {
    _TFLOpName.INPUT: common_quantize.materialize_input,
    _TFLOpName.OUTPUT: common_quantize.materialize_output,
    _TFLOpName.FULLY_CONNECTED: common_quantize.materialize_fc_conv,
    _TFLOpName.BATCH_MATMUL: common_quantize.materialize_batch_matmul,
    _TFLOpName.CONV_2D: common_quantize.materialize_fc_conv,
    _TFLOpName.DEPTHWISE_CONV_2D: common_quantize.materialize_fc_conv,
    _TFLOpName.CONV_2D_TRANSPOSE: common_quantize.materialize_conv2d_transpose,
    _TFLOpName.RESHAPE: common_quantize.materialize_reshape,
    _TFLOpName.AVERAGE_POOL_2D: common_quantize.materialize_average_pool_2d,
    _TFLOpName.EMBEDDING_LOOKUP: common_quantize.materialize_embedding_lookup,
    _TFLOpName.SOFTMAX: common_quantize.materialize_softmax_and_logistic,
    _TFLOpName.TANH: common_quantize.materialize_tanh,
    _TFLOpName.TRANSPOSE: common_quantize.materialize_transpose,
    _TFLOpName.GELU: common_quantize.materialize_gelu,
    _TFLOpName.ADD: common_quantize.materialize_add,
    _TFLOpName.SUB: common_quantize.materialize_sub,
    _TFLOpName.MUL: common_quantize.materialize_mul,
    _TFLOpName.MEAN: common_quantize.materialize_mean,
    _TFLOpName.RSQRT: common_quantize.materialize_rsqrt,
    _TFLOpName.CONCATENATION: common_quantize.materialize_concatenation,
    _TFLOpName.STRIDED_SLICE: common_quantize.materialize_strided_slice,
    _TFLOpName.SPLIT: common_quantize.materialize_split,
    _TFLOpName.LOGISTIC: common_quantize.materialize_softmax_and_logistic,
    _TFLOpName.SLICE: common_quantize.materialize_slice,
    _TFLOpName.SUM: common_quantize.materialize_sum,
    _TFLOpName.SELECT: common_quantize.materialize_select,
    _TFLOpName.SELECT_V2: common_quantize.materialize_select_v2,
    _TFLOpName.DYNAMIC_UPDATE_SLICE: (
        common_quantize.materialize_dynamic_update_slice
    ),
    _TFLOpName.STABLEHLO_COMPOSITE: common_quantize.materialize_composite,
    _TFLOpName.PAD: common_quantize.materialize_pad,
    _TFLOpName.SQUARED_DIFFERENCE: (
        common_quantize.materialize_squared_difference
    ),
    _TFLOpName.MAX_POOL_2D: common_quantize.materialize_max_pool_2d,
    _TFLOpName.RESIZE_BILINEAR: common_quantize.materialize_resize_bilinear,
    _TFLOpName.RESIZE_NEAREST_NEIGHBOR: (
        common_quantize.materialize_resize_nearest_neighbor
    ),
    _TFLOpName.GATHER_ND: common_quantize.materialize_gather_nd,
    _TFLOpName.PACK: common_quantize.materialize_pack,
    _TFLOpName.UNPACK: common_quantize.materialize_unpack,
    _TFLOpName.DIV: common_quantize.materialize_div,
    _TFLOpName.BROADCAST_TO: common_quantize.materialize_broadcast_to,
    _TFLOpName.SQRT: common_quantize.materialize_sqrt,
    _TFLOpName.GATHER: common_quantize.materialize_gather,
    _TFLOpName.HARD_SWISH: common_quantize.materialize_hard_swish,
    _TFLOpName.MAXIMUM: common_quantize.materialize_maximum,
    _TFLOpName.PADV2: common_quantize.materialize_padv2,
    _TFLOpName.REDUCE_MIN: common_quantize.materialize_reduce_min,
    _TFLOpName.EQUAL: common_quantize.materialize_equal,
    _TFLOpName.NOT_EQUAL: common_quantize.materialize_not_equal,
    _TFLOpName.MIRROR_PAD: common_quantize.materialize_mirror_pad,
    _TFLOpName.SPACE_TO_DEPTH: common_quantize.materialize_space_to_depth,
    _TFLOpName.RELU: common_quantize.materialize_relu,
}
for op_name, materialize_func in MIN_MAX_OP_NAME_MATERIALIZE_FUNC_DICT.items():
  register_quantized_op(
      AlgorithmName.MIN_MAX_UNIFORM_QUANT,
      op_name,
      naive_min_max_quantize.init_qsvs,
      calibration_func=naive_min_max_quantize.min_max_calibrate,
      # Most of the materialize op functions are common for all algorithms
      # except for the function to get scale and zero point, i.e.,
      # get_tensor_quant_params. So we use functools.partial here to pass in the
      # common utility function and thealgorithm-specific function.
      materialize_func=functools.partial(
          materialize_func,
          naive_min_max_quantize.get_tensor_quant_params,
      ),
  )

### FLOAT_CASTING ###
register_op_quant_config_validation_func(
    AlgorithmName.FLOAT_CASTING,
    float_casting.check_op_quantization_config,
)

# Register a config check policy for FLOAT_CASTING algorithm.
# TODO: b/353780772 - Replace an empty policy for FLOAT_CASTING algorithm.
register_config_check_policy_func(
    AlgorithmName.FLOAT_CASTING, qtyping.ConfigCheckPolicyDict()
)

for op_name, materialize_func in zip(
    (
        _TFLOpName.FULLY_CONNECTED,
        _TFLOpName.CONV_2D,
        _TFLOpName.DEPTHWISE_CONV_2D,
        _TFLOpName.CONV_2D_TRANSPOSE,
        _TFLOpName.EMBEDDING_LOOKUP,
    ),
    (
        float_casting.materialize_fc_conv,
        float_casting.materialize_fc_conv,
        float_casting.materialize_fc_conv,
        float_casting.materialize_conv2d_transpose,
        float_casting.materialize_embedding_lookup,
    ),
):
  register_quantized_op(
      AlgorithmName.FLOAT_CASTING,
      op_name,
      float_casting.init_qsvs,  # pyrefly: ignore[bad-argument-type]
      calibration_func=float_casting.calibrate,  # pyrefly: ignore[bad-argument-type]
      materialize_func=materialize_func,  # pyrefly: ignore[bad-argument-type]
  )

### DEQUANTIZED_WEIGHT_RECOVERY ###
register_op_quant_config_validation_func(
    AlgorithmName.DEQUANTIZED_WEIGHT_RECOVERY,
    common_quantize.check_op_quantization_config,
)

register_config_check_policy_func(
    AlgorithmName.DEQUANTIZED_WEIGHT_RECOVERY,
    default_policy.DEFAULT_CONFIG_CHECK_POLICY,
)

DEQUANTIZED_WEIGHT_RECOVERY_OP_NAME_MATERIALIZE_FUNC_DICT = {
    _TFLOpName.FULLY_CONNECTED: common_quantize.materialize_fc_conv,
    _TFLOpName.CONV_2D: common_quantize.materialize_fc_conv,
    _TFLOpName.EMBEDDING_LOOKUP: common_quantize.materialize_embedding_lookup,
}

for (
    op_name,
    materialize_func,
) in DEQUANTIZED_WEIGHT_RECOVERY_OP_NAME_MATERIALIZE_FUNC_DICT.items():
  register_quantized_op(
      algorithm_key=AlgorithmName.DEQUANTIZED_WEIGHT_RECOVERY,
      tfl_op_name=op_name,
      init_qsv_func=dequantized_weight_recovery.init_qsvs,  # pyrefly: ignore[bad-argument-type]
      calibration_func=dequantized_weight_recovery.calibrate,  # pyrefly: ignore[bad-argument-type]
      # Most of the materialize op functions are common for all algorithms
      # except for the function to get scale and zero point, i.e.,
      # get_tensor_quant_params. So we use functools.partial here to pass in the
      # common utility function and the algorithm-specific function.
      materialize_func=functools.partial(
          materialize_func,
          dequantized_weight_recovery.get_tensor_quant_params,
      ),
  )


# Register OCTAV algorithm.
register_op_quant_config_validation_func(
    AlgorithmName.OCTAV,
    common_quantize.check_op_quantization_config,
)

# Register a config check policy for OCTAV algorithm.
register_config_check_policy_func(
    AlgorithmName.OCTAV,
    default_policy.DEFAULT_CONFIG_CHECK_POLICY,
)

_OCTAV_OP_NAME_MATERIALIZE_FUNC_DICT = immutabledict({
    _TFLOpName.INPUT: common_quantize.materialize_input,
    _TFLOpName.OUTPUT: common_quantize.materialize_output,
    _TFLOpName.FULLY_CONNECTED: common_quantize.materialize_fc_conv,
    _TFLOpName.BATCH_MATMUL: common_quantize.materialize_batch_matmul,
    _TFLOpName.CONV_2D: common_quantize.materialize_fc_conv,
    _TFLOpName.DEPTHWISE_CONV_2D: common_quantize.materialize_fc_conv,
    _TFLOpName.CONV_2D_TRANSPOSE: common_quantize.materialize_conv2d_transpose,
    _TFLOpName.RESHAPE: common_quantize.materialize_reshape,
    _TFLOpName.AVERAGE_POOL_2D: common_quantize.materialize_average_pool_2d,
    _TFLOpName.EMBEDDING_LOOKUP: common_quantize.materialize_embedding_lookup,
    _TFLOpName.SOFTMAX: common_quantize.materialize_softmax_and_logistic,
    _TFLOpName.TANH: common_quantize.materialize_tanh,
    _TFLOpName.TRANSPOSE: common_quantize.materialize_transpose,
    _TFLOpName.GELU: common_quantize.materialize_gelu,
    _TFLOpName.ADD: common_quantize.materialize_add,
    _TFLOpName.SUB: common_quantize.materialize_sub,
    _TFLOpName.MUL: common_quantize.materialize_mul,
    _TFLOpName.MEAN: common_quantize.materialize_mean,
    _TFLOpName.RSQRT: common_quantize.materialize_rsqrt,
    _TFLOpName.CONCATENATION: common_quantize.materialize_concatenation,
    _TFLOpName.STRIDED_SLICE: common_quantize.materialize_strided_slice,
    _TFLOpName.SPLIT: common_quantize.materialize_split,
    _TFLOpName.LOGISTIC: common_quantize.materialize_softmax_and_logistic,
    _TFLOpName.SLICE: common_quantize.materialize_slice,
    _TFLOpName.SUM: common_quantize.materialize_sum,
    _TFLOpName.SELECT: common_quantize.materialize_select,
    _TFLOpName.SELECT_V2: common_quantize.materialize_select_v2,
    _TFLOpName.DYNAMIC_UPDATE_SLICE: (
        common_quantize.materialize_dynamic_update_slice
    ),
    _TFLOpName.STABLEHLO_COMPOSITE: common_quantize.materialize_composite,
    _TFLOpName.PAD: common_quantize.materialize_pad,
    _TFLOpName.SQUARED_DIFFERENCE: (
        common_quantize.materialize_squared_difference
    ),
    _TFLOpName.MAX_POOL_2D: common_quantize.materialize_max_pool_2d,
    _TFLOpName.RESIZE_BILINEAR: common_quantize.materialize_resize_bilinear,
    _TFLOpName.RESIZE_NEAREST_NEIGHBOR: (
        common_quantize.materialize_resize_nearest_neighbor
    ),
    _TFLOpName.GATHER_ND: common_quantize.materialize_gather_nd,
    _TFLOpName.PACK: common_quantize.materialize_pack,
    _TFLOpName.UNPACK: common_quantize.materialize_unpack,
    _TFLOpName.DIV: common_quantize.materialize_div,
    _TFLOpName.BROADCAST_TO: common_quantize.materialize_broadcast_to,
    _TFLOpName.SQRT: common_quantize.materialize_sqrt,
    _TFLOpName.GATHER: common_quantize.materialize_gather,
    _TFLOpName.HARD_SWISH: common_quantize.materialize_hard_swish,
    _TFLOpName.MAXIMUM: common_quantize.materialize_maximum,
    _TFLOpName.PADV2: common_quantize.materialize_padv2,
    _TFLOpName.REDUCE_MIN: common_quantize.materialize_reduce_min,
    _TFLOpName.EQUAL: common_quantize.materialize_equal,
    _TFLOpName.NOT_EQUAL: common_quantize.materialize_not_equal,
    _TFLOpName.MIRROR_PAD: common_quantize.materialize_mirror_pad,
    _TFLOpName.SPACE_TO_DEPTH: common_quantize.materialize_space_to_depth,
    _TFLOpName.RELU: common_quantize.materialize_relu,
})

for op_name, materialize_func in _OCTAV_OP_NAME_MATERIALIZE_FUNC_DICT.items():
  register_quantized_op(
      AlgorithmName.OCTAV,
      op_name,
      naive_min_max_quantize.init_qsvs,
      calibration_func=naive_min_max_quantize.min_max_calibrate,
      materialize_func=functools.partial(
          materialize_func,
          octav.get_tensor_quant_params,
      ),
  )

# Register the Hadamard Rotation algorithm.
register_op_quant_config_validation_func(
    AlgorithmName.HADAMARD_ROTATION,
    common_quantize.check_op_quantization_config,
)

# Register a config check policy for the Hadamard Rotation algorithm.
register_config_check_policy_func(
    AlgorithmName.HADAMARD_ROTATION,
    default_policy.DEFAULT_CONFIG_CHECK_POLICY,
)

# Register specialized hadamard rotation materialize functions.
_HADAMARD_ROTATION_OP_NAME_MATERIALIZE_FUNC_DICT = immutabledict({
    _TFLOpName.FULLY_CONNECTED: (
        hadamard_rotation.materialize_fully_connected_custom_op
    ),
    _TFLOpName.EMBEDDING_LOOKUP: (
        hadamard_rotation.materialize_embedding_lookup_custom_op
    ),
})
for (
    op_name,
    materialize_func,
) in _HADAMARD_ROTATION_OP_NAME_MATERIALIZE_FUNC_DICT.items():
  register_quantized_op(
      AlgorithmName.HADAMARD_ROTATION,
      op_name,
      naive_min_max_quantize.init_qsvs,
      calibration_func=naive_min_max_quantize.min_max_calibrate,
      materialize_func=materialize_func,  # pyrefly: ignore[bad-argument-type]
  )

register_op_quant_config_validation_func(
    AlgorithmName.DECOMPOSED_HADAMARD_ROTATION,
    common_quantize.check_op_quantization_config,
)

register_config_check_policy_func(
    AlgorithmName.DECOMPOSED_HADAMARD_ROTATION,
    default_policy.DEFAULT_CONFIG_CHECK_POLICY,
)

_DECOMPOSED_HADAMARD_ROTATION_OP_NAME_MATERIALIZE_FUNC_DICT = immutabledict({
    _TFLOpName.FULLY_CONNECTED: (
        hadamard_rotation.materialize_fully_connected_decomposed
    ),
    _TFLOpName.EMBEDDING_LOOKUP: (
        hadamard_rotation.materialize_embedding_lookup_decomposed
    ),
})
for (
    op_name,
    materialize_func,
) in _DECOMPOSED_HADAMARD_ROTATION_OP_NAME_MATERIALIZE_FUNC_DICT.items():
  register_quantized_op(
      AlgorithmName.DECOMPOSED_HADAMARD_ROTATION,
      op_name,
      naive_min_max_quantize.init_qsvs,
      calibration_func=naive_min_max_quantize.min_max_calibrate,
      materialize_func=materialize_func,  # pyrefly: ignore[bad-argument-type]
  )


# Register the MSE algorithm.
register_op_quant_config_validation_func(
    AlgorithmName.MSE,
    common_quantize.check_op_quantization_config,
)

# Register a config check policy for the MSE algorithm.
register_config_check_policy_func(
    AlgorithmName.MSE,
    default_policy.DEFAULT_CONFIG_CHECK_POLICY,
)

# Register specialized MSE materialize functions.
_MSE_OP_NAME_MATERIALIZE_FUNC_DICT = immutabledict({
    _TFLOpName.FULLY_CONNECTED: common_quantize.materialize_fc_conv,
    _TFLOpName.EMBEDDING_LOOKUP: common_quantize.materialize_embedding_lookup,
    _TFLOpName.CONV_2D: common_quantize.materialize_fc_conv,
    _TFLOpName.DEPTHWISE_CONV_2D: common_quantize.materialize_fc_conv,
    _TFLOpName.CONV_2D_TRANSPOSE: common_quantize.materialize_conv2d_transpose,
})
for (
    op_name,
    materialize_func,
) in _MSE_OP_NAME_MATERIALIZE_FUNC_DICT.items():
  register_quantized_op(
      AlgorithmName.MSE,
      op_name,
      naive_min_max_quantize.init_qsvs,
      calibration_func=naive_min_max_quantize.min_max_calibrate,
      materialize_func=functools.partial(
          materialize_func,
          mse.get_tensor_quant_params,
      ),
  )

# Register the GPTQ algorithm.
register_op_quant_config_validation_func(
    AlgorithmName.GPTQ,
    common_quantize.check_op_quantization_config,
)

# Register a config check policy for the GPTQ algorithm.
register_config_check_policy_func(
    AlgorithmName.GPTQ,
    default_policy.DEFAULT_CONFIG_CHECK_POLICY,
)

# Register specialized GPTQ materialize functions.
_GPTQ_OP_NAME_MATERIALIZE_FUNC_DICT = immutabledict({
    _TFLOpName.FULLY_CONNECTED: common_quantize.materialize_fc_conv,
})
for (
    op_name,
    materialize_func,
) in _GPTQ_OP_NAME_MATERIALIZE_FUNC_DICT.items():
  register_quantized_op(
      AlgorithmName.GPTQ,
      op_name,
      naive_min_max_quantize.init_qsvs,
      calibration_func=gptq.calibrate,  # pyrefly: ignore[bad-argument-type]
      materialize_func=functools.partial(
          materialize_func,
          gptq.get_tensor_quant_params,
      ),
      update_qsv_func=qsv_utils.gptq_and_moving_average_update,  # pyrefly: ignore[bad-argument-type]
  )

# Register the OSCAR algorithm.
register_op_quant_config_validation_func(
    AlgorithmName.OSCAR,
    common_quantize.check_op_quantization_config,
)

# Register a config check policy for the OSCAR algorithm.
register_config_check_policy_func(
    AlgorithmName.OSCAR,
    default_policy.DEFAULT_CONFIG_CHECK_POLICY,
)

# Register specialized OSCAR materialize functions.
_OSCAR_OP_NAME_MATERIALIZE_FUNC_DICT = immutabledict({
    _TFLOpName.FULLY_CONNECTED: oscar.materialize_fully_connected,
})
for (
    op_name,
    materialize_func,
) in _OSCAR_OP_NAME_MATERIALIZE_FUNC_DICT.items():
  register_quantized_op(
      AlgorithmName.OSCAR,
      op_name,
      naive_min_max_quantize.init_qsvs,
      calibration_func=oscar.calibrate,  # pyrefly: ignore[bad-argument-type]
      materialize_func=materialize_func,  # pyrefly: ignore[bad-argument-type]
      update_qsv_func=qsv_utils.oscar_and_moving_average_update,  # pyrefly: ignore[bad-argument-type]
  )
