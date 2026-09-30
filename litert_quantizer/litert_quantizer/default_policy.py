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

"""Default quantization policy."""

import collections
import copy
import json
from typing import Any, Union
from ai_edge_litert.tools import flatbuffer_utils
from litert_quantizer import qtyping

_TFLOpName = qtyping.TFLOperationName
_OpQuantizationConfig = qtyping.OpQuantizationConfig
_TensorQuantizationConfig = qtyping.TensorQuantizationConfig
_ComputePrecision = qtyping.ComputePrecision
_Granularity = qtyping.QuantGranularity
_INT = qtyping.TensorDataType.INT

# Default config check policy in JSON format. It has two keys: "configs" and
# "ops_per_config". "configs" is a dictionary mapping from config name to config
# content. "ops_per_config" is a dictionary mapping from config name to a list
# of op names.
DEFAULT_JSON_POLICY = """
{
  "configs": {
    "dynamic_wi8_afp32": {
      "weight_tensor_config": {
        "num_bits": 8,
        "symmetric": [true],
        "granularity": ["CHANNELWISE", "TENSORWISE"],
        "dtype": "INT"
      },
      "explicit_dequantize": false,
      "compute_precision": "INTEGER"
    },
    "dynamic_wi4_afp32": {
      "weight_tensor_config": {
        "num_bits": 4,
        "symmetric": [true],
        "granularity": ["CHANNELWISE", "TENSORWISE"],
        "dtype": "INT"
      },
      "explicit_dequantize": false,
      "compute_precision": "INTEGER"
    },
    "dynamic_wi4_afp32_blockwise": {
      "weight_tensor_config": {
        "num_bits": 4,
        "symmetric": [true],
        "granularity": ["BLOCKWISE_32", "BLOCKWISE_64", "BLOCKWISE_128", "BLOCKWISE_256"],
        "dtype": "INT"
      },
      "explicit_dequantize": false,
      "compute_precision": "INTEGER"
    },
    "dynamic_wi2_afp32": {
      "weight_tensor_config": {
        "num_bits": 2,
        "symmetric": [true],
        "granularity": ["CHANNELWISE", "TENSORWISE"],
        "dtype": "INT"
      },
      "explicit_dequantize": false,
      "compute_precision": "INTEGER"
    },
    "dynamic_wi2_afp32_blockwise": {
      "weight_tensor_config": {
        "num_bits": 2,
        "symmetric": [true],
        "granularity": ["BLOCKWISE_32", "BLOCKWISE_64", "BLOCKWISE_128", "BLOCKWISE_256"],
        "dtype": "INT"
      },
      "explicit_dequantize": false,
      "compute_precision": "INTEGER"
    },
    "static_wi8_ai16": {
      "activation_tensor_config": {
        "num_bits": 16,
        "symmetric": [true],
        "granularity": ["TENSORWISE"],
        "dtype": "INT"
      },
      "weight_tensor_config": {
        "num_bits": 8,
        "symmetric": [true],
        "granularity": ["CHANNELWISE", "TENSORWISE"],
        "dtype": "INT"
      },
      "explicit_dequantize": false,
      "compute_precision": "INTEGER"
    },
    "static_wi4_ai16": {
      "activation_tensor_config": {
        "num_bits": 16,
        "symmetric": [true],
        "granularity": ["TENSORWISE"],
        "dtype": "INT"
      },
      "weight_tensor_config": {
        "num_bits": 4,
        "symmetric": [true],
        "granularity": ["CHANNELWISE", "TENSORWISE"],
        "dtype": "INT"
      },
      "explicit_dequantize": false,
      "compute_precision": "INTEGER"
    },
    "static_wi8_ai8": {
      "activation_tensor_config": {
        "num_bits": 8,
        "symmetric": [true, false],
        "granularity": ["TENSORWISE"],
        "dtype": "INT"
      },
      "weight_tensor_config": {
        "num_bits": 8,
        "symmetric": [true],
        "granularity": ["CHANNELWISE", "TENSORWISE"],
        "dtype": "INT"
      },
      "explicit_dequantize": false,
      "compute_precision": "INTEGER"
    },
    "static_wi8_au8": {
      "activation_tensor_config": {
        "num_bits": 8,
        "symmetric": [false],
        "granularity": ["TENSORWISE"],
        "dtype": "UINT"
      },
      "weight_tensor_config": {
        "num_bits": 8,
        "symmetric": [true],
        "granularity": ["CHANNELWISE", "TENSORWISE"],
        "dtype": "INT"
      },
      "explicit_dequantize": false,
      "compute_precision": "INTEGER"
    },
    "static_wi4_ai8": {
      "activation_tensor_config": {
        "num_bits": 8,
        "symmetric": [true, false],
        "granularity": ["TENSORWISE"],
        "dtype": "INT"
      },
      "weight_tensor_config": {
        "num_bits": 4,
        "symmetric": [true],
        "granularity": ["CHANNELWISE", "TENSORWISE"],
        "dtype": "INT"
      },
      "explicit_dequantize": false,
      "compute_precision": "INTEGER"
    },
    "weightonly_wi8_afp32": {
      "weight_tensor_config": {
        "num_bits": 8,
        "symmetric": [true, false],
        "granularity": ["CHANNELWISE", "TENSORWISE"],
        "dtype": "INT"
      },
      "explicit_dequantize": true,
      "compute_precision": "FLOAT"
    },
    "weightonly_wi4_afp32": {
      "weight_tensor_config": {
        "num_bits": 4,
        "symmetric": [true, false],
        "granularity": ["CHANNELWISE", "TENSORWISE"],
        "dtype": "INT"
      },
      "explicit_dequantize": true,
      "compute_precision": "FLOAT"
    }
  },
  "ops_per_config": {
    "static_wi8_ai16": [
      "ADD",
      "AVERAGE_POOL_2D",
      "BATCH_MATMUL",
      "CONCATENATION",
      "CONV_2D",
      "CONV_2D_TRANSPOSE",
      "DEPTHWISE_CONV_2D",
      "FULLY_CONNECTED",
      "GELU",
      "LOGISTIC",
      "MEAN",
      "MUL",
      "RESHAPE",
      "RSQRT",
      "SOFTMAX",
      "SPLIT",
      "STRIDED_SLICE",
      "SUB",
      "TANH",
      "TRANSPOSE",
      "INPUT",
      "OUTPUT",
      "SLICE",
      "SUM",
      "SELECT",
      "SELECT_V2",
      "DYNAMIC_UPDATE_SLICE",
      "SELECT_V2",
      "STABLEHLO_COMPOSITE",
      "PAD",
      "MAX_POOL_2D",
      "RESIZE_BILINEAR",
      "RESIZE_NEAREST_NEIGHBOR",
      "GATHER_ND",
      "PACK",
      "UNPACK",
      "DIV",
      "BROADCAST_TO",
      "SQRT",
      "GATHER",
      "MAXIMUM",
      "PADV2",
      "REDUCE_MIN",
      "EQUAL",
      "NOT_EQUAL",
      "MIRROR_PAD",
      "RELU"
    ],
    "static_wi8_ai8": [
      "ADD",
      "AVERAGE_POOL_2D",
      "BATCH_MATMUL",
      "CONCATENATION",
      "FULLY_CONNECTED",
      "CONV_2D",
      "CONV_2D_TRANSPOSE",
      "DEPTHWISE_CONV_2D",
      "GELU",
      "LOGISTIC",
      "MEAN",
      "MUL",
      "RESHAPE",
      "RSQRT",
      "SOFTMAX",
      "SPLIT",
      "STRIDED_SLICE",
      "SUB",
      "TANH",
      "TRANSPOSE",
      "INPUT",
      "OUTPUT",
      "SLICE",
      "SUM",
      "SELECT",
      "SELECT_V2",
      "DYNAMIC_UPDATE_SLICE",
      "SELECT_V2",
      "STABLEHLO_COMPOSITE",
      "PAD",
      "SQUARED_DIFFERENCE",
      "MAX_POOL_2D",
      "RESIZE_BILINEAR",
      "RESIZE_NEAREST_NEIGHBOR",
      "GATHER_ND",
      "PACK",
      "UNPACK",
      "DIV",
      "BROADCAST_TO",
      "SQRT",
      "GATHER",
      "HARD_SWISH",
      "MAXIMUM",
      "PADV2",
      "REDUCE_MIN",
      "EQUAL",
      "NOT_EQUAL",
      "MIRROR_PAD",
      "SPACE_TO_DEPTH",
      "RELU"
    ],
    "static_wi4_ai8": ["FULLY_CONNECTED", "CONV_2D", "INPUT", "OUTPUT"],
    "static_wi8_au8": ["INPUT", "OUTPUT"],
    "static_wi4_ai16": ["FULLY_CONNECTED", "CONV_2D", "INPUT", "OUTPUT"],
    "dynamic_wi8_afp32": [
      "BATCH_MATMUL",
      "CONV_2D",
      "CONV_2D_TRANSPOSE",
      "DEPTHWISE_CONV_2D",
      "EMBEDDING_LOOKUP",
      "FULLY_CONNECTED"
    ],
    "dynamic_wi4_afp32": ["FULLY_CONNECTED", "EMBEDDING_LOOKUP", "CONV_2D"],
    "dynamic_wi4_afp32_blockwise": ["EMBEDDING_LOOKUP", "FULLY_CONNECTED"],
    "dynamic_wi2_afp32": ["FULLY_CONNECTED", "EMBEDDING_LOOKUP", "CONV_2D"],
    "dynamic_wi2_afp32_blockwise": ["FULLY_CONNECTED"],
    "weightonly_wi8_afp32": [
      "BATCH_MATMUL",
      "CONV_2D",
      "CONV_2D_TRANSPOSE",
      "DEPTHWISE_CONV_2D",
      "EMBEDDING_LOOKUP",
      "FULLY_CONNECTED"
    ],
    "weightonly_wi4_afp32": ["BATCH_MATMUL", "FULLY_CONNECTED", "EMBEDDING_LOOKUP", "CONV_2D"]
  }
}
"""
QUANTIZABLE_COMPOSITES = [
    "od" + "ml.npu_call",
    "od" + "ml.rms_norm",
    "od" + "ml.l2_norm",
]


def _unroll_json_config(
    json_config: dict[str, Any],
) -> list[_OpQuantizationConfig]:
  """Unrolls the config.

  Args:
    json_config: JSON config to be unrolled.

  Returns:
    quant_configs: A list of unrolled configs.
  """

  # Unroll activation configs first.
  activation_configs = []
  if "activation_tensor_config" in json_config:
    for symmetric in json_config["activation_tensor_config"]["symmetric"]:
      for granularity in json_config["activation_tensor_config"]["granularity"]:
        tensor_config = {
            "num_bits": json_config["activation_tensor_config"]["num_bits"],
            "symmetric": symmetric,
            "granularity": granularity,
            "dtype": json_config["activation_tensor_config"]["dtype"],
        }
        activation_configs.append(
            qtyping.TensorQuantizationConfig.from_dict(tensor_config)
        )

  # Then unroll weight configs and turn them into quantization configs.
  quant_configs = []
  weight_configs = []
  for symmetric in json_config["weight_tensor_config"]["symmetric"]:
    for granularity in json_config["weight_tensor_config"]["granularity"]:
      tensor_config = {
          "num_bits": json_config["weight_tensor_config"]["num_bits"],
          "symmetric": symmetric,
          "granularity": granularity,
          "dtype": json_config["weight_tensor_config"]["dtype"],
      }
      weight_configs.append(
          qtyping.TensorQuantizationConfig.from_dict(tensor_config)
      )

      if activation_configs:
        for activation_config in activation_configs:
          quant_configs.append(
              qtyping.OpQuantizationConfig(
                  activation_tensor_config=activation_config,
                  weight_tensor_config=qtyping.TensorQuantizationConfig.from_dict(
                      tensor_config
                  ),
                  compute_precision=json_config["compute_precision"],
                  explicit_dequantize=json_config["explicit_dequantize"],
              )
          )
      else:
        for weight_config in weight_configs:
          quant_configs.append(
              qtyping.OpQuantizationConfig(
                  weight_tensor_config=weight_config,
                  compute_precision=json_config["compute_precision"],
                  explicit_dequantize=json_config["explicit_dequantize"],
              )
          )

  return quant_configs


# TODO: b/401024954 - Have a better way to specify recipes based on op options.
def is_non_quantizable_composite_op(
    op: Union[qtyping.Operator, qtyping.OperatorT],
) -> bool:
  """Checks if the operator is a non-quantizable composite op.

  We may want to quantize an op only when its has certain options.
  Policies/recipes
  are not aware of op options, so it is checked here.

  Args:
    op: The operator to check.

  Returns:
    True if the operator is conditionally unquantized, False otherwise.
  """
  if opts := flatbuffer_utils.get_options_as(
      op, qtyping.StableHLOCompositeOptionsT
  ):
    name = opts.name.decode("utf-8")
    if name not in QUANTIZABLE_COMPOSITES:
      return True

  return False


def update_default_config_policy(raw_json_policy: str):
  """Updates the default config check policy."""
  json_policy_content = json.loads(raw_json_policy)

  # Unroll the json config and add the configs to the policy.
  policy = collections.OrderedDict()
  for json_policy_config in json_policy_content["ops_per_config"]:
    unrolled_configs = _unroll_json_config(
        json_policy_content["configs"][json_policy_config]
    )

    for op in json_policy_content["ops_per_config"][json_policy_config]:
      op_name = _TFLOpName(op)
      quant_configs = copy.deepcopy(unrolled_configs)
      if op in policy.keys():
        quant_configs += policy[op_name]
      policy[op_name] = quant_configs

  return policy


DEFAULT_CONFIG_CHECK_POLICY = update_default_config_policy(DEFAULT_JSON_POLICY)
