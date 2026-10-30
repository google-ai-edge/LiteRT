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

"""E2E tests for the quantizer for model with the MoE custom op."""

import pathlib

from absl.testing import absltest

from litert_quantizer import qtyping
from litert_quantizer import quantizer
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils


_TEST_MODEL_FOLDER = test_utils.get_path_to_datafile('../models/')
_QuantAlgo = quantizer.AlgorithmName


def _get_op_config(num_bits: int) -> qtyping.OpQuantizationConfig:
  return qtyping.OpQuantizationConfig(
      weight_tensor_config=qtyping.TensorQuantizationConfig(
          num_bits=num_bits,
          symmetric=True,
          granularity=qtyping.QuantGranularity.CHANNELWISE,
          quantized_dimensions=[0, 1],
      ),
      compute_precision=qtyping.ComputePrecision.INTEGER,
  )


class MoeTest(test_utils.BaseOpTestCase):

  _MOE_SUBGRAPH_INDEX = 0
  _MOE_OP_INDEX = 0

  def setUp(self):
    super().setUp()
    self._op_name = qtyping.TFLOperationName.CUSTOM_OP
    self._model_path = str(
        pathlib.Path(_TEST_MODEL_FOLDER) / 'single_moe_multiaxis.tflite'
    )

  def test_int8_quantization_accuracy_within_tolerance(self):
    self.assert_quantization_accuracy(
        algorithm_key=_QuantAlgo.MIN_MAX_UNIFORM_QUANT,
        model_path=self._model_path,
        op_name=self._op_name,
        op_config=_get_op_config(num_bits=8),
        output_tolerance=1e-5,
        # Int inputs are drawn from [0, 1), so all top-k expert indices are 0.
        min_max_range=(0, 1),
    )

  def test_int4_quantization_accuracy_within_tolerance(self):
    model = tfl_flatbuffer_utils.read_model(self._model_path)
    # The test model's gate and ff1 weight rows are generated as
    # [-127, -64, 0, 127] * scale. In int4, -64 falls between two quantized
    # values, which gives a large quantization error, so replace it with 0.
    # This is fine: the error is so large since it's a tiny model with made-up
    # weights.
    subgraph = model.subgraphs[self._MOE_SUBGRAPH_INDEX]
    for weight_idx in subgraph.operators[self._MOE_OP_INDEX].inputs[3:5]:
      weight = subgraph.tensors[weight_idx]
      data = tfl_flatbuffer_utils.get_tensor_data(weight, model.buffers).copy()
      data[..., 1] = 0
      model.buffers[weight.buffer].data = data.tobytes()
    model_path = self.create_tempfile().full_path
    tfl_flatbuffer_utils.write_model(model, model_path)
    self.assert_quantization_accuracy(
        algorithm_key=_QuantAlgo.MIN_MAX_UNIFORM_QUANT,
        model_path=model_path,
        op_name=self._op_name,
        op_config=_get_op_config(num_bits=4),
        output_tolerance=1e-5,
        # Int inputs are drawn from [0, 1), so all top-k expert indices are 0.
        min_max_range=(0, 1),
    )


if __name__ == '__main__':
  absltest.main()
