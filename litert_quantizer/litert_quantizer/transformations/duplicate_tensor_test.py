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

import pathlib

from absl.testing import absltest
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.transformations import duplicate_tensor
from litert_quantizer.transformations import transformation_utils
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile('..')


class DuplicateTensorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / 'tests/models/weight_sharing_fcs.tflite'
    )
    self.model = tfl_flatbuffer_utils.read_model(model_path)

  def _get_transformation_input(
      self,
      subgraph_idx: int,
      tensor_idx: int,
      consumers: list[int],
  ) -> transformation_utils.TransformationInput:
    return transformation_utils.TransformationInput(
        tensor_id=tensor_idx,
        model=self.model,
        consumers=consumers,
        # Dummy params below.
        subgraph=self.model.subgraphs[subgraph_idx],
        producer=-1,
        quant_params=qtyping.UniformQuantParams(
            num_bits=8,
            quantized_dimension=None,
            scale=np.ones(1),
            zero_point=np.zeros(1),
        ),
    )

  def test_constant_tensor_is_correctly_duplicated(self):
    # Duplicate the FC weight tensor in the second subgraph for the first FC.
    subgraph_idx = 1
    fc1_op_idx = 0
    prev_weight_tensor_idx = 1
    subgraph = self.model.subgraphs[subgraph_idx]
    weight_idx_in_op_inputs = list(subgraph.operators[fc1_op_idx].inputs).index(
        prev_weight_tensor_idx
    )
    prev_num_tensors = len(subgraph.tensors)
    prev_buffer_id = subgraph.tensors[prev_weight_tensor_idx].buffer
    prev_num_buffers = len(self.model.buffers)
    transformation_input = self._get_transformation_input(
        subgraph_idx, prev_weight_tensor_idx, consumers=[fc1_op_idx]
    )
    transformation_info = duplicate_tensor.duplicate_tensor(
        transformation_input
    )
    self.assertEqual(transformation_info.op_id, 0)
    self.assertEqual(transformation_info.num_ops_added, 0)
    # Check that a new tensor and buffer were added.
    self.assertLen(subgraph.tensors, prev_num_tensors + 1)
    self.assertLen(self.model.buffers, prev_num_buffers + 1)
    # Check that the duplicated tensor is the last tensor in the subgraph.
    weight_tensor_idx = transformation_info.output_tensor_id
    self.assertEqual(weight_tensor_idx, len(subgraph.tensors) - 1)
    # Compare tensors.
    original_tensor = subgraph.tensors[prev_weight_tensor_idx]
    original_tensor_name = tfl_flatbuffer_utils.get_tensor_name(original_tensor)
    duplicated_tensor = subgraph.tensors[weight_tensor_idx]
    self.assertEqual(
        duplicated_tensor.name,
        f'{original_tensor_name}_duplicated_{weight_tensor_idx}',
    )
    self.assertEqual(duplicated_tensor.type, original_tensor.type)
    self.assertTrue(np.all(duplicated_tensor.shape == original_tensor.shape))
    # Check that the new buffer is used by the duplicated tensor.
    new_buffer_id = len(self.model.buffers) - 1
    self.assertEqual(duplicated_tensor.buffer, new_buffer_id)
    # Check that the new buffer has the same data as the original one.
    self.assertTrue(
        np.all(
            np.frombuffer(
                self.model.buffers[new_buffer_id].data,
                dtype=np.float32,
            )
            == np.frombuffer(
                self.model.buffers[prev_buffer_id].data,
                dtype=np.float32,
            )
        )
    )
    # Check that first FC input tensor id was updated.
    self.assertEqual(
        subgraph.operators[fc1_op_idx].inputs[weight_idx_in_op_inputs],
        weight_tensor_idx,
    )

  def test_duplicate_tensor_raises_error_when_tensor_is_not_constant(self):
    # Duplicate the FC input tensor in the second subgraph.
    subgraph_idx = 1
    input_tensor_idx = 0
    transformation_input = self._get_transformation_input(
        subgraph_idx, input_tensor_idx, consumers=[0]
    )
    with self.assertRaisesRegex(
        ValueError,
        'Duplicate Tensor transformation supports only constant tensors.',
    ):
      duplicate_tensor.duplicate_tensor(transformation_input)

if __name__ == '__main__':
  absltest.main()
