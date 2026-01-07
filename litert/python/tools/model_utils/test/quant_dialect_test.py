# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test quant dialect transforms."""

from litert.python.mlir import ir
from absl.testing import absltest as googletest
from litert.python.tools.model_utils import testing
from litert.python.tools.model_utils.dialect import mlir
from litert.python.tools.model_utils.dialect import quant


class QuantDialectTest(testing.ModelUtilsTestCase):
  """Test quant dialect transforms."""

  def test_uniform_quantized_type(self):
    ir_type = ir.Type.parse(
        "tensor<2x2x!quant.uniform<i8<-8:7>:f32, 0.99872:127>>"
    )

    tensor_ty = mlir.RankedTensorType.from_mlir(ir_type)
    self.assertIsInstance(tensor_ty.elty, quant.UniformQuantizedType)
    self.assertEqual(tensor_ty.elty.storage_type, "i8")
    self.assertEqual(tensor_ty.elty.expressed_type, "f32")
    self.assertEqual(tensor_ty.elty.scale, 0.99872)
    self.assertEqual(tensor_ty.elty.zero_point, 127)
    self.assertEqual(tensor_ty.elty.storage_type_min, -8)
    self.assertEqual(tensor_ty.elty.storage_type_max, 7)

  def test_uniform_quantized_type_to_mlir(self):
    ir_type = ir.Type.parse(
        "tensor<2x2x!quant.uniform<i8<-8:7>:f32, 0.99872:127>>"
    )
    tensor_ty = mlir.RankedTensorType.from_mlir(ir_type)
    self.assertEqual(tensor_ty.to_mlir(), ir_type)

  def test_uniform_quantized_per_axis_type(self):
    ir_type = ir.Type.parse(
        "tensor<2x2x!quant.uniform<i8:f32:1, {2.0e+2,0.99872:120}>>"
    )
    tensor_ty = mlir.RankedTensorType.from_mlir(ir_type)
    self.assertIsInstance(tensor_ty.elty, quant.UniformQuantizedPerAxisType)
    self.assertEqual(tensor_ty.elty.storage_type, "i8")
    self.assertEqual(tensor_ty.elty.expressed_type, "f32")
    self.assertEqual(tensor_ty.elty.scales, [200.0, 0.99872])
    self.assertEqual(tensor_ty.elty.zero_points, [0, 120])
    self.assertEqual(tensor_ty.elty.quantized_dimension, 1)

  def test_uniform_quantized_per_axis_type_to_mlir(self):
    ir_type = ir.Type.parse(
        "tensor<2x2x!quant.uniform<i8:f32:1, {2.0e+2,0.99872:120}>>"
    )
    tensor_ty = mlir.RankedTensorType.from_mlir(ir_type)
    self.assertEqual(tensor_ty.to_mlir(), ir_type)


if __name__ == "__main__":
  googletest.main()
