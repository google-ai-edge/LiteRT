# Copyright 2025 The LiteRT Authors. All Rights Reserved.
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

from litert.python.mlir import ir
import numpy as np
from absl.testing import absltest as googletest
from litert.python.mlir._mlir_libs import converter_api_ext


class ConverterApiExtTest(googletest.TestCase):

  def test_py_chunked_callback_resource_attr(self):
    ir_context = ir.Context()
    converter_api_ext.prepare_mlir_context(ir_context)

    with ir_context, ir.Location.unknown():
      tensor_type = ir.RankedTensorType.get([2, 3], ir.F32Type.get())
      np_array = np.arange(6, dtype=np.float32).reshape(2, 3)

      def chunk_iterator_factory():
        for i in range(0, np_array.size, 2):
          yield np_array.flat[i : i + 2].tobytes()

      attr = converter_api_ext.get_py_chunked_callback_resource_attr(
          tensor_type, chunk_iterator_factory
      )
      self.assertIsNotNone(attr)

      attr_bytes = (
          converter_api_ext.get_py_chunked_callback_resource_attr_bytes(attr)
      )
      self.assertEqual(attr_bytes, np_array.tobytes())


if __name__ == "__main__":
  googletest.main()
