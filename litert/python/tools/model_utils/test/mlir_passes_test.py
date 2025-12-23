# Copyright 2025 Google LLC.
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
"""Check registered MLIR passes can run successfully."""

from absl.testing import absltest as googletest
from litert.python.tools.model_utils import model_builder
from litert.python.tools.model_utils import passes
from litert.python.tools.model_utils import testing
from litert.python.tools.model_utils.dialect import mlir
from litert.python.tools.model_utils.dialect import tfl


def build_sample_model() -> mlir.ModuleOp:
  @model_builder.build_module_from_py_func(
      mlir.RankedTensorType([2, 2], "f32"), mlir.RankedTensorType([2, 2], "f32")
  )
  def module(x, y):
    x = tfl.add(x, y)
    x = tfl.mul(x, y)
    return x

  return module


class MlirPassesTest(testing.ModelUtilsTestCase):
  """Test registered MLIR passes can run successfully."""

  def test_module_cleanup(self):
    module = build_sample_model()
    module.cleanup()

  def test_mlir_cse_pass(self):
    module = build_sample_model()
    passes.mlir.CsePass()(module)

  def test_mlir_canonicalize_pass(self):
    module = build_sample_model()
    passes.mlir.CanonicalizePass()(module)

  def test_tfl_optimize_pass(self):
    module = build_sample_model()
    passes.tfl.OptimizePass()(module)

  def test_tfl_prepare_quantize_pass(self):
    module = build_sample_model()
    passes.tfl.PrepareQuantizePass()(module)

  def test_tfl_propagate_qsv_pass(self):
    module = build_sample_model()
    passes.tfl.PropagateQsvPass()(module)


if __name__ == "__main__":
  googletest.main()
