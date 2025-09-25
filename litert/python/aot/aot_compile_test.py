# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

from absl.testing import absltest as googletest
from litert.python.aot import aot_compile as aot_compile_lib
from litert.python.aot.core import test_common
from litert.python.aot.core import types
from litert.python.aot.vendors.mediatek import target as mtk_target
from litert.python.aot.vendors.qualcomm import target as qnn_target


class AotCompileTest(test_common.TestWithTfliteModels):

  def test_compile(self):
    sm8450_target = qnn_target.Target(qnn_target.SocModel.SM8450)
    mt6989_target = mtk_target.Target(mtk_target.SocModel.MT6989)

    results = []
    for path in self.get_model_paths():
      results.append(
          aot_compile_lib.aot_compile(
              types.Model.create_from_path(path),
              output_dir=self.output_dir(),
              target=[sm8450_target, mt6989_target],
              keep_going=True,
          )
      )

    failed_backends = []
    model_bytes = []
    for result in results:
      if result.failed_backends:
        failed_backends.extend(result.failed_backends)
      for model in result.models:
        model_bytes.append(model.path.read_bytes())

    self.assertEmpty(failed_backends)
    for model_bytes in model_bytes:
      self.assertIn("DISPATCH_OP", str(model_bytes))


if __name__ == "__main__":
  googletest.main()
