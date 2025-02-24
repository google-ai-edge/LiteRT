# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

from google3.testing.pybase import googletest
from google3.third_party.odml.litert.litert.python.google.core import common
from google3.third_party.odml.litert.litert.python.google.core import mlir_transforms
from google3.third_party.odml.litert.litert.python.google.core import test_common


class MlirTransformsTest(test_common.TestWithTfliteModels):

  def test_call_mlir_transforms(self):
    input_model = self.get_model_path("add_simple.tflite")
    output_model = self.output_dir() / "output.tflite"
    mlir_transforms.MlirTransforms()(input_model, output_model, "")
    self.assertTrue(common.is_tflite(output_model))


if __name__ == "__main__":
  googletest.main()
