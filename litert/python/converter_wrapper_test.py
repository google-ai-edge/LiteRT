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

from absl.testing import absltest as googletest
from litert.python import _pywrap_litert_converter as converter_wrapper
from ai_edge_litert.tflite.converter import converter_flags_pb2
from ai_edge_litert.tflite.converter import model_flags_pb2


class ConverterWrapperTest(googletest.TestCase):

  def test_construct_conversion_config(self):
    config = converter_wrapper.ConversionConfig()

    self.assertEqual(
        config.original_model_type,
        converter_wrapper.ConversionConfig.ModelType.Unknown,
    )
    self.assertEqual(
        config.converter_flags, converter_flags_pb2.ConverterFlags()
    )
    self.assertEqual(config.model_flags, model_flags_pb2.ModelFlags())

  def test_model_type_values(self):
    self.assertEqual(
        converter_wrapper.ConversionConfig.ModelType.Unknown.value,
        0,
    )
    self.assertEqual(
        converter_wrapper.ConversionConfig.ModelType.Jax.value,
        6,
    )
    self.assertEqual(
        converter_wrapper.ConversionConfig.ModelType.PyTorch.value,
        7,
    )

  def test_construct_converter(self):
    config = converter_wrapper.ConversionConfig()
    converter = converter_wrapper.Converter(config)

    self.assertIsNotNone(converter)


if __name__ == "__main__":
  googletest.main()
