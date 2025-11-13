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
# ==============================================================================

import os
from unittest import mock
from absl.testing import absltest as googletest
from litert.python.aot.ai_pack import export_lib
from litert.python.aot.core import types
from litert.python.aot.vendors import fallback_backend
from litert.python.aot.vendors.google_tensor import target as google_tensor_target


def _create_test_model():
  test_model = types.Model(model_bytes=b"test_model")
  return test_model


def _create_mock_google_tensor_backend(
    soc_model: google_tensor_target.SocModel,
):
  mock_backend = mock.Mock(spec=types.Backend)
  mock_backend.target = google_tensor_target.Target(soc_model=soc_model)
  mock_backend.target_id = str(mock_backend.target)
  mock_backend.id.return_value = google_tensor_target.Target.backend_id()
  return mock_backend


def _create_mock_fallback_backend():
  mock_backend = mock.Mock(spec=types.Backend)
  mock_backend.target = fallback_backend.FallbackTarget()
  mock_backend.target_id = fallback_backend.FallbackBackend.id()
  mock_backend.id.return_value = fallback_backend.FallbackBackend.id()
  return mock_backend


class ExportLibTest(googletest.TestCase):

  def test_export_with_fallback_model_succeeds(self):
    mock_compilation_result = types.CompilationResult(
        models_with_backend=[
            (
                _create_mock_fallback_backend(),
                _create_test_model(),
            ),
        ]
    )
    output_dir = self.create_tempdir().full_path
    export_lib.export(
        compiled_models=mock_compilation_result,
        ai_pack_dir=output_dir,
        ai_pack_name="test_ai_pack",
        litert_model_name="test_litert_model",
    )

    with self.subTest(name="TargetingConfigFileContentIsCorrect"):
      self.assertTrue(
          os.path.exists(
              os.path.join(
                  output_dir,
                  "device_targeting_configuration.xml",
              )
          )
      )
      self.assertIn(
          """<config:device-targeting-config
    xmlns:config="http://schemas.android.com/apk/config">

</config:device-targeting-config>""",
          open(
              os.path.join(output_dir, "device_targeting_configuration.xml"),
              "r",
          ).read(),
      )

    with self.subTest(name="ModelFileIsWritten"):
      self.assertTrue(
          os.path.exists(
              os.path.join(
                  output_dir,
                  "test_ai_pack/src/main/assets/model#group_other/test_litert_model.tflite",
              )
          )
      )

  def test_export_without_fallback_model_fails(self):
    mock_compilation_result = types.CompilationResult()
    with self.assertRaises(AssertionError):
      export_lib.export(
          compiled_models=mock_compilation_result,
          ai_pack_dir=self.create_tempdir().full_path,
          ai_pack_name="test_ai_pack",
          litert_model_name="test_litert_model",
      )

  def test_export_with_google_tensor_target_succeeds(self):
    output_dir = self.create_tempdir().full_path
    mock_compilation_result = types.CompilationResult(
        models_with_backend=[
            (
                _create_mock_google_tensor_backend(
                    google_tensor_target.SocModel.TENSOR_G3
                ),
                _create_test_model(),
            ),
            (
                _create_mock_fallback_backend(),
                _create_test_model(),
            ),
        ]
    )

    export_lib.export(
        compiled_models=mock_compilation_result,
        ai_pack_dir=output_dir,
        ai_pack_name="test_ai_pack",
        litert_model_name="test_litert_model",
    )

    with self.subTest(name="ModelFileIsWritten"):
      self.assertTrue(
          os.path.exists(
              os.path.join(
                  output_dir,
                  "test_ai_pack/src/main/assets/model#group_Google_Tensor_G3/test_litert_model.tflite",
              )
          )
      )

    with self.subTest(name="TargetingConfigFileIsWritten"):
      self.assertTrue(
          os.path.exists(
              os.path.join(
                  output_dir,
                  "test_ai_pack/src/main/assets/model#group_other/test_litert_model.tflite",
              )
          )
      )

    with self.subTest(name="TargetingConfigFileContentIsCorrect"):
      self.assertTrue(
          os.path.exists(
              os.path.join(output_dir, "device_targeting_configuration.xml")
          )
      )
      with open(
          os.path.join(output_dir, "device_targeting_configuration.xml"), "r"
      ) as f:
        self.assertIn(
            """<config:device-targeting-config
    xmlns:config="http://schemas.android.com/apk/config">
    <config:device-group name="Google_Tensor_G3">
        <config:device-selector>
            <config:system-on-chip manufacturer="Google" model="Tensor G3"/>
        </config:device-selector>
    </config:device-group>
</config:device-targeting-config>""",
            f.read(),
        )


if __name__ == "__main__":
  googletest.main()
