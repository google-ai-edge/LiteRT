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
from litert.python.aot.core import aot_types
from litert.python.aot.vendors import fallback_backend
from litert.python.aot.vendors.google_tensor import target as google_tensor_target
from litert.python.aot.vendors.qualcomm import qualcomm_backend
from litert.python.aot.vendors.qualcomm import target as qnn_target


def _create_test_model():
  test_model = aot_types.Model(model_bytes=b"test_model")
  return test_model


def _create_mock_google_tensor_backend(
    soc_model: google_tensor_target.SocModel,
):
  mock_backend = mock.Mock(spec=aot_types.Backend)
  mock_backend.target = google_tensor_target.Target(soc_model=soc_model)
  mock_backend.target_id = str(mock_backend.target)
  mock_backend.id.return_value = google_tensor_target.Target.backend_id()
  return mock_backend


def _create_mock_qnn_backend(soc_model: qnn_target.SocModel):
  mock_backend = mock.Mock(spec=aot_types.Backend)
  mock_backend.target = qnn_target.Target(soc_model=soc_model)
  mock_backend.target_id = repr(mock_backend.target)
  mock_backend.id.return_value = qualcomm_backend.QualcommBackend.id()
  return mock_backend


def _create_mock_fallback_backend():
  mock_backend = mock.Mock(spec=aot_types.Backend)
  mock_backend.target = fallback_backend.FallbackTarget()
  mock_backend.target_id = fallback_backend.FallbackBackend.id()
  mock_backend.id.return_value = fallback_backend.FallbackBackend.id()
  return mock_backend


class ExportLibTest(googletest.TestCase):

  def test_export_with_fallback_model_succeeds(self):
    mock_compilation_result = aot_types.CompilationResult(
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
    mock_compilation_result = aot_types.CompilationResult()
    with self.assertRaises(AssertionError):
      export_lib.export(
          compiled_models=mock_compilation_result,
          ai_pack_dir=self.create_tempdir().full_path,
          ai_pack_name="test_ai_pack",
          litert_model_name="test_litert_model",
      )

  def test_export_with_google_tensor_target_succeeds(self):
    output_dir = self.create_tempdir().full_path
    mock_compilation_result = aot_types.CompilationResult(
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


class ExportLibQualcommNonMobileTest(googletest.TestCase):

  def test_is_mobile_device_backend_excludes_iot_and_auto_socs(self):
    for soc in (
        qnn_target.SocModel.QCS8275,
        qnn_target.SocModel.QCS8300,
        qnn_target.SocModel.SA8255,
        qnn_target.SocModel.SA8295,
    ):
      with self.subTest(soc=soc):
        self.assertFalse(
            export_lib._is_mobile_device_backend(_create_mock_qnn_backend(soc))
        )

  def test_is_mobile_device_backend_keeps_phone_socs(self):
    for soc in (qnn_target.SocModel.SM8650, qnn_target.SocModel.SM8750):
      with self.subTest(soc=soc):
        self.assertTrue(
            export_lib._is_mobile_device_backend(_create_mock_qnn_backend(soc))
        )

  def test_export_skips_qcs_models(self):
    output_dir = self.create_tempdir().full_path
    result = aot_types.CompilationResult(
        models_with_backend=[
            (
                _create_mock_qnn_backend(qnn_target.SocModel.QCS8275),
                _create_test_model(),
            ),
            (
                _create_mock_qnn_backend(qnn_target.SocModel.QCS8300),
                _create_test_model(),
            ),
            (
                _create_mock_qnn_backend(qnn_target.SocModel.SM8750),
                _create_test_model(),
            ),
            (_create_mock_fallback_backend(), _create_test_model()),
        ]
    )

    export_lib.export(
        compiled_models=result,
        ai_pack_dir=output_dir,
        ai_pack_name="test_ai_pack",
        litert_model_name="test_litert_model",
    )

    assets = os.path.join(output_dir, "test_ai_pack/src/main/assets")
    self.assertTrue(
        os.path.exists(
            os.path.join(
                assets, "model#group_Qualcomm_SM8750/test_litert_model.tflite"
            )
        )
    )
    for soc in ("QCS8275", "QCS8300"):
      with self.subTest(soc=soc):
        self.assertFalse(
            os.path.exists(os.path.join(assets, f"model#group_Qualcomm_{soc}"))
        )

    with open(
        os.path.join(output_dir, "device_targeting_configuration.xml")
    ) as f:
      config = f.read()
    self.assertIn('model="SM8750"', config)
    self.assertNotIn("QCS8275", config)
    self.assertNotIn("QCS8300", config)


if __name__ == "__main__":
  googletest.main()
