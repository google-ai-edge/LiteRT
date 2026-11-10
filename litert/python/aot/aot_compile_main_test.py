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

import pathlib

from absl import app
from absl.testing import flagsaver
from absl.testing import parameterized

from absl.testing import absltest as googletest
from litert.python.aot import aot_compile_main
from litert.python.aot.core import common
from litert.python.aot.vendors import fallback_backend
from litert.python.aot.vendors import import_vendor

_ADD_SIMPLE = pathlib.Path("test/testdata/add_simple.tflite")


class AotCompileMainTest(parameterized.TestCase, googletest.TestCase):

  @parameterized.named_parameters(
      ("empty_defaults_to_all", "", "all"),
      ("all", "ALL", "all"),
      ("qualcomm", "Qualcomm", "qualcomm"),
      ("mediatek", "MediaTek", "mediatek"),
      ("google_tensor", "google_tensor", "GOOGLE"),
      ("google_alias", "Google", "GOOGLE"),
      ("samsung", "Samsung", "samsung"),
      ("intel_openvino", "intel_openvino", "intel_openvino"),
      ("intel_manufacturer", "IntelOpenVINO", "intel_openvino"),
      ("fallback", "fallback", "fallback"),
  )
  def test_normalize_backend_id(self, raw_backend: str, expected: str):
    self.assertEqual(
        aot_compile_main.normalize_backend_id(raw_backend), expected
    )

  def test_parse_backend_options_coerces_types(self):
    parsed = aot_compile_main.parse_backend_options(
        [
            "quantize_recipe = srq_int8 ",
            "sdk_libs_path=/opt/qnn",
            "flag_true=true",
            "flag_false=FALSE",
            "opt_none=none",
            'config={"level": 2}',
        ],
        backend_id="qualcomm",
    )
    self.assertDictEqual(
        parsed,
        {
            "quantize_recipe": "srq_int8",
            "sdk_libs_path": "/opt/qnn",
            "flag_true": True,
            "flag_false": False,
            "opt_none": None,
            "config": {"level": 2},
        },
    )

  @parameterized.named_parameters(
      ("missing_equals", ["precision_fp16"], "expected 'key=value'"),
      ("empty_key", ["=fp16"], "option key must not be empty"),
      ("invalid_json", ["lowerings={bad}"], "Invalid JSON"),
  )
  def test_parse_backend_options_rejects_malformed_entries(
      self, entries: list[str], error_regex: str
  ):
    with self.assertRaisesRegex(ValueError, error_regex):
      aot_compile_main.parse_backend_options(entries)

  def test_parse_subgraph_indices(self):
    self.assertIsNone(aot_compile_main.parse_subgraph_indices([]))
    self.assertIsNone(aot_compile_main.parse_subgraph_indices(["", " "]))
    self.assertEqual(
        aot_compile_main.parse_subgraph_indices(["0", " 2 "]), [0, 2]
    )
    with self.assertRaisesRegex(ValueError, "non-negative integer"):
      aot_compile_main.parse_subgraph_indices(["abc"])
    with self.assertRaisesRegex(ValueError, "non-negative"):
      aot_compile_main.parse_subgraph_indices(["-1"])

  @parameterized.named_parameters(
      ("qualcomm", "qualcomm", "SM8750", "qualcomm", "Qualcomm_SM8750"),
      ("qualcomm_all_lower", "qualcomm", "all", "qualcomm", "Qualcomm_ALL"),
      ("mediatek", "mediatek", "MT6989", "mediatek", "MediaTek_MT6989"),
      (
          "google_tensor",
          "google_tensor",
          "Tensor_G5",
          "GOOGLE",
          "Google_Tensor_G5",
      ),
      ("samsung", "samsung", "E9965", "samsung", "Samsung_E9965"),
      (
          "intel_openvino",
          "intel_openvino",
          "LNL",
          "intel_openvino",
          "IntelOpenVINO_LNL",
      ),
  )
  def test_resolve_target_for_vendors(
      self,
      backend: str,
      soc_model: str,
      expected_backend_id: str,
      expected_repr: str,
  ):
    target = aot_compile_main.resolve_target(backend, soc_model)
    self.assertEqual(target.flatten()["backend_id"], expected_backend_id)
    self.assertEqual(repr(target), expected_repr)

  def test_resolve_target_for_all_fallback_and_invalid_inputs(self):
    self.assertIsInstance(
        aot_compile_main.resolve_target("all"),
        import_vendor.AllRegisteredTarget,
    )
    self.assertIsInstance(
        aot_compile_main.resolve_target("fallback"),
        fallback_backend.FallbackTarget,
    )
    with self.assertRaisesRegex(
        ValueError, "--soc_model cannot be specified with --backend=all"
    ):
      aot_compile_main.resolve_target("all", "SM8750")
    with self.assertRaisesRegex(
        ValueError, "--soc_model cannot be specified with --backend=fallback"
    ):
      aot_compile_main.resolve_target("fallback", "SM8750")
    with self.assertRaisesRegex(ValueError, "Unsupported backend"):
      aot_compile_main.resolve_target("unknown_vendor")

  def test_compile_from_cli_fallback_writes_output_model_and_dir(self):
    input_path = common.get_resource(_ADD_SIMPLE)
    work_dir = pathlib.Path(self.create_tempdir().full_path)
    out_model = work_dir / "single_out" / "compiled_fallback.tflite"

    result = aot_compile_main.compile_from_cli(
        input_path,
        output_model=out_model,
        backend="fallback",
    )
    self.assertLen(result.models_with_backend, 1)
    self.assertTrue(out_model.is_file())
    self.assertTrue(common.is_tflite(out_model))

    out_dir = work_dir / "dir_out"
    dir_result = aot_compile_main.compile_from_cli(
        input_path,
        output_dir=out_dir,
        backend="fallback",
    )
    self.assertLen(dir_result.models_with_backend, 1)
    _, dir_model = dir_result.models_with_backend[0]
    self.assertTrue(dir_model.path.is_file())
    self.assertEqual(dir_model.path.parent, out_dir)

    with self.assertRaisesRegex(ValueError, "Cannot specify both"):
      aot_compile_main.compile_from_cli(
          input_path,
          output_model=out_model,
          output_dir=out_dir,
          backend="fallback",
      )
    with self.assertRaisesRegex(FileNotFoundError, "Input model not found"):
      aot_compile_main.compile_from_cli(
          work_dir / "nonexistent.tflite",
          output_model=out_model,
          backend="fallback",
      )

  def test_main_runs_with_apply_plugin_flag_aliases(self):
    input_path = common.get_resource(_ADD_SIMPLE)
    out_model = (
        pathlib.Path(self.create_tempdir().full_path) / "aliased_out.tflite"
    )
    with flagsaver.flagsaver(
        model=str(input_path),
        o=str(out_model),
        soc_manufacturer="fallback",
        subgraphs=["0"],
    ):
      aot_compile_main.main(["aot_compile_main"])
    self.assertTrue(out_model.is_file())
    self.assertTrue(common.is_tflite(out_model))

  def test_main_rejects_missing_or_conflicting_flags(self):
    with self.assertRaisesRegex(app.UsageError, "Unexpected positional"):
      aot_compile_main.main(["aot_compile_main", "extra_arg"])

    with flagsaver.flagsaver(input_model="", model=""):
      with self.assertRaisesRegex(app.UsageError, "Must specify --input_model"):
        aot_compile_main.main(["aot_compile_main"])

    with flagsaver.flagsaver(input_model="a.tflite", model="b.tflite"):
      with self.assertRaisesRegex(app.UsageError, "conflicting values"):
        aot_compile_main.main(["aot_compile_main"])


if __name__ == "__main__":
  googletest.main()
