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


from collections.abc import Sequence
import enum
import pathlib
import subprocess
from typing import Any
from unittest import mock

from absl.testing import absltest as googletest
from litert.python.aot.core import aot_types
from litert.python.aot.core import apply_plugin
from litert.python.aot.core import test_common

_REAL_GET_RESOURCE = apply_plugin.common.get_resource


class MockCompletedProcess:

  def __init__(self, returncode: int):
    self.returncode = returncode

  @property
  def stdout(self) -> str:
    return ""

  @property
  def stderr(self) -> str:
    return ""


class ApplyPluginTest(test_common.TestWithTfliteModels):

  @property
  def output_name(self) -> str:
    return "output.tflite"

  @property
  def output_model(self) -> pathlib.Path:
    return self.output_dir() / self.output_name

  @property
  def input_model(self) -> pathlib.Path:
    return self.get_model_path("add_simple.tflite")

  @property
  def soc_manufacturer(self) -> str:
    return "ExampleSocManufacturer"

  @property
  def soc_model(self) -> str:
    return "ExampleSocModel"

  def _get_resource_for_test(
      self, litert_relative_path: pathlib.Path, is_dir: bool = False
  ) -> pathlib.Path:
    """Resolves resources while stubbing only the apply-plugin binary path."""
    # Only stub the binary lookup; model fixtures still need the real resolver.
    if litert_relative_path == apply_plugin._BINARY:
      return pathlib.Path("apply_plugin_main")
    return _REAL_GET_RESOURCE(litert_relative_path, is_dir=is_dir)

  @mock.patch.object(subprocess, "run")
  def test_apply_plugin(self, mck: mock.Mock):
    mck.side_effect = self.get_touch_side_effect(
        self.output_model, MockCompletedProcess(0)
    )
    with mock.patch.object(
        apply_plugin.common,
        "get_resource",
        side_effect=self._get_resource_for_test,
    ):
      apply_plugin.ApplyPlugin()(
          aot_types.Model.create_from_path(self.input_model),
          aot_types.Model.create_from_path(self.output_model),
          self.soc_manufacturer,
          self.soc_model,
      )
    cmd_str = " ".join(mck.call_args_list[0][0][0])
    self.assertIn(str(self.input_model), cmd_str)
    self.assertIn(str(self.output_model), cmd_str)
    self.assertIn(self.soc_manufacturer, cmd_str)
    self.assertIn(self.soc_model, cmd_str)
    self.assertIn("apply_plugin_main", cmd_str)

  @mock.patch.object(subprocess, "run", return_value=MockCompletedProcess(1))
  def test_apply_plugin_no_file(self, unused_mock: mock.Mock):
    with mock.patch.object(
        apply_plugin.common,
        "get_resource",
        side_effect=self._get_resource_for_test,
    ):
      with self.assertRaises(ValueError):
        apply_plugin.ApplyPlugin()(
            aot_types.Model.create_from_path(self.input_model),
            aot_types.Model.create_from_path(self.output_model),
            self.soc_manufacturer,
            self.soc_model,
        )


class _FakePythonPlugin:
  """Copies the input model and reports one fully offloaded subgraph.

  The copy is written to `output_model.path`, or set in memory with
  `output_model.set_bytes` if `output_bytes` is true or `output_model` is in
  memory. `output_override` replaces the copied bytes.
  """

  def __init__(
      self,
      write_output: bool = True,
      stats: Any = None,
      output_bytes: bool = False,
      output_override: bytes | None = None,
  ):
    self.calls: list[tuple[str, Sequence[int] | None, dict[str, Any]]] = []
    self._write_output = write_output
    self._stats = stats
    self._output_bytes = output_bytes
    self._output_override = output_override

  def __call__(
      self,
      input_model: aot_types.Model,
      output_model: aot_types.Model,
      soc_model: str,
      *,
      subgraphs_to_compile: Sequence[int] | None,
      **kwargs: Any,
  ) -> aot_types.PartitionStats:
    self.calls.append((soc_model, subgraphs_to_compile, kwargs))
    if self._write_output:
      if self._output_override is not None:
        data = self._output_override
      elif input_model.in_memory:
        data = input_model.model_bytes
      else:
        data = input_model.path.read_bytes()
      if self._output_bytes or output_model.in_memory:
        output_model.set_bytes(data)
      else:
        output_model.path.write_bytes(data)
    if self._stats is not None:
      return self._stats
    return aot_types.PartitionStats(
        subgraph_stats=[
            aot_types.SubgraphPartitionStats(
                subgraph_index=0,
                num_ops_offloaded=1,
                num_total_ops=1,
                num_partitions_offloaded=1,
            )
        ]
    )


class PythonPluginTest(test_common.TestWithTfliteModels):

  _MANUFACTURER = "PythonSocManufacturer"

  def setUp(self):
    super().setUp()
    self.addCleanup(apply_plugin.unregister_python_plugin, self._MANUFACTURER)
    self.input_model = aot_types.Model.create_from_path(
        self.get_model_path("add_simple.tflite")
    )
    self.output_model = aot_types.Model.create_from_path(
        self.output_dir() / "output.tflite"
    )

  def test_registry(self):
    plugin = _FakePythonPlugin()
    self.assertIsNone(apply_plugin.find_python_plugin(self._MANUFACTURER))
    apply_plugin.register_python_plugin(self._MANUFACTURER, plugin)
    self.assertIs(apply_plugin.find_python_plugin(self._MANUFACTURER), plugin)
    # Registering the same plugin again is a no-op.
    apply_plugin.register_python_plugin(self._MANUFACTURER, plugin)
    with self.assertRaisesRegex(ValueError, "already registered"):
      apply_plugin.register_python_plugin(
          self._MANUFACTURER, _FakePythonPlugin()
      )
    apply_plugin.unregister_python_plugin(self._MANUFACTURER)
    self.assertIsNone(apply_plugin.find_python_plugin(self._MANUFACTURER))

  @mock.patch.object(subprocess, "run", autospec=True)
  def test_runs_python_plugin_in_process(self, run: mock.Mock):
    plugin = _FakePythonPlugin()
    apply_plugin.register_python_plugin(self._MANUFACTURER, plugin)

    apply_plugin.ApplyPlugin(subgraphs_to_compile=[0])(
        self.input_model,
        self.output_model,
        self._MANUFACTURER,
        "SomeSocModel",
        sdk_libs_path="/sdk",
        extra_flag="value",
    )

    run.assert_not_called()
    self.assertEqual(
        plugin.calls,
        [(
            "SomeSocModel",
            [0],
            {"sdk_libs_path": "/sdk", "extra_flag": "value"},
        )],
    )
    self.assertEqual(
        self.output_model.path.read_bytes(), self.input_model.path.read_bytes()
    )
    self.assertIsNotNone(self.output_model.partition_stats)
    self.assertEqual(
        self.output_model.partition_stats.subgraph_stats[0].num_ops_offloaded, 1
    )

  def test_python_plugin_must_produce_a_model(self):
    plugins = {
        "no output": _FakePythonPlugin(write_output=False),
        "empty file": _FakePythonPlugin(output_override=b""),
        "invalid file": _FakePythonPlugin(output_override=b"not a TFLite"),
        "empty bytes": _FakePythonPlugin(
            output_bytes=True, output_override=b""
        ),
        "invalid bytes": _FakePythonPlugin(
            output_bytes=True, output_override=b"not a TFLite"
        ),
    }
    for name, plugin in plugins.items():
      with self.subTest(name):
        output_model = aot_types.Model.create_from_path(
            self.output_dir() / "output.tflite"
        )
        output_model.path.unlink(missing_ok=True)
        apply_plugin.unregister_python_plugin(self._MANUFACTURER)
        apply_plugin.register_python_plugin(self._MANUFACTURER, plugin)
        with self.assertRaisesRegex(
            ValueError,
            "the Python compiler plugin did not produce a TFLite model",
        ):
          apply_plugin.ApplyPlugin()(
              self.input_model, output_model, self._MANUFACTURER, "SocModel"
          )
        self.assertIsNone(output_model.partition_stats)

  def test_python_plugin_with_in_memory_models(self):
    plugin = _FakePythonPlugin()
    apply_plugin.register_python_plugin(self._MANUFACTURER, plugin)
    model_bytes = self.input_model.path.read_bytes()
    input_model = aot_types.Model.create_from_bytes(model_bytes)
    output_model = aot_types.Model.create_from_bytes(b"placeholder")

    apply_plugin.ApplyPlugin()(
        input_model, output_model, self._MANUFACTURER, "SocModel"
    )

    self.assertEqual(output_model.model_bytes, model_bytes)
    self.assertIsNotNone(output_model.partition_stats)

  def test_str_enum_manufacturer_matches_its_value(self):
    class SocManufacturer(enum.StrEnum):
      PYTHON = self._MANUFACTURER

    plugin = _FakePythonPlugin()
    apply_plugin.register_python_plugin(SocManufacturer.PYTHON, plugin)
    self.assertIs(apply_plugin.find_python_plugin(self._MANUFACTURER), plugin)

    apply_plugin.ApplyPlugin()(
        self.input_model, self.output_model, self._MANUFACTURER, "SocModel"
    )

    self.assertLen(plugin.calls, 1)
    apply_plugin.unregister_python_plugin(self._MANUFACTURER)
    self.assertIsNone(apply_plugin.find_python_plugin(SocManufacturer.PYTHON))

  def test_empty_subgraphs_to_compile_means_all_subgraphs(self):
    plugin = _FakePythonPlugin()
    apply_plugin.register_python_plugin(self._MANUFACTURER, plugin)

    apply_plugin.ApplyPlugin(subgraphs_to_compile=[])(
        self.input_model, self.output_model, self._MANUFACTURER, "SocModel"
    )

    self.assertEqual(plugin.calls, [("SocModel", None, {})])

  def test_python_plugin_must_return_partition_stats(self):
    apply_plugin.register_python_plugin(
        self._MANUFACTURER, _FakePythonPlugin(stats="not stats")
    )
    with self.assertRaisesRegex(TypeError, "expected PartitionStats"):
      apply_plugin.ApplyPlugin()(
          self.input_model, self.output_model, self._MANUFACTURER, "SocModel"
      )

  @mock.patch.object(subprocess, "run", autospec=True)
  def test_other_manufacturers_still_use_the_binary(self, run: mock.Mock):
    apply_plugin.register_python_plugin(self._MANUFACTURER, _FakePythonPlugin())
    run.side_effect = self.get_touch_side_effect(
        self.output_model.path, MockCompletedProcess(0)
    )
    with mock.patch.object(
        apply_plugin.common,
        "get_resource",
        return_value=pathlib.Path("apply_plugin_main"),
        autospec=True,
    ):
      apply_plugin.ApplyPlugin()(
          self.input_model,
          self.output_model,
          "ExampleSocManufacturer",
          "ExampleSocModel",
      )
    run.assert_called_once()


if __name__ == "__main__":
  googletest.main()
