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

import pathlib
import sys
import types
from unittest import mock

from absl.testing import absltest as googletest
from litert.python.aot.core import common

_TFL_MODEL = pathlib.Path("test/testdata/add_simple.tflite")


class CommonTest(googletest.TestCase):

  def test_get_resource(self):
    resource = common.get_resource(_TFL_MODEL)
    self.assertTrue(resource.exists())
    self.assertTrue(resource.is_file())

  def test_get_resource_non_existent(self):
    with self.assertRaises(FileNotFoundError):
      common.get_resource(pathlib.Path("non_existent.tflite"))

  def test_get_resource_from_vendor_sdk_package(self):
    tmp_dir = pathlib.Path(self.create_tempdir().full_path)
    plugin_rel = pathlib.Path(
        "vendors/qualcomm/compiler/libLiteRtCompilerPlugin_Qualcomm.so"
    )
    plugin_file = tmp_dir / plugin_rel
    plugin_file.parent.mkdir(parents=True)
    plugin_file.write_bytes(b"plugin")
    fake_sdk = types.ModuleType("ai_edge_litert_sdk_qualcomm")
    fake_sdk.__file__ = str(tmp_dir / "__init__.py")
    with mock.patch.dict(
        sys.modules, {"ai_edge_litert_sdk_qualcomm": fake_sdk}
    ):
      self.assertEqual(common.get_resource(plugin_rel), plugin_file.resolve())

  def test_get_resource_finds_vendor_sdk_without_importing_it(self):
    site_dir = pathlib.Path(self.create_tempdir().full_path)
    pkg_dir = site_dir / "ai_edge_litert_sdk_mediatek"
    plugin_rel = pathlib.Path(
        "vendors/mediatek/compiler/libLiteRtCompilerPlugin_MediaTek.so"
    )
    (pkg_dir / plugin_rel).parent.mkdir(parents=True)
    (pkg_dir / plugin_rel).write_bytes(b"plugin")
    # Importing the package would raise; discovery must not execute it.
    (pkg_dir / "__init__.py").write_text("raise RuntimeError('imported')\n")
    with mock.patch.object(sys, "path", [str(site_dir)] + sys.path):
      self.assertEqual(
          common.get_resource(plugin_rel), (pkg_dir / plugin_rel).resolve()
      )
    self.assertNotIn("ai_edge_litert_sdk_mediatek", sys.modules)

  def test_get_resource_ignores_vendor_sdk_outside_vendors_dir(self):
    tmp_dir = pathlib.Path(self.create_tempdir().full_path)
    (tmp_dir / _TFL_MODEL).parent.mkdir(parents=True)
    (tmp_dir / _TFL_MODEL).write_bytes(b"not the real model")
    fake_sdk = types.ModuleType("ai_edge_litert_sdk_qualcomm")
    fake_sdk.__file__ = str(tmp_dir / "__init__.py")
    with mock.patch.dict(
        sys.modules, {"ai_edge_litert_sdk_qualcomm": fake_sdk}
    ):
      self.assertNotEqual(
          common.get_resource(_TFL_MODEL), (tmp_dir / _TFL_MODEL).resolve()
      )

  def test_is_tflite(self):
    resource = common.get_resource(_TFL_MODEL)
    self.assertTrue(common.is_tflite(resource))


if __name__ == "__main__":
  googletest.main()
