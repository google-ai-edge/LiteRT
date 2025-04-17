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
"""Builds torch sources for litert."""

import subprocess

from ai_edge_torch.aot import prepare_for_npu
from etils import epath
import fire

from google3.pyglib.contrib.gpathlib import gpath
from google3.third_party.odml.infra.testing.odml_flow.commands import torch_exports

_TORCH_CNS_TESTS = [
    "basic_RMSNorm.tflite",
]


def get_g3_path(path):
  return gpath.GPath((epath.g3_path() / path).as_posix())


def torch_cns_path(model_name: str) -> gpath.GPath:
  parent = torch_exports.latest_model_export_parent_dir()
  model_id = model_name.removesuffix(".tflite")
  return parent / model_id / model_name


LITERT_SCRIPT_ROOT = get_g3_path("litert/google/")

LITERT_SCRIPT = LITERT_SCRIPT_ROOT / "litert_scripts.sh"


def build_torch_sources(work_dir: str):
  """Builds torch sources for litert.

  Args:
    work_dir: The directory to store the built sources.
  """
  tempdir_path = gpath.GPath(work_dir)

  for model_name in _TORCH_CNS_TESTS:
    # Clone source model to the tempdir.
    torch_path = torch_cns_path(model_name)
    model_path = tempdir_path / model_name
    torch_exports.clone_saved_model_dirs_from_cns(
        [str(torch_path)], str(model_path)
    )

    # Apply litert AOT flow.
    try:
      prepare_for_npu.prepare_for_npu(
          str(model_path), str(model_path.parent), backend_id="qualcomm"
      )
    except (RuntimeError, subprocess.CalledProcessError) as e:
      print(f"{model_name} cannot be compiled due to: ", str(e))
      continue


def main(_):
  fire.Fire(build_torch_sources)


if __name__ == "__main__":
  fire.run()
