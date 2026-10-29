# Copyright 2026 The LiteRT Authors. All Rights Reserved.
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
"""Configure the Arm emulation wheel for child processes only."""
import importlib.util
import os
from pathlib import Path


def layer_environment(icd):
    spec = importlib.util.find_spec("emulation_layer")
    if spec is None:
        raise RuntimeError("Install the pinned emulation-layer wheel first")
    deploy = Path(spec.origin).parent / "deploy"
    manifests = deploy / "share/vulkan/explicit_layer.d"
    for name in ["VkLayer_Graph.json", "VkLayer_Tensor.json"]:
        if not (manifests / name).is_file():
            raise FileNotFoundError(manifests / name)
    if not icd.is_file():
        raise FileNotFoundError(icd)
    env = os.environ.copy()
    env.pop("VK_ICD_FILENAMES", None)
    env.pop("VK_ADD_DRIVER_FILES", None)
    env.pop("VK_ADD_LAYER_PATH", None)
    env.update(
        VK_DRIVER_FILES=str(icd),
        VK_LAYER_PATH=str(manifests),
        VK_INSTANCE_LAYERS=
        "VK_LAYER_ML_Graph_Emulation:VK_LAYER_ML_Tensor_Emulation",
        LD_LIBRARY_PATH=str(deploy / "lib") +
        (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""),
    )
    return env
