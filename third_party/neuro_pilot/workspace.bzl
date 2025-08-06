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

"""Workspace definition for Neuro Pilot."""

load("//litert/sdk_util:repo.bzl", "configurable_repo")

def neuro_pilot():
    configurable_repo(
        name = "neuro_pilot",
        build_file = "@//third_party/neuro_pilot:neuro_pilot.BUILD",
        local_path_env = "LITERT_NEURO_PILOT_SDK",
        strip_prefix = "neuro_pilot",
        url = "https://s3.ap-southeast-1.amazonaws.com/mediatek.neuropilot.com/57c17aa0-90b4-4871-a7b6-cdcdc678b3aa.gz",
        symlink_mapping = {
            "v8_latest": "v8_0_8",
            # Just let the compilation pass, we don't expect it to work...
            # TODO: Remove this once we have a working V7 & V9 version.
            "v7_latest": "v8_0_8",
            "v9_latest": "v8_0_8",
        },
    )
