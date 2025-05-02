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
        allow_local = True,
        build_file = "@//third_party/neuro_pilot:neuro_pilot.BUILD",
        local_path_env = "NEURO_PILOT_SDK",
    )
