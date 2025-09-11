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

"""Workspace definition for QAIRT."""

load("//litert/sdk_util:repo.bzl", "configurable_repo")

def qairt():
    configurable_repo(
        name = "qairt",
        build_file = "@//third_party/qairt:qairt.BUILD",
        local_path_env = "LITERT_QAIRT_SDK",
        strip_prefix = "latest",
        url = "https://storage.googleapis.com/litert/litert_qualcomm_sdk_2_37_1_release.tar.gz",
    )
