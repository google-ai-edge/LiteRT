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

# LINT.IfChange(bazel_qairt_sdk_version)
def qairt():
    configurable_repo(
        name = "qairt",
        build_file = "@//third_party/qairt:qairt.BUILD",
        local_path_env = "LITERT_QAIRT_SDK",
        strip_prefix = "qairt/2.42.0.251225",
        url = "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.42.0.251225/v2.42.0.251225.zip",
        file_extension = "zip",
    )

# LINT.ThenChange(
#     ../../ci/tools/python/vendor_sdk/qualcomm/setup.py:wheel_qairt_sdk_version,
#     ../../../litert/google/npu_runtime_libraries/fetch_qualcomm_library.sh:fetch_qairt_sdk_version,
#     ../../../litert/google/npu_runtime_libraries/fetch_qualcomm_library_jit.sh:fetch_qairt_sdk_version,
#     ../../../litert/vendors/CMakeLists.txt:qairt_headers_dir,
# )
