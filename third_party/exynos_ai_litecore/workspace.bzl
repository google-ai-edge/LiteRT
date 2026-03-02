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


load("//litert/sdk_util:repo.bzl", "configurable_repo")

def exynos_ai_litecore():
    configurable_repo(
        name = "exynos_ai_litecore",
        build_file = "@//third_party/exynos_ai_litecore:exynos_ai_litecore.BUILD",
        local_path_env = "EXYNOS_AI_LITECORE_ROOT",
        # TODO: Internal link (Not available). Please change to official link when release.
        url =
          "https://soc-developer.semiconductor.samsung.com/api/v1/resource/download-file/1.1.0/ai-litecore-ubuntu2404-v1.1.0.tar.gz",

    )

