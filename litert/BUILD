# Copyright 2024 Google LLC.
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

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = [
        "//litert:__subpackages__",
    ],
)

# VISIBILITY #######################################################################################

package_group(
    name = "litert_internal_users",
    packages = [
        # Self packages.
        "//litert/c/...",
        "//litert/cc/...",
        "//litert/compiler/...",
        "//litert/core/...",
        "//litert/google/...",
        "//litert/integration_test/...",
        "//litert/kotlin/...",
        "//litert/python/...",
        "//litert/runtime/...",
        "//litert/test/...",
        "//litert/tools/...",
        # Users with access to core internal apis.
    ],
)

package_group(
    name = "litert_stable_abi_users",
    includes = [
        ":litert_internal_users",
    ],
    packages = [
        "//litert/vendors/...",
    ],
)

package_group(
    name = "litert_python_internal_users",
    packages = [
        "//litert/python/...",
    ],
)

package_group(
    name = "litert_python_users",
    includes = [
        ":litert_internal_users",
    ],
    packages = [],
)
