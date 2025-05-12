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

# These are the internal packages that have access to the cc api linked against the runtime .so.
package_group(
    name = "litert_internal_users_with_public_cc",
    packages = [
        # Self packages.
        "//ci/tools/...",
        "//litert/cc/...",
        "//litert/google/...",
        "//litert/integration_test/...",
        "//litert/kotlin/...",
        "//litert/python/...",
        "//litert/test/...",
        "//litert/sdk_util/...",
        # Users with access to core internal apis.
        # copybara:uncomment "//third_party/odml/infra/genai/conversion/...",
        # copybara:uncomment "//platforms/darwinn/devtools/google_tensor_tpu_delegate_executor/...",
    ],
)

# These are the internal packages that have access to the cc api linked statically.
package_group(
    name = "litert_internal_users_with_internal_cc",
    packages = [
        # copybara:uncomment "//third_party/darwinn/litert/...",
        # copybara:uncomment "//third_party/darwinn/tflite/...",
        # copybara:uncomment "@mediapipe//mediapipe/calculators/tensor/...",
        # copybara:uncomment "//third_party/odml/infra/genai/inference/...",
        # copybara:uncomment "//third_party/odml/infra/ml_drift_delegate/...",
        "//litert/c/...",
        "//litert/compiler/...",
        "//litert/core/...",
        "//litert/runtime/...",
        "//litert/tools/...",
        "//litert/vendors/cc/...",
        "//litert/vendors/qualcomm/...",
        # copybara:uncomment "//third_party/odml/litert_lm/...",
    ],
)

# All packages with visibility to the internal apis.
package_group(
    name = "litert_internal_users",
    includes = [
        ":litert_internal_users_with_internal_cc",
        ":litert_internal_users_with_public_cc",
    ],
)

# Users with access to only the public stable abi + wrappers.
package_group(
    name = "litert_external_users",
    packages = [
        # copybara:uncomment "INTERNAL_IMAGE_COLLECTION_INTEGRATION/...",
        # copybara:uncomment "//platforms/darwinn/compiler/...",
        # copybara:uncomment "@mediapipe//mediapipe/tasks/javatests/...",
        "//litert/samples/...",
        "//litert/vendors/...",
        # copybara:uncomment "//third_party/odml/litert/llm/...",
    ],
)

# All litert users, least restrictive option.
package_group(
    name = "litert_public",
    includes = [
        ":litert_external_users",
        ":litert_internal_users",
    ],
)

package_group(
    name = "litert_python_internal_users",
    packages = [
        "//litert/python/...",
        "//ci/tools/...",
        # copybara:uncomment "//third_party/py/ai_edge_torch/aot/...",
    ],
)

package_group(
    name = "litert_python_users",
    includes = [
        ":litert_internal_users",
    ],
    packages = [],
)
