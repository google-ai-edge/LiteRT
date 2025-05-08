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
        "//ci/tools/...",
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
        "//litert/sdk_util/...",
        "//litert/tools/...",
        # Users with access to core internal apis.
        # copybara:uncomment "//third_party/odml/infra/ml_drift_delegate/...",
        # copybara:uncomment "//third_party/odml/infra/genai/conversion/...",
        # copybara:uncomment "//third_party/darwinn/tflite/...",
        # copybara:uncomment "//platforms/darwinn/devtools/google_tensor_tpu_delegate_executor/...",
        # copybara:uncomment "//third_party/darwinn/litert/...",
    ],
)

package_group(
    name = "litert_stable_abi_users",
    includes = [
        ":litert_internal_users",
    ],
    packages = [
        # copybara:uncomment "INTERNAL_IMAGE_COLLECTION_INTEGRATION/...",
        # copybara:uncomment "//platforms/darwinn/compiler/...",
        # copybara:uncomment "@mediapipe//mediapipe/calculators/tensor/...",
        # copybara:uncomment "@mediapipe//mediapipe/tasks/javatests/...",
        # copybara:uncomment "//third_party/odml/infra/genai/inference/...",
        "//litert/samples/...",
        "//litert/vendors/...",
        # copybara:uncomment "//third_party/odml/litert/llm/...",
        # copybara:uncomment "//third_party/odml/litert_lm/...",
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
