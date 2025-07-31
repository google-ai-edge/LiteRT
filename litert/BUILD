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
    # copybara:uncomment includes = ["//third_party/py/executorch:execufriends"],
    packages = [
        # Self packages.
        "//ci/tools/...",
        "//litert/cc/...",
        "//litert/google/...",
        "//litert/integration_test/...",
        "//litert/kotlin/...",
        "//litert/python/...",
        "//litert/rust/...",
        "//litert/test/...",
        "//litert/sdk_util/...",
        # When litert_lm_link_capi_so=false
        # copybara:uncomment "//third_party/odml/litert_lm/...",
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
        "//litert/cts/...",
        "//litert/tools/...",
        "//litert/vendors/cc/...",
        "//litert/vendors/qualcomm/...",
        # When litert_lm_link_capi_so=false
        # copybara:uncomment "//third_party/odml/litert_lm/...",
        # copybara:uncomment "//platforms/darwinn/devtools/google_tensor_tpu_delegate_executor/...",
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
        "//litert/runtime/...",
        "//litert/samples/...",
        "//litert/tools/...",
        "//litert/vendors/...",
        # copybara:uncomment "//third_party/odml/litert/llm/...",
        # copybara:uncomment "//third_party/odml/litert/tensor/...",
    ],
)

# All litert users, least restrictive option.
package_group(
    name = "litert_public",
    # copybara:uncomment_begin(oss)
    # includes = [
    # ":litert_external_users",
    # ":litert_internal_users",
    # ],
    # copybara:uncomment_end_and_comment_begin
    packages = ["//..."],
    # copybara:comment_end
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

exports_files(
    glob(["*.gwsq"]),
    visibility = ["//visibility:public"],
)

# If litert_link_capi_so is not defined, each target links LiteRT C API either statically or
# dynamically based on its own requirements.
config_setting(
    name = "litert_link_capi_so",
    values = {"define": "litert_link_capi_so=true"},
)

config_setting(
    name = "litert_link_capi_static",
    values = {"define": "litert_link_capi_so=false"},
)
