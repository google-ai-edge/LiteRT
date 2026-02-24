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

load("//litert/build_common:special_rule.bzl", "litert_cc_users", "litert_friends")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = [
        "//litert:__subpackages__",
    ],
)

# Controls visibility into LiteRT C++ and C APIs.
# Access to LiteRT C++ API is controlled by :litert_cc_users_dynamic_link and
# :litert_cc_users_static_link.
# Access to LiteRT C API is controlled by :litert_c_users.

# Users with access to the LiteRT C++ API linked dynamically (litert/cc/dynamic_runtime).
# Requires linking against libLiteRt.so.
# For statically linked C++ API, use :litert_cc_users_static_link.
#
# Note: External users should use litert/cc:litert_api_with_dynamic_runtime which is litert_public.
# This should be only used for internal LiteRT targets.
package_group(
    name = "litert_cc_users_dynamic_link",
    packages = [
        "//litert/cc/...",
        "//litert/compiler/plugin/...",
        "//litert/google/...",
        "//litert/integration_test/...",
        "//litert/kotlin/...",
        "//litert/python/...",
        "//litert/runtime/accelerators/gpu/...",
        "//litert/runtime/dispatch/...",
        "//litert/runtime/ml_drift_delegate/...",
        "//litert/rust/...",
        "//litert/samples/...",
        "//litert/sdk_util/...",
        "//litert/test/...",
        "//litert/tools/...",
        "//litert/vendors/...",
        # When litert_link_capi_so=false
        "//ci/tools/...",
        # copybara:uncomment "//third_party/odml/litert/tensor/...",
    ] + litert_cc_users() + litert_friends(),
)

# Users with access to the LiteRT C++ API linked statically (i.e. litert/cc/).
# For dynamically linked C++ API, use :litert_cc_users_dynamic_link.
package_group(
    name = "litert_cc_users_static_link",
    packages = [
        "//litert/ats/...",
        "//litert/google/...",
        "//litert/integration_test/...",
        "//litert/js/...",
        "//litert/kotlin/...",
        "//litert/python/...",
        "//third_party/odml/litert/tensor/...",
        "//litert/samples/semantic_similarity/...",
        # TODO(b/424778444): Internal runtime would better not call cc runtime APIs. Remove below targets.
        "//litert/runtime/...",
        "//litert/cc/...",
        "//litert/c/...",
        "//third_party/odml/litert/weight_loader/...",
        "//litert/c/options/...",
        "//litert/compiler/plugin/...",
        "//litert/core/model/...",
        "//litert/test/...",
        "//litert/tools/...",
    ] + litert_cc_users() + litert_friends(),
)

# All packages with visibility to the internal APIs
# such as litert/cc/internal and litert/c/internal.
package_group(
    name = "litert_internal_users",
    packages = [
        "//ci/tools/...",
        "//litert/ats/...",
        "//litert/c/...",
        "//litert/cc/...",
        "//litert/compiler/...",
        "//litert/core/...",
        "//litert/google/...",
        "//litert/integration_test/...",
        "//litert/kotlin/...",
        "//litert/python/...",
        "//litert/runtime/...",
        "//litert/sdk_util/...",
        "//litert/test/...",
        "//litert/tools/...",
        "//litert/vendors/...",
        "//third_party/odml/litert/weight_loader/...",
    ] + litert_friends(),
)

# All packages with visibility to the internal APIs
# such as litert/cc/internal and litert/c/internal.
package_group(
    name = "litert_friends",
    packages = litert_friends(),
)

# Users with access to the LiteRT C API.
package_group(
    name = "litert_c_users",
    includes = [
        ":litert_internal_users",
    ],
    packages = [
        "//litert/kotlin/...",
        "//litert/rust/...",
        "//litert/vendors/...",
    ],
)

# All litert users, least restrictive option.
package_group(
    name = "litert_public",
    # copybara:uncomment_begin(oss)
    # includes = [
    # ":litert_c_users",
    # ":litert_cc_users_dynamic_link",
    # ":litert_cc_users_static_link",
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
        # copybara:uncomment "//third_party/py/litert_torch/aot/...",
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

# Configuration for building with built-in kernels (default)
config_setting(
    name = "with_builtin_ops",
    values = {"define": "litert_builtin_ops=true"},
)

# Configuration for building without built-in kernels
config_setting(
    name = "without_builtin_ops",
    values = {"define": "litert_builtin_ops=false"},
)

# Configuration for building Android builds that can't depend on libandroid.so
config_setting(
    name = "litert_android_no_jni",
    constraint_values = ["@platforms//os:android"],
    values = {
        "define": "litert_android_no_jni=true",
        "crosstool_top": "//external:android/crosstool",  # copybara:comment
    },
)
