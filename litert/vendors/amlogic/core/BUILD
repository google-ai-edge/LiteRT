# Copyright 2026 Amlogic, Inc.
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
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:private",
    ],
)

# ---------------------------------------------------------------------------
# Compatibility aliases.
# Compiler core / nnsdk: @aml_npu_nn_runtime (GitHub nn_runtime / LITERT_AML_NN_RUNTIME).
# ---------------------------------------------------------------------------
alias(
    name = "Aml_common_lib",
    actual = "@aml_npu_nn_runtime//:aml_compiler_core_lib",
    visibility = [
        "//ci/tools/python/wheel:__subpackages__",
        "//litert:__subpackages__",
    ],
)

alias(
    name = "aml_compiler_core_lib",
    actual = "@aml_npu_nn_runtime//:aml_compiler_core_lib",
    visibility = [
        "//ci/tools/python/wheel:__subpackages__",
        "//litert:__subpackages__",
    ],
)

alias(
    name = "aml_compiler_core",
    actual = "@aml_npu_nn_runtime//:aml_compiler_core",
    visibility = ["//visibility:public"],
)

alias(
    name = "Aml_common",
    actual = "@aml_npu_nn_runtime//:aml_compiler_core_lib",
    visibility = ["//visibility:public"],
)

# ---------------------------------------------------------------------------
# Runtime-only pieces used by LiteRtDispatch_Amlogic (inference).
# Must NOT depend on libcompiler_adla.so.
# ---------------------------------------------------------------------------
cc_library(
    name = "amlogic_dispatch_info",
    srcs = ["amlogic_dispatch_info.cc"],
    hdrs = ["amlogic_dispatch_info.h"],
    visibility = [
        "//ci/tools/python/wheel:__subpackages__",
        "//litert:__subpackages__",
    ],
)

# dlopen shim for libaml_compiler_core.so (function table, resolved at run
# time). Headers-only dependency on the SDK, so processes that never compile
# never map libaml_compiler_core.so / its libcompiler_adla.so dependency.
cc_library(
    name = "aml_compiler_core_loader",
    srcs = ["aml_compiler_core_loader.cc"],
    hdrs = ["aml_compiler_core_loader.h"],
    deps = [
        "//litert/c/internal:litert_logging",
        "@aml_npu_nn_runtime//:headers",
    ],
    visibility = [
        "//ci/tools/python/wheel:__subpackages__",
        "//litert:__subpackages__",
    ],
)

cc_library(
    name = "Amlogic_runtime_lib",
    srcs = [
        "nnsdk_func.cc",
    ],
    hdrs = [
        "nnsdk_func.h",
    ],
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/c/options:litert_amlogic_options",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_opaque_options",
        "//litert/cc:litert_options",
        "//litert/cc/options:litert_amlogic_options",
        ":amlogic_dispatch_info",
        "@com_google_absl//absl/strings:string_view",
        "@aml_npu_nn_runtime//:nnsdk_headers",
    ],
    visibility = [
        "//ci/tools/python/wheel:__subpackages__",
        "//litert:__subpackages__",
    ],
)
