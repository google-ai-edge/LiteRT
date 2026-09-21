# Copyright (C) 2023 Amlogic, Inc. All rights reserved.
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

# Runtime logging used by LiteRtDispatch_Aml.
# A copy also exists under tflite/delegates/aml for the TFLite Interpreter
# delegate; LiteRT vendor code must depend on this target, not the delegate.
cc_library(
    name = "aml_adla_log",
    srcs = [
        "adla_log.cc",
    ],
    hdrs = [
        "adla_log.h",
    ],
    includes = ["."],
    visibility = [
        "//ci/tools/python/wheel:__subpackages__",
        "//litert:__subpackages__",
    ],
)

# Runtime-only pieces used by LiteRtDispatch_Aml (inference).
# Must NOT depend on libcompiler_adla.so.
# nnsdk2.h / nnsdk.h come from the external AML SDK, not this tree.
cc_library(
    name = "Aml_runtime_lib",
    srcs = [
        "nnsdk_func.cc",
    ],
    hdrs = [
        "nnsdk_func.h",
    ],
    includes = ["."],
    deps = [
        ":aml_adla_log",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/options:litert_aml_options",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_opaque_options",
        "//litert/cc:litert_options",
        "//litert/cc/options:litert_aml_options",
        "//litert/vendors/aml/compiler:aml_dispatch_info",
        "//litert/vendors/c:litert_compiler_plugin",
        "@com_google_absl//absl/strings:string_view",
    ],
    visibility = [
        "//ci/tools/python/wheel:__subpackages__",
        "//litert:__subpackages__",
    ],
)
