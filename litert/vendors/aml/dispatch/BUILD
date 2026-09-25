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
    default_visibility = ["//visibility:private"],
)

cc_library(
    name = "litert_dispatch_aml_lib",
    srcs = [
        "dispatch_api.cc",
        "litert_dispatch_device_context.cc",
        "litert_dispatch_invocation_context.cc",
    ],
    hdrs = [
        "litert_dispatch_device_context.h",
        "litert_dispatch_invocation_context.h",
        "registry.h",
    ],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/c/options:litert_aml_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_element_type",
        "//litert/core/util:tensor_type_util",
        "//litert/vendors/c:litert_dispatch_c_api",
        # Runtime only — do not pull libcompiler_adla.so into dispatch.
        "//litert/vendors/aml/core:Aml_runtime_lib",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "LiteRtDispatch_Aml",
    deps = [":litert_dispatch_aml_lib"],
    linkopts = [
        "-Wl,-soname=libLiteRtDispatch_Aml.so",
        "-Wl,-rpath,$$ORIGIN",
    ] + select({
        "@org_tensorflow//tensorflow:android": [
            "-Wl,-lc++abi",
            "-Wl,-lc++_shared",
        ],
        "//conditions:default": [],
    }),
    linkshared = 1,
    linkstatic = 0,
    visibility = ["//visibility:public"],
)
