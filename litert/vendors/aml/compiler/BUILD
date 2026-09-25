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

load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib")
load("//litert/build_common:special_rule.bzl", "litert_android_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:private",
    ],
)

# Packager / runtime bytecode format (AML_Dispatch_Info).
# Leaf library: no ADLA / compiler_adla dependency.
cc_library(
    name = "aml_dispatch_info",
    srcs = [
        "aml_dispatch_info.cc",
    ],
    hdrs = [
        "aml_dispatch_info.h",
    ],
    visibility = [
        "//ci/tools/python/wheel:__subpackages__",
        "//litert:__subpackages__",
    ],
)

# LiteRT frontend: map LiteRT graph -> Context API, AmlContextCompile, pack
# DispatchInfo. compiler_core_*.h and libaml_compiler_core.so come from the
# external AML SDK (AML_SDK_ROOT). Does NOT link libcompiler_adla.so.
litert_dynamic_lib(
    name = "aml_compiler_plugin",
    srcs = [
        "aml_compiler_plugin.cc",
        "node_binary_info.cc",
    ],
    hdrs = [
        "//litert/vendors/c:litert_compiler_plugin.h",
        "aml_dispatch_info.h",
        "node_binary_info.h",
    ],
    # Keep RTTI on for NodeBuilder::SetParameter dynamic_cast across
    # libLiteRtCompilerPlugin_Aml.so <-> libcompiler_adla.so.
    copts = [
        "-frtti",
    ],
    # Do NOT use export_litert_only: that hides adla typeinfo and breaks
    # cross-DSO dynamic_cast. Use curated aml version-script instead:
    # export LiteRt* + RTTI (_ZTI*/_ZTS*/_ZTV*), hide std::filesystem etc.
    export_litert_only = False,
    version_script = ":export_aml_compiler_plugin_linux.lds",
    linkopts = litert_android_linkopts() + [
        "-Wl,-soname=libLiteRtCompilerPlugin_Aml.so",
        "-Wl,-rpath,$$ORIGIN",
        "-ldl",
    ],
    shared_lib_name = "aml_compiler_plugin_so",
    so_name = "libLiteRtCompilerPlugin_Aml.so",
    ungrte = True,
    visibility = [
        "//ci/tools/python/wheel:__subpackages__",
        "//litert:__subpackages__",
    ],
    deps = [
        ":aml_dispatch_info",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/c:litert_opaque_options",
        "//litert/c:litert_options",
        "//litert/c/options:litert_aml_options",
        "//litert/c:litert_op_options",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_model_types",
        "//litert/cc/internal:litert_extended_model",
        "//litert/cc/internal:litert_handle",
        "//litert/cc/internal:litert_op_options",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)
