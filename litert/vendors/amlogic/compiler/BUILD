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

load("@rules_cc//cc:cc_shared_library.bzl", "cc_shared_library")
load(
    "//litert/build_common:litert_build_defs.bzl",
    "export_lrt_only_linkopt",
    "export_lrt_only_script",
    "litert_lib",
    "litert_test",
    "make_rpaths",
)
load("//litert/build_common:special_rule.bzl", "litert_android_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:private",
    ],
)

# LiteRT-only frontend: map LiteRT graph -> Context API, AmlContextCompile, pack DispatchInfo.
# dlopen()s libaml_compiler_core.so at run time (aml_compiler_core_loader);
# @aml_npu_nn_runtime provides headers only, so processes that never compile
# (e.g. the x86 plugin test) never map the SDK .so / libcompiler_adla.so.
#
# Hand-expansion of `litert_dynamic_lib` (the macro does not accept per-.so
# linkopts): litert_lib + cc_shared_library + filegroup. The soname /
# exclude-libs / $ORIGIN flags live in `user_link_flags`, which apply only to
# libLiteRtCompilerPlugin_Amlogic.so and do NOT propagate to dependents;
# cc_library `linkopts` would propagate (that is how the test binary
# previously inherited the plugin soname and shadowed the real .so).
litert_lib(
    name = "amlogic_compiler_plugin",
    srcs = [
        "amlogic_compiler_plugin.cc",
        "node_binary_info.cc",
    ],
    hdrs = [
        "//litert/vendors/c:litert_compiler_plugin.h",
        "node_binary_info.h",
    ],
    # C ABI only. adla C++ RTTI / dynamic_cast stays inside libcompiler_adla.so.
    copts = [
        "-fno-rtti",
    ],
    linkopts = litert_android_linkopts() + select({
        "@org_tensorflow//tensorflow:linux_x86_64": [
            make_rpaths([
                "external/aml_npu_nn_runtime/aml_compiler_core/libs/linux/x86_64",
            ]),
        ],
        "//conditions:default": [],
    }) + [
        "-ldl",
    ],
    ungrte = True,
    visibility = [
        "//ci/tools/python/wheel:__subpackages__",
        "//litert:__subpackages__",
    ],
    deps = [
        "//litert/vendors/amlogic/core:aml_compiler_core_loader",
        "//litert/vendors/amlogic/core:amlogic_dispatch_info",
        "//litert/vendors/c:litert_compiler_plugin_api",
        "@aml_npu_nn_runtime//:headers",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/c/internal:litert_logging_helper_with_compiler_context",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/c:litert_opaque_options",
        "//litert/c:litert_options",
        "//litert/c/options:litert_amlogic_options",
        "//litert/c:litert_op_options",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_model_types",
        "//litert/compiler/cc:litert_model",
        "//litert/core/model",
        "//litert/cc/internal:litert_handle",
        "//litert/cc/internal:litert_op_options",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_shared_library(
    name = "amlogic_compiler_plugin_so",
    shared_lib_name = "libLiteRtCompilerPlugin_Amlogic.so",
    user_link_flags = export_lrt_only_linkopt() + [
        # libc++ is linked statically. Keep it out of .dynsym so
        # libcompiler_adla.so cannot bind locale / iostream to this .so.
        "-Wl,--exclude-libs,libc++_static.a",
        "-Wl,--exclude-libs,libc++abi.a",
        "-Wl,-soname=libLiteRtCompilerPlugin_Amlogic.so",
        # Wheel package places the shared library in the root directory.
        "-Wl,-rpath,$$ORIGIN",
    ],
    additional_linker_inputs = export_lrt_only_script(),
    visibility = [
        "//ci/tools/python/wheel:__subpackages__",
        "//litert:__subpackages__",
    ],
    deps = [":amlogic_compiler_plugin"],
)

filegroup(
    name = "libLiteRtCompilerPlugin_Amlogic.so",
    srcs = [":amlogic_compiler_plugin_so"],
    visibility = [
        "//ci/tools/python/wheel:__subpackages__",
        "//litert:__subpackages__",
    ],
)

litert_test(
    name = "amlogic_compiler_plugin_test",
    srcs = ["compiler_plugin_test.cc"],
    linkopts = select({
        "@org_tensorflow//tensorflow:linux_x86_64": [
            make_rpaths([
                "external/aml_npu_nn_runtime/aml_compiler_core/libs/linux/x86_64",
            ]),
        ],
        "//conditions:default": [],
    }) + ["-ldl"],
    target_compatible_with = select({
        "//litert:android": [],
        "//litert:linux_x86_64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":amlogic_compiler_plugin",
        "//litert/c:litert_common",
        "//litert/c/internal:litert_abi_header",
        "//litert/vendors/c:litert_compiler_plugin",
        "//litert/vendors/c:litert_compiler_plugin_api",
    ],
)
