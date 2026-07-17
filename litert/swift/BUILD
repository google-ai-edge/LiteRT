# Copyright 2026 Google LLC.
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

load("@build_bazel_rules_apple//apple:apple.bzl", "apple_universal_binary", "apple_xcframework")
load("@build_bazel_rules_apple//apple:macos.bzl", "macos_dylib")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_interop_hint", "swift_library", "swift_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//litert/build_common:ios.bzl", "strip_common_include_path_prefix")
load("//litert/build_common:special_rule.bzl", "litert_gpu_accelerator_prebuilts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "CLiteRT_objc",
    hdrs = ["Sources/LiteRtC.h"],
    aspect_hints = [":CLiteRT_swift_hint"],
    deps = [
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_compiled_model",
        "//litert/c:litert_custom_op_kernel",
        "//litert/c:litert_environment",
        "//litert/c:litert_environment_options",
        "//litert/c:litert_layout",
        "//litert/c:litert_model",
        "//litert/c:litert_model_types",
        "//litert/c:litert_opaque_options",
        "//litert/c:litert_options",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_requirements",
    ],
    alwayslink = True,
)

swift_interop_hint(
    name = "CLiteRT_swift_hint",
    module_name = "CLiteRT",
)

swift_library(
    name = "LiteRT_Swift",
    srcs = glob(["Sources/*.swift"]),
    copts = ["-enable-testing"],
    module_name = "LiteRT",
    deps = [
        ":CLiteRT_objc",
        # Dependency placeholder: apple_frameworks Darwin,
        # Dependency placeholder: swift foundation,
    ],
)

swift_test(
    name = "LiteRT_Swift_Tests",
    srcs = glob(["Tests/*.swift"]),
    data = [
        "//litert/test:testdata/simple_model.tflite",
    ] + litert_gpu_accelerator_prebuilts(),
    deps = [
        ":CLiteRT_objc",
        ":LiteRT_Swift",
        # Dependency placeholder: apple_frameworks Metal,
        # Dependency placeholder: swift foundation,
        # Dependency placeholder: swift xctest,
    ],
)

strip_common_include_path_prefix(
    name = "CLiteRT_headers",
    hdr_labels = [
        "//litert/build_common:build_config.h",
        "//litert/c:litert_any.h",
        "//litert/c:litert_common.h",
        "//litert/c:litert_compiled_model.h",
        "//litert/c:litert_custom_op_kernel.h",
        "//litert/c:litert_custom_tensor_buffer.h",
        "//litert/c:litert_environment.h",
        "//litert/c:litert_environment_options.h",
        "//litert/c:litert_gl_types.h",
        "//litert/c:litert_layout.h",
        "//litert/c:litert_model.h",
        "//litert/c:litert_model_types.h",
        "//litert/c:litert_op_code.h",
        "//litert/c:litert_opaque_options.h",
        "//litert/c:litert_opencl_types.h",
        "//litert/c:litert_options.h",
        "//litert/c:litert_tensor_buffer.h",
        "//litert/c:litert_tensor_buffer_requirements.h",
        "//litert/c:litert_tensor_buffer_types.h",
        "//litert/c:litert_webgpu_types.h",
        "//litert/c/internal:litert_scheduling_info.h",
    ],
)

apple_xcframework(
    name = "CLiteRT",
    bundle_id = "com.google.odml.litert.CLiteRT",
    features = ["exported_symbols"],
    infoplists = ["Sources/Info.plist"],
    ios = {
        "simulator": [
            "arm64",
        ],
        "device": [
            "arm64",
        ],
    },
    minimum_os_versions = {
        "ios": "15.0",
    },
    public_hdrs = [
        ":CLiteRT_headers",
    ],
    deps = [
        ":CLiteRT_objc",
    ],
)

macos_dylib(
    name = "CLiteRT_dylib",
    features = ["exported_symbols"],
    linkopts = [
        "-undefined dynamic_lookup",
        "-install_name @rpath/libCLiteRT_mac.dylib",
    ],
    minimum_os_version = "12.0",
    visibility = ["//visibility:public"],
    deps = [":CLiteRT_objc"],
)

apple_universal_binary(
    name = "CLiteRT_universal",
    binary = ":CLiteRT_dylib",
    forced_cpus = [
        "arm64",
        "x86_64",
    ],
    minimum_os_version = "12.0",
    platform_type = "macos",
)

genrule(
    name = "CLiteRT_mac",
    srcs = [
        ":CLiteRT_universal",
        ":CLiteRT_headers",
    ],
    outs = ["CLiteRT_mac.zip"],
    cmd = """
       mkdir -p CLiteRT_mac/Headers
       # Copy the universal dylib
       cp $(location :CLiteRT_universal) CLiteRT_mac/libCLiteRT_mac.dylib
       # Copy the headers
       cp $(locations :CLiteRT_headers) CLiteRT_mac/Headers/
       # Create module map
       echo "module CLiteRT {" > CLiteRT_mac/Headers/module.modulemap
       for FILE in $(locations :CLiteRT_headers); do
         echo "   header \\"$$(basename $$FILE)\\"" >> CLiteRT_mac/Headers/module.modulemap
       done
       echo "   export *" >> CLiteRT_mac/Headers/module.modulemap
       echo "}" >> CLiteRT_mac/Headers/module.modulemap
       # Zip it all up
       zip -q -r $@ CLiteRT_mac
       rm -rf CLiteRT_mac
    """,
    visibility = ["//visibility:public"],
)
