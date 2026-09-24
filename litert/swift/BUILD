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

load("@build_bazel_rules_apple//apple:apple.bzl", "apple_static_xcframework", "apple_universal_binary", "apple_xcframework")
load("@build_bazel_rules_apple//apple:macos.bzl", "macos_dylib")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_interop_hint", "swift_library", "swift_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//litert:version.bzl", "LITERT_EXPERIMENTAL_VERSION")
load("//litert/build_common:ios.bzl", "strip_common_include_path_prefix")
load("//litert/build_common:special_rule.bzl", "litert_gpu_accelerator_prebuilts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "CLiteRT_objc",
    hdrs = ["Sources/LiteRT/LiteRtC.h"],
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
        "//litert/c/options:litert_cpu_options",
    ],
    alwayslink = True,
)

swift_interop_hint(
    name = "CLiteRT_swift_hint",
    module_name = "CLiteRT",
)

swift_library(
    name = "LiteRT_Swift",
    srcs = glob(["Sources/LiteRT/*.swift"]),
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
    srcs = glob(["Tests/LiteRT/*.swift"]),
    data = [
        "//litert/test:testdata/dynamic_add_model.tflite",
        "//litert/test:testdata/simple_model.tflite",
        "//litert/test:testdata/simple_quantized_ops.tflite",
    ] + litert_gpu_accelerator_prebuilts(),
    deps = [
        ":CLiteRT_objc",
        ":LiteRT_Swift",
        # Dependency placeholder: apple_frameworks Metal,
        # Dependency placeholder: swift foundation,
        "//third_party/swift/swift-testing",
        # Dependency placeholder: swift xctest,
    ],
)

# The TensorFlowLiteC module, copied from
# //tflite/ios:tensorflow_lite_c. It is duplicated rather
# than depended on because //tflite/ios is not exported to
# the open-source LiteRT repository, so the Swift package has to be able to
# define the module itself. Only the module definition is duplicated; the
# underlying C libraries are still depended on directly.
#
# The dep list matches upstream except for
# //tflite/profiling/telemetry/c:telemetry_setting, which
# build_cleaner drops here: the umbrella header does not include
# telemetry_setting.h directly, and it arrives transitively through :profiler.
cc_library(
    name = "TensorFlowLiteC_objc",
    # Every header this module vends is listed explicitly, not just the
    # umbrella, so that the module map generated from `hdrs` claims ownership
    # of them.
    hdrs = [
        "Sources/TensorFlowLiteC/TensorFlowLiteC.h",
        "//tflite:builtin_ops.h",
        "//tflite/converter/core/c:tflite_types.h",
        "//tflite/core/async/c:types.h",
        "//tflite/core/c:c_api.h",
        "//tflite/core/c:c_api_experimental.h",
        "//tflite/core/c:c_api_opaque.h",
        "//tflite/core/c:c_api_types.h",
        "//tflite/core/c:common.h",
        "//tflite/core/c:operator.h",
        "//tflite/delegates/xnnpack:xnnpack_delegate.h",
        "//tflite/profiling/telemetry/c:profiler.h",
        "//tflite/profiling/telemetry/c:telemetry_setting.h",
    ],
    aspect_hints = [":TensorFlowLiteC_swift_hint"],
    tags = [
        "nobuilder",
    ],
    deps = [
        "//tflite:builtin_ops",
        "//tflite/c:c_api",
        "//tflite/c:c_api_types",
        "//tflite/converter/core/c:tflite_types",
        "//tflite/core/async/c:types",
        "//tflite/core/c:c_api",
        "//tflite/core/c:c_api_experimental",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/core/c:operator",
        "//tflite/delegates/xnnpack:xnnpack_delegate",
        "//tflite/profiling/telemetry/c:profiler",
    ],
)

swift_interop_hint(
    name = "TensorFlowLiteC_swift_hint",
    module_name = "TensorFlowLiteC",
)

# An `apple_static_xcframework` flattens every header into one directory, so the
# `#include` paths have to be rewritten to match. Mirrors
# `strip_common_include_path_core` in //tflite/ios.
strip_common_include_path_prefix(
    name = "TensorFlowLiteC_headers",
    hdr_labels = [
        "//tflite/converter/core/c:tflite_types.h",
        "//tflite:builtin_ops.h",
        "//tflite/core/async/c:types.h",
        "//tflite/core/c:c_api.h",
        "//tflite/core/c:c_api_experimental.h",
        "//tflite/core/c:c_api_opaque.h",
        "//tflite/core/c:c_api_types.h",
        "//tflite/core/c:common.h",
        "//tflite/core/c:operator.h",
        "//tflite/delegates/xnnpack:xnnpack_delegate.h",
        "//tflite/profiling/telemetry/c:profiler.h",
        "//tflite/profiling/telemetry/c:telemetry_setting.h",
    ],
)

config_setting(
    name = "use_coreml_delegate",
    define_values = {"use_coreml_delegate": "1"},
)

config_setting(
    name = "use_metal_delegate",
    define_values = {"use_metal_delegate": "1"},
)

# By default this builds with no delegates.
# To build with the Metal delegate pass --define=use_metal_delegate=1
# To build with the CoreML delegate pass --define=use_coreml_delegate=1
swift_library(
    name = "TensorFlowLite_Swift",
    srcs = glob(
        ["Sources/TensorFlowLite/*.swift"],
        exclude = [
            "Sources/TensorFlowLite/CoreMLDelegate.swift",
            "Sources/TensorFlowLite/MetalDelegate.swift",
        ],
    ) + select({
        ":use_coreml_delegate": ["Sources/TensorFlowLite/CoreMLDelegate.swift"],
        "//conditions:default": [],
    }) + select({
        ":use_metal_delegate": ["Sources/TensorFlowLite/MetalDelegate.swift"],
        "//conditions:default": [],
    }),
    # `library_evolution = True` below makes swiftc emit a `.swiftinterface`,
    # which needs both of these flags:
    #
    # `-swift-version 5` pins the language mode recorded in the interface's
    # `swift-module-flags` line. Without it swiftc warns that "emitting module
    # interface files requires '-language-mode'" and the shipped interface is
    # parsed in whatever mode the consumer's toolchain defaults to, which breaks
    # once that default becomes Swift 6.
    #
    # `-no-verify-emitted-module-interface` skips the post-compile verification
    # pass, the equivalent of Xcode's SWIFT_VERIFY_EMITTED_MODULE_INTERFACE=NO.
    # The interface exposes `TensorFlowLiteC` types publicly, so it begins with
    # `import TensorFlowLiteC`, but that module is only visible to this compile
    # through `-Xcc -fmodule-map-file=`, and `-Xcc` flags are deliberately not
    # recorded in the interface. The verification sub-invocation therefore fails
    # with "no such module 'TensorFlowLiteC'". Consumers are unaffected: they
    # resolve the module from `:TensorFlowLiteC` below, which is
    # where the interface is compiled for real.
    copts = [
        "-swift-version",
        "5",
        "-no-verify-emitted-module-interface",
    ],
    library_evolution = True,
    linkopts = select({
        ":use_coreml_delegate": ["-Wl,-weak_framework,CoreML"],
        "//conditions:default": [],
    }) + select({
        ":use_metal_delegate": ["-Wl,-weak_framework,Metal"],
        "//conditions:default": [],
    }),
    module_name = "TensorFlowLite",
    deps = [
        ":TensorFlowLiteC_objc",
        # Dependency placeholder: apple_frameworks Darwin,
        # Dependency placeholder: swift foundation,
    ] + select({
        ":use_coreml_delegate": ["//tflite/delegates/coreml:coreml_delegate"],
        "//conditions:default": [],
    }) + select({
        ":use_metal_delegate": ["//tflite/delegates/gpu:metal_delegate"],
        "//conditions:default": [],
    }),
)

# bazel build -c opt --config=ios //litert/swift:TensorFlowLite
apple_static_xcframework(
    name = "TensorFlowLite",
    # The C runtime is deliberately kept out of this bundle and shipped as
    # `:TensorFlowLiteC` below. An `apple_static_xcframework` built
    # from Swift sources can only vend a single module -- the one named by
    # `bundle_name` -- so there is no way to also carry the `TensorFlowLiteC`
    # headers and modulemap in here, and `TensorFlowLite.swiftinterface`
    # imports `TensorFlowLiteC` (see Sources/TensorFlowLite/Delegate.swift,
    # which exposes `TfLiteDelegate` publicly). Linking the runtime in anyway
    # would bloat the archive and cause duplicate symbols for apps that also
    # use the C or Objective-C API.
    avoid_deps = [
        ":TensorFlowLiteC_objc",
    ],
    bundle_name = "TensorFlowLite",
    ios = {
        "simulator": [
            "x86_64",
            "arm64",
        ],
        "device": ["arm64"],
    },
    minimum_os_versions = {"ios": "15.0"},
    deps = [
        ":TensorFlowLite_Swift",
    ],
)

# The TensorFlow Lite C runtime that `:TensorFlowLite` links against,
# packaged so that the `TensorFlowLiteC` binary target in Package.swift has an
# artifact to resolve. Symbol hiding is not applied, matching
# `//tflite/ios:TensorFlowLiteC_static_framework`.
#
# bazel build -c opt --config=ios //litert/swift:TensorFlowLiteC
apple_static_xcframework(
    name = "TensorFlowLiteC",
    bundle_name = "TensorFlowLiteC",
    ios = {
        "simulator": [
            "x86_64",
            "arm64",
        ],
        "device": ["arm64"],
    },
    minimum_os_versions = {"ios": "15.0"},
    public_hdrs = [
        ":TensorFlowLiteC_headers",
    ],
    deps = [
        ":TensorFlowLiteC_objc",
    ],
)

# The full source list, including both delegates. Consumed by the all-delegates
# variant that the tests build against, which lives in its own package so the
# two `TensorFlowLite` modules don't share an output directory.
filegroup(
    name = "TensorFlowLite_Swift_srcs",
    srcs = glob(["Sources/TensorFlowLite/*.swift"]),
    visibility = ["//litert/swift/Tests/TensorFlowLite:__pkg__"],
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
        "//litert/c/options:litert_cpu_options.h",
    ],
)

# Fills in the version keys of the framework Info.plist from the LiteRT release
# version, so the framework reports the same version as the other release
# artifacts (Maven, Python wheel) instead of a hardcoded placeholder.
genrule(
    name = "CLiteRT_info_plist",
    srcs = ["Sources/LiteRT/Info.plist"],
    outs = ["CLiteRT_Info.plist"],
    cmd = "sed 's/@VERSION@/{version}/g' $< > $@".format(
        version = LITERT_EXPERIMENTAL_VERSION,
    ),
)

apple_xcframework(
    name = "CLiteRT",
    bundle_id = "com.google.odml.litert.CLiteRT",
    bundle_name = "CLiteRT",
    features = ["exported_symbols"],
    infoplists = [":CLiteRT_info_plist"],
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
    visibility = ["//visibility:public"],
    deps = [
        ":CLiteRT_objc",
    ],
)

apple_static_xcframework(
    name = "CLiteRT_static",
    bundle_name = "CLiteRT",
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
    visibility = ["//visibility:public"],
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
    ] + litert_gpu_accelerator_prebuilts(),
    outs = ["CLiteRT_mac.zip"],
    cmd = """
       mkdir -p CLiteRT_mac/Headers
       # Copy the universal dylib
       cp $(location :CLiteRT_universal) CLiteRT_mac/libCLiteRT_mac.dylib
       # Copy the accelerator dylib if present
       for FILE in $(SRCS); do
         if [[ "$$FILE" == *.dylib && "$$FILE" != *CLiteRT* ]]; then
           cp $$FILE CLiteRT_mac/
         fi
       done
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

# copybara:comment_begin(oss-only)
genrule(
    name = "LiteRtMetalAccelerator",
    srcs = [
        "@litert_prebuilts//:ios_arm64/libLiteRtMetalAccelerator.dylib",
        "@litert_prebuilts//:ios_sim_arm64/libLiteRtMetalAccelerator.dylib",
    ],
    outs = ["LiteRtMetalAccelerator.xcframework.zip"],
    # The prebuilt Metal accelerator is shipped as a plain dylib
    # (libLiteRtMetalAccelerator.dylib), but a framework bundle must hold an
    # executable named exactly like the bundle itself. Tools consuming the
    # xcframework rely on that: rules_apple, for instance, only treats a file
    # as the framework binary when its basename equals the bundle name (see
    # `_classify_framework_imports` in rules_apple's
    # apple/internal/framework_import_support.bzl), and fails the import
    # otherwise. The dylib is therefore renamed on its way into the bundle,
    # its install name is rewritten to match, and the Info.plists are filled
    # in with the keys Xcode and App Store validation expect from an
    # embeddable dynamic framework.
    cmd = """
       OUT_ZIP=$$(pwd)/$@
       XCFRAMEWORK=metal_tmp/LiteRtMetalAccelerator.xcframework
       mkdir -p $$XCFRAMEWORK/ios-arm64/LiteRtMetalAccelerator.framework
       mkdir -p $$XCFRAMEWORK/ios-arm64-simulator/LiteRtMetalAccelerator.framework
       cp $(location @litert_prebuilts//:ios_arm64/libLiteRtMetalAccelerator.dylib) $$XCFRAMEWORK/ios-arm64/LiteRtMetalAccelerator.framework/LiteRtMetalAccelerator
       cp $(location @litert_prebuilts//:ios_sim_arm64/libLiteRtMetalAccelerator.dylib) $$XCFRAMEWORK/ios-arm64-simulator/LiteRtMetalAccelerator.framework/LiteRtMetalAccelerator
       # The prebuilts are linked with an install name of
       # `@rpath/libLiteRtMetalAccelerator.dylib`. The linker copies that
       # string verbatim into every app that links the framework, so it has to
       # name the binary at its location inside the bundle, otherwise dyld
       # cannot find it at launch. install_name_tool refreshes the ad-hoc
       # linker signature of the arm64 slices on its own, so no separate
       # codesign step is needed.
       for SLICE in ios-arm64 ios-arm64-simulator; do
         BINARY=$$XCFRAMEWORK/$$SLICE/LiteRtMetalAccelerator.framework/LiteRtMetalAccelerator
         chmod +w $$BINARY
         xcrun install_name_tool -id @rpath/LiteRtMetalAccelerator.framework/LiteRtMetalAccelerator $$BINARY
       done
       # The version keys carry the LiteRT release version, matching the
       # CLiteRT framework shipped alongside this one.
       cat << 'EOF' > metal_tmp/framework_Info.plist
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>LiteRtMetalAccelerator</string>
    <key>CFBundleIdentifier</key>
    <string>com.google.odml.litert.LiteRtMetalAccelerator</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>LiteRtMetalAccelerator</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleShortVersionString</key>
    <string>{version}</string>
    <key>CFBundleSupportedPlatforms</key>
    <array>
        <string>@PLATFORM@</string>
    </array>
    <key>CFBundleVersion</key>
    <string>{version}</string>
    <key>MinimumOSVersion</key>
    <string>15.0</string>
</dict>
</plist>
EOF
       sed 's/@PLATFORM@/iPhoneOS/' metal_tmp/framework_Info.plist > $$XCFRAMEWORK/ios-arm64/LiteRtMetalAccelerator.framework/Info.plist
       sed 's/@PLATFORM@/iPhoneSimulator/' metal_tmp/framework_Info.plist > $$XCFRAMEWORK/ios-arm64-simulator/LiteRtMetalAccelerator.framework/Info.plist
       cat << 'EOF' > $$XCFRAMEWORK/Info.plist
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>AvailableLibraries</key>
    <array>
        <dict>
            <key>BinaryPath</key>
            <string>LiteRtMetalAccelerator.framework/LiteRtMetalAccelerator</string>
            <key>LibraryIdentifier</key>
            <string>ios-arm64</string>
            <key>LibraryPath</key>
            <string>LiteRtMetalAccelerator.framework</string>
            <key>SupportedArchitectures</key>
            <array>
                <string>arm64</string>
            </array>
            <key>SupportedPlatform</key>
            <string>ios</string>
        </dict>
        <dict>
            <key>BinaryPath</key>
            <string>LiteRtMetalAccelerator.framework/LiteRtMetalAccelerator</string>
            <key>LibraryIdentifier</key>
            <string>ios-arm64-simulator</string>
            <key>LibraryPath</key>
            <string>LiteRtMetalAccelerator.framework</string>
            <key>SupportedArchitectures</key>
            <array>
                <string>arm64</string>
            </array>
            <key>SupportedPlatform</key>
            <string>ios</string>
            <key>SupportedPlatformVariant</key>
            <string>simulator</string>
        </dict>
    </array>
    <key>CFBundlePackageType</key>
    <string>XFWK</string>
    <key>XCFrameworkFormatVersion</key>
    <string>1.0</string>
</dict>
</plist>
EOF
       cd metal_tmp && zip -q -r "$$OUT_ZIP" LiteRtMetalAccelerator.xcframework && cd ..
       rm -rf metal_tmp
    """.format(version = LITERT_EXPERIMENTAL_VERSION),
    visibility = ["//visibility:public"],
)
# copybara:comment_end
