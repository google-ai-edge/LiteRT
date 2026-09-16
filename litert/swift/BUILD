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
