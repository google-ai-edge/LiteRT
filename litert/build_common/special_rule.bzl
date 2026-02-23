# Copyright 2025 Google LLC.
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

"""External versions of LiteRT build rules that differ outside of Google."""

def litert_cc_users():
    """Visibility to LiteRT C++ API.

    Return the package group declaration for internal code locations that need
    visibility to LiteRT C++ API."""

    return []

def litert_friends():
    """Internal visibility for packages outside of LiteRT code location.

    Return the package group declaration for internal code locations that need
    visibility to all LiteRT APIs"""

    return []

def gl_native_deps():
    """This is a no-op outside of Google."""
    return []

def gles_deps():
    """This is a no-op outside of Google."""
    return []

def gles_headers():
    """This is a no-op outside of Google."""
    return []

def gles_linkopts():
    return select({
        "@platforms//os:android": [
            "-lGLESv3",
            "-lEGL",
        ],
        "//conditions:default": [],
    })

def litert_android_linkopts():
    return select({
        "//litert:litert_android_no_jni": ["-lnativewindow"],
        "@platforms//os:android": ["-landroid"],
        "//conditions:default": [],
    })

def litert_metal_opts():
    return select({
        "@platforms//os:ios": ["-ObjC++"],
        "@platforms//os:macos": ["-ObjC++"],
        "//conditions:default": [],
    })

def litert_metal_linkopts():
    """This is a no-op outside of Google."""
    return []

def litert_metal_deps_without_gpu_environment():
    return select({
        "@platforms//os:ios": ["//tflite/delegates/gpu/metal:metal_device"],
        "@platforms//os:macos": ["//tflite/delegates/gpu/metal:metal_device"],
        "//conditions:default": [],
    })

def litert_metal_deps():
    return litert_metal_deps_without_gpu_environment() + select({
        "@platforms//os:ios": ["//litert/runtime:metal_info"],
        "@platforms//os:macos": ["//litert/runtime:metal_info"],
        "//conditions:default": [],
    })

# Dependencies for GPU accelerators for each platform.
def litert_gpu_accelerator_deps():
    return []

# Prebuilt dependencies for GPU accelerators for each platform.
def litert_gpu_accelerator_prebuilts():
    return select({
        "//litert:linux_x86_64": [
            "@litert_prebuilts//:linux_x86_64/libLiteRtWebGpuAccelerator.so",  # copybara:comment
        ],
        "//litert:linux_aarch64": [
            "@litert_prebuilts//:linux_arm64/libLiteRtWebGpuAccelerator.so",  # copybara:comment
        ],
        "//litert:macos_arm64": [
            "@litert_prebuilts//:macos_arm64/libLiteRtMetalAccelerator.dylib",  # copybara:comment
        ],
        "@platforms//os:windows": [
            "@litert_prebuilts//:windows_x86_64/libLiteRtWebGpuAccelerator.dll",  # copybara:comment
        ],
        "//litert:android_arm64": [
            "@litert_prebuilts//:android_arm64/libLiteRtClGlAccelerator.so",  # copybara:comment
        ],
        "//conditions:default": [],
    })

# Prebuilt dependencies for LiteRT runtime shared library for each platform.
def litert_runtime_prebuilts():
    return select({
        "//litert:linux_x86_64": [
            "@litert_prebuilts//:linux_x86_64/libLiteRt.so",  # copybara:comment
        ],
        "//litert:linux_aarch64": [
            "@litert_prebuilts//:linux_arm64/libLiteRt.so",  # copybara:comment
        ],
        "//litert:macos_arm64": [
            "@litert_prebuilts//:macos_arm64/libLiteRt.dylib",  # copybara:comment
        ],
        "@platforms//os:windows": [
            "@litert_prebuilts//:windows_x86_64/libLiteRt.dll",  # copybara:comment
        ],
        "//litert:android_arm64": [
            "@litert_prebuilts//:android_arm64/libLiteRt.so",  # copybara:comment
        ],
        "//conditions:default": [],
    })
