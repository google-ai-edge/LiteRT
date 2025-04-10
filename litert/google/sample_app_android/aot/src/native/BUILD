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

load("//mobile/build:build_defs.bzl", "android_jni_library")
load("//third_party/arcore/build_defs:android_rules.bzl", "android_cc_library")
load("//tools/build_defs/java_native_headers:java_native_headers.bzl", "java_native_headers")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/google/sample_app_android/aot:__subpackages__"],
)

java_native_headers(
    name = "jni_headers",
    outs = [
        "org_odml_litert_litert_aot_LiteRtJni.h",
    ],
    dep = "//litert/google/sample_app_android/aot:testapp_lib_no_jni",
)

android_cc_library(
    name = "utils",
    hdrs = ["utils.h"],
    linkopts = ["-landroid"],
    deps = [
        "//util/java:jni_helper",
    ],
)

android_jni_library(
    name = "litert_native",
    srcs = [
        "org_odml_litert_litert_aot_LiteRtJni.cc",
        "org_odml_litert_litert_aot_LiteRtJni.h",
    ],
    force_android = True,
    lds = "jni.lds",
    linkopts = ["-landroid"],
    target_compatible_with = select({
        "@platforms//os:android": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":utils",
        "//litert/cc:litert_compiled_model",
        "//litert/core/model:model_buffer",
        "//util/java:jni_helper",
    ],
)
