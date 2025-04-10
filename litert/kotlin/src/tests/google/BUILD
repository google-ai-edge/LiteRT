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

load("//devtools/deviceinfra/api/builddefs/device:rules.bzl", "satellite_lab_device")
load("//devtools/mobileharness/api/builddef/ait:ait.bzl", "android_logcat_decorator")
load(
    "//tools/build_defs/android:rules.bzl",
    "android_binary",
    "android_instrumentation_test",
    "android_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//third_party/odml/litert:__subpackages__",
    ],
)

android_binary(
    name = "litert_dummy_app_for_test",
    testonly = True,
    manifest = "AndroidManifest.xml",
    deps = ["//litert/kotlin:litert_kotlin_api"],
)

android_library(
    name = "litert_kotlin_api_test_lib",
    testonly = True,
    srcs = ["kotlin/com/google/ai/edge/litert/LiteRtTest.kt"],
    deps = [
        "//third_party/android/androidx_test/core",
        "//third_party/android/androidx_test/ext/junit",
        "//third_party/android/androidx_test/runner/android_junit_runner",
        "//third_party/java/junit:junit-android",
        "//third_party/java/truth:truth-android",
        "//third_party/kotlin/kotlinx_coroutines:kotlinx_coroutines_test",
        "//litert/kotlin:litert_kotlin_api",
    ],
)

android_binary(
    name = "litert_test_app",
    testonly = True,
    assets = [
        "//litert/test:testdata/simple_model.tflite",
    ],
    assets_dir = "testdata",
    instruments = ":litert_dummy_app_for_test",
    manifest = "AndroidManifest-test.xml",
    deps = [
        ":litert_kotlin_api_test_lib",
    ],
)

android_instrumentation_test(
    name = "litert_kotlin_api_emulator_test",
    target_device = "//tools/mobile/devices/android/generic_phone:android_21_x86",
    test_app = ":litert_test_app",
    test_class = "com.google.ai.edge.litert.LiteRtTest",
)

android_library(
    name = "litert_kotlin_api_mh_test_lib",
    testonly = True,
    srcs = ["kotlin/com/google/ai/edge/litert/LiteRtMhTest.kt"],
    deps = [
        "//third_party/android/androidx_test/core",
        "//third_party/android/androidx_test/ext/junit",
        "//third_party/android/androidx_test/runner/android_junit_runner",
        "//third_party/java/junit:junit-android",
        "//third_party/java/truth:truth-android",
        "//third_party/kotlin/kotlinx_coroutines:kotlinx_coroutines_test",
        "//litert/kotlin:litert_kotlin_api",
    ],
)

android_binary(
    name = "litert_mh_test_app",
    testonly = True,
    assets = [
        "//litert/test:testdata/simple_model.tflite",
    ],
    assets_dir = "testdata",
    instruments = ":litert_dummy_app_for_test",
    manifest = "AndroidManifest-test.xml",
    deps = [
        ":litert_kotlin_api_mh_test_lib",
    ],
)

# go/omnilab-ait-doc#android-logcat-decorator
android_logcat_decorator(
    name = "logcat_decorator",
    log_filter_specs = ["tflite:V"],
    log_to_file = True,
)

satellite_lab_device(
    name = "odml_test_lab_device",
    dimensions = {
        "label": "odml-test",
    },
    type = "AndroidRealDevice",
)

# blaze test --config=android_arm64 --test_output=streamed --notest_loasd --nocache_test_results \
#   //litert/kotlin:litert_kotlin_api_mh_test
android_instrumentation_test(
    name = "litert_kotlin_api_mh_test",
    timeout = "long",
    args = [
        "--allocation_exit_strategy=FAIL_FAST_NO_MATCH",
        "--run_as=odml-team",
    ],
    tags = [
        "local",
        "manual",
        "notap",
    ],
    target_device = ":odml_test_lab_device",
    test_app = ":litert_mh_test_app",
    test_class = "com.google.ai.edge.litert.LiteRtMhTest",
    utp_host_plugins = [
        ":logcat_decorator",
    ],
)

# MH tests for NPU
android_library(
    name = "litert_kotlin_api_npu_mh_test_lib",
    testonly = True,
    srcs = ["kotlin/com/google/ai/edge/litert/LiteRtNpuMhTest.kt"],
    deps = [
        "//third_party/android/androidx_test/core",
        "//third_party/android/androidx_test/ext/junit",
        "//third_party/android/androidx_test/runner/android_junit_runner",
        "//third_party/java/junit:junit-android",
        "//third_party/java/truth:truth-android",
        "//third_party/kotlin/kotlinx_coroutines:kotlinx_coroutines_test",
        "//litert/kotlin:litert_kotlin_api",
    ],
)

android_binary(
    name = "litert_npu_mh_test_app",
    testonly = True,
    instruments = ":litert_dummy_app_for_test",
    manifest = "AndroidManifest-test.xml",
    deps = [
        ":litert_kotlin_api_npu_mh_test_lib",
        "//litert/kotlin/src/tests/google/npu_models:qualcomm_v75",
        "//litert/vendors/qualcomm/google:qualcomm_npu_runtime_v75",
    ],
)

satellite_lab_device(
    name = "odml_test_lab_qualcomm_device",
    dimensions = {
        "label": "odml-test",
    },
    models = ["sm-s921u"],  # Samsung S24
    type = "AndroidRealDevice",
)

# blaze test --config=android_arm64 --test_output=streamed --notest_loasd --nocache_test_results \
#   //litert/kotlin:litert_kotlin_api_npu_mh_test
android_instrumentation_test(
    name = "litert_kotlin_api_npu_mh_test",
    timeout = "long",
    args = [
        "--allocation_exit_strategy=FAIL_FAST_NO_MATCH",
        "--run_as=odml-team",
    ],
    tags = [
        "local",
        "manual",
        "notap",
    ],
    target_device = ":odml_test_lab_qualcomm_device",
    test_app = ":litert_npu_mh_test_app",
    test_class = "com.google.ai.edge.litert.LiteRtNpuMhTest",
    utp_host_plugins = [
        ":logcat_decorator",
    ],
)
