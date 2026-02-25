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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//litert/ats:ats.bzl", "litert_define_ats")
load("//litert/build_common:litert_build_defs.bzl", "commandline_flag_copts", "if_oss", "litert_test")
load("//litert/integration_test:litert_device.bzl", "litert_device_test")
load("//litert/integration_test:litert_device_script.bzl", "make_download_model_provider")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",  # copybara:comment_replace "//litert:litert_public",
    ],
)

# SUITES ###########################################################################################

litert_define_ats(
    name = "cpu_ats",
    backend = "cpu",
    compile_only_suffix = None,
    dont_register = [
        "hf_all_minilm_l6_v2",
        "u2net_lite",
        "whisper_tiny_decoder",
        "yolo11n",
    ],
    jit_suffix = "",
)

litert_define_ats(
    name = "example_ats",
    backend = "example",
    compile_only_suffix = "_aot",
    do_register = [
        "sub.*f32",
        "mul.*f32",
    ],
    dont_register = [
        "ExtraModel",
    ],
    jit_suffix = "",
)

litert_define_ats(
    name = "qualcomm_ats",
    backend = "qualcomm",
    compile_only_suffix = "_aot",
    do_register = [
        "SingleOp",
        "hf_all_minilm_l6_v2",
        "hf_mobilevit_small",
        "qai_hub_midas",
        "qai_hub_real_esrgan_x4plus",
        "torchvision_mobilenet_v2",
        "u2net_lite",
        "torchvision_resnet18",
        "torchvision_squeezenet1_1",
        "whisper_tiny_encoder",
        "yamnet",
        "yolo11n",
    ],
    dont_register = [
        "hf_all_minilm_l6_v2",
        "whisper_tiny_decoder",
        ".*5d.*",
        ".*6d.*",
    ],
    extra_flags = [
        "--data_seed=42",
        "--qualcomm_optimization_level=O3",
        "--qualcomm_use_htp_preference=true",
        "--qualcomm_htp_performance_mode=burst",
        "--qualcomm_log_level=info",
        "--qualcomm_use_fold_relu=true",
        "--qualcomm_use_conv_hmx=true",
    ],
    jit_suffix = "",
)

litert_define_ats(
    name = "mediatek_ats",
    backend = "mediatek",
    compile_only_suffix = "_aot",
    do_register = [
        "SingleOp",
        "hf_mobilevit_small",
        "qai_hub_midas",
        "torchvision_mobilenet_v2",
        "u2net_lite",
        "torchvision_resnet18",
        "torchvision_squeezenet1_1",
        "yamnet",
        "yolo11n",
    ],
    dont_register = [
        ".*5d.*",
        ".*6d.*",
        "whisper_tiny_decoder",
        "whisper_tiny_encoder",
        "hf_all_minilm_l6_v2",
        "qai_hub_real_esrgan_x4plus",
    ],
    extra_flags = [
        "--data_seed=42",
        "--mediatek_enable_gemma_compiler_optimizations=true",
        "--mediatek_performance_mode_type=turbo_boost",
        "--mediatek_optimization_hint=low_latency",
        "--mediatek_enable_l1_cache_optimizations=true",
    ],
    jit_suffix = "",
)

litert_define_ats(
    name = "intel_openvino_ats",
    backend = "intel_openvino",
    compile_only_suffix = None,
    do_register = [
        "SingleOp",
        "hf_all_minilm_l6_v2",
        "hf_mobilevit_small",
        "qai_hub_real_esrgan_x4plus",
        "torchvision_mobilenet_v2",
        "u2net_lite",
        "torchvision_resnet18",
        "torchvision_squeezenet1_1",
        "whisper_tiny_encoder",
        "yolo11n",
    ],
    dont_register = [
        ".*SingleOp_inference_BinaryNoBroadcast.*",
        ".*SingleOp_inference_Unary.*",
        "qai_hub_midas",
        "yamnet",
    ],
    jit_suffix = "",
)

# SOURCE CODES #####################################################################################

litert_test(
    name = "ats",
    srcs = ["ats.cc"],
    copts = commandline_flag_copts(),
    linkstatic = True,
    no_main = True,
    tags = [
        "notap",
    ],
    ungrte = if_oss(True, False),
    use_sys_malloc = True,
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/c:litert_op_code",
        "//litert/c/internal:litert_logging",
        "//litert/cc/internal:litert_c_types_printing",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators",
        "//litert/test/generators:common",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "configure",
    testonly = True,
    srcs = ["configure.cc"],
    hdrs = ["configure.h"],
    copts = commandline_flag_copts() + [
        "-DINCLUDE_QUALCOMM_COMPILE_FLAGS",
        "-DINCLUDE_QUALCOMM_RUNTIME_FLAGS",
        "-DINCLUDE_MEDIATEK_COMPILE_FLAGS",
        "-DINCLUDE_MEDIATEK_RUNTIME_FLAGS",
    ],
    deps = [
        ":common",
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_common",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_options",
        "//litert/cc/internal:litert_rng",
        "//litert/compiler/plugin:compiler_plugin",
        "//litert/core:filesystem_testonly",
        "//litert/core/model:model_serialize",
        "//litert/tools/flags/vendors:mediatek_flags",
        "//litert/tools/flags/vendors:qualcomm_flags",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "executor",
    testonly = True,
    hdrs = ["executor.h"],
    deps = [
        ":inference_capture",
        "//litert/c:litert_common",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_options",
        "//litert/cc/internal:litert_compiled_model_next",
        "//litert/test:simple_buffer",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "register",
    testonly = True,
    hdrs = ["register.h"],
    deps = [
        ":common",
        ":configure",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/internal:litert_detail",
        "//litert/cc/internal:litert_rng",
        "//litert/core:filesystem",
        "//litert/test/generators",
    ],
)

cc_test(
    name = "executor_test",
    srcs = ["executor_test.cc"],
    deps = [
        ":executor",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_common",
        "//litert/cc:litert_options",
        "//litert/core/model",
        "//litert/test:matchers",
        "//litert/test:simple_buffer",
        "//litert/test/generators:graph_helpers",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "inference_fixture",
    testonly = True,
    hdrs = ["inference_fixture.h"],
    deps = [
        ":common",
        ":configure",
        ":executor",
        ":inference_capture",
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_c_types_printing",
        "//litert/core/model",
        "//litert/test:matchers_oss",
        "//litert/test:rng_fixture_oss",
        "//litert/test:simple_buffer",
        "//litert/test/generators:common",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "compile_fixture",
    testonly = True,
    hdrs = ["compile_fixture.h"],
    deps = [
        ":capture_common",
        ":common",
        ":compile_capture",
        ":configure",
        "//litert/c/internal:litert_logging",
        "//litert/cc/internal:litert_c_types_printing",
        "//litert/compiler/plugin:compiler_plugin",
        "//litert/core:filesystem_testonly",
        "//litert/core/model",
        "//litert/test:matchers_oss",
        "//litert/test/generators:common",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

litert_device_test(
    name = "executor_test_npu",
    srcs = ["executor_test.cc"],
    backend_id = "qualcomm",
    data = ["//litert/test:testdata/simple_model.tflite"],
    defines = ["_TEST_NPU"],
    deps = [
        ":executor",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_common",
        "//litert/cc:litert_options",
        "//litert/core/model",
        "//litert/test:matchers",
        "//litert/test:simple_buffer",
        "//litert/test/generators:graph_helpers",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "check_ats",
    srcs = ["check_ats.cc"],
    copts = commandline_flag_copts(),
    data = [
        "//litert/test:testdata/simple_model.tflite",
        "//litert/vendors/examples:example_dispatch_so",
        "//litert/vendors/examples:example_plugin_so",
    ],
    # Re-enable sans once test libs can be linked in a sans friendly way.
    tags = [
        "noasan",
        "nomsan",
        "nosan",
        "notsan",
    ],
    deps = [
        ":common",
        ":compile_fixture",
        ":configure",
        ":executor",
        ":inference_fixture",
        ":register",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_detail",
        "//litert/core:filesystem",
        "//litert/core/model",
        "//litert/core/model:model_load",
        "//litert/test:common",
        "//litert/test:simple_buffer",
        "//litert/test/generators",
        "//litert/test/generators:common",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/flags:reflection",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "inference_capture",
    testonly = True,
    hdrs = ["inference_capture.h"],
    deps = [
        ":capture_common",
        ":common",
        ":print",
        "//litert/cc/internal:litert_detail",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "inference_capture_test",
    srcs = ["inference_capture_test.cc"],
    deps = [
        ":inference_capture",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "common",
    testonly = True,
    hdrs = ["common.h"],
    deps = [
        "//litert/core/model",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "print",
    testonly = True,
    hdrs = ["print.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "print_test",
    srcs = ["print_test.cc"],
    deps = [
        ":print",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "compile_capture",
    testonly = True,
    hdrs = ["compile_capture.h"],
    deps = [
        ":capture_common",
        ":common",
        ":print",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "compile_capture_test",
    srcs = ["compile_capture_test.cc"],
    deps = [
        ":common",
        ":compile_capture",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "capture_common",
    testonly = True,
    hdrs = ["capture_common.h"],
    deps = [
        ":common",
        ":configure",
        ":print",
        "//litert/core/model",
        "@com_google_absl//absl/strings:string_view",
    ],
)

make_download_model_provider(
    name = "ats_models_provider",
    url = "https://storage.googleapis.com/litert/ats_models.tar.gz",
)
