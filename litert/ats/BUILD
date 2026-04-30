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

# Note: The `litert_define_ats` targets below define test suites that execute tests on-device
# using the underlying base `ats` binary compiled further down in this file.

litert_define_ats(
    name = "cpu_ats",
    backend = "cpu",
    compile_only_suffix = None,
    dont_register = [
        "^(?!.*(Pad|Reduction|Transpose)).*i32",  # XNNPACK does not support int32 math (except Pad/Reduction/Transpose static constants)
        "OneHot",  # XNNPACK does not support OneHot
        "Transformer",  # XNNPACK does not support TransformerLayer flex/custom ops
        "tfl\\.floor_div",  # XNNPACK does not support FloorDiv
        "tfl\\.pow",  # XNNPACK does not support Pow
        "tfl\\.prelu",  # XNNPACK requires PRelu alpha to be statically allocated
        "tfl\\.zeros_like",  # XNNPACK does not support ZerosLike
        "tfl\\.sign",  # XNNPACK does not support Sign
        "tfl\\.relu_0_to_1",  # XNNPACK does not support Relu0To1
        "tfl\\.elu",  # XNNPACK does not support Elu
        "tfl\\.reduce_prod",  # XNNPACK does not support ReduceProd
        "LogSoftmax",  # XNNPACK does not support LogSoftmax
        "Pooling",  # XNNPACK rejects pooling when stride > filter size
        "FullyConnected.*(i8|u8|SHUFFLED)",  # XNNPACK delegate at ToT does not yet support these FullyConnected presets
        "FullyConnected.*,2d_f32.*,1d_f32",
        "FullyConnected.*,2d_f32[^c]",
        "FullyConnected.*_cst.*,1d_f32[^c]",
        "Transpose.*([(][78]d|[(][1-8]d_(i1|i4|i16|u16|i32|u32|i64|u64)<)",  # XNNPACK Transpose supports FP32, FP16, INT8, and UINT8 types up to rank 6
        # Explicit ExtraModel exclusions (unsupported by XNNPACK):
        "simple_add_dynamic_shape",
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
        "5d",  # Qualcomm QNN HTP only supports up to 4D/6D tensor layouts
        "6d",  # Qualcomm QNN HTP only supports up to 4D/6D tensor layouts
        "7d",  # Qualcomm QNN HTP only supports up to 4D/6D tensor layouts
        "8d",  # Qualcomm QNN HTP only supports up to 4D/6D tensor layouts
        "Binary.*i32",  # Qualcomm QNN does not support int32 binary arithmetic
        "NoOp.*i32",  # Qualcomm QNN does not support int32 NoOp (add)
        "OneHot",  # Qualcomm QNN compiler does not support dynamic depth/axis OneHot
        "tfl[.]zeros_like",  # Qualcomm QNN compiler does not support ZerosLike
        "tfl[.]reduce_prod",  # Qualcomm QNN reduce_prod exhibits numerical accumulation divergence
        "tfl[.]ceil.*f32",  # Qualcomm QNN ceil exhibits -0.0/-1.0 negative zero boundary mismatch on f32
        "Transpose.*[(]1d_",  # Qualcomm QNN Transpose requires multidimensional tensors (rank >= 2)
        "Transpose.*i4",  # Qualcomm QNN MemHandle registration does not support int4 graph I/O buffers
        "Transpose.*[(][1-8]d_(i32|u32|i64|u64)<",  # Qualcomm QNN Transpose does not support 32-bit and 64-bit integer I/O buffers
        "Transpose.*f32",  # Qualcomm QNN executes Transpose in FP16 precision
        "Pooling.*filter_height=14",  # Qualcomm QNN max pool filter size exceeds 11x11 hardware limit
        "Reduction.*sum.*->[(]0d",  # Qualcomm QNN does not support full reduction to 0D scalar
        "Pad.*4d_f32<4x1x5x5>",  # Boundary Pad permutation
        "FullyConnected.*u8",  # Qualcomm QNN does not support quantized UINT8 GEMM
        "FullyConnected.*,2d_f32[^c]",  # Qualcomm QNN does not support dynamic weights for FP32 FC
        "FullyConnected.*_cst.*,1d_f32[^c]",  # Qualcomm QNN does not support dynamic bias for FP32 FC
    ],
    extra_flags = [
        "--data_seed=42",
        "--qualcomm_optimization_level=O3",
        "--qualcomm_htp_performance_mode=burst",
        "--qualcomm_log_level=info",
        "--qualcomm_use_fold_relu=true",
        "--qualcomm_use_conv_hmx=true",
    ],
    jit_suffix = "",
)

# copybara:uncomment_begin(google-only)
# litert_define_ats(
#     name = "google_tensor_ats",
#     backend = "google_tensor",
#     compile_only_suffix = "_aot",
#     do_register = [
#         "SingleOp",
#         "hf_all_minilm_l6_v2",
#         "hf_mobilevit_small",
#         "qai_hub_midas",
#         "qai_hub_real_esrgan_x4plus",
#         "torchvision_mobilenet_v2",
#         "u2net_lite",
#         "torchvision_resnet18",
#         "torchvision_squeezenet1_1",
#         "whisper_tiny_encoder",
#         "yamnet",
#         "yolo11n",
#     ],
#     dont_register = [
#         "hf_all_minilm_l6_v2",
#         "whisper_tiny_decoder",
#         "5d",
#         "6d",
#     ],
#     extra_flags = ["--soc_model=Tensor_G5"],
#     jit_suffix = "",
# )
# copybara:uncomment_end

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
        "5d",
        "6d",
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
    name = "gpu_ats",
    backend = "gpu",
    compile_only_suffix = "_aot",
    do_register = [
        "SingleOp",
        "hf_mobilevit_small",
        "qai_hub_midas",
        "torchvision_mobilenet_v2",
        "u2net_lite",
        "torchvision_squeezenet1_1",
        "yamnet",
        "yolo11n",
    ],
    dont_register = [
        "Binary.*i32",  # OpenCL delegate does not support int32 binary arithmetic
        "NoOp.*i32",  # OpenCL delegate does not support int32 NoOp (add)
        "OneHot",  # OpenCL delegate does not support OneHot
        "LogSoftmax",  # OpenCL delegate does not support LogSoftmax
        "tfl[.]prelu",  # OpenCL delegate requires PRelu alpha to be statically allocated
        "tfl[.]zeros_like",  # OpenCL delegate does not support ZerosLike
        "tfl[.]relu_0_to_1",  # OpenCL delegate does not support Relu0To1
        "tfl[.]floor_div",  # OpenCL evaluates floor_div with FP16 precision boundary roundoffs
        "tfl[.]ceil.*f32",  # OpenCL ceil exhibits -0.0/-1.0 negative zero boundary mismatch on f32
        "5d",  # OpenCL delegate only supports up to 4D tensor layouts
        "6d",  # OpenCL delegate only supports up to 4D tensor layouts
        "7d",  # OpenCL delegate only supports up to 4D tensor layouts
        "8d",  # OpenCL delegate only supports up to 4D tensor layouts
        "Pad.*[(][123]d",  # OpenCL Pad kernel requires 4D tensor layouts
        "Transpose.*[(]1d_",  # OpenCL Transpose requires multidimensional tensors (rank >= 2)
        "Transpose.*[(][1-8]d_(f32|i1|i4|i8|u8|i16|u16|i32|u32|i64|u64)<",  # OpenCL Transpose executes in FP16 texture buffers; only explicit FP16 Transpose is supported
        "FullyConnected.*(i8|u8)",  # OpenCL delegate does not support quantized INT8/UINT8 GEMM
        "FullyConnected.*_cst.*,1d_f32[^c]",  # OpenCL does not support dynamic bias
        "FullyConnected.*,2d_f32[^c]",  # OpenCL does not support dynamic weights
        "FullyConnected.*,2d_f32.*,1d_f32",  # OpenCL dynamic bias permutations
        "BatchMatmul.*2d.*,[34]d",  # OpenCL does not support rank-asymmetric batch broadcasting
        "BatchMatmul.*3d",  # ATS generator forces 1xB broadcast on 3D BatchMatmul
        "BatchMatmul.*4d",  # ATS generator forces 1xB broadcast on 4D BatchMatmul
    ],
    extra_flags = [
        "--data_seed=42",
    ],
    jit_suffix = "",
)

litert_define_ats(
    name = "intel_openvino_ats",
    backend = "intel_openvino",
    compile_only_suffix = None,
    do_register = [
        "SingleOp",
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
        "6d",
        "hf_all_minilm_l6_v2",
        "whisper_tiny_decoder",
        "qai_hub_real_esrgan_x4plus",
    ],
    jit_suffix = "",
)

litert_define_ats(
    name = "samsung_ats",
    backend = "samsung",
    compile_only_suffix = "_aot",
    do_register = [
    ],
    dont_register = [
        "whisper_tiny_decoder",
        "hf_all_minilm_l6_v2",
        "hf_mobilevit_small",
        "5d",
        "6d",
        "tfl.elu",
        "tfl.neg",
        "tfl.square",
        "tfl.zeros_like",
        "tfl.round",
        "tfl.sign",
    ],
    extra_flags = [
    ],
    jit_suffix = "",
)

# SOURCE CODES #####################################################################################

# Note: The `litert_test` target below (`ats`) is the base test binary executed directly on the host
# machine. In contrast, the `litert_define_ats` targets defined above generate test
# suites that wrap this binary and execute it on-device.
litert_test(
    name = "ats",
    srcs = ["ats.cc"],
    args = [
        "--dont_register=^(?!.*(Pad|Reduction|Transpose)).*i32",  # XNNPACK does not support int32 math (except Pad/Reduction/Transpose static constants)
        "--dont_register=OneHot",  # XNNPACK does not support OneHot
        "--dont_register=Transformer",  # XNNPACK does not support TransformerLayer flex/custom ops
        "--dont_register=tfl\\.floor_div",  # XNNPACK does not support FloorDiv
        "--dont_register=tfl\\.pow",  # XNNPACK does not support Pow
        "--dont_register=tfl\\.prelu",  # XNNPACK requires PRelu alpha to be statically allocated
        "--dont_register=tfl\\.zeros_like",  # XNNPACK does not support ZerosLike
        "--dont_register=tfl\\.sign",  # XNNPACK does not support Sign
        "--dont_register=tfl\\.relu_0_to_1",  # XNNPACK does not support Relu0To1
        "--dont_register=tfl\\.elu",  # XNNPACK does not support Elu
        "--dont_register=tfl\\.reduce_prod",  # XNNPACK does not support ReduceProd
        "--dont_register=LogSoftmax",  # XNNPACK does not support LogSoftmax
        "--dont_register=Pooling",  # XNNPACK rejects pooling when stride > filter size
        "--dont_register=FullyConnected.*(i8|u8|SHUFFLED)",  # XNNPACK delegate at ToT does not yet support these FullyConnected presets
        "--dont_register=FullyConnected.*,2d_f32.*,1d_f32",
        "--dont_register=FullyConnected.*,2d_f32[^c]",
        "--dont_register=FullyConnected.*_cst.*,1d_f32[^c]",
        "--dont_register=Transpose.*([(][78]d|[(][1-8]d_(i1|i4|i16|u16|i32|u32|i64|u64)<)",  # XNNPACK Transpose supports FP32, FP16, INT8, and UINT8 types up to rank 6
        # Explicit ExtraModel exclusions (unsupported by XNNPACK):
        "--dont_register=simple_add_dynamic_shape",
        "--dont_register=hf_all_minilm_l6_v2",
        "--dont_register=u2net_lite",
        "--dont_register=whisper_tiny_decoder",
        "--dont_register=yolo11n",
    ],
    copts = commandline_flag_copts(),
    linkstatic = True,
    no_main = True,
    target_compatible_with = select({
        "//litert:windows": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
    ungrte = if_oss(True, False),
    use_sys_malloc = True,
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/c:litert_op_code",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_environment",
        "//litert/cc/internal:litert_c_types_printing",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators",
        "//litert/test/generators:common",
        "//litert/test/generators:one_hot",
        "//tensor:arithmetic",
        "//tensor:arithmetic_graph",
        "//tensor:datatypes",
        "//tflite/schema:schema_fbs",
        "//tflite/types:half",
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
        "-DINCLUDE_SAMSUNG_COMPILE_FLAGS",
        "-DINCLUDE_SAMSUNG_RUNTIME_FLAGS",
    ],
    deps = [
        ":common",
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_common",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_options",
        "//litert/cc/internal:litert_rng",
        "//litert/compiler/plugin:compiler_plugin",
        "//litert/core:filesystem_testonly",
        "//litert/core/model:model_serialize",
        "//litert/tools/flags/vendors:mediatek_flags",
        "//litert/tools/flags/vendors:qualcomm_flags",
        "//litert/tools/flags/vendors:samsung_flags",
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
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "executor_test",
    srcs = ["executor_test.cc"],
    target_compatible_with = select({
        "//litert:windows": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
    deps = [
        ":executor",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_common",
        "//litert/cc:litert_environment",
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
        "//tflite/types:half",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
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
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_common",
        "//litert/cc:litert_environment",
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
    args = [
        "--dont_register=add.*i32",  # XNNPACK does not support int32 add
    ],
    copts = commandline_flag_copts(),
    data = [
        "//litert/test:testdata/simple_model.tflite",
        "//litert/vendors/examples:example_dispatch_so",
        "//litert/vendors/examples:example_plugin_so_data",
    ],
    # Re-enable sans once test libs can be linked in a sans friendly way.
    tags = [
        "noasan",
        "nomsan",
        "nosan",
        "notsan",
    ],
    target_compatible_with = select({
        "//litert:windows": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
    deps = [
        ":common",
        ":compile_fixture",
        ":configure",
        ":executor",
        ":inference_fixture",
        ":register",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_environment",
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
        "//litert/test/generators:one_hot",
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
        "@com_google_absl//absl/strings:str_format",
    ],
)

make_download_model_provider(
    name = "ats_models_provider",
    url = "https://storage.googleapis.com/litert/ats_models.tar.gz",
)
