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

# copybara:uncomment load(INTERNAL_PHYSICAL_MOBILE_TESTING_INFRA, "guitar", "guitar_workflow_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//litert/ats:ats.bzl", "litert_define_ats")

# copybara:uncomment load("//litert/ats:models.bzl", "CLASSICAL_ALL_MODELS")
load("//litert/build_common:litert_build_defs.bzl", "commandline_flag_copts", "if_oss", "litert_test")
load("//litert/integration_test:litert_device.bzl", "litert_device_test")
# copybara:uncomment load("//litert/integration_test/google:litert_device_guitar.bzl", "litert_macos_mh_guitar_test", "litert_mediatek_mh_guitar_test", "litert_npu_aot_guitar_test", "litert_pixel_9_mh_guitar_test", "litert_qualcomm_mh_guitar_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",  # copybara:comment_replace "//litert:litert_public",
    ],
)

# SUITES ###########################################################################################

# Note: The `litert_define_ats` targets below define test suites that execute tests on-device
# using the underlying base `ats` binary compiled further down in this file.

XNNPACK_MODELS_DONT_REGISTER = [
    "simple_add_dynamic_shape",
    "depth_anything_v2",
    "hf_all_minilm_l6_v2",
    "hf_deberta_large_mnli",
    "hf_detr_resnet50_dc5",
    "hf_distil_whisper_small_no_cache",
    "hf_distil_whisper_small_with_cache_init",
    "hf_distilbert",
    "hf_dpt_large",
    "hf_emotion_english_distilroberta_base",
    "hf_mbart_large_50",
    "hf_mobilebert",
    "hf_mobilevit_small_e2e",
    "hf_pyannote_segmentation_3_0",
    "hf_segformer_512",
    "hf_splade_v2_distil",
    "hf_tinyclip_vit_8m_16_text_3m_yfcc15m",
    "hf_tinyroberta_squad2",
    "hf_tinyroberta_squad2_dynamic_batch",
    "hf_wav2vec2_base_960h",
    "hf_wav2vec2_base_960h_dynamic",
    "hf_yolos_tiny",
    "qai_hub_hf_wavlm_base_plus",
    "qai_hub_hf_whisper_base_decoder",
    "qai_hub_hf_whisper_base_encoder",
    "qai_hub_inception_v3",
    "qai_hub_openai_clip",
    "qai_hub_real_esrgan_x4plus",
    "qai_hub_resnet101",
    "qai_hub_xlsr",
    "snapml_StyleTransferNet",
    "timm_efficientformerv2_s0",
    "timm_pvt_v2_b1",
    "timm_pvt_v2_b3",
    "timm_resnest14d",
    "torchaudio_conformer",
    "torchaudio_deepspeech",
    "torchaudio_wav2vec2_base",
    "torchvision_convnext_tiny",
    "torchvision_deeplabv3_resnet101",
    "torchvision_deeplabv3_resnet50",
    "torchvision_densenet121",
    "torchvision_fcn_resnet50",
    "torchvision_inception_v3",
    "torchvision_resnet152",
    "torchvision_resnet18",
    "torchvision_resnet50",
    "torchvision_shufflenet_v2",
    "torchvision_swin_tiny",
    "torchvision_video_resnet2plus1d_18",
    "torchvision_video_resnet3d_18",
    "torchvision_video_resnetmc3_18",
    "torchvision_video_swin3d_tiny",
    "torchvision_wide_resnet101_2",
    "torchvision_wide_resnet50_2",
    "u2net_full",
    "u2net_lite",
    "whisper_tiny_decoder",
    "yolo11n",
    "yolox_tiny",
]

XNNPACK_DONT_REGISTER = [
    "(_i32<[^>]+>(?!_cst)|->[(][^)]*_i32)",  # XNNPACK does not support dynamic int32 math (allows int32 static constants)
    "Comparison",  # XNNPACK does not support Comparison ops
    "OneHot",  # XNNPACK does not support OneHot
    "Transformer",  # XNNPACK does not support TransformerLayer flex/custom ops
    "SdpaTransposed",  # XNNPACK does not support SdpaTransposed composite op
    "QkvNormRope",  # XNNPACK does not support QkvNormRope composite op
    "RmsNorm",  # XNNPACK does not support RmsNorm composite op
    "Swiglu",  # XNNPACK does not support SwiGLU op
    "SelectV2",  # XNNPACK does not support SelectV2
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
    "FullyConnected.*([23]d_i8,2d_i8|u8|SHUFFLED)",  # XNNPACK delegate at ToT does not yet support these FullyConnected presets
    "Sdpa.*f16",  # XNNPACK delegate does not support f16 SDPA
    "Sdpa.*x8x[0-9]+>.*x1x[0-9]+>",  # XNNPACK delegate does not support 8-head MQA decode
    "FullyConnected.*,2d_f32<[^>]+>(?!_cst)",
    "FullyConnected.*,1d_f32<[^>]+>(?!_cst)",
    "Transpose.*([(][78]d|[(][1-8]d_(i1|i4|i16|u16|i32|u32|i64|u64)<)",  # XNNPACK Transpose supports FP32, FP16, INT8, and UINT8 types up to rank 6
]

YNNPACK_MODELS_DONT_REGISTER = [
    "simple_add_dynamic_shape",
    "depth_anything_v2",
    "hf_all_minilm_l6_v2",
    "hf_deberta_large_mnli",
    "hf_detr_resnet50_dc5",
    "hf_distil_whisper_small_no_cache",
    "hf_distil_whisper_small_with_cache_init",
    "hf_distilbert",
    "hf_dpt_large",
    "hf_emotion_english_distilroberta_base",
    "hf_mbart_large_50",
    "hf_mobilebert",
    "hf_mobilevit_small",
    "hf_pyannote_segmentation_3_0",
    "hf_segformer_512",
    "hf_splade_v2_distil",
    "hf_tinyclip_vit_8m_16_text_3m_yfcc15m",
    "hf_tinyroberta_squad2",
    "hf_tinyroberta_squad2_dynamic_batch",
    "hf_yolos_tiny",
    "isnet",
    "qai_hub_hf_wavlm_base_plus",
    "qai_hub_hf_whisper_base_decoder",
    "qai_hub_hf_whisper_base_encoder",
    "qai_hub_midas",
    "qai_hub_openai_clip",
    "qai_hub_real_esrgan_x4plus",
    "snapml_StyleTransferNet",
    "timm_efficientformerv2_s0",
    "torchaudio_conformer",
    "torchaudio_deepspeech",
    "torchaudio_wav2letter",
    "torchvision_deeplabv3_mobilenet_v3_large",
    "torchvision_deeplabv3_resnet101",
    "torchvision_deeplabv3_resnet50",
    "torchvision_fcn_resnet50",
    "torchvision_lraspp_mobilenet_v3_large",
    "torchvision_resnet18",
    "torchvision_swin_tiny",
    "torchvision_video_resnet2plus1d_18",
    "torchvision_video_resnet3d_18",
    "torchvision_video_resnetmc3_18",
    "torchvision_video_swin3d_tiny",
    "u2net_full",
    "u2net_lite",
    "whisper_tiny_decoder",
    "whisper_tiny_encoder",
    "yamnet",
    "yolo11n",
    "yolox_tiny",
]

YNNPACK_DONT_REGISTER = [
    "Binary.*i32",  # YNNPACK does not support int32 binary arithmetic
    "Comparison",  # YNNPACK does not support Comparison ops
    "OneHot",  # YNNPACK does not support OneHot
    "SelectV2",  # YNNPACK does not support SelectV2
    "tfl\\.floor_div",  # YNNPACK does not support FloorDiv
    "tfl\\.zeros_like",  # YNNPACK does not support ZerosLike
    "tfl\\.reduce_prod",  # YNNPACK does not support ReduceProd
    "tfl\\.tile",  # YNNPACK does not support Tile (used in MQA TransformerLayer)
    "Pad.*<([0-9]+x)*1(x[0-9]+)*>",  # YNNPACK does not support padding on extent-1 dimensions
    "Sdpa.*f16",  # YNNPACK does not support mixed fp32_fp16_fp32 dot in f16 SDPA
    "SdpaTransposed",  # SdpaTransposed generator does not yet set logit_cap attribute for YNNPACK
    "FullyConnected.*(i4|u8)",  # YNNPACK delegate requires even output dims for INT4 weights and does not support UINT8 FullyConnected
    "Transpose.*[(][1-8]d_(i1|i16|u16|u32|i64|u64)<",  # YNNPACK Transpose supports FP32, FP16, INT4, INT8, UINT8, and INT32 types
]

BUILTIN_MODELS_DONT_REGISTER = [
    "simple_add_dynamic_shape",
    "hf_all_minilm_l6_v2",
    "hf_deberta_large_mnli",
    "hf_distil_whisper_small_no_cache",
    "hf_distil_whisper_small_with_cache_init",
    "hf_distilbert",
    "hf_emotion_english_distilroberta_base",
    "hf_mbart_large_50",
    "hf_mobilebert",
    "hf_splade_v2_distil",
    "hf_tinyclip_vit_8m_16_text_3m_yfcc15m",
    "hf_tinyroberta_squad2",
    "hf_tinyroberta_squad2_dynamic_batch",
    "qai_hub_hf_whisper_base_decoder",
    "torchaudio_conformer",
    "whisper_tiny_decoder",
]

GPU_MODELS_DONT_REGISTER = [
    "simple_add_dynamic_shape",
    "depth_anything_v2",
    "hf_all_minilm_l6_v2",
    "hf_deberta_large_mnli",
    "hf_detr_resnet50_dc5",
    "hf_distil_whisper_small_no_cache",
    "hf_distil_whisper_small_with_cache_init",
    "hf_distilbert",
    "hf_dpt_large",
    "hf_emotion_english_distilroberta_base",
    "hf_mbart_large_50",
    "hf_mobilebert",
    "hf_splade_v2_distil",
    "hf_tinyclip_vit_8m_16_text_3m_yfcc15m",
    "hf_tinyroberta_squad2",
    "hf_tinyroberta_squad2_dynamic_batch",
    "hf_wav2vec2_base_960h",
    "hf_wav2vec2_base_960h_dynamic",
    "hf_yolos_tiny",
    "qai_hub_hf_wavlm_base_plus",
    "qai_hub_hf_whisper_base_decoder",
    "qai_hub_openai_clip",
    "qai_hub_real_esrgan_x4plus",
    "qai_hub_resnet101",
    "qai_hub_xlsr",
    "snapml_StyleTransferNet",
    "timm_pvt_v2_b1",
    "timm_pvt_v2_b3",
    "timm_resnest14d",
    "timm_vit_small_path16_224",
    "torchaudio_conformer",
    "torchaudio_wav2vec2_base",
    "torchvision_convnext_tiny",
    "torchvision_deeplabv3_resnet101",
    "torchvision_deeplabv3_resnet50",
    "torchvision_densenet121",
    "torchvision_fcn_resnet50",
    "torchvision_maxvit_t",
    "torchvision_resnet152",
    "torchvision_resnet18",
    "torchvision_resnet50",
    "torchvision_shufflenet_v2",
    "torchvision_swin_tiny",
    "torchvision_video_resnet2plus1d_18",
    "torchvision_video_resnet3d_18",
    "torchvision_video_resnetmc3_18",
    "torchvision_video_swin3d_tiny",
    "torchvision_vit_base_patch16",
    "torchvision_wide_resnet101_2",
    "torchvision_wide_resnet50_2",
    "whisper_tiny_decoder",
    "yolox_tiny",
    # FP16 precision overflow (NaN) on OpenCL GPU:
    "hf_hubert_large_ls960_ft",
    "timm_efficientnet",
    "torchvision_efficientnet_b0",
    "torchvision_efficientnet_b1",
    "torchvision_efficientnet_b3",
    "torchvision_efficientnet_b4",
    "torchvision_efficientnet_b5",
    "torchvision_efficientnet_b6",
    "torchvision_efficientnet_b7",
]

BUILTIN_DONT_REGISTER = [
    "_f16",  # TFLite built-in kernels do not support FLOAT16 tensors
    "wf=SHUFFLED4x16INT8",  # Built-in FC needs a 2nd workspace output for this format
]

BUILTIN_EXTRA_FLAGS = [
    "--cpu_kernel_mode=builtin",
    "--data_seed=42",
]

litert_define_ats(
    name = "cpu_ats",
    backend = "cpu",
    compile_only_suffix = None,
    dont_register = XNNPACK_DONT_REGISTER,
    jit_suffix = "",
)

litert_define_ats(
    name = "ynnpack_cpu_ats",
    backend = "cpu",
    compile_only_suffix = None,
    dont_register = YNNPACK_DONT_REGISTER,
    extra_flags = [
        "--enable_ynnpack=true",
    ],
    jit_suffix = "",
)

# Runs the TFLite built-in CPU kernels (no delegate) against the generator
# references. Used to guard refactoring of the built-in kernels.
litert_define_ats(
    name = "builtin_cpu_ats",
    backend = "cpu",
    compile_only_suffix = None,
    dont_register = BUILTIN_DONT_REGISTER,
    extra_flags = BUILTIN_EXTRA_FLAGS,
    jit_suffix = "",
)

litert_define_ats(
    name = "cpu_models_ats",
    backend = "cpu",
    compile_only_suffix = None,
    do_register = ["ExtraModel"],
    dont_register = XNNPACK_MODELS_DONT_REGISTER,
    extra_flags = [
        "--iters_per_test=1",
    ],
    extra_models = ["https://storage.googleapis.com/litert/ats_models.tar.gz"],  # copybara:comment_replace extra_models = CLASSICAL_ALL_MODELS,
    jit_suffix = "",
)

litert_define_ats(
    name = "ynnpack_cpu_models_ats",
    backend = "cpu",
    compile_only_suffix = None,
    do_register = ["ExtraModel"],
    dont_register = YNNPACK_MODELS_DONT_REGISTER,
    extra_flags = [
        "--enable_ynnpack=true",
        "--iters_per_test=1",
    ],
    extra_models = ["https://storage.googleapis.com/litert/ats_models.tar.gz"],  # copybara:comment_replace extra_models = CLASSICAL_ALL_MODELS,
    jit_suffix = "",
)

litert_define_ats(
    name = "builtin_cpu_models_ats",
    backend = "cpu",
    compile_only_suffix = None,
    do_register = ["ExtraModel"],
    dont_register = BUILTIN_MODELS_DONT_REGISTER,
    extra_flags = [
        "--cpu_kernel_mode=builtin",
        "--iters_per_test=1",
    ],
    extra_models = ["https://storage.googleapis.com/litert/ats_models.tar.gz"],  # copybara:comment_replace extra_models = CLASSICAL_ALL_MODELS,
    jit_suffix = "",
)

litert_define_ats(
    name = "gpu_models_ats",
    backend = "gpu",
    compile_only_suffix = None,
    do_register = ["ExtraModel"],
    dont_register = GPU_MODELS_DONT_REGISTER,
    extra_flags = [
        "--iters_per_test=1",
    ],
    extra_models = ["https://storage.googleapis.com/litert/ats_models.tar.gz"],  # copybara:comment_replace extra_models = CLASSICAL_ALL_MODELS,
    jit_suffix = "",
)

litert_define_ats(
    name = "example_ats",
    backend = "example",
    compile_only_suffix = "_aot",
    do_register = [
        "tfl[.]sub.*f32",
        "tfl[.]mul.*f32",
    ],
    dont_register = [
    ],
    jit_suffix = "",
)

litert_define_ats(
    name = "qualcomm_ats",
    aot_shard_count = 32,
    aot_tags = [
        "manual",
        "notap",
    ],
    backend = "qualcomm",
    compile_only_suffix = "_aot",
    dont_register = [
        "CompositeOp",
        "Transformer",
        "hf_all_minilm_l6_v2",
        "whisper_tiny_decoder",
        "5d",  # Qualcomm QNN HTP only supports up to 4D/6D tensor layouts
        "6d",  # Qualcomm QNN HTP only supports up to 4D/6D tensor layouts
        "7d",  # Qualcomm QNN HTP only supports up to 4D/6D tensor layouts
        "8d",  # Qualcomm QNN HTP only supports up to 4D/6D tensor layouts
        "Binary.*i32",  # Qualcomm QNN does not support int32 binary arithmetic
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
#     aot_shard_count = 50,
#     backend = "google_tensor",
#     compile_only_suffix = "_aot",
#     dont_register = [
#         "CompositeOp",
#         "Transformer",
#         "hf_all_minilm_l6_v2",
#         "whisper_tiny_decoder",
#         "5d",
#         "6d",
#         "7d",
#         "8d",
#         "OneHot",  # Google Tensor compiler does not support OneHot
#         "tfl[.]prelu",  # Google Tensor compiler requires PRelu alpha to be statically allocated
#         "tfl[.]reduce_prod",  # Google Tensor compiler does not support ReduceProd
#         "Conv2d.*f16",  # Google Tensor compiler does not support FP16 Conv2d/DepthwiseConv2d
#         "FullyConnected.*(f16|i4|u8|SHUFFLED)",  # Google Tensor compiler does not support FP16, INT4, UINT8, or SHUFFLED FullyConnected
#         "FullyConnected.*,2d_f32<[^>]+>(?!_cst)",  # Google Tensor compiler does not support dynamic weights for FP32 FullyConnected
#         "FullyConnected.*,1d_f32<[^>]+>(?!_cst)",  # Google Tensor compiler does not support dynamic bias for FP32 FullyConnected
#         "LogSoftmax.*f16",  # Google Tensor compiler does not support FP16 LogSoftmax
#         "Pad.*f16",  # Google Tensor compiler does not support FP16 Pad
#         "average_pool_2d.*fa=RELU",  # Google Tensor compiler does not support fused RELU on AveragePool2d
#         "Transpose.*[(][1-8]d_(i1|i4|i64|u64)<",  # Google Tensor Transpose does not support i1, i4, i64, or u64 types
#     ],
#     extra_flags = ["--soc_model=Tensor_G5"],
#     jit_suffix = "",
# )
# copybara:uncomment_end

litert_define_ats(
    name = "mediatek_ats",
    aot_shard_count = 32,
    aot_tags = [
        "manual",
        "notap",
    ],
    backend = "mediatek",
    compile_only_suffix = "_aot",
    dont_register = [
        "CompositeOp",
        "Transformer",
        "5d",
        "6d",
        "7d",
        "8d",
        "Binary.*i32",  # MediaTek NeuroPilot does not support int32 binary arithmetic
        "BinaryBroadcast.*[234]d_",  # MediaTek NeuroPilot rejects ATS bidirectional/rank-mismatched broadcasts
        "BatchMatmul.*[34]d",  # MediaTek NeuroPilot rejects ATS 1xB batch broadcasts on 3D/4D BatchMatmul
        "Comparison",  # MediaTek NeuroPilot does not support Comparison ops
        "OneHot",  # MediaTek NeuroPilot does not support OneHot
        "SelectV2",  # MediaTek NeuroPilot does not support SelectV2
        "LogSoftmax",  # MediaTek NeuroPilot does not support LogSoftmax
        "tfl[.]floor_div",  # MediaTek NeuroPilot does not support FloorDiv
        "tfl[.]pow",  # MediaTek NeuroPilot does not support Pow
        "tfl[.]prelu",  # MediaTek NeuroPilot requires static PRelu alpha
        "tfl[.]reduce_min",  # MediaTek NeuroPilot does not support ReduceMin
        "tfl[.]reduce_prod",  # MediaTek NeuroPilot does not support ReduceProd
        "tfl[.](cos|sin|ceil|elu|exp|floor|log|neg|relu6|relu_0_to_1|relu_n1_to_1|round|sign|square|zeros_like)[{]",  # Unsupported Unary ops in MediaTek plugin
        "FullyConnected.*,1d_f32(?:<[^>]+>)?(?!_cst)",  # MediaTek NeuroPilot does not support dynamic bias for FP32 FullyConnected
        "FullyConnected.*f32(?:<[^>]+>)?,2d_i[48]",  # MediaTek NeuroPilot does not support hybrid FP32 activation with INT4/INT8 weights
        "FullyConnected.*2d_f16(?:<[^>]+>)?_cst[)]",  # MediaTek NeuroPilot requires bias tensor for FP16 FullyConnected
        "Slice.*4d_i32",  # MediaTek NeuroPilot does not support 4D INT32 Slice
        "Transpose.*[(][1-8]d_(i1|i4|i32|u32|i64|u64)",  # MediaTek NeuroPilot Transpose does not support i1, i4, 32-bit, or 64-bit integer tensors
        "Pooling.*(filter_width=(10|14)|filter_height=(12|14)|stride_h=3,filter_width=2)",  # MediaTek NeuroPilot pooling kernel size/stride hardware limits
        "average_pool_2d.*(filter_width=12|fa=RELU,padding=SAME,stride_w=3,stride_h=2)",  # MediaTek NeuroPilot AveragePool2d kernel size/fused activation limits
        "whisper_tiny_decoder",
        "whisper_tiny_encoder",
        "hf_all_minilm_l6_v2",
        "qai_hub_real_esrgan_x4plus",
    ],
    extra_flags = [
        "--data_seed=42",
        "--soc_model=mt6989",
        "--mediatek_performance_mode_type=turbo_boost",
        "--mediatek_optimization_hint=low_latency",
        "--mediatek_enable_l1_cache_optimizations=true",
    ],
    jit_suffix = "",
)

litert_define_ats(
    name = "gpu_ats",
    backend = "gpu",
    compile_only_suffix = None,
    dont_register = [
        "Transformer",
        "scaled_dot_product_attention",
        "Binary.*i32",  # OpenCL delegate does not support int32 binary arithmetic
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
        "Transpose[^d].*[(]1d_",  # OpenCL Transpose requires multidimensional tensors (rank >= 2)
        "Transpose[^d].*[(][1-8]d_(f32|i1|i4|i8|u8|i16|u16|i32|u32|i64|u64)<",  # OpenCL Transpose executes in FP16 texture buffers; only explicit FP16 Transpose is supported
        "FullyConnected.*(i4|i8|u8)",  # OpenCL delegate does not support quantized INT4/INT8/UINT8 GEMM
        "FullyConnected.*_cst.*,1d_f32[^c]",  # OpenCL does not support dynamic bias
        "FullyConnected.*,2d_f32[^c]",  # OpenCL does not support dynamic weights
        "FullyConnected.*,2d_f32.*,1d_f32",  # OpenCL dynamic bias permutations
        "BatchMatmul.*2d.*,[34]d",  # OpenCL does not support rank-asymmetric batch broadcasting
        "BatchMatmul.*3d",  # ATS generator forces 1xB broadcast on 3D BatchMatmul
        "BatchMatmul.*4d",  # ATS generator forces 1xB broadcast on 4D BatchMatmul
        "Swiglu.*3d_f(16|32)<(?!1x)",  # GPU accelerator does not support SwiGLU with batch > 1
        "QkvNormRope.*4d_f(16|32)<(?!1x)",  # GPU accelerator does not support QkvNormRope with batch > 1
        "SelectV2",  # OpenCL SelectV2 rejects multidimensional if/else broadcasting and uses kExact on FP16-rounded f32
        "Slice",  # OpenCL Slice uses kExact byte comparison on FP16-rounded f32 buffers
        "simple_add_dynamic_shape",
        "hf_all_minilm_l6_v2",
        "whisper_tiny_decoder",
        "torchvision_resnet18",
        "whisper_tiny_encoder",
        "qai_hub_real_esrgan_x4plus",
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
    dont_register = [
        "CompositeOp",
        "Transformer",
        "6d",
        "hf_all_minilm_l6_v2",
        "whisper_tiny_decoder",
        "qai_hub_real_esrgan_x4plus",
    ],
    jit_suffix = "",
)

litert_define_ats(
    name = "samsung_ats",
    aot_shard_count = 32,
    aot_tags = [
        "manual",
        "notap",
    ],
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

litert_define_ats(
    name = "cpu_macos_ats",
    backend = "cpu",
    compile_only_suffix = None,
    dont_register = [
        "(_i32<[^>]+>(?!_cst)|->[(][^)]*_i32)",  # XNNPACK does not support dynamic int32 math (allows int32 static constants)
        "Comparison",  # XNNPACK does not support Comparison ops
        "OneHot",  # XNNPACK does not support OneHot
        "Transformer",  # XNNPACK does not support TransformerLayer flex/custom ops
        "SdpaTransposed",  # XNNPACK does not support SdpaTransposed composite op
        "QkvNormRope",  # XNNPACK does not support QkvNormRope composite op
        "RmsNorm",  # XNNPACK does not support RmsNorm composite op
        "Swiglu",  # XNNPACK does not support SwiGLU op
        "SelectV2",  # XNNPACK does not support SelectV2
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
        "Conv2d.*f16",  # XNNPACK delegate at ToT does not yet support f16 Conv2d
        "BatchMatmul.*f16",  # XNNPACK delegate at ToT does not yet support f16 BatchMatmul
        "FullyConnected.*(f16|[23]d_i8,2d_i8|u8|SHUFFLED)",  # XNNPACK delegate at ToT does not yet support these FullyConnected presets
        "Sdpa.*f16",  # XNNPACK delegate does not support f16 SDPA
        "Sdpa.*x8x[0-9]+>.*x1x[0-9]+>",  # XNNPACK delegate does not support 8-head MQA decode
        "FullyConnected.*,2d_f32<[^>]+>(?!_cst)",
        "FullyConnected.*,1d_f32<[^>]+>(?!_cst)",
        "Transpose.*([(][78]d|[(][1-8]d_(i1|i4|i16|u16|i32|u32|i64|u64)<)",  # XNNPACK Transpose supports FP32, FP16, INT8, and UINT8 types up to rank 6
        # Explicit ExtraModel exclusions (unsupported by XNNPACK):
        "simple_add_dynamic_shape",
        "hf_all_minilm_l6_v2",
        "u2net_lite",
        "whisper_tiny_decoder",
        "yolo11n",
    ],
    extra_flags = [
        "--data_seed=42",
    ],
    jit_suffix = "",
    platform = "macos",
)

litert_define_ats(
    name = "ynnpack_cpu_macos_ats",
    backend = "cpu",
    compile_only_suffix = None,
    dont_register = YNNPACK_DONT_REGISTER,
    extra_flags = [
        "--enable_ynnpack=true",
    ],
    jit_suffix = "",
    platform = "macos",
)

litert_define_ats(
    name = "metal_macos_ats",
    backend = "metal",
    compile_only_suffix = None,
    dont_register = [
        "Binary.*i32",
        "OneHot",
        "LogSoftmax",
        "Transpose.*[(]1d_",
        "Transpose.*[(][1-8]d_(f32|i1|i4|i8|u8|i16|u16|i32|u32|i64|u64)<",
        "FullyConnected",
        "BatchMatmul",
        "tfl[.]prelu",
        "tfl[.]zeros_like",
        "tfl[.]relu_0_to_1",
        "tfl[.]floor_div",
        "tfl[.]floor[{]",
        "tfl[.]ceil",
        "tfl[.]round",
        "5d",
        "6d",
        "7d",
        "8d",
        "Pad.*[(][123]d",
        "Swiglu.*3d_f(16|32)<(?!1x)",  # Metal accelerator does not support SwiGLU with batch > 1
        "QkvNormRope.*4d_f(16|32)<(?!1x)",  # Metal accelerator does not support QkvNormRope with batch > 1
        "SelectV2",
        "Slice",
        "scaled_dot_product_attention",
    ],
    extra_flags = [
        "--data_seed=42",
    ],
    jit_suffix = "",
    platform = "macos",
)

litert_define_ats(
    name = "webgpu_macos_ats",
    backend = "webgpu",
    compile_only_suffix = None,
    dont_register = [
        "Binary.*i32",
        "OneHot",
        "LogSoftmax",
        "Transpose.*[(]1d_",
        "Transpose.*[(][1-8]d_(f32|i1|i4|i8|u8|i16|u16|i32|u32|i64|u64)<",
        "FullyConnected",
        "BatchMatmul",
        "tfl[.]prelu",
        "tfl[.]zeros_like",
        "tfl[.]relu_0_to_1",
        "tfl[.]floor_div",
        "tfl[.]floor[{]",
        "tfl[.]ceil",
        "tfl[.]round",
        "5d",
        "6d",
        "7d",
        "8d",
        "Pad.*[(][123]d",
        "Swiglu.*3d_f(16|32)<(?!1x)",  # WebGPU accelerator does not support SwiGLU with batch > 1
        "QkvNormRope.*4d_f(16|32)<(?!1x)",  # WebGPU accelerator does not support QkvNormRope with batch > 1
        "SelectV2",
        "Slice",
        "scaled_dot_product_attention",
    ],
    extra_flags = [
        "--data_seed=42",
    ],
    jit_suffix = "",
    platform = "macos",
)

litert_define_ats(
    name = "cpu_macos_models_ats",
    backend = "cpu",
    compile_only_suffix = None,
    do_register = ["ExtraModel"],
    dont_register = XNNPACK_MODELS_DONT_REGISTER,
    extra_flags = [
        "--iters_per_test=1",
    ],
    extra_models = ["https://storage.googleapis.com/litert/ats_models.tar.gz"],  # copybara:comment_replace extra_models = CLASSICAL_ALL_MODELS,
    jit_suffix = "",
    platform = "macos",
)

litert_define_ats(
    name = "ynnpack_cpu_macos_models_ats",
    backend = "cpu",
    compile_only_suffix = None,
    do_register = ["ExtraModel"],
    dont_register = YNNPACK_MODELS_DONT_REGISTER,
    extra_flags = [
        "--enable_ynnpack=true",
        "--iters_per_test=1",
    ],
    extra_models = ["https://storage.googleapis.com/litert/ats_models.tar.gz"],  # copybara:comment_replace extra_models = CLASSICAL_ALL_MODELS,
    jit_suffix = "",
    platform = "macos",
)

litert_define_ats(
    name = "builtin_cpu_macos_models_ats",
    backend = "cpu",
    compile_only_suffix = None,
    do_register = ["ExtraModel"],
    dont_register = BUILTIN_MODELS_DONT_REGISTER,
    extra_flags = [
        "--cpu_kernel_mode=builtin",
        "--iters_per_test=1",
    ],
    extra_models = ["https://storage.googleapis.com/litert/ats_models.tar.gz"],  # copybara:comment_replace extra_models = CLASSICAL_ALL_MODELS,
    jit_suffix = "",
    platform = "macos",
)

litert_define_ats(
    name = "metal_macos_models_ats",
    backend = "metal",
    compile_only_suffix = None,
    do_register = ["ExtraModel"],
    dont_register = GPU_MODELS_DONT_REGISTER,
    extra_flags = [
        "--iters_per_test=1",
    ],
    extra_models = ["https://storage.googleapis.com/litert/ats_models.tar.gz"],  # copybara:comment_replace extra_models = CLASSICAL_ALL_MODELS,
    jit_suffix = "",
    platform = "macos",
)

litert_define_ats(
    name = "webgpu_macos_models_ats",
    backend = "webgpu",
    compile_only_suffix = None,
    do_register = ["ExtraModel"],
    dont_register = GPU_MODELS_DONT_REGISTER,
    extra_flags = [
        "--iters_per_test=1",
    ],
    extra_models = ["https://storage.googleapis.com/litert/ats_models.tar.gz"],  # copybara:comment_replace extra_models = CLASSICAL_ALL_MODELS,
    jit_suffix = "",
    platform = "macos",
)

# SOURCE CODES #####################################################################################

# Note: The `litert_test` target below (`ats`) is the base test binary executed directly on the host
# machine. In contrast, the `litert_define_ats` targets defined above generate test
# suites that wrap this binary and execute it on-device.
litert_test(
    name = "ats",
    srcs = ["ats.cc"],
    args = ["--dont_register=" + p for p in XNNPACK_DONT_REGISTER],
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
        ":register_composite_ops",
        ":register_single_ops",
        ":register_transformer_layer",
        "//litert/c/internal:litert_logging",
        "//litert/cc/internal:litert_detail",
        "//litert/runtime/accelerators/ynnpack:ynnpack_accelerator",  # buildcleaner: keep
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

# Like `ats` above, `ynnpack_ats` is executed directly on the host machine with
# YNNPACK enabled, whereas `ynnpack_cpu_ats` and `ynnpack_cpu_macos_ats` above
# wrap `ats` for on-device and macOS execution.
litert_test(
    name = "ynnpack_ats",
    srcs = ["ats.cc"],
    args = [
        "--enable_ynnpack=true",
    ] + ["--dont_register=" + p for p in YNNPACK_DONT_REGISTER],
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
        ":register_composite_ops",
        ":register_single_ops",
        ":register_transformer_layer",
        "//litert/c/internal:litert_logging",
        "//litert/cc/internal:litert_detail",
        "//litert/runtime/accelerators/ynnpack:ynnpack_accelerator",  # buildcleaner: keep
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

# Like `ats` above, `builtin_ats` is executed directly on the host machine, but
# with the TFLite built-in CPU kernels instead of XNNPACK. `builtin_cpu_ats`
# above wraps `ats` for on-device execution.
litert_test(
    name = "builtin_ats",
    srcs = ["ats.cc"],
    args = BUILTIN_EXTRA_FLAGS + ["--dont_register=" + p for p in BUILTIN_DONT_REGISTER],
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
        ":register_composite_ops",
        ":register_single_ops",
        ":register_transformer_layer",
        "//litert/c/internal:litert_logging",
        "//litert/cc/internal:litert_detail",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "register_single_ops",
    testonly = True,
    srcs = ["register_single_ops.cc"],
    hdrs = ["register_single_ops.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register_batch_matmul",
        ":register_binary_broadcast",
        ":register_binary_no_bcast",
        ":register_comparison",
        ":register_concatenation",
        ":register_conv_2d",
        ":register_depthwise_conv_2d",
        ":register_fully_connected",
        ":register_mean",
        ":register_one_hot",
        ":register_pad",
        ":register_pooling",
        ":register_reduction",
        ":register_reshape",
        ":register_select_v2",
        ":register_slice",
        ":register_softmax",
        ":register_transpose",
        ":register_unary",
    ],
)

cc_library(
    name = "register_composite_ops",
    testonly = True,
    srcs = ["register_composite_ops.cc"],
    hdrs = ["register_composite_ops.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register_qkv_norm_rope",
        ":register_rms_norm",
        ":register_sdpa",
        ":register_sdpa_transposed",
        ":register_swiglu",
    ],
)

cc_library(
    name = "register_batch_matmul",
    testonly = True,
    srcs = ["register_batch_matmul.cc"],
    hdrs = ["register_batch_matmul.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:batch_matmul",
        "//litert/test/generators:common",
        "//tflite/types:half",
    ],
)

cc_library(
    name = "register_binary_broadcast",
    testonly = True,
    srcs = ["register_binary_broadcast.cc"],
    hdrs = ["register_binary_broadcast.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/c:litert_op_code",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:binary_broadcast",
        "//litert/test/generators:common",
        "//tflite/schema:schema_fbs",
    ],
)

cc_library(
    name = "register_binary_no_bcast",
    testonly = True,
    srcs = ["register_binary_no_bcast.cc"],
    hdrs = ["register_binary_no_bcast.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/c:litert_op_code",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:binary_no_bcast",
        "//litert/test/generators:common",
        "//tflite/schema:schema_fbs",
    ],
)

cc_library(
    name = "register_comparison",
    testonly = True,
    srcs = ["register_comparison.cc"],
    hdrs = ["register_comparison.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/c:litert_op_code",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:common",
        "//litert/test/generators:comparison",
        "//tflite/types:half",
    ],
)

cc_library(
    name = "register_concatenation",
    testonly = True,
    srcs = ["register_concatenation.cc"],
    hdrs = ["register_concatenation.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/c:litert_op_code",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:common",
        "//litert/test/generators:concatenation",
        "//tflite/schema:schema_fbs",
    ],
)

cc_library(
    name = "register_conv_2d",
    testonly = True,
    srcs = ["register_conv_2d.cc"],
    hdrs = ["register_conv_2d.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/c:litert_op_code",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:common",
        "//litert/test/generators:conv_2d",
        "//tflite/schema:schema_fbs",
        "//tflite/types:half",
    ],
)

cc_library(
    name = "register_depthwise_conv_2d",
    testonly = True,
    srcs = ["register_depthwise_conv_2d.cc"],
    hdrs = ["register_depthwise_conv_2d.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/c:litert_op_code",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:common",
        "//litert/test/generators:depthwise_conv_2d",
        "//tflite/schema:schema_fbs",
        "//tflite/types:half",
    ],
)

cc_library(
    name = "register_fully_connected",
    testonly = True,
    srcs = ["register_fully_connected.cc"],
    hdrs = ["register_fully_connected.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/c:litert_op_code",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:common",
        "//litert/test/generators:fully_connected",
        "//tensor:arithmetic_graph",
        "//tensor:datatypes",
        "//tflite/schema:schema_fbs",
        "//tflite/types:half",
    ],
)

cc_library(
    name = "register_mean",
    testonly = True,
    srcs = ["register_mean.cc"],
    hdrs = ["register_mean.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/c:litert_op_code",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:common",
        "//litert/test/generators:mean",
        "//tflite/types:half",
    ],
)

cc_library(
    name = "register_one_hot",
    testonly = True,
    srcs = ["register_one_hot.cc"],
    hdrs = ["register_one_hot.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:common",
        "//litert/test/generators:one_hot",
    ],
)

cc_library(
    name = "register_pad",
    testonly = True,
    srcs = ["register_pad.cc"],
    hdrs = ["register_pad.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:common",
        "//litert/test/generators:pad",
        "//tflite/types:half",
    ],
)

cc_library(
    name = "register_pooling",
    testonly = True,
    srcs = ["register_pooling.cc"],
    hdrs = ["register_pooling.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/c:litert_op_code",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:common",
        "//litert/test/generators:pooling",
        "//tflite/schema:schema_fbs",
    ],
)

cc_library(
    name = "register_reduction",
    testonly = True,
    srcs = ["register_reduction.cc"],
    hdrs = ["register_reduction.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/c:litert_op_code",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:common",
        "//litert/test/generators:reduction",
    ],
)

cc_library(
    name = "register_reshape",
    testonly = True,
    srcs = ["register_reshape.cc"],
    hdrs = ["register_reshape.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:common",
        "//litert/test/generators:reshape",
    ],
)

cc_library(
    name = "register_qkv_norm_rope",
    testonly = True,
    srcs = ["register_qkv_norm_rope.cc"],
    hdrs = ["register_qkv_norm_rope.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:qkv_norm_rope",
        "//tflite/types:half",
    ],
)

cc_library(
    name = "register_rms_norm",
    testonly = True,
    srcs = ["register_rms_norm.cc"],
    hdrs = ["register_rms_norm.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:rms_norm",
        "//tflite/types:half",
    ],
)

cc_library(
    name = "register_sdpa",
    testonly = True,
    srcs = ["register_sdpa.cc"],
    hdrs = ["register_sdpa.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:sdpa",
        "//tflite/types:half",
    ],
)

cc_library(
    name = "register_sdpa_transposed",
    testonly = True,
    srcs = ["register_sdpa_transposed.cc"],
    hdrs = ["register_sdpa_transposed.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:sdpa_transposed",
        "//tflite/types:half",
    ],
)

cc_library(
    name = "register_swiglu",
    testonly = True,
    srcs = ["register_swiglu.cc"],
    hdrs = ["register_swiglu.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:swiglu",
        "//tflite/types:half",
    ],
)

cc_library(
    name = "register_slice",
    testonly = True,
    srcs = ["register_slice.cc"],
    hdrs = ["register_slice.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:common",
        "//litert/test/generators:slice",
        "//tflite/types:half",
    ],
)

cc_library(
    name = "register_select_v2",
    testonly = True,
    srcs = ["register_select_v2.cc"],
    hdrs = ["register_select_v2.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:common",
        "//litert/test/generators:select_v2",
        "//tflite/types:half",
    ],
)

cc_library(
    name = "register_softmax",
    testonly = True,
    srcs = ["register_softmax.cc"],
    hdrs = ["register_softmax.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/c:litert_op_code",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:common",
        "//litert/test/generators:softmax",
        "//tflite/types:half",
    ],
)

cc_library(
    name = "register_transformer_layer",
    testonly = True,
    srcs = ["register_transformer_layer.cc"],
    hdrs = ["register_transformer_layer.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:transformer_layer",
    ],
)

cc_library(
    name = "register_transpose",
    testonly = True,
    srcs = ["register_transpose.cc"],
    hdrs = ["register_transpose.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:common",
        "//litert/test/generators:transpose",
        "//tflite/types:half",
    ],
)

cc_library(
    name = "register_unary",
    testonly = True,
    srcs = ["register_unary.cc"],
    hdrs = ["register_unary.h"],
    copts = commandline_flag_copts(),
    deps = [
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/c:litert_op_code",
        "//litert/cc/internal:litert_detail",
        "//litert/test/generators:common",
        "//litert/test/generators:unary",
        "//tflite/types:half",
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
        "//litert/c/options:litert_cpu_options",
        "//litert/cc:litert_common",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_options",
        "//litert/cc/internal:litert_rng",
        "//litert/cc/options:litert_mediatek_options",
        "//litert/cc/options:litert_qualcomm_options",
        "//litert/cc/options:litert_samsung_options",
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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
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
        ":register",
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
        "@com_google_absl//absl/strings",
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
        ":register",
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
        "//litert/c:litert_op_code",
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

cc_test(
    name = "register_ops_contract_test",
    srcs = ["register_ops_contract_test.cc"],
    copts = commandline_flag_copts(),
    target_compatible_with = select({
        "//litert:windows": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
    deps = [
        ":common",
        ":compile_fixture",
        ":configure",
        ":inference_fixture",
        ":register_composite_ops",
        ":register_single_ops",
        "//litert/c:litert_op_code",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
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
        "//litert/c:litert_any",
        "//litert/cc/internal:litert_compiled_model_next",
        "//litert/core/model",
        "@com_google_absl//absl/strings:str_format",
    ],
)

# copybara:uncomment_begin(google-only)
# # GUITAR WORKFLOWS #################################################################################
#
# PIXEL_9_ATS_GUITAR_TARGETS = [
#     ":gpu_ats_remote",
#     ":cpu_ats_remote",
#     ":ynnpack_cpu_ats_remote",
#     ":builtin_cpu_ats_remote",
# ]
#
# guitar_workflow_test(
#     name = "litert_ats_guitar_pixel_9",
#     # This flag is required by guitar::update_culprit_finding_config and shard_count.
#     extract_deps = True,
#     integration_test = guitar.IntegrationTest(
#         tests = [
#             litert_pixel_9_mh_guitar_test(
#                 targets = PIXEL_9_ATS_GUITAR_TARGETS,
#             ),
#         ],
#     ),
#     tags = [
#         "external",
#         "guitar",
#         "manual",
#         "notap",
#     ],
# )
#
# QUALCOMM_S24_ATS_GUITAR_TARGETS = [
#     ":gpu_ats_remote",
#     ":cpu_ats_remote",
#     ":ynnpack_cpu_ats_remote",
#     ":builtin_cpu_ats_remote",
# ]
#
# guitar_workflow_test(
#     name = "litert_ats_guitar_qualcomm_s24",
#     # This flag is required by guitar::update_culprit_finding_config and shard_count.
#     extract_deps = True,
#     integration_test = guitar.IntegrationTest(
#         tests = [
#             litert_qualcomm_mh_guitar_test(
#                 targets = QUALCOMM_S24_ATS_GUITAR_TARGETS,
#             ),
#         ],
#     ),
#     tags = [
#         "external",
#         "guitar",
#         "manual",
#         "notap",
#     ],
# )
#
# MEDIATEK_MT6989_ATS_GUITAR_TARGETS = [
#     ":gpu_ats_remote",
#     ":cpu_ats_remote",
#     ":ynnpack_cpu_ats_remote",
#     ":builtin_cpu_ats_remote",
# ]
#
# guitar_workflow_test(
#     name = "litert_ats_guitar_mediatek_mt6989",
#     # This flag is required by guitar::update_culprit_finding_config and shard_count.
#     extract_deps = True,
#     integration_test = guitar.IntegrationTest(
#         tests = [
#             litert_mediatek_mh_guitar_test(
#                 dimension_hardware = "mt6989",
#                 targets = MEDIATEK_MT6989_ATS_GUITAR_TARGETS,
#             ),
#         ],
#     ),
#     tags = [
#         "external",
#         "guitar",
#         "manual",
#         "notap",
#     ],
# )
#
# MACOS_ATS_GUITAR_TARGETS = [
#     ":metal_macos_ats_remote",
#     ":webgpu_macos_ats_remote",
#     ":cpu_macos_ats_remote",
#     ":ynnpack_cpu_macos_ats_remote",
# ]
#
# guitar_workflow_test(
#     name = "litert_ats_guitar_macos",
#     # This flag is required by guitar::update_culprit_finding_config and shard_count.
#     extract_deps = True,
#     integration_test = guitar.IntegrationTest(
#         tests = [
#             litert_macos_mh_guitar_test(
#                 targets = MACOS_ATS_GUITAR_TARGETS,
#             ),
#         ],
#     ),
#     tags = [
#         "external",
#         "guitar",
#         "manual",
#         "notap",
#     ],
# )
#
# PIXEL_9_ATS_MODELS_GUITAR_TARGETS = [
#     ":gpu_models_ats_remote",
#     ":cpu_models_ats_remote",
#     ":ynnpack_cpu_models_ats_remote",
#     ":builtin_cpu_models_ats_remote",
# ]
#
# guitar_workflow_test(
#     name = "litert_ats_models_guitar_pixel_9",
#     # This flag is required by guitar::update_culprit_finding_config and shard_count.
#     extract_deps = True,
#     integration_test = guitar.IntegrationTest(
#         tests = [
#             litert_pixel_9_mh_guitar_test(
#                 targets = PIXEL_9_ATS_MODELS_GUITAR_TARGETS,
#             ),
#         ],
#     ),
#     tags = [
#         "external",
#         "guitar",
#         "manual",
#         "notap",
#     ],
# )
#
# QUALCOMM_S24_ATS_MODELS_GUITAR_TARGETS = [
#     ":gpu_models_ats_remote",
#     ":cpu_models_ats_remote",
#     ":ynnpack_cpu_models_ats_remote",
#     ":builtin_cpu_models_ats_remote",
# ]
#
# guitar_workflow_test(
#     name = "litert_ats_models_guitar_qualcomm_s24",
#     # This flag is required by guitar::update_culprit_finding_config and shard_count.
#     extract_deps = True,
#     integration_test = guitar.IntegrationTest(
#         tests = [
#             litert_qualcomm_mh_guitar_test(
#                 targets = QUALCOMM_S24_ATS_MODELS_GUITAR_TARGETS,
#             ),
#         ],
#     ),
#     tags = [
#         "external",
#         "guitar",
#         "manual",
#         "notap",
#     ],
# )
#
# MEDIATEK_MT6989_ATS_MODELS_GUITAR_TARGETS = [
#     ":gpu_models_ats_remote",
#     ":cpu_models_ats_remote",
#     ":ynnpack_cpu_models_ats_remote",
#     ":builtin_cpu_models_ats_remote",
# ]
#
# guitar_workflow_test(
#     name = "litert_ats_models_guitar_mediatek_mt6989",
#     # This flag is required by guitar::update_culprit_finding_config and shard_count.
#     extract_deps = True,
#     integration_test = guitar.IntegrationTest(
#         tests = [
#             litert_mediatek_mh_guitar_test(
#                 dimension_hardware = "mt6989",
#                 targets = MEDIATEK_MT6989_ATS_MODELS_GUITAR_TARGETS,
#             ),
#         ],
#     ),
#     tags = [
#         "external",
#         "guitar",
#         "manual",
#         "notap",
#     ],
# )
#
# MACOS_ATS_MODELS_GUITAR_TARGETS = [
#     ":metal_macos_models_ats_remote",
#     ":webgpu_macos_models_ats_remote",
#     ":cpu_macos_models_ats_remote",
#     ":ynnpack_cpu_macos_models_ats_remote",
#     ":builtin_cpu_macos_models_ats_remote",
# ]
#
# guitar_workflow_test(
#     name = "litert_ats_models_guitar_macos",
#     # This flag is required by guitar::update_culprit_finding_config and shard_count.
#     extract_deps = True,
#     integration_test = guitar.IntegrationTest(
#         tests = [
#             litert_macos_mh_guitar_test(
#                 targets = MACOS_ATS_MODELS_GUITAR_TARGETS,
#             ),
#         ],
#     ),
#     tags = [
#         "external",
#         "guitar",
#         "manual",
#         "notap",
#     ],
# )
#
# NPU_AOT_ATS_GUITAR_TARGETS = [
#     ":google_tensor_ats_aot",
#     ":example_ats_aot",
# ]
#
# guitar_workflow_test(
#     name = "litert_ats_guitar_npu_aot",
#     # This flag is required by guitar::update_culprit_finding_config and shard_count.
#     extract_deps = True,
#     integration_test = guitar.IntegrationTest(
#         tests = [
#             litert_npu_aot_guitar_test(
#                 targets = NPU_AOT_ATS_GUITAR_TARGETS,
#             ),
#         ],
#     ),
#     tags = [
#         "external",
#         "guitar",
#         "manual",
#         "notap",
#     ],
# )
# copybara:uncomment_end
