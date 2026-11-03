// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/tools/vision_eval/execution.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <sstream>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/btree_set.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_cpu_options.h"
#include "litert/c/options/litert_gpu_options.h"
#include "litert/c/options/litert_intel_openvino_options.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/options/litert_arm_vulkan_ml_options.h"
#include "litert/tools/vision_eval/data.h"
#include "litert/tools/vision_eval/image.h"
#include "google/protobuf/text_format.h"  // from @com_google_protobuf
#include "tflite/converter/schema/schema_generated.h"

namespace litert::vision {
namespace {

void WriteAddModel(const std::filesystem::path& path, tflite::TensorType type,
                   bool quantized, float scale, int64_t zero_point,
                   absl::Span<const uint8_t> rhs_bytes) {
  tflite::ModelT model;
  model.version = 3;
  model.buffers.push_back(std::make_unique<tflite::BufferT>());
  auto rhs_buf = std::make_unique<tflite::BufferT>();
  rhs_buf->data.assign(rhs_bytes.begin(), rhs_bytes.end());
  model.buffers.push_back(std::move(rhs_buf));

  const bool use_squeeze = !quantized && type == tflite::TensorType_INT8;
  auto op_code = std::make_unique<tflite::OperatorCodeT>();
  op_code->builtin_code = use_squeeze ? tflite::BuiltinOperator_SQUEEZE
                                      : tflite::BuiltinOperator_ADD;
  op_code->version =
      (!use_squeeze &&
       (type == tflite::TensorType_INT8 || type == tflite::TensorType_UINT8))
          ? 2
          : 1;
  model.operator_codes.push_back(std::move(op_code));

  auto graph = std::make_unique<tflite::SubGraphT>();
  graph->inputs = {0};
  graph->outputs = {2};
  const char* names[] = {"x", "bias", "y"};
  const int buffers[] = {0, 1, 0};
  for (int i = 0; i < 3; ++i) {
    auto tensor = std::make_unique<tflite::TensorT>();
    tensor->name = names[i];
    tensor->type = type;
    tensor->shape = (use_squeeze && i == 0) ? std::vector<int32_t>{1, 1, 2}
                                            : std::vector<int32_t>{1, 2};
    tensor->buffer = buffers[i];
    if (quantized) {
      tensor->quantization =
          std::make_unique<tflite::QuantizationParametersT>();
      tensor->quantization->scale = {scale};
      tensor->quantization->zero_point = {zero_point};
    }
    graph->tensors.push_back(std::move(tensor));
  }
  auto op = std::make_unique<tflite::OperatorT>();
  op->opcode_index = 0;
  if (use_squeeze) {
    op->inputs = {0};
    op->outputs = {2};
    tflite::SqueezeOptionsT squeeze_opts;
    squeeze_opts.squeeze_dims = {1};
    op->builtin_options.Set(squeeze_opts);
  } else {
    op->inputs = {0, 1};
    op->outputs = {2};
    op->builtin_options.Set(tflite::AddOptionsT());
  }
  graph->operators.push_back(std::move(op));
  model.subgraphs.push_back(std::move(graph));

  auto signature = std::make_unique<tflite::SignatureDefT>();
  signature->signature_key = "serving_default";
  auto in_map = std::make_unique<tflite::TensorMapT>();
  in_map->name = "x";
  in_map->tensor_index = 0;
  signature->inputs.push_back(std::move(in_map));
  auto out_map = std::make_unique<tflite::TensorMapT>();
  out_map->name = "y";
  out_map->tensor_index = 2;
  signature->outputs.push_back(std::move(out_map));
  model.signature_defs.push_back(std::move(signature));

  flatbuffers::FlatBufferBuilder builder;
  tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, &model));
  ASSERT_TRUE(WriteBytes(path, builder.GetBufferPointer(), builder.GetSize()));
}

TEST(BackendTest, CpuBaselineUsesReferenceKernelsWithoutVendorOptions) {
  EvaluationConfig config;
  config.set_baseline_accelerator("cpu");
  config.set_baseline_num_threads(2);
  auto* backend = config.mutable_backend();
  backend->set_accelerator("npu,gpu");
  backend->set_num_threads(4);
  backend->set_xnnpack_only(true);
  backend->set_arm_vulkan_ml_enable_jit(true);
  backend->set_compiler_plugin_library_dir("/plugins");
  backend->set_dispatch_library_dir("/dispatch");
  backend->set_runtime_library_dir("/runtime");
  backend->set_openvino_device("npu");
  backend->set_gpu_use_ir_model("true");
  backend->set_gpu_backend("opencl");
  backend->set_gpu_precision("fp16");
  backend->set_gpu_benchmark_mode(true);
  backend->set_require_full_acceleration(true);
  ASSERT_TRUE(ValidateBackendOptions(*backend));

  const auto empty_env = BuildEnvironmentOptions(BackendOptions{});
  EXPECT_FALSE(
      empty_env.GetOption(EnvironmentOptions::Tag::kRuntimeLibraryDir)
          .HasValue());
  EXPECT_FALSE(
      empty_env.GetOption(EnvironmentOptions::Tag::kDispatchLibraryDir)
          .HasValue());
  EXPECT_FALSE(
      empty_env.GetOption(EnvironmentOptions::Tag::kCompilerPluginLibraryDir)
          .HasValue());

  const auto populated_env = BuildEnvironmentOptions(*backend);
  const auto runtime_opt =
      populated_env.GetOption(EnvironmentOptions::Tag::kRuntimeLibraryDir);
  ASSERT_TRUE(runtime_opt);
  ASSERT_TRUE(std::holds_alternative<const char*>(*runtime_opt));
  EXPECT_STREQ(std::get<const char*>(*runtime_opt), "/runtime");
  const auto dispatch_opt =
      populated_env.GetOption(EnvironmentOptions::Tag::kDispatchLibraryDir);
  ASSERT_TRUE(dispatch_opt);
  ASSERT_TRUE(std::holds_alternative<const char*>(*dispatch_opt));
  EXPECT_STREQ(std::get<const char*>(*dispatch_opt), "/dispatch");
  const auto plugin_opt = populated_env.GetOption(
      EnvironmentOptions::Tag::kCompilerPluginLibraryDir);
  ASSERT_TRUE(plugin_opt);
  ASSERT_TRUE(std::holds_alternative<const char*>(*plugin_opt));
  EXPECT_STREQ(std::get<const char*>(*plugin_opt), "/plugins");

  auto baseline = BaselineOptions(config);
  EXPECT_EQ(baseline.accelerator(), "cpu");
  EXPECT_EQ(baseline.num_threads(), 2);
  EXPECT_FALSE(baseline.xnnpack_only());
  EXPECT_FALSE(baseline.arm_vulkan_ml_enable_jit());
  EXPECT_TRUE(baseline.compiler_plugin_library_dir().empty());
  EXPECT_TRUE(baseline.dispatch_library_dir().empty());
  EXPECT_EQ(baseline.runtime_library_dir(), "/runtime");
  EXPECT_TRUE(baseline.openvino_device().empty());
  EXPECT_EQ(baseline.gpu_backend(), "automatic");
  EXPECT_EQ(baseline.gpu_precision(), "default");
  EXPECT_EQ(baseline.gpu_use_ir_model(), "default");
  EXPECT_FALSE(baseline.gpu_benchmark_mode());
  EXPECT_FALSE(baseline.require_full_acceleration());

  auto options = CreateOptions(baseline, CpuKernelPreference::kReference);
  ASSERT_TRUE(options);
  auto hw = options->GetHardwareAccelerators();
  ASSERT_TRUE(hw);
  EXPECT_EQ(*hw, kLiteRtHwAcceleratorCpu);
  auto cpu = options->GetCpuOptions();
  ASSERT_TRUE(cpu);
  auto threads = cpu->GetNumThreads();
  ASSERT_TRUE(threads);
  EXPECT_EQ(*threads, 2);
  auto ynn = cpu->GetEnableYNNPack();
  ASSERT_TRUE(ynn);
  EXPECT_FALSE(*ynn);
  auto mode = cpu->GetKernelMode();
  ASSERT_TRUE(mode);
  EXPECT_EQ(*mode, kLiteRtCpuKernelModeReference);

  BackendOptions default_threads;
  default_threads.set_accelerator("cpu");
  default_threads.set_num_threads(0);
  auto default_opts =
      CreateOptions(default_threads, CpuKernelPreference::kOptimized);
  ASSERT_TRUE(default_opts);
  auto default_cpu = default_opts->GetCpuOptions();
  ASSERT_TRUE(default_cpu);
  int unset_threads = -1;
  EXPECT_EQ(LrtGetCpuOptionsNumThread(default_cpu->Get(), &unset_threads),
            kLiteRtStatusErrorNotFound);

  auto ref_default_opts =
      CreateOptions(default_threads, CpuKernelPreference::kReference);
  ASSERT_TRUE(ref_default_opts);
  auto ref_default_cpu = ref_default_opts->GetCpuOptions();
  ASSERT_TRUE(ref_default_cpu);
  auto ref_mode = ref_default_cpu->GetKernelMode();
  ASSERT_TRUE(ref_mode);
  EXPECT_EQ(*ref_mode, kLiteRtCpuKernelModeReference);

  default_threads.set_num_threads(1);
  auto single_thread_opts =
      CreateOptions(default_threads, CpuKernelPreference::kOptimized);
  ASSERT_TRUE(single_thread_opts);
  auto single_cpu = single_thread_opts->GetCpuOptions();
  ASSERT_TRUE(single_cpu);
  auto single_threads = single_cpu->GetNumThreads();
  ASSERT_TRUE(single_threads);
  EXPECT_EQ(*single_threads, 1);
}

TEST(BackendTest, XnnpackOnlyConfiguresDelegateKernelModeAndFlags) {
  BackendOptions backend;
  backend.set_accelerator("cpu");
  backend.set_xnnpack_only(true);
  auto options = CreateOptions(backend, CpuKernelPreference::kOptimized);
  ASSERT_TRUE(options);
  auto cpu = options->GetCpuOptions();
  ASSERT_TRUE(cpu);
  auto mode = cpu->GetKernelMode();
  ASSERT_TRUE(mode);
  EXPECT_EQ(*mode, kLiteRtCpuKernelModeDelegate);
  auto ynn = cpu->GetEnableYNNPack();
  ASSERT_TRUE(ynn);
  EXPECT_FALSE(*ynn);
  auto flags = cpu->GetXNNPackFlags();
  ASSERT_TRUE(flags);
  EXPECT_EQ(*flags, 0x00000001u);
}

TEST(BackendTest, GpuAndOpenVinoBackendOptionsAreValidatedAndApplied) {
  const std::pair<const char*, LiteRtHwAcceleratorSet> accel_cases[] = {
      {"cpu", kLiteRtHwAcceleratorCpu},
      {"gpu", kLiteRtHwAcceleratorGpu},
      {"npu", kLiteRtHwAcceleratorNpu},
      {"cpu,gpu,npu", kLiteRtHwAcceleratorCpu | kLiteRtHwAcceleratorGpu |
                          kLiteRtHwAcceleratorNpu},
  };
  for (const auto& [accel_str, expected_hw] : accel_cases) {
    BackendOptions backend;
    backend.set_accelerator(accel_str);
    auto options = CreateOptions(backend);
    ASSERT_TRUE(options);
    auto hw = options->GetHardwareAccelerators();
    ASSERT_TRUE(hw);
    EXPECT_EQ(*hw, expected_hw);
  }

  BackendOptions cpu_only;
  cpu_only.set_accelerator("cpu");
  cpu_only.set_gpu_benchmark_mode(true);
  auto cpu_opts = CreateOptions(cpu_only);
  ASSERT_TRUE(cpu_opts);
  auto untouched_gpu = cpu_opts->GetGpuOptions();
  ASSERT_TRUE(untouched_gpu);
  bool untouched_bench = false;
  EXPECT_EQ(
      LrtGetGpuOptionsBenchmarkMode(&untouched_bench, untouched_gpu->Get()),
      kLiteRtStatusErrorNotFound);

  const std::pair<const char*, LiteRtGpuBackend> gpu_backends[] = {
      {"automatic", kLiteRtGpuBackendAutomatic},
      {"webgpu", kLiteRtGpuBackendWebGpu},
      {"opengl", kLiteRtGpuBackendOpenGl},
      {"gl", kLiteRtGpuBackendOpenGl},
      {"opencl", kLiteRtGpuBackendOpenCl},
      {"cl", kLiteRtGpuBackendOpenCl},
  };
  const std::pair<const char*, LiteRtDelegatePrecision> precisions[] = {
      {"default", kLiteRtDelegatePrecisionDefault},
      {"fp32", kLiteRtDelegatePrecisionFp32},
      {"fp16", kLiteRtDelegatePrecisionFp16},
  };
  for (const auto& [gpu_backend, expected_backend] : gpu_backends) {
    for (const auto& [precision, expected_precision] : precisions) {
      for (const char* ir : {"default", "true", "false"}) {
        for (bool bench : {false, true}) {
          BackendOptions backend;
          backend.set_accelerator("cpu,gpu");
          backend.set_gpu_backend(gpu_backend);
          backend.set_gpu_precision(precision);
          backend.set_gpu_use_ir_model(ir);
          backend.set_gpu_benchmark_mode(bench);
          ASSERT_TRUE(ValidateBackendOptions(backend));
          auto options = CreateOptions(backend);
          ASSERT_TRUE(options);
          auto gpu = options->GetGpuOptions();
          ASSERT_TRUE(gpu);

          LiteRtGpuBackend actual_backend = kLiteRtGpuBackendAutomatic;
          if (std::string(gpu_backend) == "automatic") {
            EXPECT_EQ(LrtGetGpuOptionsGpuBackend(&actual_backend, gpu->Get()),
                      kLiteRtStatusErrorNotFound);
          } else {
            ASSERT_EQ(LrtGetGpuOptionsGpuBackend(&actual_backend, gpu->Get()),
                      kLiteRtStatusOk);
            EXPECT_EQ(actual_backend, expected_backend);
          }

          LiteRtDelegatePrecision actual_prec = kLiteRtDelegatePrecisionDefault;
          if (std::string(precision) == "default") {
            EXPECT_EQ(LrtGetGpuAcceleratorCompilationOptionsPrecision(
                          &actual_prec, gpu->Get()),
                      kLiteRtStatusErrorNotFound);
          } else {
            ASSERT_EQ(LrtGetGpuAcceleratorCompilationOptionsPrecision(
                          &actual_prec, gpu->Get()),
                      kLiteRtStatusOk);
            EXPECT_EQ(actual_prec, expected_precision);
          }

          bool actual_ir = false;
          if (std::string(ir) == "default") {
#if defined(LITERT_GPU_USE_IR_MODEL)
            ASSERT_EQ(LrtGetGpuOptionsUseIrModel(&actual_ir, gpu->Get()),
                      kLiteRtStatusOk);
            EXPECT_EQ(actual_ir, static_cast<bool>(LITERT_GPU_USE_IR_MODEL));
#else
            EXPECT_EQ(LrtGetGpuOptionsUseIrModel(&actual_ir, gpu->Get()),
                      kLiteRtStatusErrorNotFound);
#endif
          } else {
            ASSERT_EQ(LrtGetGpuOptionsUseIrModel(&actual_ir, gpu->Get()),
                      kLiteRtStatusOk);
            EXPECT_EQ(actual_ir, std::string(ir) == "true");
          }

          bool actual_bench = !bench;
          ASSERT_EQ(LrtGetGpuOptionsBenchmarkMode(&actual_bench, gpu->Get()),
                    kLiteRtStatusOk);
          EXPECT_EQ(actual_bench, bench);
        }
      }
    }
  }

  BackendOptions empty_ov;
  empty_ov.set_accelerator("npu");
  auto empty_ov_opts = CreateOptions(empty_ov);
  ASSERT_TRUE(empty_ov_opts);
  auto empty_ov_ext = empty_ov_opts->GetIntelOpenVinoOptions();
  ASSERT_TRUE(empty_ov_ext);
  EXPECT_FALSE(empty_ov_ext->GetGraphBackend(-1).HasValue());

  const std::pair<const char*, LiteRtIntelOpenVinoGraphBackend> ov_cases[] = {
      {"cpu", kLiteRtIntelOpenVinoGraphBackendCPU},
      {"gpu", kLiteRtIntelOpenVinoGraphBackendGPU},
      {"npu", kLiteRtIntelOpenVinoGraphBackendNPU},
  };
  for (const auto& [device, expected_ov] : ov_cases) {
    BackendOptions backend;
    backend.set_accelerator("npu");
    backend.set_openvino_device(device);
    auto options = CreateOptions(backend);
    ASSERT_TRUE(options);
    auto ov = options->GetIntelOpenVinoOptions();
    ASSERT_TRUE(ov);
    auto graph_backend = ov->GetGraphBackend(-1);
    ASSERT_TRUE(graph_backend);
    EXPECT_EQ(*graph_backend, expected_ov);
  }

  BackendOptions auto_ov;
  auto_ov.set_accelerator("npu");
  auto_ov.set_openvino_device("auto");
  auto auto_res = CreateOptions(auto_ov);
  ASSERT_FALSE(auto_res);
  EXPECT_TRUE(absl::StrContains(auto_res.Error().Message(),
                                "openvino_device=auto is not supported"));

  BackendOptions unknown_ov;
  unknown_ov.set_accelerator("npu");
  unknown_ov.set_openvino_device("unknown");
  auto unknown_res = CreateOptions(unknown_ov);
  ASSERT_FALSE(unknown_res);
  EXPECT_TRUE(absl::StrContains(unknown_res.Error().Message(),
                                "Unknown openvino_device"));
}

TEST(BackendTest, InheritedBaselinePreservesOptionsAndArmJitIsOptIn) {
  EvaluationConfig config;
  config.mutable_backend()->set_accelerator("npu,cpu");
  config.mutable_backend()->set_arm_vulkan_ml_enable_jit(true);
  ASSERT_TRUE(ValidateBackendOptions(config.backend()));
  auto baseline = BaselineOptions(config);
  std::string baseline_text;
  std::string expected_text;
  ASSERT_TRUE(google::protobuf::TextFormat::PrintToString(baseline, &baseline_text));
  ASSERT_TRUE(
      google::protobuf::TextFormat::PrintToString(config.backend(), &expected_text));
  EXPECT_EQ(baseline_text, expected_text);
  auto enabled = CreateOptions(baseline);
  ASSERT_TRUE(enabled);
  baseline.set_arm_vulkan_ml_enable_jit(false);
  auto disabled = CreateOptions(baseline);
  ASSERT_TRUE(disabled);
  auto arm = enabled->GetOptions<arm_vulkan_ml::ArmVulkanMLOptions>();
  ASSERT_TRUE(arm);
  auto jit_enabled = arm->GetEnableJustInTime();
  ASSERT_TRUE(jit_enabled);
  EXPECT_TRUE(*jit_enabled);
  auto other = disabled->GetOptions<arm_vulkan_ml::ArmVulkanMLOptions>();
  ASSERT_TRUE(other);
  auto jit_disabled = other->GetEnableJustInTime();
  ASSERT_TRUE(jit_disabled);
  EXPECT_FALSE(*jit_disabled);
}

TEST(BackendTest, RejectsMalformedAcceleratorOptions) {
  BackendOptions backend;
  EXPECT_TRUE(ValidateBackendOptions(backend));

  backend.set_accelerator("");
  EXPECT_FALSE(ValidateBackendOptions(backend));

  backend.set_accelerator(" ");
  EXPECT_FALSE(ValidateBackendOptions(backend));

  backend.set_accelerator("cpu,,gpu");
  auto bad_accel = ValidateBackendOptions(backend);
  ASSERT_FALSE(bad_accel);
  EXPECT_TRUE(absl::StrContains(bad_accel.Error().Message(),
                                "Unknown accelerator"));

  backend.set_accelerator("cpu,gpu");
  backend.set_arm_vulkan_ml_enable_jit(true);
  auto bad_jit = ValidateBackendOptions(backend);
  ASSERT_FALSE(bad_jit);
  EXPECT_TRUE(absl::StrContains(bad_jit.Error().Message(),
                                "Arm Vulkan ML JIT requires"));
  backend.set_arm_vulkan_ml_enable_jit(false);

  backend.set_num_threads(-1);
  auto bad_threads = ValidateBackendOptions(backend);
  ASSERT_FALSE(bad_threads);
  EXPECT_TRUE(absl::StrContains(bad_threads.Error().Message(),
                                "num_threads must be nonnegative"));
  backend.set_num_threads(0);

  backend.set_gpu_use_ir_model("maybe");
  auto bad_ir = ValidateBackendOptions(backend);
  ASSERT_FALSE(bad_ir);
  EXPECT_TRUE(
      absl::StrContains(bad_ir.Error().Message(), "gpu_use_ir_model"));
  backend.set_gpu_use_ir_model("false");
  EXPECT_TRUE(ValidateBackendOptions(backend));

  backend.set_gpu_backend("vulkan");
  auto bad_gpu = ValidateBackendOptions(backend);
  ASSERT_FALSE(bad_gpu);
  EXPECT_TRUE(
      absl::StrContains(bad_gpu.Error().Message(), "Unknown gpu_backend"));
  backend.set_gpu_backend("automatic");

  backend.set_gpu_precision("int8");
  auto bad_prec = ValidateBackendOptions(backend);
  ASSERT_FALSE(bad_prec);
  EXPECT_TRUE(
      absl::StrContains(bad_prec.Error().Message(), "Unknown gpu_precision"));
}

TEST(BackendTest, EncodeImageHandlesFloatInt8Uint8AndRejectsBadInputs) {
  TensorSpec float_spec;
  float_spec.set_type(FLOAT32);
  float_spec.add_shape(2);
  auto nan_res = EncodeImage({std::nanf(""), 1.0f}, float_spec);
  ASSERT_FALSE(nan_res);
  EXPECT_TRUE(
      absl::StrContains(nan_res.Error().Message(), "Nonfinite preprocessed"));
  auto size_res = EncodeImage({1.0f}, float_spec);
  ASSERT_FALSE(size_res);
  EXPECT_TRUE(absl::StrContains(size_res.Error().Message(),
                                "Preprocessed image byte count mismatch"));
  auto encoded_float = EncodeImage({1.5f, -2.0f}, float_spec);
  ASSERT_TRUE(encoded_float);
  ASSERT_EQ(encoded_float->size(), 2 * sizeof(float));
  float unpacked[2] = {};
  std::memcpy(unpacked, encoded_float->data(), sizeof(unpacked));
  EXPECT_FLOAT_EQ(unpacked[0], 1.5f);
  EXPECT_FLOAT_EQ(unpacked[1], -2.0f);

  TensorSpec int8_spec;
  int8_spec.set_type(INT8);
  int8_spec.add_shape(2);
  auto no_scale = EncodeImage({1.0f, 2.0f}, int8_spec);
  ASSERT_FALSE(no_scale);
  EXPECT_TRUE(absl::StrContains(no_scale.Error().Message(),
                                "missing quantization"));
  int8_spec.set_scale(0);
  EXPECT_FALSE(EncodeImage({1.0f, 2.0f}, int8_spec));
  int8_spec.set_scale(0.25f);
  int8_spec.set_zero_point(2);
  auto encoded_int8 = EncodeImage({-1.0f, 100.0f}, int8_spec);
  ASSERT_TRUE(encoded_int8);
  EXPECT_EQ(*encoded_int8,
            (std::vector<uint8_t>{static_cast<uint8_t>(-2), 127}));

  TensorSpec uint8_spec;
  uint8_spec.set_type(UINT8);
  uint8_spec.add_shape(2);
  EXPECT_FALSE(EncodeImage({1.0f, 2.0f}, uint8_spec));
  uint8_spec.set_scale(0.25f);
  uint8_spec.set_zero_point(128);
  auto encoded_uint8 = EncodeImage({-1.0f, 100.0f}, uint8_spec);
  ASSERT_TRUE(encoded_uint8);
  EXPECT_EQ(*encoded_uint8, (std::vector<uint8_t>{124, 255}));

  TensorSpec int32_spec;
  int32_spec.set_type(INT32);
  int32_spec.add_shape(2);
  auto bad_type = EncodeImage(std::vector<float>(8, 0.0f), int32_spec);
  ASSERT_FALSE(bad_type);
  EXPECT_TRUE(absl::StrContains(bad_type.Error().Message(),
                                "Unsupported image input type"));
  int32_spec.set_scale(0.25f);
  EXPECT_FALSE(EncodeImage(std::vector<float>(8, 0.0f), int32_spec));
}

TEST(BackendTest, ExecutionCreatesInvokesAndValidatesModelContracts) {
  const auto dir = Utf8ToPath(testing::TempDir()) / "exec_models";
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  ASSERT_FALSE(ec);

  auto env = Environment::Create({});
  ASSERT_TRUE(env);

  const float fp32_bias[] = {0.5f, -1.5f};
  std::vector<uint8_t> fp32_rhs(sizeof(fp32_bias));
  std::memcpy(fp32_rhs.data(), fp32_bias, sizeof(fp32_bias));
  const auto fp32_path = dir / "fp32_add.tflite";
  ASSERT_NO_FATAL_FAILURE(WriteAddModel(fp32_path, tflite::TensorType_FLOAT32,
                                        /*quantized=*/false, 0.0f, 0,
                                        fp32_rhs));

  BackendOptions cpu_backend;
  cpu_backend.set_accelerator("cpu");
  cpu_backend.set_num_threads(1);
  cpu_backend.set_require_full_acceleration(true);

  EXPECT_FALSE(Execution::Create(*env, cpu_backend, dir / "missing.tflite",
                                 /*signature_index=*/0,
                                 CpuKernelPreference::kOptimized));
  EXPECT_FALSE(Execution::Create(*env, cpu_backend, fp32_path,
                                 /*signature_index=*/99,
                                 CpuKernelPreference::kOptimized));

  auto fp32_exec = Execution::Create(*env, cpu_backend, fp32_path,
                                     /*signature_index=*/0,
                                     CpuKernelPreference::kOptimized);
  ASSERT_TRUE(fp32_exec) << fp32_exec.Error().Message();
  const ExecutionIdentity& fp32_id = (*fp32_exec)->identity();
  ASSERT_EQ((*fp32_exec)->inputs().size(), 1u);
  ASSERT_EQ((*fp32_exec)->outputs().size(), 1u);
  EXPECT_EQ((*fp32_exec)->inputs()[0].name(), "x");
  EXPECT_EQ((*fp32_exec)->outputs()[0].name(), "y");
  EXPECT_FALSE(fp32_id.model().path().empty());
  EXPECT_EQ(fp32_id.options().num_threads(), 1);
  EXPECT_TRUE(fp32_id.fully_accelerated());
  EXPECT_GT(fp32_id.setup_ms(), 0.0);
  EXPECT_GE(fp32_id.total_runtime_ops(), 1u);
  ASSERT_GE(fp32_id.accelerators_size(), 1);
  EXPECT_FALSE(fp32_id.accelerators(0).name().empty());
  EXPECT_EQ(fp32_id.accelerators(0).hardware_type(), kLiteRtHwAcceleratorCpu);
  EXPECT_GE(fp32_id.accelerators(0).delegated_ops(), 1u);
  EXPECT_GE(fp32_id.accelerators(0).partitions(), 1u);

  auto count_err = (*fp32_exec)->Invoke({}, /*warmups=*/0);
  ASSERT_FALSE(count_err);
  EXPECT_TRUE(absl::StrContains(count_err.Error().Message(),
                                "Runtime input count mismatch"));

  auto byte_err =
      (*fp32_exec)->Invoke({std::vector<uint8_t>(4, 0)}, /*warmups=*/0);
  ASSERT_FALSE(byte_err);
  EXPECT_TRUE(absl::StrContains(byte_err.Error().Message(),
                                "Runtime input byte count mismatch"));

  auto valid_in = EncodeImage({1.0f, 2.0f}, (*fp32_exec)->inputs()[0]);
  ASSERT_TRUE(valid_in);
  auto fp32_out = (*fp32_exec)->Invoke({*valid_in}, /*warmups=*/1);
  ASSERT_TRUE(fp32_out) << fp32_out.Error().Message();
  EXPECT_GE(fp32_out->latency_ms, 0.0);
  ASSERT_EQ(fp32_out->outputs.size(), 1u);
  ASSERT_EQ(fp32_out->outputs[0].size(), 2u);
  EXPECT_FLOAT_EQ(fp32_out->outputs[0][0], 1.5f);
  EXPECT_FLOAT_EQ(fp32_out->outputs[0][1], 0.5f);

  const float nan_in[] = {std::nanf(""), 1.0f};
  std::vector<uint8_t> nan_bytes(sizeof(nan_in));
  std::memcpy(nan_bytes.data(), nan_in, sizeof(nan_in));
  auto nan_out = (*fp32_exec)->Invoke({nan_bytes}, /*warmups=*/0);
  ASSERT_FALSE(nan_out);
  EXPECT_TRUE(absl::StrContains(nan_out.Error().Message(),
                                "Nonfinite runtime output: y"));

  auto ref_strict_res = Execution::Create(
      *env, cpu_backend, fp32_path, 0, CpuKernelPreference::kReference);
  ASSERT_FALSE(ref_strict_res);
  EXPECT_TRUE(absl::StrContains(ref_strict_res.Error().Message(),
                                "Incomplete acceleration"));

  cpu_backend.set_require_full_acceleration(false);
  auto ref_fallback = Execution::Create(
      *env, cpu_backend, fp32_path, 0, CpuKernelPreference::kReference);
  ASSERT_TRUE(ref_fallback) << ref_fallback.Error().Message();
  EXPECT_FALSE((*ref_fallback)->identity().fully_accelerated());

  const int32_t int32_bias[] = {1, 2};
  std::vector<uint8_t> int32_rhs(sizeof(int32_bias));
  std::memcpy(int32_rhs.data(), int32_bias, sizeof(int32_bias));
  const auto int32_path = dir / "int32_add.tflite";
  ASSERT_NO_FATAL_FAILURE(WriteAddModel(int32_path, tflite::TensorType_INT32,
                                        /*quantized=*/false, 0.0f, 0,
                                        int32_rhs));
  auto int32_res = Execution::Create(
      *env, cpu_backend, int32_path, 0, CpuKernelPreference::kReference);
  ASSERT_FALSE(int32_res);
  EXPECT_TRUE(absl::StrContains(
      int32_res.Error().Message(),
      "Model interfaces support FP32, INT8 and UINT8"));

  const auto unquant_i8_path = dir / "unquant_i8.tflite";
  ASSERT_NO_FATAL_FAILURE(WriteAddModel(
      unquant_i8_path, tflite::TensorType_INT8, /*quantized=*/false, 0.0f, 0,
      {0, 0}));
  auto unquant_res = Execution::Create(
      *env, cpu_backend, unquant_i8_path, 0, CpuKernelPreference::kReference);
  ASSERT_FALSE(unquant_res);
  EXPECT_TRUE(absl::StrContains(unquant_res.Error().Message(),
                                "Model IO requires per-tensor quantization"));

  const auto i8_path = dir / "i8_add.tflite";
  // Real rhs = {0.0f, 1.0f}.
  ASSERT_NO_FATAL_FAILURE(WriteAddModel(i8_path, tflite::TensorType_INT8,
                                        /*quantized=*/true, 0.25f, 2, {2, 6}));
  auto i8_exec = Execution::Create(
      *env, cpu_backend, i8_path, 0, CpuKernelPreference::kReference);
  ASSERT_TRUE(i8_exec) << i8_exec.Error().Message();
  auto i8_in = EncodeImage({-2.0f, 1.0f}, (*i8_exec)->inputs()[0]);
  ASSERT_TRUE(i8_in);
  auto i8_out = (*i8_exec)->Invoke({*i8_in}, 0);
  ASSERT_TRUE(i8_out) << i8_out.Error().Message();
  EXPECT_FLOAT_EQ(i8_out->outputs[0][0], -2.0f);
  EXPECT_FLOAT_EQ(i8_out->outputs[0][1], 2.0f);

  const auto u8_path = dir / "u8_add.tflite";
  // Real rhs = {1.0f, 2.0f}.
  ASSERT_NO_FATAL_FAILURE(WriteAddModel(
      u8_path, tflite::TensorType_UINT8, /*quantized=*/true, 0.25f, 128,
      {132, 136}));
  auto u8_exec = Execution::Create(
      *env, cpu_backend, u8_path, 0, CpuKernelPreference::kReference);
  ASSERT_TRUE(u8_exec) << u8_exec.Error().Message();
  auto u8_in = EncodeImage({1.0f, 2.0f}, (*u8_exec)->inputs()[0]);
  ASSERT_TRUE(u8_in);
  auto u8_out = (*u8_exec)->Invoke({*u8_in}, 0);
  ASSERT_TRUE(u8_out) << u8_out.Error().Message();
  EXPECT_FLOAT_EQ(u8_out->outputs[0][0], 2.0f);
  EXPECT_FLOAT_EQ(u8_out->outputs[0][1], 4.0f);
}

TEST(BackendTest, DeviceInfoAndRuntimeLibrariesReportHostAndPluginFiles) {
  {
    std::istringstream x86_stream(
        "processor\t: 0\nmodel name\t: Test X86 CPU\nmodel name\t: Second\n");
    const DeviceIdentity x86_id = internal::ParseCpuInfoHardware(x86_stream);
    EXPECT_EQ(x86_id.hardware(), " Test X86 CPU");
  }
  {
    std::istringstream arm_stream(
        "not model name : Wrong\nnot Hardware : Wrong\nHardware\t: Test ARM\n");
    const DeviceIdentity arm_id = internal::ParseCpuInfoHardware(arm_stream);
    EXPECT_EQ(arm_id.hardware(), " Test ARM");
  }
  {
    std::istringstream no_colon(
        "model name without colon\nHardware\t: Ignored\n");
    const DeviceIdentity malformed_id =
        internal::ParseCpuInfoHardware(no_colon);
    EXPECT_FALSE(malformed_id.has_hardware());
  }
  {
    std::istringstream unrelated("bogomips\t: 100.0\n");
    const DeviceIdentity empty_id = internal::ParseCpuInfoHardware(unrelated);
    EXPECT_FALSE(empty_id.has_hardware());
  }

  const auto info = DeviceInfo();
#if defined(__unix__) || defined(__APPLE__)
  EXPECT_GT(info.operating_system().size(), 1u);
  EXPECT_FALSE(info.architecture().empty());
#endif
#if defined(__linux__) && !defined(__ANDROID__)
  EXPECT_FALSE(info.hardware().empty());
#endif
#if defined(__linux__)
  auto host_libs = RuntimeLibraries(EvaluationConfig{});
  ASSERT_TRUE(host_libs);
  EXPECT_FALSE(host_libs->empty());
  for (const auto& lib : *host_libs) {
    EXPECT_TRUE(absl::StrContains(lib.path(), ".so")) << lib.path();
  }
#endif

  const auto dir = Utf8ToPath(testing::TempDir()) / "runtime_libs";
  const auto runtime_dir = dir / "runtime";
  const auto dispatch_dir = dir / "dispatch";
  const auto plugin_dir = dir / "plugin";
  const auto fake_dir_so = dir / "fake_dir.so";
  std::error_code ec;
  std::filesystem::create_directories(runtime_dir, ec);
  ASSERT_FALSE(ec);
  std::filesystem::create_directories(dispatch_dir, ec);
  ASSERT_FALSE(ec);
  std::filesystem::create_directories(plugin_dir, ec);
  ASSERT_FALSE(ec);
  std::filesystem::create_directories(fake_dir_so, ec);
  ASSERT_FALSE(ec);

  const auto dll_path = runtime_dir / "runtime.DLL";
  const auto so_path = dispatch_dir / "libcustom_dispatch.so.1";
  const auto dylib_path = plugin_dir / "libcompiler.DYLIB";
  const auto txt_path = dispatch_dir / "notes.txt";
  const char payload[] = "stub";
  ASSERT_TRUE(WriteBytes(dll_path, payload, sizeof(payload)));
  ASSERT_TRUE(WriteBytes(so_path, payload, sizeof(payload)));
  ASSERT_TRUE(WriteBytes(dylib_path, payload, sizeof(payload)));
  ASSERT_TRUE(WriteBytes(txt_path, payload, sizeof(payload)));
  {
    std::istringstream fake_maps(
        "7f000000-7f001000 r-xp 00000000 00:00 0 [vdso]\n"
        "7f001000-7f002000 r-xp 00000000 00:00 0 relative.so\n"
        "7f002000-7f003000 r--p 00000000 08:01 1 " +
        PathToUtf8(txt_path) +
        "\n7f003000-7f004000 r--p 00000000 08:01 2 " +
        PathToUtf8(fake_dir_so) +
        "\n7f004000-7f005000 r--p 00000000 08:01 3 " +
        PathToUtf8(dir / "deleted.so") +
        "\n7f005000-7f006000 r-xp 00000000 08:01 4 " + PathToUtf8(so_path) +
        "\n");
    EXPECT_EQ(internal::ParseProcMapsLibraries(fake_maps),
              (absl::btree_set<std::string>{PathToUtf8(so_path)}));
  }
  const auto expected_dll = PathToUtf8(std::filesystem::absolute(dll_path, ec));
  ASSERT_FALSE(ec);
  const auto expected_so = PathToUtf8(std::filesystem::absolute(so_path, ec));
  ASSERT_FALSE(ec);
  const auto expected_dylib =
      PathToUtf8(std::filesystem::absolute(dylib_path, ec));
  ASSERT_FALSE(ec);
  const auto expected_txt = PathToUtf8(std::filesystem::absolute(txt_path, ec));
  ASSERT_FALSE(ec);

  EvaluationConfig config;
  config.mutable_backend()->set_runtime_library_dir(PathToUtf8(runtime_dir));
  config.mutable_backend()->set_dispatch_library_dir(PathToUtf8(dispatch_dir));
  config.mutable_backend()->set_compiler_plugin_library_dir(
      PathToUtf8(plugin_dir));
  auto libs = RuntimeLibraries(config);
  ASSERT_TRUE(libs);
  bool found_dll = false;
  bool found_so = false;
  bool found_dylib = false;
  bool found_txt = false;
  for (const auto& lib : *libs) {
    if (lib.path() == expected_dll) found_dll = true;
    if (lib.path() == expected_so) found_so = true;
    if (lib.path() == expected_dylib) found_dylib = true;
    if (lib.path() == expected_txt) found_txt = true;
  }
  EXPECT_TRUE(found_dll);
  EXPECT_TRUE(found_so);
  EXPECT_TRUE(found_dylib);
  EXPECT_FALSE(found_txt);

  config.mutable_backend()->set_runtime_library_dir(
      PathToUtf8(dir / "missing_subdir"));
  auto missing_res = RuntimeLibraries(config);
  ASSERT_FALSE(missing_res);
  EXPECT_TRUE(absl::StrContains(missing_res.Error().Message(),
                                "Cannot list library directory"));
}
}  // namespace
}  // namespace litert::vision
