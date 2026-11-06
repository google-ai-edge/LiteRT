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
#include <system_error>  // NOLINT(build/c++11)
#include <vector>

#include <gtest/gtest.h>
#include "litert/c/options/litert_cpu_options.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_arm_vulkan_ml_options.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "litert/tools/vision_eval/data.h"

namespace litert::vision {
namespace {

TEST(BackendTest, CpuBaselineUsesReferenceKernelsWithoutVendorOptions) {
  EvaluationConfig config;
  config.set_baseline_accelerator("cpu");
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
  backend->set_require_full_acceleration(true);
  ASSERT_TRUE(ValidateBackendOptions(*backend));
  auto baseline = BaselineOptions(config);
  EXPECT_EQ(baseline.accelerator(), "cpu");
  EXPECT_EQ(baseline.num_threads(), 1);
  EXPECT_FALSE(baseline.arm_vulkan_ml_enable_jit());
  EXPECT_TRUE(baseline.compiler_plugin_library_dir().empty());
  EXPECT_TRUE(baseline.dispatch_library_dir().empty());
  EXPECT_EQ(baseline.runtime_library_dir(), "/runtime");
  EXPECT_TRUE(baseline.openvino_device().empty());
  EXPECT_EQ(baseline.gpu_use_ir_model(), "default");
  EXPECT_FALSE(baseline.require_full_acceleration());
  auto options = CreateOptions(baseline, true);
  ASSERT_TRUE(options);
  auto cpu = options->GetCpuOptions();
  ASSERT_TRUE(cpu);
  auto mode = cpu->GetKernelMode();
  ASSERT_TRUE(mode);
  EXPECT_EQ(*mode, kLiteRtCpuKernelModeReference);
}

TEST(BackendTest, XnnpackOnlyConfiguresDelegateKernelModeAndFlags) {
  BackendOptions backend;
  backend.set_accelerator("cpu");
  backend.set_xnnpack_only(true);
  auto options = CreateOptions(backend, /*reference_cpu=*/false);
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
  for (const char* gpu_backend :
       {"automatic", "webgpu", "opengl", "gl", "opencl", "cl"}) {
    for (const char* precision : {"default", "fp32", "fp16"}) {
      for (const char* ir : {"default", "true", "false"}) {
        BackendOptions backend;
        backend.set_accelerator("gpu");
        backend.set_gpu_backend(gpu_backend);
        backend.set_gpu_precision(precision);
        backend.set_gpu_use_ir_model(ir);
        backend.set_gpu_benchmark_mode(1);
        EXPECT_TRUE(ValidateBackendOptions(backend));
        EXPECT_TRUE(CreateOptions(backend));
      }
    }
  }
  for (const char* device : {"cpu", "gpu", "npu"}) {
    BackendOptions backend;
    backend.set_accelerator("npu");
    backend.set_openvino_device(device);
    EXPECT_TRUE(CreateOptions(backend));
  }
  for (const char* bad_device : {"auto", "unknown"}) {
    BackendOptions backend;
    backend.set_accelerator("npu");
    backend.set_openvino_device(bad_device);
    EXPECT_FALSE(CreateOptions(backend));
  }
}

TEST(BackendTest, InheritedBaselinePreservesOptionsAndArmJitIsOptIn) {
  EvaluationConfig config;
  config.mutable_backend()->set_accelerator("npu");
  config.mutable_backend()->set_arm_vulkan_ml_enable_jit(true);
  auto baseline = BaselineOptions(config);
  EXPECT_EQ(baseline.SerializeAsString(), config.backend().SerializeAsString());
  auto enabled = CreateOptions(baseline);
  ASSERT_TRUE(enabled);
  baseline.set_arm_vulkan_ml_enable_jit(false);
  auto disabled = CreateOptions(baseline);
  ASSERT_TRUE(disabled);
  auto arm = enabled->GetOptions<arm_vulkan_ml::ArmVulkanMLOptions>();
  ASSERT_TRUE(arm);
  ASSERT_TRUE(arm->GetEnableJustInTime());
  auto jit_enabled = arm->GetEnableJustInTime();
  EXPECT_TRUE(*jit_enabled);
  auto other = disabled->GetOptions<arm_vulkan_ml::ArmVulkanMLOptions>();
  ASSERT_TRUE(other);
  ASSERT_TRUE(other->GetEnableJustInTime());
  auto jit_disabled = other->GetEnableJustInTime();
  EXPECT_FALSE(*jit_disabled);
}

TEST(BackendTest, RejectsMalformedAcceleratorOptions) {
  BackendOptions backend;
  EXPECT_TRUE(ValidateBackendOptions(backend));
  backend.set_accelerator("cpu,,gpu");
  EXPECT_FALSE(ValidateBackendOptions(backend));
  backend.set_accelerator("cpu");
  backend.set_arm_vulkan_ml_enable_jit(true);
  EXPECT_FALSE(ValidateBackendOptions(backend));
  backend.set_arm_vulkan_ml_enable_jit(false);
  backend.set_num_threads(-1);
  EXPECT_FALSE(ValidateBackendOptions(backend));
  backend.set_num_threads(0);
  backend.set_gpu_use_ir_model("maybe");
  EXPECT_FALSE(ValidateBackendOptions(backend));
  backend.set_gpu_use_ir_model("false");
  EXPECT_TRUE(ValidateBackendOptions(backend));
  backend.set_gpu_backend("vulkan");
  EXPECT_FALSE(ValidateBackendOptions(backend));
  backend.set_gpu_backend("automatic");
  backend.set_gpu_precision("int8");
  EXPECT_FALSE(ValidateBackendOptions(backend));
}

TEST(BackendTest, EncodeImageHandlesFloatInt8Uint8AndRejectsBadInputs) {
  TensorSpec float_spec;
  float_spec.set_type(FLOAT32);
  float_spec.add_shape(2);
  EXPECT_FALSE(EncodeImage({std::nanf(""), 1.0f}, float_spec));
  EXPECT_FALSE(EncodeImage({1.0f}, float_spec));
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
  EXPECT_FALSE(EncodeImage({1.0f, 2.0f}, int8_spec));
  int8_spec.set_scale(0);
  EXPECT_FALSE(EncodeImage({1.0f, 2.0f}, int8_spec));
  int8_spec.set_scale(0.25f);
  int8_spec.set_zero_point(0);
  auto encoded_int8 = EncodeImage({-1.0f, 100.0f}, int8_spec);
  ASSERT_TRUE(encoded_int8);
  EXPECT_EQ(*encoded_int8,
            (std::vector<uint8_t>{static_cast<uint8_t>(-4), 127}));

  TensorSpec uint8_spec;
  uint8_spec.set_type(UINT8);
  uint8_spec.add_shape(2);
  uint8_spec.set_scale(0.25f);
  uint8_spec.set_zero_point(128);
  auto encoded_uint8 = EncodeImage({-1.0f, 100.0f}, uint8_spec);
  ASSERT_TRUE(encoded_uint8);
  EXPECT_EQ(*encoded_uint8, (std::vector<uint8_t>{124, 255}));
}

TEST(BackendTest, DeviceInfoAndRuntimeLibrariesReportHostAndPluginFiles) {
  const auto info = DeviceInfo();
#if defined(__unix__) || defined(__APPLE__)
  EXPECT_FALSE(info.operating_system().empty());
  EXPECT_FALSE(info.architecture().empty());
#endif

  const auto dir = Utf8ToPath(testing::TempDir()) / "runtime_libs";
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  ASSERT_FALSE(ec);
  const auto so_path = dir / "libcustom_dispatch.so";
  const auto txt_path = dir / "notes.txt";
  const char payload[] = "stub";
  ASSERT_TRUE(WriteBytes(so_path, payload, sizeof(payload)));
  ASSERT_TRUE(WriteBytes(txt_path, payload, sizeof(payload)));
  const auto expected_so = PathToUtf8(std::filesystem::absolute(so_path, ec));
  ASSERT_FALSE(ec);
  const auto expected_txt = PathToUtf8(std::filesystem::absolute(txt_path, ec));
  ASSERT_FALSE(ec);

  EvaluationConfig config;
  config.mutable_backend()->set_dispatch_library_dir(PathToUtf8(dir));
  auto libs = RuntimeLibraries(config);
  ASSERT_TRUE(libs);
  bool found_so = false;
  bool found_txt = false;
  for (const auto& lib : *libs) {
    if (lib.path() == expected_so) found_so = true;
    if (lib.path() == expected_txt) found_txt = true;
  }
  EXPECT_TRUE(found_so);
  EXPECT_FALSE(found_txt);

  config.mutable_backend()->set_runtime_library_dir(
      PathToUtf8(dir / "missing_subdir"));
  EXPECT_FALSE(RuntimeLibraries(config));
}
}  // namespace
}  // namespace litert::vision
