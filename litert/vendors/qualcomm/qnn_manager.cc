// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/qnn_manager.h"

#include <array>
#include <charconv>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/qualcomm/core/backends/dsp_backend.h"
#include "litert/vendors/qualcomm/core/backends/gpu_backend.h"
#include "litert/vendors/qualcomm/core/backends/htp_backend.h"
#include "litert/vendors/qualcomm/core/backends/ir_backend.h"
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/qnn_api_loader.h"
#include "GPU/QnnGpuContext.h"  // from @qairt
#include "HTP/QnnHtpContext.h"  // from @qairt
#include "HTP/QnnHtpProfile.h"  // from @qairt
#include "QnnCommon.h"  // from @qairt
#include "QnnContext.h"  // from @qairt
#include "QnnProfile.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt
#include "System/QnnSystemContext.h"  // from @qairt

namespace litert::qnn {

namespace {

// Compile registers the custom op package against the "CPU" target; dispatch
// uses the caller-specified target.
constexpr char kCustomOpPackageCompileTarget[] = "CPU";

// Registers the configured custom op package (if any) against `qnn_manager`.
// No-op for the IR backend. Called at the tail of QnnManager::Create.
LiteRtStatus RegisterCustomOpPackage(QnnManager& qnn_manager,
                                     QnnManager::Mode mode) {
  const auto& custom_op_package = qnn_manager.GetOptions().GetCustomOpPackage();
  if (custom_op_package.name.empty()) {
    return kLiteRtStatusOk;
  }
  if (qnn_manager.GetOptions().GetBackendType() ==
      ::qnn::BackendType::kIrBackend) {
    LITERT_LOG(LITERT_INFO,
               "Custom op package is not supported in IrBackend. Ignore.");
    return kLiteRtStatusOk;
  }
  // Compile and dispatch consume different fields of the same option.
  const bool is_compile = mode == QnnManager::Mode::kCompile;
  const std::string& package_path =
      is_compile ? custom_op_package.compile_package_path
                 : custom_op_package.dispatch_package_path;
  const char* target = is_compile ? kCustomOpPackageCompileTarget
                                  : custom_op_package.target.c_str();
  if (auto status = qnn_manager.Api()->backendRegisterOpPackage(
          qnn_manager.BackendHandle(), package_path.c_str(),
          custom_op_package.interface_provider.c_str(), target);
      status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to register op package. Error code: %d",
               status);
    return kLiteRtStatusErrorRuntimeFailure;
  }
  LITERT_LOG(LITERT_INFO, "Op package loaded successfully.");
  return kLiteRtStatusOk;
}

}  // namespace

Expected<QnnManager> QnnManager::Create(const QnnApiLoader& loader,
                                        std::optional<::qnn::SocInfo> soc_info,
                                        QnnManager::Mode mode) {
  const ::qnn::Options& options = loader.GetOptions();
  std::unique_ptr<::qnn::QnnBackend> backend;
  ::qnn::SocInfo bound_soc = ::qnn::kSocInfos[0];
  switch (options.GetBackendType()) {
    case ::qnn::BackendType::kGpuBackend: {
      backend = std::make_unique<::qnn::GpuBackend>(loader.Api());
      if (!backend->Init(options, std::nullopt)) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "Failed to init GPU backend");
      }
      break;
    }
    case ::qnn::BackendType::kHtpBackend: {
      auto htp_backend = std::make_unique<::qnn::HtpBackend>(loader.Api());
      if (!htp_backend->Init(options, soc_info)) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "Failed to init HTP backend");
      }
      // Only HTP is SoC-specific; other backends ignore `soc_info`.
      bound_soc = htp_backend->GetSocInfo();
      backend = std::move(htp_backend);
      break;
    }
    case ::qnn::BackendType::kIrBackend: {
      backend = std::make_unique<::qnn::IrBackend>(loader.Api());
      if (!backend->Init(options, std::nullopt)) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "Failed to init IR backend");
      }
      break;
    }
    case ::qnn::BackendType::kDspBackend: {
      backend = std::make_unique<::qnn::DspBackend>(loader.Api());
      if (!backend->Init(options, soc_info)) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "Failed to init DSP backend");
      }
      break;
    }
    default:
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Unsupported backend type");
  }
  LITERT_ASSIGN_OR_RETURN(SdkVersion sdk_version,
                          ParseSdkVersion(loader.GetBuildId().c_str()));
  QnnManager qnn_manager(loader.Api(), loader.SystemApi(), &options,
                         sdk_version, std::move(backend), bound_soc, mode);
  // Fresh backend handle -- apply the configured custom op package.
  LITERT_RETURN_IF_ERROR(RegisterCustomOpPackage(qnn_manager, mode));
  return qnn_manager;
}

Expected<SdkVersion> QnnManager::ParseSdkVersion(const char* build_id) {
  // A generic error to be returned on any parsing failure.
  const auto parsing_error =
      Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to parse build ID");

  std::string_view version_str = build_id;
  if (!build_id) return parsing_error;

  // Check for and remove the 'v' prefix.
  if (version_str.empty() || version_str.front() != 'v') {
    return parsing_error;
  }
  version_str.remove_prefix(1);

  SdkVersion version{};
  const char* current = version_str.data();
  const char* const end = version_str.data() + version_str.size();

  auto parse_component = [&current, &end](int& component) {
    auto [ptr, ec] = std::from_chars(current, end, component);
    if (ec != std::errc()) {
      return false;
    }
    current = ptr;
    return true;
  };

  // Parse major, minor, and patch versions, checking for dots in between.
  if (!parse_component(version.major)) return parsing_error;

  if (current == end || *current++ != '.') return parsing_error;
  if (!parse_component(version.minor)) return parsing_error;

  if (current == end || *current++ != '.') return parsing_error;
  if (!parse_component(version.patch)) return parsing_error;

  return version;
}

LiteRtStatus QnnManager::ValidateOp(::qnn::OpWrapper& op) {
  // TODO(jiunkaiy): Remove version check and break backward compatibility when
  // acceptable.
  // Bypass RmsNorm OP validation.
  if (SdkVersion{2, 35, 0} <= sdk_version_ &&
      sdk_version_ < SdkVersion{2, 37, 0} &&
      op.IsOpCode(::qnn::QnnOpCode::kRmsNorm)) {
    LITERT_LOG(LITERT_WARNING,
               "SDK version is in [2.35.0, 2.37.0); RmsNorm OP validation is "
               "bypassed.");
    return kLiteRtStatusOk;
  }
  // Bypass L2Norm OP validation.
  if (SdkVersion{2, 39, 0} <= sdk_version_ &&
      sdk_version_ < SdkVersion{2, 43, 0} &&
      op.IsOpCode(::qnn::QnnOpCode::kL2Norm)) {
    LITERT_LOG(LITERT_WARNING,
               "SDK version is in [2.39.0, 2.43.0); L2Norm OP validation is "
               "bypassed.");
    return kLiteRtStatusOk;
  }
  // Bypass Quantize OP validation.
  if (SdkVersion{2, 35, 0} <= sdk_version_ &&
      sdk_version_ < SdkVersion{2, 38, 0} &&
      op.IsOpCode(::qnn::QnnOpCode::kQuantize) &&
      op.GetInputTensor(0).IsF32() && op.GetOutputTensor(0).IsQuantI16()) {
    LITERT_LOG(LITERT_WARNING,
               "SDK version is in [2.35.0, 2.38.0); Quantize OP validation is "
               "bypassed.");
    return kLiteRtStatusOk;
  }
  // Bypass Split OP validation.
  if (SdkVersion{2, 35, 0} <= sdk_version_ &&
      sdk_version_ < SdkVersion{2, 37, 0} &&
      op.IsOpCode(::qnn::QnnOpCode::kSplit)) {
    LITERT_LOG(
        LITERT_WARNING,
        "SDK version is in [2.35.0, 2.37.0); Split OP validation is bypassed.");
    return kLiteRtStatusOk;
  }
  const auto op_config = op.GetOpConfig();
  if (Qnn_ErrorHandle_t error =
          Api()->backendValidateOpConfig(BackendHandle(), op_config);
      QNN_SUCCESS != error) {
    LITERT_LOG(LITERT_ERROR, "Failed to validate op %s\n, error: %lld",
               op_config.v1.name, static_cast<long long>(error));
    return kLiteRtStatusErrorInvalidLegalization;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus QnnManager::GenerateContextBinary(
    Qnn_ContextHandle_t context_handle, std::vector<char>& buffer) {
  Qnn_ContextBinarySize_t bin_size = 0;
  if (QNN_SUCCESS != Api()->contextGetBinarySize(context_handle, &bin_size)) {
    LITERT_LOG(LITERT_ERROR, "%s", "Failed to get context bin size\n");
    return kLiteRtStatusErrorNotFound;
  }
  buffer.clear();
  buffer.resize(bin_size);
  Qnn_ContextBinarySize_t written_bin_size = 0;
  if (QNN_SUCCESS != Api()->contextGetBinary(context_handle, buffer.data(),
                                             buffer.size(),
                                             &written_bin_size)) {
    LITERT_LOG(LITERT_ERROR, "%s", "Failed to generated context binary \n");
    return kLiteRtStatusErrorNotFound;
  }
  LITERT_LOG(LITERT_INFO, "Serialized a context bin of size (bytes): %lu\n",
             written_bin_size);
  return kLiteRtStatusOk;
}

Expected<QnnManager::ContextHandle> QnnManager::CreateContextHandle(
    absl::Span<const QnnContext_Config_t*> configs,
    ::qnn::Profiling profiling_level) {
  const QnnApi* api = Api();
  Qnn_ContextHandle_t context_handle;
  if (auto status = api->contextCreate(
          BackendHandle(), DeviceHandle(),
          // `configs` should be null-terminated. For empty `configs`, most
          // backend libraries accept nullptr so we use nullptr directly instead
          // of a array which contains only one nullptr.
          configs.size() <= 1 ? nullptr : configs.data(), &context_handle);
      status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to create QNN context: %d", status);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to create QNN context");
  }
  auto context_deleter = api->contextFree;

  // Return empty profile handle if profiling is off.
  if (profiling_level == ::qnn::Profiling::kOff) {
    return ContextHandle{context_handle, nullptr, context_deleter, nullptr};
  }

  // Create profile handle.
  Qnn_ProfileHandle_t profile_handle = nullptr;
  uint32_t profiling = static_cast<uint32_t>(profiling_level);
  if (profiling_level == ::qnn::Profiling::kLinting) {
    profiling = QNN_HTP_PROFILE_LEVEL_LINTING;
  } else if (profiling_level == ::qnn::Profiling::kOptrace) {
    profiling = QNN_PROFILE_LEVEL_DETAILED;
  }
  if (auto status =
          api->profileCreate(BackendHandle(), profiling, &profile_handle);
      status != QNN_SUCCESS) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to create profile handle");
  }

  // Handle Optrace profile config.
  if (profiling_level == ::qnn::Profiling::kOptrace) {
    static const QnnProfile_Config_t profile_config = {
        .option = QNN_PROFILE_CONFIG_OPTION_ENABLE_OPTRACE, .enableOptrace = 1};
    static std::array<const QnnProfile_Config_t*, 2> results = {&profile_config,
                                                                nullptr};
    if (auto status = api->profileSetConfig(profile_handle, results.data());
        status != QNN_SUCCESS) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to set profile configs");
    }
  }

  return ContextHandle{context_handle, profile_handle, context_deleter,
                       api->profileFree};
}

Expected<QnnManager::ContextHandle> QnnManager::CreateContextHandle(
    absl::Span<const QnnContext_Config_t*> configs,
    absl::Span<const uint8_t> bytecode, Qnn_ProfileHandle_t profile_handle) {
  const QnnApi* api = Api();
  Qnn_ContextHandle_t context_handle;
  if (auto status = api->contextCreateFromBinary(
          BackendHandle(), DeviceHandle(), configs.data(), bytecode.data(),
          bytecode.size(), &context_handle, profile_handle);
      status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to create QNN context: %d", status);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to create QNN context");
  }
  auto context_deleter = api->contextFree;
  auto profile_deleter = api->profileFree;
  return ContextHandle{context_handle, profile_handle, context_deleter,
                       profile_deleter};
}

Expected<QnnManager::SystemContextHandle>
QnnManager::CreateSystemContextHandle() {
  QnnSystemContext_Handle_t system_context_handle;
  if (auto status = SystemApi()->systemContextCreate(&system_context_handle);
      status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to create QNN system context: %d", status);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to create QNN system context");
  }
  auto deleter = SystemApi()->systemContextFree;
  return SystemContextHandle{system_context_handle, deleter};
}

absl::Span<const QnnContext_Config_t*> QnnManager::DefaultContextConfigs() {
  static const QnnContext_Config_t* configs[] = {nullptr};
  return absl::MakeSpan(configs);
}

absl::Span<const QnnContext_Config_t*>
QnnManager::WeightSharingContextConfigs() {
  static QnnHtpContext_CustomConfig_t customConfig =
      QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT;
  customConfig.option = QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED;
  customConfig.weightSharingEnabled = true;
  static QnnContext_Config_t contextConfig = QNN_CONTEXT_CONFIG_INIT;
  contextConfig.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
  contextConfig.customConfig = &customConfig;
  static const QnnContext_Config_t* configs[2] = {&contextConfig, nullptr};
  return absl::MakeSpan(configs);
}

absl::Span<const QnnContext_Config_t*> QnnManager::GpuPerformanceContextConfigs(
    ::qnn::GpuPerformanceMode performance_mode) {
  static QnnGpuContext_CustomConfig_t customConfig =
      QNN_GPU_CONTEXT_CUSTOM_CONFIG_INIT;
  customConfig.option = QNN_GPU_CONTEXT_CONFIG_OPTION_PERF_HINT;
  switch (performance_mode) {
    case ::qnn::GpuPerformanceMode::kHigh:
      customConfig.perfHint = QNN_GPU_CONTEXT_PERF_HINT_HIGH;
      break;
    case ::qnn::GpuPerformanceMode::kNormal:
      customConfig.perfHint = QNN_GPU_CONTEXT_PERF_HINT_NORMAL;
      break;
    case ::qnn::GpuPerformanceMode::kLow:
      customConfig.perfHint = QNN_GPU_CONTEXT_PERF_HINT_LOW;
      break;
    case ::qnn::GpuPerformanceMode::kDefault:
    default:
      return DefaultContextConfigs();
  }

  static QnnContext_Config_t contextConfig = QNN_CONTEXT_CONFIG_INIT;
  contextConfig.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
  contextConfig.customConfig = &customConfig;
  static const QnnContext_Config_t* configs[2] = {&contextConfig, nullptr};
  return absl::MakeSpan(configs);
}

}  // namespace litert::qnn
