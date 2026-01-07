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

#include <stdlib.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_shared_library.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/dynamic_loading.h"
#include "litert/vendors/qualcomm/common.h"
#include "litert/vendors/qualcomm/core/backends/dsp_backend.h"
#include "litert/vendors/qualcomm/core/backends/htp_backend.h"
#include "litert/vendors/qualcomm/core/backends/ir_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/qnn_saver_utils.h"
#include "HTP/QnnHtpContext.h"  // from @qairt
#include "HTP/QnnHtpProfile.h"  // from @qairt
#include "QnnCommon.h"  // from @qairt
#include "QnnContext.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt
#include "QnnProfile.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt
#include "System/QnnSystemCommon.h"  // from @qairt
#include "System/QnnSystemContext.h"  // from @qairt
#include "System/QnnSystemInterface.h"  // from @qairt

namespace {
static constexpr int kRequiredNumProviders{1};
}
namespace litert::qnn {

namespace {

RtldFlags GetRtldFlags() {
#if defined(__ANDROID__)
  // Race condition segfault without NoDelete on android.
  return RtldFlags::Lazy().Local().NoDelete();
#else
  return RtldFlags::Default();
#endif
}

constexpr char kLibQnnGetProvidersSymbol[] = "QnnInterface_getProviders";

constexpr char kLibQnnSystemGetProvidersSymbol[] =
    "QnnSystemInterface_getProviders";

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(
    const QnnInterface_t*** provider_list, uint32_t* num_providers);

typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(
    const QnnSystemInterface_t***, uint32_t*);

Expected<absl::Span<const QnnInterface_t*>> LoadProvidersFromLib(
    SharedLibrary& lib) {
  QnnInterfaceGetProvidersFn_t get_providers = nullptr;
  LITERT_ASSIGN_OR_RETURN(get_providers,
                          lib.LookupSymbol<QnnInterfaceGetProvidersFn_t>(
                              kLibQnnGetProvidersSymbol));
  const QnnInterface_t** interface_providers = nullptr;
  uint32_t num_providers = 0;
  if (QNN_SUCCESS != get_providers(&interface_providers, &num_providers)) {
    return Error(kLiteRtStatusErrorRuntimeFailure, "Failed to get providers");
  }
  return absl::MakeSpan(interface_providers, num_providers);
}

Expected<absl::Span<const QnnSystemInterface_t*>> LoadSystemProvidersFromLib(
    SharedLibrary& lib) {
  LITERT_ASSIGN_OR_RETURN(QnnSystemInterfaceGetProvidersFn_t get_providers,
                          lib.LookupSymbol<QnnSystemInterfaceGetProvidersFn_t>(
                              kLibQnnSystemGetProvidersSymbol));
  const QnnSystemInterface_t** interface_providers = nullptr;
  uint32_t num_providers = 0;
  if (QNN_SUCCESS != get_providers(&interface_providers, &num_providers)) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to get system providers");
  }
  return absl::MakeSpan(interface_providers, num_providers);
}

}  // namespace

QnnManager::~QnnManager() = default;

LiteRtStatus QnnManager::LoadLib(absl::string_view path) {
  auto saver_output_dir = options_.GetSaverOutputDir();
  if (saver_output_dir.empty()) {
    LITERT_LOG(LITERT_INFO, "Loading qnn shared library from \"%s\"",
               path.data());
    LITERT_ASSIGN_OR_RETURN(lib_, SharedLibrary::Load(path, GetRtldFlags()));
  } else {
    path = kSaverLibraryName;
    LITERT_LOG(LITERT_INFO, "Loading qnn shared library from \"%s\"",
               path.data());
    LITERT_ASSIGN_OR_RETURN(lib_, SharedLibrary::Load(path, GetRtldFlags()));
    LITERT_RETURN_IF_ERROR(InitSaver(lib_, saver_output_dir));
  }
  LITERT_LOG(LITERT_INFO, "Loaded qnn shared library", "");
  return kLiteRtStatusOk;
}

LiteRtStatus QnnManager::LoadSystemLib(absl::string_view path) {
  auto lib_system_or = SharedLibrary::Load(path, GetRtldFlags());
  if (!lib_system_or) {
    LITERT_LOG(LITERT_ERROR, "%s", lib_system_or.Error().Message().data());
    return lib_system_or.Error().Status();
  }
  lib_system_ = std::move(lib_system_or.Value());
  return kLiteRtStatusOk;
}

const QnnApi* QnnManager::Api() const {
  if (interface_ == nullptr) {
    return nullptr;
  }
  return &interface_->QNN_INTERFACE_VER_NAME;
}

LiteRtStatus QnnManager::ResolveApi(Qnn_Version_t expected_qnn_version) {
  if (!lib_.Loaded()) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "Cannot resolve functions: libQnn*.so has not been loaded.\n");
    return kLiteRtStatusErrorDynamicLoading;
  }

  LITERT_ASSIGN_OR_RETURN(auto providers, LoadProvidersFromLib(lib_));
  if (providers.size() != kRequiredNumProviders) {
    LITERT_LOG(LITERT_ERROR, "Found %zu providers, expected %u",
               providers.size(), kRequiredNumProviders);
    return kLiteRtStatusErrorDynamicLoading;
  }

  auto qnn_version = providers[0]->apiVersion;
  // Check api version
  if (qnn_version.coreApiVersion.major != QNN_API_VERSION_MAJOR) {
    LITERT_LOG(LITERT_ERROR,
               "Qnn library version %u.%u.%u is not supported. "
               "The minimum supported version is %u.%u.%u. Please make "
               "sure you have the correct library version.",
               qnn_version.coreApiVersion.major,
               qnn_version.coreApiVersion.minor,
               qnn_version.coreApiVersion.patch, QNN_API_VERSION_MAJOR,
               QNN_API_VERSION_MINOR, QNN_API_VERSION_PATCH);
    return kLiteRtStatusErrorDynamicLoading;
  }

  if ((qnn_version.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
       qnn_version.coreApiVersion.minor < QNN_API_VERSION_MINOR)) {
    LITERT_LOG(LITERT_ERROR,
               "Qnn library version %u.%u.%u is mismatched. "
               "The minimum supported version is %u.%u.%u. Please make "
               "sure you have the correct library version.",
               qnn_version.coreApiVersion.major,
               qnn_version.coreApiVersion.minor,
               qnn_version.coreApiVersion.patch, QNN_API_VERSION_MAJOR,
               QNN_API_VERSION_MINOR, QNN_API_VERSION_PATCH);
    return kLiteRtStatusErrorDynamicLoading;
  }

  if (qnn_version.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
      qnn_version.coreApiVersion.minor > QNN_API_VERSION_MINOR) {
    LITERT_LOG(LITERT_WARNING,
               "Qnn library version %u.%u.%u is used. "
               "The version LiteRT using is %u.%u.%u.",
               qnn_version.coreApiVersion.major,
               qnn_version.coreApiVersion.minor,
               qnn_version.coreApiVersion.patch, QNN_API_VERSION_MAJOR,
               QNN_API_VERSION_MINOR, QNN_API_VERSION_PATCH);
  }

  if (!options_.GetSaverOutputDir().empty()) {
    expected_qnn_version = GetExpectedSaverVersion();
  }
  // Check backend version
  if (qnn_version.backendApiVersion.major != expected_qnn_version.major) {
    LITERT_LOG(LITERT_ERROR,
               "Qnn backend library version %u.%u.%u is not supported. "
               "The minimum supported version is %u.%u.%u. Please make "
               "sure you have the correct library version.",
               qnn_version.backendApiVersion.major,
               qnn_version.backendApiVersion.minor,
               qnn_version.backendApiVersion.patch, expected_qnn_version.major,
               expected_qnn_version.minor, expected_qnn_version.patch);
    return kLiteRtStatusErrorDynamicLoading;
  }

  if ((qnn_version.backendApiVersion.major == expected_qnn_version.major &&
       qnn_version.backendApiVersion.minor < expected_qnn_version.minor)) {
    LITERT_LOG(LITERT_ERROR,
               "Qnn backend library version %u.%u.%u is mismatched. "
               "The minimum supported version is %u.%u.%u. Please make "
               "sure you have the correct library version.",
               qnn_version.backendApiVersion.major,
               qnn_version.backendApiVersion.minor,
               qnn_version.backendApiVersion.patch, expected_qnn_version.major,
               expected_qnn_version.minor, expected_qnn_version.patch);
    return kLiteRtStatusErrorDynamicLoading;
  }

  if (qnn_version.backendApiVersion.major == expected_qnn_version.major &&
      qnn_version.backendApiVersion.minor > expected_qnn_version.minor) {
    LITERT_LOG(LITERT_WARNING,
               "Qnn backend library version %u.%u.%u is used. "
               "The version LiteRT using is %u.%u.%u.",
               qnn_version.backendApiVersion.major,
               qnn_version.backendApiVersion.minor,
               qnn_version.backendApiVersion.patch, expected_qnn_version.major,
               expected_qnn_version.minor, expected_qnn_version.patch);
  }
  interface_ = providers[0];

  if (interface_ == nullptr) {
    LITERT_LOG(LITERT_ERROR, "%s", "No valid interface was provided\n");
    return kLiteRtStatusErrorDynamicLoading;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus QnnManager::ResolveSystemApi() {
  LITERT_ASSIGN_OR_RETURN(auto system_providers,
                          LoadSystemProvidersFromLib(lib_system_));
  if (system_providers.size() != kRequiredNumProviders) {
    LITERT_LOG(LITERT_ERROR, "Found %zu system providers, expected %u",
               system_providers.size(), kRequiredNumProviders);
    return kLiteRtStatusErrorDynamicLoading;
  }

  auto qnn_system_version = system_providers[0]->systemApiVersion;
  if (qnn_system_version.major != QNN_SYSTEM_API_VERSION_MAJOR) {
    LITERT_LOG(LITERT_ERROR,
               "Qnn System library version %u.%u.%u is not supported. "
               "The minimum supported version is %u.%u.%u. Please make "
               "sure you have the correct library version.",
               qnn_system_version.major, qnn_system_version.minor,
               qnn_system_version.patch, QNN_SYSTEM_API_VERSION_MAJOR,
               QNN_SYSTEM_API_VERSION_MINOR, QNN_SYSTEM_API_VERSION_PATCH);
    return kLiteRtStatusErrorDynamicLoading;
  }

  if ((qnn_system_version.major == QNN_SYSTEM_API_VERSION_MAJOR &&
       qnn_system_version.minor < QNN_SYSTEM_API_VERSION_MINOR)) {
    LITERT_LOG(LITERT_ERROR,
               "Qnn System library version %u.%u.%u is mismatched. "
               "The minimum supported version is %u.%u.%u. Please make "
               "sure you have the correct library version.",
               qnn_system_version.major, qnn_system_version.minor,
               qnn_system_version.patch, QNN_SYSTEM_API_VERSION_MAJOR,
               QNN_SYSTEM_API_VERSION_MINOR, QNN_SYSTEM_API_VERSION_PATCH);
    return kLiteRtStatusErrorDynamicLoading;
  }

  if (qnn_system_version.major == QNN_SYSTEM_API_VERSION_MAJOR &&
      qnn_system_version.minor > QNN_SYSTEM_API_VERSION_MINOR) {
    LITERT_LOG(LITERT_WARNING,
               "Qnn System library version %u.%u.%u is used. "
               "The version LiteRT using is %u.%u.%u.",
               qnn_system_version.major, qnn_system_version.minor,
               qnn_system_version.patch, QNN_SYSTEM_API_VERSION_MAJOR,
               QNN_SYSTEM_API_VERSION_MINOR, QNN_SYSTEM_API_VERSION_PATCH);
  }
  system_interface_ = system_providers[0];

  if (system_interface_ == nullptr) {
    LITERT_LOG(LITERT_ERROR, "%s", "No valid system interface was provided\n");
    return kLiteRtStatusErrorDynamicLoading;
  }

  return kLiteRtStatusOk;
}

const QnnSystemApi* QnnManager::SystemApi() const {
  if (system_interface_ == nullptr) {
    return nullptr;
  }
  return &system_interface_->QNN_SYSTEM_INTERFACE_VER_NAME;
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

LiteRtStatus QnnManager::ValidateOp(const Qnn_OpConfig_t& op_config) {
  // TODO: Unblock QNN validation for RMSNorm
  if (absl::StrContains(op_config.v1.name, "RmsNorm") ||
      absl::StrContains(op_config.v1.name, "L2Norm")) {
    return kLiteRtStatusOk;
  }

  if (Qnn_ErrorHandle_t error =
          Api()->backendValidateOpConfig(BackendHandle(), op_config);
      QNN_SUCCESS != error) {
    LITERT_LOG(LITERT_ERROR, "Failed to validate op %s\n, error: %lld",
               op_config.v1.name, static_cast<long long>(error));
    return kLiteRtStatusErrorInvalidLegalization;
  }

  return kLiteRtStatusOk;
}


LiteRtStatus QnnManager::Init(std::optional<std::string> shared_library_dir,
                              std::optional<::qnn::SocInfo> soc_info,
                              const ::qnn::Options& options) {
  // If shared_library_dir is provided, add it to the path as it may contain
  // libs to be loaded.
  // TOOD: This should probably be done upstream in litert_dispatch.
  if (shared_library_dir) {
    LITERT_LOG(LITERT_INFO, "Adding shared library dir to path: %s",
               shared_library_dir->c_str());

    // Always overwrite the environment variable as we want to use the
    // provided library paths only.
    static constexpr char kAdsp[] = "ADSP_LIBRARY_PATH";
    const char* adsp_library_path = getenv(kAdsp);
    if (adsp_library_path == nullptr) {
      setenv(kAdsp, shared_library_dir->data(), /*overwrite=*/1);
    } else {
      auto new_adsp_library_path =
          absl::StrCat(shared_library_dir.value(), ";", adsp_library_path);
      setenv(kAdsp, new_adsp_library_path.c_str(), /*overwrite=*/1);
    }
    LITERT_LOG(LITERT_DEBUG, "ADSP_LIBRARY_PATH: %s", getenv(kAdsp));

    // TODO: Put dynamic loading module in cc or vendor/cc.
    litert::internal::PutLibOnLdPath(*shared_library_dir,
                                     ::qnn::HtpBackend::GetLibraryName());
  }

  LITERT_RETURN_IF_ERROR(LoadSystemLib(kLibQnnSystemSo));
  LITERT_RETURN_IF_ERROR(ResolveSystemApi());

  options_ = options;
  switch (options_.GetBackendType()) {
    case ::qnn::BackendType::kHtpBackend: {
      LITERT_RETURN_IF_ERROR(LoadLib(::qnn::HtpBackend::GetLibraryName()));
      LITERT_RETURN_IF_ERROR(
          ResolveApi(::qnn::HtpBackend::GetExpectedBackendVersion()));

      auto htp_backend = std::make_unique<::qnn::HtpBackend>(Api());
      LITERT_RETURN_IF_ERROR(htp_backend->Init(options_, soc_info));
      soc_info_ = htp_backend->GetSocInfo();
      backend_ = std::move(htp_backend);

      break;
    }
    case ::qnn::BackendType::kIrBackend: {
      LITERT_RETURN_IF_ERROR(LoadLib(::qnn::IrBackend::GetLibraryName()));
      LITERT_RETURN_IF_ERROR(
          ResolveApi(::qnn::IrBackend::GetExpectedBackendVersion()));

      backend_ = std::make_unique<::qnn::IrBackend>(Api());
      LITERT_RETURN_IF_ERROR(backend_->Init(options_, std::nullopt));

      break;
    }
    case ::qnn::BackendType::kDspBackend: {
      LITERT_RETURN_IF_ERROR(LoadLib(::qnn::DspBackend::GetLibraryName()));
      LITERT_RETURN_IF_ERROR(
          ResolveApi(::qnn::DspBackend::GetExpectedBackendVersion()));

      backend_ = std::make_unique<::qnn::DspBackend>(Api());
      LITERT_RETURN_IF_ERROR(backend_->Init(options_, soc_info));

      break;
    }
    default: {
      LITERT_LOG(LITERT_ERROR, "Unsupported backend type: %d",
                 options_.GetBackendType());
      return kLiteRtStatusErrorRuntimeFailure;
    }
  }

  return kLiteRtStatusOk;
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

Expected<QnnManager::ContextHandle> QnnManager::CreateContextHandle(
    absl::Span<const QnnContext_Config_t*> configs,
    ::qnn::Profiling profiling_level) {
  Qnn_ContextHandle_t context_handle;
  if (auto status = Api()->contextCreate(
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
  auto context_deleter = Api()->contextFree;

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
          Api()->profileCreate(BackendHandle(), profiling, &profile_handle);
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
    if (auto status = Api()->profileSetConfig(profile_handle, results.data());
        status != QNN_SUCCESS) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to set profile configs");
    }
  }

  return ContextHandle{context_handle, profile_handle, context_deleter,
                       Api()->profileFree};
}

Expected<QnnManager::ContextHandle> QnnManager::CreateContextHandle(
    absl::Span<const QnnContext_Config_t*> configs,
    absl::Span<const uint8_t> bytecode, Qnn_ProfileHandle_t profile_handle) {
  Qnn_ContextHandle_t context_handle;
  if (auto status = Api()->contextCreateFromBinary(
          BackendHandle(), DeviceHandle(), configs.data(), bytecode.data(),
          bytecode.size(), &context_handle, profile_handle);
      status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to create QNN context: %d", status);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to create QNN context");
  }
  auto context_deleter = Api()->contextFree;
  auto profile_deleter = Api()->profileFree;
  return ContextHandle{context_handle, profile_handle, context_deleter,
                       profile_deleter};
}

Expected<QnnManager::Ptr> QnnManager::Create(
    const ::qnn::Options& options,
    std::optional<std::string> shared_library_dir,
    std::optional<::qnn::SocInfo> soc_info) {
  Ptr qnn_manager(new QnnManager);
  if (auto status = qnn_manager->Init(shared_library_dir, soc_info, options);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to set up QNN manager");
  }
  return qnn_manager;
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

};  // namespace litert::qnn
