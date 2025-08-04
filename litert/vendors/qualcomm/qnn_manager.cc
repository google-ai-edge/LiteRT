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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "HTP/QnnHtpContext.h"          // from @qairt
#include "HTP/QnnHtpDevice.h"           // from @qairt
#include "QnnBackend.h"                 // from @qairt
#include "QnnCommon.h"                  // from @qairt
#include "QnnContext.h"                 // from @qairt
#include "QnnDevice.h"                  // from @qairt
#include "QnnInterface.h"               // from @qairt
#include "QnnLog.h"                     // from @qairt
#include "QnnTypes.h"                   // from @qairt
#include "System/QnnSystemCommon.h"     // from @qairt
#include "System/QnnSystemContext.h"    // from @qairt
#include "System/QnnSystemInterface.h"  // from @qairt
#include "absl/strings/match.h"         // from @com_google_absl
#include "absl/strings/string_view.h"   // from @com_google_absl
#include "absl/types/span.h"            // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_shared_library.h"
#include "litert/core/dynamic_loading.h"
#include "litert/vendors/qualcomm/common.h"
#include "litert/vendors/qualcomm/core/backends/htp_backend.h"
#include "litert/vendors/qualcomm/core/backends/ir_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"

namespace {
static constexpr int kRequiredNumProviders{1};
}
namespace litert::qnn {

namespace {

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

QnnManager::~QnnManager() {
  if (backend_) backend_->GetPerfControl().Terminate();
}

LiteRtStatus QnnManager::LoadLib(absl::string_view path) {
  LITERT_LOG(LITERT_INFO, "Loading qnn shared library from \"%s\"",
             path.data());
  LITERT_ASSIGN_OR_RETURN(lib_,
                          SharedLibrary::Load(path, RtldFlags::Default()));
  LITERT_LOG(LITERT_INFO, "Loaded qnn shared library", "");
  return kLiteRtStatusOk;
}

LiteRtStatus QnnManager::LoadSystemLib(absl::string_view path) {
  LITERT_ASSIGN_OR_RETURN(lib_system_,
                          SharedLibrary::Load(path, RtldFlags::Default()));
  return kLiteRtStatusOk;
}

const QnnApi* QnnManager::Api() const {
  if (interface_ == nullptr) {
    return nullptr;
  }
  return &interface_->QNN_INTERFACE_VER_NAME;
}

LiteRtStatus QnnManager::ResolveApi(Qnn_Version_t golden_qnn_version) {
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

  // Check backend version
  if (qnn_version.backendApiVersion.major != golden_qnn_version.major) {
    LITERT_LOG(LITERT_ERROR,
               "Qnn backend library version %u.%u.%u is not supported. "
               "The minimum supported version is %u.%u.%u. Please make "
               "sure you have the correct library version.",
               qnn_version.backendApiVersion.major,
               qnn_version.backendApiVersion.minor,
               qnn_version.backendApiVersion.patch, golden_qnn_version.major,
               golden_qnn_version.minor, golden_qnn_version.patch);
    return kLiteRtStatusErrorDynamicLoading;
  }

  if ((qnn_version.backendApiVersion.major == golden_qnn_version.major &&
       qnn_version.backendApiVersion.minor < golden_qnn_version.minor)) {
    LITERT_LOG(LITERT_ERROR,
               "Qnn backend library version %u.%u.%u is mismatched. "
               "The minimum supported version is %u.%u.%u. Please make "
               "sure you have the correct library version.",
               qnn_version.backendApiVersion.major,
               qnn_version.backendApiVersion.minor,
               qnn_version.backendApiVersion.patch, golden_qnn_version.major,
               golden_qnn_version.minor, golden_qnn_version.patch);
    return kLiteRtStatusErrorDynamicLoading;
  }

  if (qnn_version.backendApiVersion.major == golden_qnn_version.major &&
      qnn_version.backendApiVersion.minor > golden_qnn_version.minor) {
    LITERT_LOG(LITERT_WARNING,
               "Qnn backend library version %u.%u.%u is used. "
               "The version LiteRT using is %u.%u.%u.",
               qnn_version.backendApiVersion.major,
               qnn_version.backendApiVersion.minor,
               qnn_version.backendApiVersion.patch, golden_qnn_version.major,
               golden_qnn_version.minor, golden_qnn_version.patch);
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
  if (absl::StrContains(op_config.v1.name, "RmsNorm")) {
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

std::optional<::qnn::SocInfo> FindSocInfo(
    const ::qnn::SnapdragonModel& soc_model) {
  for (auto i = 0; i < ::qnn::kNumSocInfos; ++i) {
    if (soc_model == ::qnn::kSocInfos[i].soc_model) {
      return ::qnn::kSocInfos[i];
    }
  }
  LITERT_LOG(LITERT_ERROR, "Failed to find available SoC!");
  return std::nullopt;
}

LiteRtStatus QnnManager::Init(absl::Span<const QnnBackend_Config_t*> configs,
                              std::optional<std::string> shared_library_dir,
                              std::optional<::qnn::SocInfo> soc_info,
                              const ::qnn::Options& options) {
  // If shared_library_dir is provided, add it to the path as it may contain
  // libs to be loaded.
  // TOOD: This should probably be done upstream in litert_dispatch.
  if (shared_library_dir) {
    LITERT_LOG(LITERT_INFO, "Adding shared library dir to path: %s",
               shared_library_dir->c_str());

    static constexpr char kAdsp[] = "ADSP_LIBRARY_PATH";
    if (getenv(kAdsp) == nullptr) {
      setenv(kAdsp, shared_library_dir->data(), /*overwrite=*/1);
    }

    // TODO: Put dynamic loading module in cc or vendor/cc.
    litert::internal::PutLibOnLdPath(shared_library_dir->data(),
                                     ::qnn::HtpBackend::GetLibraryName());
  }

  LITERT_RETURN_IF_ERROR(LoadSystemLib(kLibQnnSystemSo));
  LITERT_RETURN_IF_ERROR(ResolveSystemApi());

  switch (options.GetBackendType()) {
    case ::qnn::BackendType::kHtpBackend: {
      LITERT_RETURN_IF_ERROR(LoadLib(::qnn::HtpBackend::GetLibraryName()));
      LITERT_RETURN_IF_ERROR(
          ResolveApi(::qnn::HtpBackend::GetExpectedBackendVersion()));

      backend_ = std::make_unique<::qnn::HtpBackend>(Api());
      backend_->Init(options, soc_info);
      soc_info_ = static_cast<::qnn::HtpBackend*>(backend_.get())->GetSocInfo();

      break;
    }
    case ::qnn::BackendType::kIrBackend: {
      LITERT_RETURN_IF_ERROR(LoadLib(::qnn::IrBackend::GetLibraryName()));
      LITERT_RETURN_IF_ERROR(
          ResolveApi(::qnn::IrBackend::GetExpectedBackendVersion()));

      backend_ = std::make_unique<::qnn::IrBackend>(Api());
      backend_->Init(options, std::nullopt);

      break;
    }
    default: {
      LITERT_LOG(LITERT_ERROR, "Unsupported backend type: %d",
                 options.GetBackendType());
      return kLiteRtStatusErrorRuntimeFailure;
      break;
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
    absl::Span<const QnnContext_Config_t*> configs) {
  Qnn_ContextHandle_t context_handle;
  if (auto status = Api()->contextCreate(
          BackendHandle(), DeviceHandle(),
          configs.size() == 1 ? nullptr : configs.data(), &context_handle);
      status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to create QNN context: %d", status);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to create QNN context");
  }
  auto deleter = Api()->contextFree;
  return ContextHandle{context_handle, /*profile=*/nullptr, deleter, nullptr};
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
    absl::Span<const QnnBackend_Config_t*> configs,
    const ::qnn::Options& options,
    std::optional<std::string> shared_library_dir,
    std::optional<::qnn::SocInfo> soc_info) {
  Ptr qnn_manager(new QnnManager);
  if (auto status =
          qnn_manager->Init(configs, shared_library_dir, soc_info, options);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to set up QNN manager");
  }
  return qnn_manager;
}

absl::Span<const QnnBackend_Config_t*> QnnManager::DefaultBackendConfigs() {
  static const QnnBackend_Config_t* configs[] = {nullptr};
  return absl::MakeSpan(configs);
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
