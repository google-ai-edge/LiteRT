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
#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "GPU/QnnGpuContext.h"  // from @qairt
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
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_shared_library.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/filesystem.h"
#include "litert/vendors/qualcomm/common.h"
#include "litert/vendors/qualcomm/core/backends/dsp_backend.h"
#include "litert/vendors/qualcomm/core/backends/gpu_backend.h"
#include "litert/vendors/qualcomm/core/backends/htp_backend.h"
#include "litert/vendors/qualcomm/core/backends/ir_backend.h"
#include "litert/vendors/qualcomm/core/backends/lpai_backend.h"
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/qnn_saver_utils.h"

namespace {
static constexpr int kRequiredNumProviders{1};
}
namespace litert::qnn {

namespace {

LiteRtStatus SetEnvVar(const char* name, const char* value) {
#if defined(_WIN32)
  if (_putenv_s(name, value) != 0) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
#else
  if (setenv(name, value, /*overwrite=*/1) != 0) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
#endif
  return kLiteRtStatusOk;
}

RtldFlags GetRtldFlags(bool needs_global_symbols) {
#if defined(__ANDROID__)
  // Race condition segfault without NoDelete on android.
  return RtldFlags::Lazy().Local().NoDelete();
#else
  return needs_global_symbols ? RtldFlags::Lazy().Global()
                              : RtldFlags::Default();
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

LiteRtStatus QnnManager::LoadSharedLibHelper(absl::string_view path,
                                             bool needs_global_symbols,
                                             SharedLibrary& out_lib) {
  const auto rtld_flags = GetRtldFlags(needs_global_symbols);

  // 1. Try explicit qnn_lib_dir from Options
  if (!options_.GetQnnLibDir().empty()) {
    std::string resolved_path =
        litert::internal::Join({options_.GetQnnLibDir(), path});
    LITERT_LOG(LITERT_INFO, "Loading qnn shared library from \"%s\"",
               resolved_path.c_str());
    auto lib_or = SharedLibrary::Load(resolved_path, rtld_flags);
    if (lib_or) {
      out_lib = std::move(lib_or.Value());
      return kLiteRtStatusOk;
    }
    LITERT_LOG(LITERT_INFO, "Falling back from qnn_lib_dir to loading \"%s\"",
               path.data());
  }

  // 2. Try shared_library_dir_ (plugin directory)
  if (shared_library_dir_ && !shared_library_dir_->empty() &&
      *shared_library_dir_ != options_.GetQnnLibDir()) {
    std::string resolved_path =
        litert::internal::Join({*shared_library_dir_, path});
    LITERT_LOG(LITERT_INFO, "Loading qnn shared library from \"%s\"",
               resolved_path.c_str());
    auto lib_or = SharedLibrary::Load(resolved_path, rtld_flags);
    if (lib_or) {
      out_lib = std::move(lib_or.Value());
      return kLiteRtStatusOk;
    }
    LITERT_LOG(LITERT_INFO,
               "Falling back from shared_library_dir to loading \"%s\"",
               path.data());
  }

  // 3. Fallback to path directly (system loader paths)
  LITERT_LOG(LITERT_INFO, "Loading qnn shared library from \"%s\"",
             path.data());
  auto lib_or = SharedLibrary::Load(path, rtld_flags);
  if (!lib_or) {
    LITERT_LOG(LITERT_ERROR,
               "Failed to load qnn shared library from \"%s\": %s", path.data(),
               lib_or.Error().Message().data());
    return lib_or.Error().Status();
  }
  out_lib = std::move(lib_or.Value());
  return kLiteRtStatusOk;
}

LiteRtStatus QnnManager::LoadLib(absl::string_view path) {
  auto saver_output_dir = options_.GetSaverOutputDir();
  const bool needs_global_symbols = !options_.GetCustomOpPackage().name.empty();
  if (saver_output_dir.empty()) {
    LITERT_RETURN_IF_ERROR(
        LoadSharedLibHelper(path, needs_global_symbols, lib_));
  } else {
    path = kSaverLibraryName;
    LITERT_RETURN_IF_ERROR(
        LoadSharedLibHelper(path, needs_global_symbols, lib_));
    LITERT_RETURN_IF_ERROR(InitSaver(lib_, saver_output_dir));
  }
  LITERT_LOG(LITERT_INFO, "Loaded qnn shared library", "");
  return kLiteRtStatusOk;
}

LiteRtStatus QnnManager::LoadSystemLib(absl::string_view path) {
  const bool needs_global_symbols = !options_.GetCustomOpPackage().name.empty();
  return LoadSharedLibHelper(path, needs_global_symbols, lib_system_);
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

  auto providers_or = LoadProvidersFromLib(lib_);
  if (!providers_or) {
    LITERT_LOG(LITERT_ERROR, "Failed to load providers from library: %s",
               providers_or.Error().Message().data());
    return providers_or.Error().Status();
  }
  auto providers = std::move(providers_or.Value());

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
  auto system_providers_or = LoadSystemProvidersFromLib(lib_system_);
  if (!system_providers_or) {
    LITERT_LOG(LITERT_ERROR, "Failed to load system providers: %s",
               system_providers_or.Error().Message().data());
    return system_providers_or.Error().Status();
  }
  auto system_providers = std::move(system_providers_or.Value());

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

LiteRtStatus QnnManager::ValidateOp(::qnn::QnnBackend& qnn_backend,
                                    ::qnn::OpWrapper& op) {
  // TODO(jiunkaiy): Remove version check and break backward compatibility when
  // acceptable.
  const auto sdk_version = GetSdkVersion();
  using ::qnn::SdkVersion;
  // Bypass RmsNorm OP validation.
  if (SdkVersion{2, 35, 0} <= sdk_version &&
      sdk_version < SdkVersion{2, 37, 0} &&
      op.IsOpCode(::qnn::QnnOpCode::kRmsNorm)) {
    LITERT_LOG(LITERT_WARNING,
               "SDK version is in [2.35.0, 2.37.0); RmsNorm OP validation is "
               "bypassed.");
    return kLiteRtStatusOk;
  }
  // Bypass L2Norm OP validation.
  if (SdkVersion{2, 39, 0} <= sdk_version &&
      sdk_version < SdkVersion{2, 43, 0} &&
      op.IsOpCode(::qnn::QnnOpCode::kL2Norm)) {
    LITERT_LOG(LITERT_WARNING,
               "SDK version is in [2.39.0, 2.43.0); L2Norm OP validation is "
               "bypassed.");
    return kLiteRtStatusOk;
  }
  // Bypass Quantize OP validation.
  if (SdkVersion{2, 35, 0} <= sdk_version &&
      sdk_version < SdkVersion{2, 38, 0} &&
      op.IsOpCode(::qnn::QnnOpCode::kQuantize) &&
      op.GetInputTensor(0).IsF32() && op.GetOutputTensor(0).IsQuantI16()) {
    LITERT_LOG(LITERT_WARNING,
               "SDK version is in [2.35.0, 2.38.0); Quantize OP validation is "
               "bypassed.");
    return kLiteRtStatusOk;
  }
  // Bypass Split OP validation.
  if (SdkVersion{2, 35, 0} <= sdk_version &&
      sdk_version < SdkVersion{2, 37, 0} &&
      op.IsOpCode(::qnn::QnnOpCode::kSplit)) {
    LITERT_LOG(
        LITERT_WARNING,
        "SDK version is in [2.35.0, 2.37.0); Split OP validation is bypassed.");
    return kLiteRtStatusOk;
  }

  if (op.IsOpCode(::qnn::QnnOpCode::kFullyConnected) &&
      op.GetInputTensor(0).IsQuantI8() && op.GetInputTensor(1).IsQuantI8() &&
      op.GetInputTensor(1).IsQuantBitwidth(::qnn::kQuantBitWidth2) &&
      op.GetOutputTensor(0).IsQuantI8() &&
      SdkVersion{2, 47, 0} <= sdk_version &&
      sdk_version < SdkVersion{2, 49, 0}) {
    LITERT_LOG(LITERT_WARNING,
               "SDK version is in [2.47.0, 2.49.0); A8W2 FC OP validation is "
               "bypassed.");
    return kLiteRtStatusOk;
  }

  const auto op_config = op.GetOpConfig();
  if (Qnn_ErrorHandle_t error = Api()->backendValidateOpConfig(
          qnn_backend.GetBackendHandle(), op_config);
      QNN_SUCCESS != error) {
    // Detailed message on the failure path only: which op failed, the QNN
    // error code, and every operand's dtype / dims / quantization kind.
    LITERT_LOG(
        LITERT_ERROR,
        "\nQNN op validation failed: %s\n"
        "  QNN error=%lld. See the QNN validator log lines above for the "
        "specific reason.",
        op.ToString().c_str(), static_cast<long long>(error));
    return kLiteRtStatusErrorInvalidLegalization;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus QnnManager::Init(std::optional<std::string> shared_library_dir,
                              const ::qnn::Options& options) {
  shared_library_dir_ = shared_library_dir;
  options_ = options;
  auto backend_type = options_.GetBackendType();

  // Determine ADSP library directory:
  // 1. Explicit dsp_skel_dir from options
  // 2. Explicit qnn_lib_dir from options
  // 3. Fallback shared_library_dir (plugin dir)
  std::string adsp_dir;
  if (!options_.GetDspSkelDir().empty()) {
    adsp_dir = std::string(options_.GetDspSkelDir());
  } else if (!options_.GetQnnLibDir().empty()) {
    adsp_dir = std::string(options_.GetQnnLibDir());
  } else if (shared_library_dir.has_value() && !shared_library_dir->empty()) {
    adsp_dir = *shared_library_dir;
  }

  if (!adsp_dir.empty()) {
    LITERT_LOG(LITERT_INFO, "Configuring ADSP_LIBRARY_PATH with dir: %s",
               adsp_dir.c_str());
    static constexpr char kAdsp[] = "ADSP_LIBRARY_PATH";
    const char* adsp_library_path = getenv(kAdsp);
    if (adsp_library_path == nullptr || adsp_library_path[0] == '\0') {
      LITERT_RETURN_IF_ERROR(SetEnvVar(kAdsp, adsp_dir.c_str()));
    } else {
      bool found = false;
      for (absl::string_view part : absl::StrSplit(adsp_library_path, ';')) {
        if (part == adsp_dir) {
          found = true;
          break;
        }
      }
      if (!found) {
        auto new_adsp_library_path =
            absl::StrCat(adsp_dir, ";", adsp_library_path);
        LITERT_RETURN_IF_ERROR(SetEnvVar(kAdsp, new_adsp_library_path.c_str()));
      }
    }
    LITERT_LOG(LITERT_DEBUG, "ADSP_LIBRARY_PATH: %s", getenv(kAdsp));
  }

  LITERT_RETURN_IF_ERROR(LoadSystemLib(kLibQnnSystemSo));
  LITERT_RETURN_IF_ERROR(ResolveSystemApi());

  switch (backend_type) {
    case ::qnn::BackendType::kGpuBackend: {
      LITERT_RETURN_IF_ERROR(LoadLib(::qnn::GpuBackend::GetLibraryName()));
      LITERT_RETURN_IF_ERROR(
          ResolveApi(::qnn::GpuBackend::GetExpectedBackendVersion()));
      break;
    }
    case ::qnn::BackendType::kHtpBackend: {
      LITERT_RETURN_IF_ERROR(LoadLib(::qnn::HtpBackend::GetLibraryName()));
      LITERT_RETURN_IF_ERROR(
          ResolveApi(::qnn::HtpBackend::GetExpectedBackendVersion()));
      break;
    }
    case ::qnn::BackendType::kIrBackend: {
      LITERT_RETURN_IF_ERROR(LoadLib(::qnn::IrBackend::GetLibraryName()));
      LITERT_RETURN_IF_ERROR(
          ResolveApi(::qnn::IrBackend::GetExpectedBackendVersion()));
      break;
    }
    case ::qnn::BackendType::kDspBackend: {
      LITERT_RETURN_IF_ERROR(LoadLib(::qnn::DspBackend::GetLibraryName()));
      LITERT_RETURN_IF_ERROR(
          ResolveApi(::qnn::DspBackend::GetExpectedBackendVersion()));
      break;
    }
    case ::qnn::BackendType::kLpaiBackend: {
      LITERT_RETURN_IF_ERROR(LoadLib(::qnn::LpaiBackend::GetLibraryName()));
      LITERT_RETURN_IF_ERROR(
          ResolveApi(::qnn::LpaiBackend::GetExpectedBackendVersion()));
      break;
    }
    default: {
      LITERT_LOG(LITERT_ERROR, "Unsupported backend type: %d",
                 options_.GetBackendType());
      return kLiteRtStatusErrorRuntimeFailure;
    }
  }

  // Get SDK version from build ID.
  const char* build_id;
  Api()->backendGetBuildId(&build_id);
  auto parsed_version = ::qnn::ParseSdkVersion(build_id);
  if (!parsed_version) {
    LITERT_LOG(LITERT_ERROR, "Failed to parse build ID", "");
    return kLiteRtStatusErrorRuntimeFailure;
  }
  sdk_version_ = *parsed_version;
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
    ::qnn::QnnBackend& qnn_backend,
    absl::Span<const QnnContext_Config_t*> configs,
    ::qnn::Profiling profiling_level) {
  Qnn_ContextHandle_t context_handle;
  if (auto status = Api()->contextCreate(
          qnn_backend.GetBackendHandle(), qnn_backend.GetDeviceHandle(),
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
  if (auto status = Api()->profileCreate(qnn_backend.GetBackendHandle(),
                                         profiling, &profile_handle);
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
    ::qnn::QnnBackend& qnn_backend,
    absl::Span<const QnnContext_Config_t*> configs,
    absl::Span<const uint8_t> bytecode, Qnn_ProfileHandle_t profile_handle) {
  std::vector<const QnnContext_Config_t*> effective_configs;
  for (const auto* cfg : configs) {
    if (cfg != nullptr) {
      effective_configs.push_back(cfg);
    }
  }
  QnnHtpContext_CustomConfig_t htp_read_budget_custom_config =
      QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT;
  QnnContext_Config_t htp_read_budget_config = QNN_CONTEXT_CONFIG_INIT;
  QnnHtpContext_CustomConfig_t htp_init_accel_custom_config =
      QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT;
  QnnContext_Config_t htp_init_accel_config = QNN_CONTEXT_CONFIG_INIT;
  if (options_.GetBackendType() == ::qnn::BackendType::kHtpBackend) {
    constexpr uint64_t kDefaultFileReadMemoryBudgetInMb = 16;
    htp_read_budget_custom_config.option =
        QNN_HTP_CONTEXT_CONFIG_OPTION_FILE_READ_MEMORY_BUDGET;
    htp_read_budget_custom_config.fileReadMemoryBudgetInMb =
        kDefaultFileReadMemoryBudgetInMb;
    htp_read_budget_config.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    htp_read_budget_config.customConfig = &htp_read_budget_custom_config;
    effective_configs.push_back(&htp_read_budget_config);

    // Let the DSP use all of its hardware threads to deserialize the context
    // binary. Deserialization is the longest single phase of
    // contextCreateFromBinary for large AOT contexts (the loading thread is
    // asleep waiting on the DSP for most of the call), and no graph of this
    // context can execute concurrently with its own creation, so the
    // documented caveat about degrading concurrent graph execution does not
    // apply here. LITERT_QNN_INIT_ACCELERATION=0 disables it (for A/B
    // measurements).
    const char* init_accel_env = getenv("LITERT_QNN_INIT_ACCELERATION");
    if (init_accel_env == nullptr || absl::string_view(init_accel_env) != "0") {
      htp_init_accel_custom_config.option =
          QNN_HTP_CONTEXT_CONFIG_OPTION_INIT_ACCELERATION;
      htp_init_accel_custom_config.initAcceleration = true;
      htp_init_accel_config.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
      htp_init_accel_config.customConfig = &htp_init_accel_custom_config;
      effective_configs.push_back(&htp_init_accel_config);
    }
  }
  effective_configs.push_back(nullptr);

#if !defined(_WIN32)
  const int64_t page_size = sysconf(_SC_PAGESIZE);
  uintptr_t aligned_start = 0;
  uintptr_t aligned_end = 0;
  if (page_size > 0 && !bytecode.empty()) {
    const uintptr_t mask = static_cast<uintptr_t>(page_size) - 1;
    const uintptr_t start = reinterpret_cast<uintptr_t>(bytecode.data());
    const uintptr_t end = start + bytecode.size();
    aligned_start = (start + mask) & ~mask;
    aligned_end = end & ~mask;
#if defined(MADV_NOHUGEPAGE)
    const uintptr_t outer_start = start & ~mask;
    const uintptr_t outer_end = (end + mask) & ~mask;
    if (outer_end > outer_start) {
      madvise(reinterpret_cast<void*>(outer_start), outer_end - outer_start,
              MADV_NOHUGEPAGE);
    }
#endif
  }

  // Prefetch the context binary into the page cache while libQnnHtp streams
  // it. The bytecode is a file-backed mapping (a section of the model file):
  // with the 16 MB read budget above, libQnnHtp copies the binary into DSP
  // memory in 16 MB chunks, and every chunk whose pages are not yet cached
  // stalls the loading thread on synchronous storage reads (observed as
  // ~40 ms of folio_wait_bit_common inside a cold 178 MiB vision-encoder
  // load). The helper thread below reads ahead of that copy loop.
  //
  // Pacing matters more than aggressiveness: issuing the whole range as
  // readahead at once (e.g. one MADV_WILLNEED per 2 MiB over 178 MiB) floods
  // the UFS request queue and the loading thread then waits behind our own
  // prefetch (blk_mq_get_tag / folio_wait_bit_common grew, and the metadata
  // parse before the DSP copy went from 17 ms to 119 ms). So the helper
  // populates one 8 MiB window at a time with the synchronous
  // MADV_POPULATE_READ, which keeps exactly one prefetch stream in flight
  // next to the consumer's own reads, and immediately drops the window's
  // page-table entries again with MADV_DONTNEED: the pages stay in the page
  // cache for libQnnHtp (its next access is a minor fault), while this
  // process's RSS only grows by one window instead of the whole binary,
  // preserving the peak-RSS bound from the 16 MB read budget. Kernels
  // without MADV_POPULATE_READ (< 5.14) fall back to one MADV_WILLNEED per
  // window; WILLNEED is asynchronous so the fallback is best-effort only.
  // Setting LITERT_QNN_CONTEXT_PREFETCH=0 disables the helper (for A/B
  // measurements). The helper is joined before returning so it never touches
  // a mapping that has gone away.
  std::thread prefetch_thread;
  const char* prefetch_env = getenv("LITERT_QNN_CONTEXT_PREFETCH");
  const bool prefetch_enabled =
      prefetch_env == nullptr || absl::string_view(prefetch_env) != "0";
  if (prefetch_enabled && aligned_end > aligned_start) {
    prefetch_thread = std::thread([aligned_start, aligned_end] {
      constexpr uintptr_t kPrefetchWindow = 8 * 1024 * 1024;
#if defined(MADV_POPULATE_READ)
      bool populate_supported = true;
#else
      bool populate_supported = false;
      constexpr int MADV_POPULATE_READ = 22;  // Linux 5.14+.
#endif
      for (uintptr_t cursor = aligned_start; cursor < aligned_end;
           cursor += kPrefetchWindow) {
        void* window = reinterpret_cast<void*>(cursor);
        const size_t len = static_cast<size_t>(
            std::min(kPrefetchWindow, aligned_end - cursor));
        if (populate_supported) {
          if (madvise(window, len, MADV_POPULATE_READ) == 0) {
            // Pages are now cached; release our mapping of them so the
            // prefetch does not inflate this process's resident set.
            madvise(window, len, MADV_DONTNEED);
            continue;
          }
          // EINVAL: kernel predates MADV_POPULATE_READ. Anything else (e.g.
          // the mapping is being torn down) also ends the prefetch.
          if (errno != EINVAL) {
            break;
          }
          populate_supported = false;
        }
        if (madvise(window, len, MADV_WILLNEED) != 0) {
          break;  // Advisory only; no prefetch, not a broken load.
        }
      }
    });
  }
#endif

  Qnn_ContextHandle_t context_handle;
  const Qnn_ErrorHandle_t create_status = Api()->contextCreateFromBinary(
      qnn_backend.GetBackendHandle(), qnn_backend.GetDeviceHandle(),
      effective_configs.data(), bytecode.data(), bytecode.size(),
      &context_handle, profile_handle);

#if !defined(_WIN32)
  if (prefetch_thread.joinable()) {
    prefetch_thread.join();
  }
#endif

  if (create_status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to create QNN context: %d", create_status);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to create QNN context");
  }

#if !defined(_WIN32)
  if (aligned_end > aligned_start) {
    madvise(reinterpret_cast<void*>(aligned_start), aligned_end - aligned_start,
            MADV_DONTNEED);
  }
#endif

  auto context_deleter = Api()->contextFree;
  auto profile_deleter = Api()->profileFree;
  return ContextHandle{context_handle, profile_handle, context_deleter,
                       profile_deleter};
}

Expected<QnnManager::Ptr> QnnManager::Create(
    const ::qnn::Options& options,
    std::optional<std::string> shared_library_dir) {
  Ptr qnn_manager(new QnnManager);
  if (auto status = qnn_manager->Init(shared_library_dir, options);
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

};  // namespace litert::qnn
