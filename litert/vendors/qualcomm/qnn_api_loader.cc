// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/qnn_api_loader.h"

#include <stdlib.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_shared_library.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/dynamic_loading.h"
#include "litert/core/filesystem.h"
#include "litert/vendors/qualcomm/common.h"
#include "litert/vendors/qualcomm/core/backends/dsp_backend.h"
#include "litert/vendors/qualcomm/core/backends/gpu_backend.h"
#include "litert/vendors/qualcomm/core/backends/htp_backend.h"
#include "litert/vendors/qualcomm/core/backends/ir_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/qnn_saver_utils.h"
#include "litert/vendors/qualcomm/qnn_sdk_version.h"
#include "QnnCommon.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt
#include "System/QnnSystemCommon.h"  // from @qairt
#include "System/QnnSystemInterface.h"  // from @qairt

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

// The per-backend facts LoadLibraries() needs before any backend instance
// exists: which .so to load, which API version to resolve against, and
// whether the .so must be pushed onto the loader path when a custom
// shared-library dir is set (true for HTP/IR/DSP; GPU does not need it).
struct BackendLibraryInfo {
  const char* library_name;
  Qnn_Version_t expected_version;
  bool needs_lib_on_ld_path;
};

std::optional<BackendLibraryInfo> GetBackendLibraryInfo(
    ::qnn::BackendType type) {
  switch (type) {
    case ::qnn::BackendType::kGpuBackend:
      return BackendLibraryInfo{::qnn::GpuBackend::GetLibraryName(),
                                ::qnn::GpuBackend::GetExpectedBackendVersion(),
                                /*needs_lib_on_ld_path=*/true};
    case ::qnn::BackendType::kHtpBackend:
      return BackendLibraryInfo{::qnn::HtpBackend::GetLibraryName(),
                                ::qnn::HtpBackend::GetExpectedBackendVersion(),
                                /*needs_lib_on_ld_path=*/true};
    case ::qnn::BackendType::kIrBackend:
      return BackendLibraryInfo{::qnn::IrBackend::GetLibraryName(),
                                ::qnn::IrBackend::GetExpectedBackendVersion(),
                                /*needs_lib_on_ld_path=*/true};
    case ::qnn::BackendType::kDspBackend:
      return BackendLibraryInfo{::qnn::DspBackend::GetLibraryName(),
                                ::qnn::DspBackend::GetExpectedBackendVersion(),
                                /*needs_lib_on_ld_path=*/true};
    default:
      return std::nullopt;
  }
}

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

QnnApiLoader::~QnnApiLoader() = default;

LiteRtStatus QnnApiLoader::LoadLib(absl::string_view path) {
  auto saver_output_dir = options_.GetSaverOutputDir();
  const bool needs_global_symbols = !options_.GetCustomOpPackage().name.empty();
  if (saver_output_dir.empty()) {
    LITERT_LOG(LITERT_INFO, "Loading qnn shared library from \"%s\"",
               path.data());
    auto lib_or = SharedLibrary::Load(path, GetRtldFlags(needs_global_symbols));
    if (!lib_or) {
      LITERT_LOG(LITERT_ERROR,
                 "Failed to load qnn shared library from \"%s\": %s",
                 path.data(), lib_or.Error().Message().data());
      return lib_or.Error().Status();
    }
    lib_ = std::move(lib_or.Value());
  } else {
    path = kSaverLibraryName;
    LITERT_LOG(LITERT_INFO, "Loading qnn shared library from \"%s\"",
               path.data());
    auto lib_or = SharedLibrary::Load(path, GetRtldFlags(needs_global_symbols));
    if (!lib_or) {
      LITERT_LOG(LITERT_ERROR,
                 "Failed to load qnn shared library from \"%s\": %s",
                 path.data(), lib_or.Error().Message().data());
      return lib_or.Error().Status();
    }
    lib_ = std::move(lib_or.Value());
    LITERT_RETURN_IF_ERROR(InitSaver(lib_, saver_output_dir));
  }
  LITERT_LOG(LITERT_INFO, "Loaded qnn shared library", "");
  return kLiteRtStatusOk;
}

LiteRtStatus QnnApiLoader::LoadSystemLib(absl::string_view path) {
  const bool needs_global_symbols = !options_.GetCustomOpPackage().name.empty();
  if (shared_library_dir_) {
    std::string resolved_path =
        litert::internal::Join({*shared_library_dir_, path});
    LITERT_LOG(LITERT_INFO, "Loading qnn system shared library from \"%s\"",
               resolved_path.c_str());
    auto lib_system_or =
        SharedLibrary::Load(resolved_path, GetRtldFlags(needs_global_symbols));
    if (lib_system_or) {
      lib_system_ = std::move(lib_system_or.Value());
      return kLiteRtStatusOk;
    }
    LITERT_LOG(LITERT_INFO,
               "Falling back to loading qnn system shared library from \"%s\"",
               path.data());
  }
  LITERT_LOG(LITERT_INFO, "Loading qnn system shared library from \"%s\"",
             path.data());

  auto lib_system_or =
      SharedLibrary::Load(path, GetRtldFlags(needs_global_symbols));
  if (!lib_system_or) {
    LITERT_LOG(LITERT_ERROR, "%s", lib_system_or.Error().Message().data());
    return lib_system_or.Error().Status();
  }
  lib_system_ = std::move(lib_system_or.Value());
  return kLiteRtStatusOk;
}

const QnnApi* QnnApiLoader::Api() const {
  if (interface_ == nullptr) {
    return nullptr;
  }
  return &interface_->QNN_INTERFACE_VER_NAME;
}

LiteRtStatus QnnApiLoader::ResolveApi(Qnn_Version_t expected_qnn_version) {
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
  // Core API version check.
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
  // Backend API version check.
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

LiteRtStatus QnnApiLoader::ResolveSystemApi() {
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

const QnnSystemApi* QnnApiLoader::SystemApi() const {
  if (system_interface_ == nullptr) {
    return nullptr;
  }
  return &system_interface_->QNN_SYSTEM_INTERFACE_VER_NAME;
}

LiteRtStatus QnnApiLoader::LoadLibraries(
    std::optional<std::string> shared_library_dir,
    const ::qnn::Options& options) {
  shared_library_dir_ = shared_library_dir;
  options_ = options;
  auto backend_type = options_.GetBackendType();

  auto lib_info = GetBackendLibraryInfo(backend_type);
  if (!lib_info) {
    LITERT_LOG(LITERT_ERROR, "Unsupported backend type: %d",
               static_cast<int>(backend_type));
    return kLiteRtStatusErrorRuntimeFailure;
  }

  // If shared_library_dir is provided, prepend it to the loader search paths.
  // TODO: This should probably be done upstream in litert_dispatch.
  if (shared_library_dir) {
    LITERT_LOG(LITERT_INFO, "Adding shared library dir to path: %s",
               shared_library_dir->c_str());

    // Always overwrite to enforce that only the provided paths are used.
    static constexpr char kAdsp[] = "ADSP_LIBRARY_PATH";
    const char* adsp_library_path = getenv(kAdsp);
    if (adsp_library_path == nullptr) {
      LITERT_RETURN_IF_ERROR(SetEnvVar(kAdsp, shared_library_dir->c_str()));
    } else {
      bool found = false;
      for (absl::string_view part : absl::StrSplit(adsp_library_path, ';')) {
        if (part == shared_library_dir.value()) {
          found = true;
          break;
        }
      }
      if (!found) {
        auto new_adsp_library_path =
            absl::StrCat(shared_library_dir.value(), ";", adsp_library_path);
        LITERT_RETURN_IF_ERROR(SetEnvVar(kAdsp, new_adsp_library_path.c_str()));
      }
    }
    LITERT_LOG(LITERT_DEBUG, "ADSP_LIBRARY_PATH: %s", getenv(kAdsp));

    if (lib_info->needs_lib_on_ld_path) {
      // TODO: Put dynamic loading module in cc or vendor/cc.
      litert::internal::PutLibOnLdPath(*shared_library_dir,
                                       lib_info->library_name);
    }
  }

  LITERT_RETURN_IF_ERROR(LoadSystemLib(kLibQnnSystemSo));
  LITERT_RETURN_IF_ERROR(ResolveSystemApi());

  LITERT_RETURN_IF_ERROR(LoadLib(lib_info->library_name));
  LITERT_RETURN_IF_ERROR(ResolveApi(lib_info->expected_version));

  // Build id is SoC-independent, so parse it here rather than in QnnManager.
  const char* build_id;
  if (Api()->backendGetBuildId(&build_id) != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "%s", "Failed to get QNN backend build ID");
    return kLiteRtStatusErrorRuntimeFailure;
  }
  LITERT_ASSIGN_OR_RETURN(sdk_version_, ParseSdkVersion(build_id));

  return kLiteRtStatusOk;
}

Expected<QnnApiLoader::Ptr> QnnApiLoader::Create(
    const ::qnn::Options& options,
    std::optional<std::string> shared_library_dir) {
  Ptr loader(new QnnApiLoader);
  if (auto status =
          loader->LoadLibraries(std::move(shared_library_dir), options);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to load QNN libraries");
  }
  return loader;
}

}  // namespace litert::qnn
