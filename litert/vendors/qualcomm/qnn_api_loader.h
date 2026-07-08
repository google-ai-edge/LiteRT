// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_API_LOADER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_API_LOADER_H_

#include <memory>
#include <optional>
#include <string>
#include <tuple>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_shared_library.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/qualcomm/common.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "QnnCommon.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt
#include "System/QnnSystemInterface.h"  // from @qairt

//===----------------------------------------------------------------------===//
//
//                                                                QnnApiLoader
//
// The QNN library layer: locates and loads the QNN SDK .so files, resolves
// the backend + system API function tables, and parses the SoC-independent
// build ID into an SdkVersion. Owns the loaded libraries and releases them on
// destruction. Holds no SoC or backend handles -- construct a
// litert::qnn::QnnManager (see qnn_manager.h) to bind a SoC.
//
//===----------------------------------------------------------------------===//

namespace litert::qnn {

struct SdkVersion {
  int major, minor, patch;

  friend constexpr bool operator==(const SdkVersion& lhs,
                                   const SdkVersion& rhs) noexcept {
    return std::tie(lhs.major, lhs.minor, lhs.patch) ==
           std::tie(rhs.major, rhs.minor, rhs.patch);
  }
  friend constexpr bool operator!=(const SdkVersion& lhs,
                                   const SdkVersion& rhs) noexcept {
    return !(lhs == rhs);
  }
  friend constexpr bool operator<(const SdkVersion& lhs,
                                  const SdkVersion& rhs) noexcept {
    return std::tie(lhs.major, lhs.minor, lhs.patch) <
           std::tie(rhs.major, rhs.minor, rhs.patch);
  }
  friend constexpr bool operator>(const SdkVersion& lhs,
                                  const SdkVersion& rhs) noexcept {
    return rhs < lhs;
  }
  friend constexpr bool operator<=(const SdkVersion& lhs,
                                   const SdkVersion& rhs) noexcept {
    return !(rhs < lhs);
  }
  friend constexpr bool operator>=(const SdkVersion& lhs,
                                   const SdkVersion& rhs) noexcept {
    return !(lhs < rhs);
  }
};

// Parses a QNN SDK build ID string (e.g. "v2.37.0") into an SdkVersion.
Expected<SdkVersion> ParseSdkVersion(const char* build_id);

class QnnApiLoader;

namespace internal {

std::string Dump(const QnnApiLoader& loader);

}  // namespace internal

class QnnApiLoader {
  friend std::string internal::Dump(const QnnApiLoader& loader);

 public:
  using Ptr = std::unique_ptr<QnnApiLoader>;

  ~QnnApiLoader();

  // Loads and resolves the QNN libraries. Does not bind a SoC -- call
  // QnnManager::Create() for that. The SDK version is populated on success.
  static Expected<Ptr> Create(
      const ::qnn::Options& options,
      std::optional<std::string> shared_library_dir = std::nullopt);

  // Resolved backend API function table. Non-null after a successful Create().
  const QnnApi* Api() const;

  // Resolved system API function table. Non-null after a successful Create().
  const QnnSystemApi* SystemApi() const;

  const ::qnn::Options& GetOptions() const { return options_; }

  // SDK version parsed from the backend build ID.
  SdkVersion GetSdkVersion() const { return sdk_version_; }

 private:
  QnnApiLoader() = default;

  // Sets ADSP path, loads libraries, resolves API tables, reads build ID.
  // Runs once from Create().
  LiteRtStatus LoadLibraries(std::optional<std::string> shared_library_dir,
                             const ::qnn::Options& options);

  //
  // Library loading
  //

  // Loads libQnn*.so from `path`.
  LiteRtStatus LoadLib(absl::string_view path);

  // Loads libQnnSystem.so from `path`.
  LiteRtStatus LoadSystemLib(absl::string_view path);

  //
  // Function-table resolution
  //

  // Resolves the backend API from the already-loaded library. Requires
  // exactly one provider whose version is compatible with
  // `expected_qnn_version`.
  LiteRtStatus ResolveApi(Qnn_Version_t expected_qnn_version);

  // Resolves the system API from the already-loaded system library. Requires
  // exactly one provider.
  LiteRtStatus ResolveSystemApi();

  // Backend .so. Released when the loader is destroyed.
  SharedLibrary lib_;

  // System .so. Released when the loader is destroyed.
  SharedLibrary lib_system_;

  const QnnInterface_t* interface_ = nullptr;
  const QnnSystemInterface_t* system_interface_ = nullptr;

  ::qnn::Options options_;
  std::optional<std::string> shared_library_dir_;
  SdkVersion sdk_version_{};
};

}  // namespace litert::qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_API_LOADER_H_
