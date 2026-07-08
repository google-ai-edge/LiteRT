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

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_MANAGER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_MANAGER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/qualcomm/common.h"
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "QnnCommon.h"  // from @qairt
#include "QnnContext.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt
#include "QnnProfile.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt
#include "System/QnnSystemContext.h"  // from @qairt
#include "System/QnnSystemInterface.h"  // from @qairt

//===----------------------------------------------------------------------===//
//
//                                                                  QnnManager
//
// The SoC layer: a QNN backend + device bound to one SoC, created from an
// already-loaded QnnApiLoader. Api/SystemApi/Options alias the loader's
// storage, so a QnnManager must not outlive its loader. Move-only.
//
//===----------------------------------------------------------------------===//

namespace litert::qnn {

class QnnApiLoader;

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

class QnnManager {
 public:
  // RAII wrapper for a QNN system-context handle.
  using SystemContextHandle =
      std::unique_ptr<std::remove_pointer<QnnSystemContext_Handle_t>::type,
                      QnnSystemContext_FreeFn_t>;

  // RAII wrapper for a QNN context handle plus its optional profile. Not a
  // std::unique_ptr because the free fn takes the profile as a second arg and
  // the profile must be freed before the context.
  class ContextHandle {
   public:
    ContextHandle() = default;

    ContextHandle(Qnn_ContextHandle_t context_handle,
                  Qnn_ProfileHandle_t profile, QnnContext_FreeFn_t free_fn,
                  QnnProfile_FreeFn_t profile_free_fn)
        : context_handle_(context_handle),
          profile_(profile),
          free_fn_(free_fn),
          profile_free_fn_(profile_free_fn) {}

    ~ContextHandle() {
      if (profile_ && profile_free_fn_) {
        if (auto status = profile_free_fn_(profile_); status != QNN_SUCCESS) {
          LITERT_LOG(LITERT_ERROR, "%s", "Failed to free profile handle\n");
        }
        profile_ = nullptr;
      }
      if (context_handle_ && free_fn_) {
        if (auto status = free_fn_(context_handle_, profile_);
            status != QNN_SUCCESS) {
          LITERT_LOG(LITERT_ERROR, "%s", "Failed to free context handle\n");
        }
        context_handle_ = nullptr;
      }
    }

    ContextHandle(ContextHandle&& other) { *this = std::move(other); }

    ContextHandle(const ContextHandle& other) = delete;

    ContextHandle& operator=(ContextHandle&& other) {
      std::swap(context_handle_, other.context_handle_);
      std::swap(profile_, other.profile_);
      std::swap(free_fn_, other.free_fn_);
      std::swap(profile_free_fn_, other.profile_free_fn_);
      return *this;
    }

    ContextHandle& operator=(const ContextHandle& other) = delete;

    Qnn_ContextHandle_t Get() const noexcept { return context_handle_; }
    Qnn_ProfileHandle_t get_profile_handle() const noexcept { return profile_; }
    explicit operator bool() const noexcept {
      return context_handle_ != nullptr;
    }

   private:
    Qnn_ContextHandle_t context_handle_ = nullptr;
    Qnn_ProfileHandle_t profile_ = nullptr;
    QnnContext_FreeFn_t free_fn_ = nullptr;
    QnnProfile_FreeFn_t profile_free_fn_ = nullptr;
  };

  // Selects which custom-op-package fields Create() consumes.
  enum class Mode {
    kCompile,
    kDispatch,
  };

  // Binds `soc_info` to a ready-to-use QnnManager using `loader`'s libraries.
  // Callable repeatedly with different SoCs without reloading libraries.
  static Expected<QnnManager> Create(const QnnApiLoader& loader,
                                     std::optional<::qnn::SocInfo> soc_info,
                                     Mode mode);

  // Context config builders: empty/default, HTP weight-sharing, and GPU
  // performance-hint (falls back to default for kDefault).
  static absl::Span<const QnnContext_Config_t*> DefaultContextConfigs();
  static absl::Span<const QnnContext_Config_t*> WeightSharingContextConfigs();
  static absl::Span<const QnnContext_Config_t*> GpuPerformanceContextConfigs(
      ::qnn::GpuPerformanceMode performance_mode);

  // Move-only: backend_ uniquely owns the QNN backend + device handles.
  QnnManager(QnnManager&&) = default;
  QnnManager& operator=(QnnManager&&) = default;
  QnnManager(const QnnManager&) = delete;
  QnnManager& operator=(const QnnManager&) = delete;
  ~QnnManager() = default;

  Qnn_BackendHandle_t BackendHandle() { return backend_->GetBackendHandle(); }
  const ::qnn::SocInfo& GetSocInfo() const { return soc_info_; }

  // Whether this SoC supports FP16 operations.
  // TODO(jiunkaiy): Remove once the SDK enforces this itself.
  bool IsFp16Supported() const {
    return soc_info_.dsp_arch != ::qnn::DspArch::V68 &&
           soc_info_.soc_model != ::qnn::SnapdragonModel::SAR2230P;
  }

  LiteRtStatus ValidateOp(::qnn::OpWrapper& op);

  Expected<ContextHandle> CreateContextHandle(
      absl::Span<const QnnContext_Config_t*> configs,
      ::qnn::Profiling profiling_level);
  Expected<ContextHandle> CreateContextHandle(
      absl::Span<const QnnContext_Config_t*> configs,
      absl::Span<const uint8_t> bytecode, Qnn_ProfileHandle_t profile_handle);

  LiteRtStatus GenerateContextBinary(Qnn_ContextHandle_t context_handle,
                                     std::vector<char>& buffer);

  const QnnApi* Api() const { return api_; }
  const QnnSystemApi* SystemApi() const { return system_api_; }
  const ::qnn::Options& GetOptions() const { return *options_; }
  SdkVersion GetSdkVersion() const { return sdk_version_; }
  Mode GetMode() const { return mode_; }

  // Parses a QNN SDK build ID string (e.g. "v2.37.0") into an SdkVersion.
  static Expected<SdkVersion> ParseSdkVersion(const char* build_id);

  Expected<SystemContextHandle> CreateSystemContextHandle();

 private:
  QnnManager(const QnnApi* api, const QnnSystemApi* system_api,
             const ::qnn::Options* options, SdkVersion sdk_version,
             std::unique_ptr<::qnn::QnnBackend> backend,
             ::qnn::SocInfo soc_info, Mode mode)
      : api_(api),
        system_api_(system_api),
        options_(options),
        sdk_version_(sdk_version),
        backend_(std::move(backend)),
        soc_info_(soc_info),
        mode_(mode) {}

  Qnn_DeviceHandle_t DeviceHandle() { return backend_->GetDeviceHandle(); }

  const QnnApi* api_ = nullptr;
  const QnnSystemApi* system_api_ = nullptr;
  // Aliases the parent QnnApiLoader, which must outlive this QnnManager.
  const ::qnn::Options* options_ = nullptr;
  SdkVersion sdk_version_{};
  // Owns the underlying QNN backend + device handles.
  std::unique_ptr<::qnn::QnnBackend> backend_;
  ::qnn::SocInfo soc_info_ = ::qnn::kSocInfos[0];
  Mode mode_ = Mode::kCompile;
};

}  // namespace litert::qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_MANAGER_H_
