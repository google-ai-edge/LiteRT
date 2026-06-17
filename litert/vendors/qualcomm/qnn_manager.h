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
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/qualcomm/common.h"
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/qnn_api_loader.h"
#include "litert/vendors/qualcomm/qnn_handles.h"
#include "litert/vendors/qualcomm/qnn_sdk_version.h"
#include "QnnCommon.h"  // from @qairt
#include "QnnContext.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt
#include "System/QnnSystemInterface.h"  // from @qairt

//===----------------------------------------------------------------------===//
//
//                                                                  QnnManager
//
// The SoC layer: a QNN backend + device bound to one SoC, produced by
// QnnManager::Create() against an already-loaded QnnApiLoader. All accessors
// are non-null after a successful Create(). Api / SystemApi / Options alias
// storage owned by the parent QnnApiLoader, so a QnnManager must not outlive
// the loader that created it. Move-only.
//
//===----------------------------------------------------------------------===//

namespace litert::qnn {

// Which side of the compile/dispatch flow this QnnManager is bound to.
// Determines which fields of the custom-op-package option are consumed by
// Create().
enum class QnnManagerMode {
  kCompile,
  kDispatch,
};

class QnnManager {
 public:
  // Binds `soc_info` to a ready-to-use QnnManager using the libraries owned
  // by `loader`. Callable repeatedly with different SoCs -- libraries are not
  // reloaded. `mode` selects which fields of the custom-op-package option
  // are consumed.
  static Expected<QnnManager> Create(const QnnApiLoader& loader,
                                     std::optional<::qnn::SocInfo> soc_info,
                                     QnnManagerMode mode);

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
  QnnManagerMode GetMode() const { return mode_; }

  Expected<SystemContextHandle> CreateSystemContextHandle();

 private:
  QnnManager(const QnnApi* api, const QnnSystemApi* system_api,
             const ::qnn::Options* options, SdkVersion sdk_version,
             std::unique_ptr<::qnn::QnnBackend> backend,
             ::qnn::SocInfo soc_info, QnnManagerMode mode)
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
  QnnManagerMode mode_ = QnnManagerMode::kCompile;
};

}  // namespace litert::qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_MANAGER_H_
