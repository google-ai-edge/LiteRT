// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_HANDLES_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_HANDLES_H_

#include <memory>
#include <type_traits>
#include <utility>

#include "litert/c/internal/litert_logging.h"
#include "QnnCommon.h"  // from @qairt
#include "QnnContext.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt (for QnnContext_FreeFn_t / QnnProfile_FreeFn_t)
#include "QnnProfile.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt
#include "System/QnnSystemContext.h"  // from @qairt
#include "System/QnnSystemInterface.h"  // from @qairt (for QnnSystemContext_FreeFn_t)

namespace litert::qnn {

// RAII wrapper for a QNN system-context handle.
using SystemContextHandle =
    std::unique_ptr<std::remove_pointer<QnnSystemContext_Handle_t>::type,
                    QnnSystemContext_FreeFn_t>;

// RAII wrapper for a QNN context handle plus its optional profile handle. Not
// a std::unique_ptr because QnnContext_FreeFn_t takes the profile handle as a
// second argument, and the profile must be freed before the context.
class ContextHandle {
 public:
  ContextHandle() = default;

  ContextHandle(Qnn_ContextHandle_t context_handle, Qnn_ProfileHandle_t profile,
                QnnContext_FreeFn_t free_fn,
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
  explicit operator bool() const noexcept { return context_handle_ != nullptr; }

 private:
  Qnn_ContextHandle_t context_handle_ = nullptr;
  Qnn_ProfileHandle_t profile_ = nullptr;
  QnnContext_FreeFn_t free_fn_ = nullptr;
  QnnProfile_FreeFn_t profile_free_fn_ = nullptr;
};

}  // namespace litert::qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_HANDLES_H_
