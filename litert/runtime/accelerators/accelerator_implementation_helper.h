// Copyright 2025 Google LLC.
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

#ifndef ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_ACCELERATOR_IMPLEMENTATION_HELPER_H_
#define ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_ACCELERATOR_IMPLEMENTATION_HELPER_H_

#include <memory>
#include <utility>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_accelerator_registration.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert::internal {

struct AcceleratorDestructor {
  void operator()(LiteRtAccelerator accelerator) {
    LiteRtDestroyAccelerator(accelerator);
  }
};

// RAII wrapper for accelerator handles.
using AcceleratorGuard =
    std::unique_ptr<std::pointer_traits<LiteRtAccelerator>::element_type,
                    AcceleratorDestructor>;

// Helps setting up an accelerator handle for accelerators that use the
// `AcceleratorImplementationHelper` template as a base class.
template <class T>
Expected<void> SetAcceleratorBoilerplateFunctions(
    AcceleratorGuard& accelerator) {
  LITERT_RETURN_IF_ERROR(
      LiteRtSetAcceleratorGetName(accelerator.get(), T::GetName));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetAcceleratorGetVersion(accelerator.get(), T::GetVersion));
  LITERT_RETURN_IF_ERROR(LiteRtSetAcceleratorGetHardwareSupport(
      accelerator.get(), T::GetHardwareSupport));
  LITERT_RETURN_IF_ERROR(LiteRtSetDelegateFunction(
      accelerator.get(), T::CreateDelegate, T::DestroyDelegate));
  return {};
}

// Helps accelerator implementation by providing a lot of the boilerplate
// needed.
//
// Warning: The provided Ptr assumes that AcceleratorClass instances are
// created using `operator new`.
//
// Warning: `version` should be incremented every time the code of this
// accelerator is updated according to semanting versioning.
//
// Pre C++20, it's unable to use struct as non-type template parameter.
// The following example will be used instead of templating on LiteRtApiVersion
// directly.
//
// struct LiteRtApiVersionTrait {
//   static constexpr int kMajor = 0;
//   static constexpr int kMinor = 0;
//   static constexpr int kPatch = 0;
//   static constexpr LiteRtApiVersion version = {kMajor, kMinor, kPatch};
// };
template <class AcceleratorClass, const char* name_, typename VersionTrait,
          LiteRtHwAcceleratorSet hardware_support_>
class AcceleratorImplementationHelper {
 public:
  // The accelerator name returned by `GetName`.
  constexpr static const absl::string_view kName = name_;
  // The accelerator version returned by `GetVersion`.
  constexpr static const LiteRtApiVersion kVersion = VersionTrait::version;
  // The accelerator hardware support returned by `GetHardwareSupport`.
  constexpr static const LiteRtHwAcceleratorSet kHwSupport = hardware_support_;

  struct Deleter {
    void operator()(AcceleratorClass* accelerator_impl) {
      delete accelerator_impl;
    }
  };

  // Owning pointer wrapping the accelerator.
  using Ptr = std::unique_ptr<AcceleratorClass, Deleter>;

  // Creates a new instance of the accelerator implementation.
  template <class... Args>
  static Ptr Allocate(Args&&... args) {
    return Ptr(new AcceleratorClass(std::forward<Args>(args)...));
  }

  // Deletes the accelerator data.
  static void Destroy(void* accelerator_impl) {
    Deleter()(reinterpret_cast<AcceleratorClass*>(accelerator_impl));
  }

  // Returns the accelerator's name by setting `name`.
  static LiteRtStatus GetName(LiteRtAccelerator accelerator,
                              const char** name) {
    LITERT_RETURN_IF_ERROR(accelerator != nullptr,
                           ErrorStatusBuilder::InvalidArgument())
        << "Accelerator handle is invalid.";
    LITERT_RETURN_IF_ERROR(name != nullptr,
                           ErrorStatusBuilder::InvalidArgument())
        << "Name pointer is null.";
    *name = kName.data();
    return kLiteRtStatusOk;
  }

  // Returns the accelerator's version by setting `version`.
  static LiteRtStatus GetVersion(LiteRtAccelerator accelerator,
                                 LiteRtApiVersion* version) {
    LITERT_RETURN_IF_ERROR(accelerator != nullptr,
                           ErrorStatusBuilder::InvalidArgument())
        << "Accelerator handle is invalid.";
    LITERT_RETURN_IF_ERROR(version != nullptr,
                           ErrorStatusBuilder::InvalidArgument())
        << "Version pointer is null.";
    *version = kVersion;
    return kLiteRtStatusOk;
  }

  // Returns the accelerator's hardware support by setting `hw_set`.
  static LiteRtStatus GetHardwareSupport(LiteRtAccelerator accelerator,
                                         LiteRtHwAcceleratorSet* hw_set) {
    LITERT_RETURN_IF_ERROR(accelerator != nullptr,
                           ErrorStatusBuilder::InvalidArgument())
        << "Accelerator handle is invalid.";
    LITERT_RETURN_IF_ERROR(hw_set != nullptr,
                           ErrorStatusBuilder::InvalidArgument())
        << "Hardware support pointer is null.";
    *hw_set = kHwSupport;
    return kLiteRtStatusOk;
  }
};

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_ACCELERATOR_IMPLEMENTATION_HELPER_H_
