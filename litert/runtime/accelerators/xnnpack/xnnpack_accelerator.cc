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

#include "litert/runtime/accelerators/xnnpack/xnnpack_accelerator.h"

#include <memory>

#include "litert/c/internal/litert_accelerator_registration.h"
#include "litert/c/internal/litert_delegate_wrapper.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/options/litert_cpu_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/accelerator.h"
#include "litert/runtime/accelerators/accelerator_implementation_helper.h"
#include "tflite/c/c_api_types.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"

namespace litert {
namespace {

constexpr const char kCpuAcceleratorName[] = "CpuAccelerator";

struct CpuAcceleratorVersion {
  static constexpr int kMajor = 1;
  static constexpr int kMinor = 0;
  static constexpr int kPatch = 0;
  static constexpr LiteRtApiVersion version = {kMajor, kMinor, kPatch};  // NOLINT
};

class CpuAccelerator final
    : public internal::AcceleratorImplementationHelper<
          CpuAccelerator, kCpuAcceleratorName, CpuAcceleratorVersion,
          kLiteRtHwAcceleratorCpu> {
 public:
  CpuAccelerator() = default;

  static Expected<Ptr> Create() { return Allocate(); }

  // C API

  // Creates a Dispatch delegate instance.
  static LiteRtStatus CreateDelegate(LiteRtAccelerator accelerator,
                                     LiteRtOptions options,
                                     LiteRtDelegateWrapper* delegate_wrapper) {
    LITERT_RETURN_IF_ERROR(delegate_wrapper != nullptr,
                           ErrorStatusBuilder::InvalidArgument())
        << "Delegate wrapper pointer is null.";
    LITERT_RETURN_IF_ERROR(accelerator != nullptr,
                           ErrorStatusBuilder::InvalidArgument())
        << "Accelerator handle is invalid.";
    LITERT_RETURN_IF_ERROR(accelerator->env != nullptr,
                           ErrorStatusBuilder::InvalidArgument())
        << "Accelerator is not registered to an environment.";

    LiteRtOpaqueOptions opaque_options;
    LITERT_RETURN_IF_ERROR(LiteRtGetOpaqueOptions(options, &opaque_options));
    LiteRtCpuOptions cpu_options;
    const auto options_data_status = LiteRtFindOpaqueOptionsData(
        opaque_options, LiteRtGetCpuOptionsIdentifier(),
        reinterpret_cast<void**>(&cpu_options));

    switch (options_data_status) {
      case kLiteRtStatusOk:
        break;
      case kLiteRtStatusErrorNotFound:
        cpu_options = nullptr;
        break;
      default:
        return options_data_status;
    }

    // TODO: b/403547017 - Make the CPU accelerator configurable using the
    // compilation options.
    auto xnn_options = TfLiteXNNPackDelegateOptionsDefault();
    if (cpu_options != nullptr) {
      LiteRtGetCpuOptionsNumThread(cpu_options, &xnn_options.num_threads);
      LiteRtGetCpuOptionsXNNPackFlags(cpu_options, &xnn_options.flags);
      LITERT_RETURN_IF_ERROR(LiteRtGetCpuOptionsXnnPackWeightCachePath(
          cpu_options, &xnn_options.weight_cache_file_path));
    }
    TfLiteOpaqueDelegate* xnnpack_delegate =
        TfLiteXNNPackDelegateCreate(&xnn_options);
    LITERT_RETURN_IF_ERROR(xnnpack_delegate != nullptr,
                           ErrorStatusBuilder(kLiteRtStatusErrorRuntimeFailure))
        << "XNNPack delegate failed to be created.";
    LITERT_RETURN_IF_ERROR(
        LiteRtWrapDelegate(xnnpack_delegate, delegate_wrapper));

    return kLiteRtStatusOk;
  }

  // Destroys an XNNPack delegate instance.
  static void DestroyDelegate(LiteRtDelegateWrapper delegate_wrapper) {
    TfLiteOpaqueDelegate* xnnpack_delegate;
    LiteRtUnwrapDelegate(delegate_wrapper, &xnnpack_delegate);
    TfLiteXNNPackDelegateDelete(xnnpack_delegate);
  }
};

}  // namespace
}  // namespace litert

extern "C" {

LiteRtStatus LiteRtRegisterCpuAccelerator(LiteRtEnvironment environment) {
  LITERT_RETURN_IF_ERROR(environment != nullptr,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "environment handle is null";

  LiteRtAccelerator accelerator_handle;
  LITERT_RETURN_IF_ERROR(LiteRtCreateAccelerator(&accelerator_handle));
  litert::internal::AcceleratorGuard accelerator(accelerator_handle);

  LITERT_RETURN_IF_ERROR(litert::internal::SetAcceleratorBoilerplateFunctions<
                         litert::CpuAccelerator>(accelerator));

  LITERT_ASSIGN_OR_RETURN(auto accelerator_impl,
                          litert::CpuAccelerator::Create());

  LITERT_RETURN_IF_ERROR(LiteRtRegisterAccelerator(
      environment, accelerator.release(), accelerator_impl.release(),
      litert::CpuAccelerator::Destroy));

  return kLiteRtStatusOk;
}

}  // extern "C"
