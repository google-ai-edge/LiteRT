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

#include <cstring>

#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/options/litert_cpu_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/accelerator.h"
#include "litert/runtime/accelerators/accelerator_implementation_helper.h"
#include "litert/runtime/litert_cpu_options.h"
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
  static LiteRtStatus CreateDelegate(LiteRtRuntimeContext* runtime_context,
                                     LiteRtEnvironment env,
                                     LiteRtAccelerator accelerator,
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
    LITERT_RETURN_IF_ERROR(
        runtime_context->get_opaque_options(options, &opaque_options));
    const void* cpu_options_data = nullptr;
    const auto options_data_status = runtime_context->find_opaque_options_data(
        opaque_options, LrtGetCpuOptionsIdentifier(),
        const_cast<void**>(&cpu_options_data));

    LiteRtCpuOptionsT parsed_options;
    switch (options_data_status) {
      case kLiteRtStatusOk:
        if (cpu_options_data) {
          const char* toml_str = static_cast<const char*>(cpu_options_data);
          LITERT_RETURN_IF_ERROR(litert::internal::ParseLiteRtCpuOptions(
              toml_str, strlen(toml_str), &parsed_options));
        }
        break;
      case kLiteRtStatusErrorNotFound:
        break;
      default:
        return options_data_status;
    }

    if (parsed_options.kernel_mode != kLiteRtCpuKernelModeXnnpack) {
      *delegate_wrapper = nullptr;
      return kLiteRtStatusOk;
    }

    auto xnn_options = parsed_options.xnn;
    TfLiteOpaqueDelegate* xnnpack_delegate =
        TfLiteXNNPackDelegateCreate(&xnn_options);
    LITERT_RETURN_IF_ERROR(xnnpack_delegate != nullptr,
                           ErrorStatusBuilder(kLiteRtStatusErrorRuntimeFailure))
        << "XNNPack delegate failed to be created.";
    LITERT_RETURN_IF_ERROR(
        runtime_context->wrap_delegate(xnnpack_delegate, delegate_wrapper));

    return kLiteRtStatusOk;
  }

  // Destroys an XNNPack delegate instance.
  static void DestroyDelegate(LiteRtRuntimeContext* runtime_context,
                              LiteRtDelegateWrapper delegate_wrapper) {
    if (delegate_wrapper == nullptr) {
      return;
    }
    TfLiteOpaqueDelegate* xnnpack_delegate;
    runtime_context->unwrap_delegate(delegate_wrapper, &xnnpack_delegate);
    TfLiteXNNPackDelegateDelete(xnnpack_delegate);
  }

  // Returns true to indicate the XNNPack delegate is responsible for JIT
  // compilation.
  static LiteRtStatus IsTfLiteDelegateResponsibleForJitCompilation(
      LiteRtAcceleratorT* accelerator, bool* does_jit_compilation) {
    LITERT_RETURN_IF_ERROR(does_jit_compilation,
                           litert::ErrorStatusBuilder::InvalidArgument())
        << "`does_jit_compilation` pointer is null.";
#if defined(__EMSCRIPTEN__)
    // Xnnpack is always applied to Web.
    *does_jit_compilation = false;
#else
    *does_jit_compilation = true;
#endif  // defined(__EMSCRIPTEN__)
    return kLiteRtStatusOk;
  }
};

}  // namespace
}  // namespace litert

extern "C" {

// Discovery C object for the CPU (Xnnpack) accelerator by LiteRT.
// This object is used by the LiteRT environment constructor.
static LiteRtAcceleratorDef LiteRtCpuAcceleratorImpl = {
    .version = 1,  // LiteRtAcceleratorDefV1
    .get_name = litert::CpuAccelerator::GetName,
    .get_version = litert::CpuAccelerator::GetVersion,
    .get_hardware_support = litert::CpuAccelerator::GetHardwareSupport,
    .is_tflite_delegate_responsible_for_jit_compilation =
        litert::CpuAccelerator::IsTfLiteDelegateResponsibleForJitCompilation,
    .create_delegate = litert::CpuAccelerator::CreateDelegate,
    .destroy_delegate = litert::CpuAccelerator::DestroyDelegate,
    .buffer_handlers =
        {
            .create_func = nullptr,
            .destroy_func = nullptr,
            .lock_func = nullptr,
            .unlock_func = nullptr,
            .clear_func = nullptr,
            .import_func = nullptr,
            .device_tag = kLiteRtEnvOptionTagNull,
            .queue_tag = kLiteRtEnvOptionTagNull,
            .num_supported_buffer_types = 0,
        },
};

// Accelerator definition pointer referenced by auto_registration.cc.
LiteRtAcceleratorDef* LiteRtStaticLinkedAcceleratorCpuDef =
    &LiteRtCpuAcceleratorImpl;

}  // extern "C"
