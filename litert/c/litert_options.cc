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

#include "litert/c/litert_options.h"

#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_op_kernel.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_handle.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/cc/options/litert_runtime_options.h"
#include "litert/core/options.h"

#define LRT_CHECK_NON_NULL(handle)                          \
  if (!(handle)) {                                          \
    LITERT_LOG(LITERT_ERROR, #handle " must not be null."); \
    return kLiteRtStatusErrorInvalidArgument;               \
  }

extern "C" {

LiteRtStatus LiteRtCreateOptions(LiteRtOptions* options) {
  LRT_CHECK_NON_NULL(options);
  *options = new LiteRtOptionsT;
  return kLiteRtStatusOk;
}

void LiteRtDestroyOptions(LiteRtOptions options) { delete options; }

LiteRtStatus LiteRtSetOptionsHardwareAccelerators(
    LiteRtOptions options, LiteRtHwAcceleratorSet hardware_accelerators) {
  LRT_CHECK_NON_NULL(options);
  if ((hardware_accelerators &
       (kLiteRtHwAcceleratorCpu | kLiteRtHwAcceleratorGpu |
        kLiteRtHwAcceleratorNpu)) != hardware_accelerators) {
    LITERT_LOG(LITERT_ERROR,
               "Invalid bitfield value for hardware accelerator set: %d.",
               hardware_accelerators);
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->hardware_accelerators = hardware_accelerators;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetOptionsHardwareAccelerators(
    LiteRtOptions options, LiteRtHwAcceleratorSet* hardware_accelerators) {
  LRT_CHECK_NON_NULL(options);
  LRT_CHECK_NON_NULL(hardware_accelerators);
  *hardware_accelerators = options->hardware_accelerators;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtAddOpaqueOptions(LiteRtOptions options,
                                    LiteRtOpaqueOptions opaque_options) {
  LRT_CHECK_NON_NULL(options);
  LRT_CHECK_NON_NULL(opaque_options);
  LITERT_RETURN_IF_ERROR(options->options.Append(
      litert::OpaqueOptions(opaque_options, litert::OwnHandle::kNo)));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtAddRuntimeOptions(LiteRtOptions options,
                                     LiteRtRuntimeOptions runtime_options) {
  LRT_CHECK_NON_NULL(options);
  LRT_CHECK_NON_NULL(runtime_options);
  options->runtime_options =
      litert::RuntimeOptions(runtime_options, litert::OwnHandle::kNo);
  return kLiteRtStatusOk;
}

// Retrieves the head of the accelerator compilation option list.
//
// Note: The following elements may be retrieved with
// `LiteRtGetNextAcceleratorCompilationOptions`.
LiteRtStatus LiteRtGetOpaqueOptions(LiteRtOptions options,
                                    LiteRtOpaqueOptions* opaque_options) {
  LRT_CHECK_NON_NULL(options);
  LRT_CHECK_NON_NULL(opaque_options);
  *opaque_options = options->options.Get();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetRuntimeOptions(LiteRtOptions options,
                                     LiteRtRuntimeOptions* runtime_options) {
  LRT_CHECK_NON_NULL(options);
  LRT_CHECK_NON_NULL(runtime_options);
  *runtime_options = options->runtime_options.Get();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtAddCustomOpKernelOption(
    LiteRtOptions options, const char* custom_op_name, int custom_op_version,
    const LiteRtCustomOpKernel* custom_op_kernel,
    void* custom_op_kernel_user_data) {
  LRT_CHECK_NON_NULL(options);
  LRT_CHECK_NON_NULL(custom_op_name);
  LRT_CHECK_NON_NULL(custom_op_kernel);
  // TODO(b/330649488): Check if the custom op kernel already exists.
  options->custom_op_options.push_back({
      /*.op_name=*/custom_op_name,
      /*.op_version=*/custom_op_version,
      /*.user_data=*/custom_op_kernel_user_data,
      /*.op_kernel=*/*custom_op_kernel,
  });
  return kLiteRtStatusOk;
}

}  // extern "C"
