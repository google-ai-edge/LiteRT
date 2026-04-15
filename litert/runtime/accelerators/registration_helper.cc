// Copyright 2026 Google LLC.
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

#include "litert/runtime/accelerators/registration_helper.h"

#include <cstddef>

#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/c/internal/litert_accelerator_registration.h"
#include "litert/c/internal/litert_tensor_buffer_registry.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_macros.h"

namespace litert::internal {

LiteRtStatus RegisterAcceleratorFromDef(
    LiteRtEnvironment env, const LiteRtAcceleratorDef* accelerator_def) {
  if (accelerator_def->version != LITERT_ACCELERATOR_DEF_CURRENT_VERSION)
    return kLiteRtStatusErrorWrongVersion;

  if (accelerator_def->get_name == nullptr ||
      accelerator_def->get_version == nullptr ||
      accelerator_def->get_hardware_support == nullptr ||
      accelerator_def->is_tflite_delegate_responsible_for_jit_compilation ==
          nullptr ||
      accelerator_def->create_delegate == nullptr ||
      accelerator_def->destroy_delegate == nullptr ||
      accelerator_def->num_supported_buffer_types >=
          LITERT_ACCELERATOR_DEF_MAX_SUPPORTED_BUFFER_TYPES) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  LiteRtAccelerator accelerator;
  LITERT_RETURN_IF_ERROR(LiteRtCreateAccelerator(&accelerator));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetAcceleratorGetName(accelerator, accelerator_def->get_name));
  LITERT_RETURN_IF_ERROR(LiteRtSetAcceleratorGetVersion(
      accelerator, accelerator_def->get_version));
  LITERT_RETURN_IF_ERROR(LiteRtSetAcceleratorGetHardwareSupport(
      accelerator, accelerator_def->get_hardware_support));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDelegateFunction(accelerator, accelerator_def->create_delegate,
                                accelerator_def->destroy_delegate));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetIsAcceleratorDelegateResponsibleForJitCompilation(
          accelerator,
          accelerator_def->is_tflite_delegate_responsible_for_jit_compilation));

  LITERT_RETURN_IF_ERROR(
      LiteRtRegisterAccelerator(env, accelerator, nullptr, nullptr));

  for (size_t i = 0; i < accelerator_def->num_supported_buffer_types; ++i) {
    auto handler_status = LiteRtRegisterTensorBufferHandlers(
        env, accelerator_def->supported_buffer_types[i],
        accelerator_def->create_func, accelerator_def->destroy_func,
        accelerator_def->lock_func, accelerator_def->unlock_func,
        accelerator_def->clear_func, accelerator_def->import_func,
        accelerator_def->device_tag, accelerator_def->queue_tag);
    if (handler_status != kLiteRtStatusOk &&
        handler_status != kLiteRtStatusErrorAlreadyExists) {
      return handler_status;
    }
  }

  return kLiteRtStatusOk;
}

}  // namespace litert::internal
