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

#include "litert/c/internal/litert_tensor_buffer_registry.h"

#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/environment.h"
#include "litert/runtime/tensor_buffer_registry.h"

LiteRtStatus LiteRtRegisterTensorBufferHandlers(
    LiteRtEnvironment env, LiteRtTensorBufferType buffer_type,
    CreateCustomTensorBuffer create_func,
    DestroyCustomTensorBuffer destroy_func, LockCustomTensorBuffer lock_func,
    UnlockCustomTensorBuffer unlock_func,
    ImportCustomTensorBuffer import_func) {
  auto& registry = env->GetTensorBufferRegistry();
  litert::internal::CustomTensorBufferHandlers handlers = {
      .create_func = create_func,
      .destroy_func = destroy_func,
      .lock_func = lock_func,
      .unlock_func = unlock_func,
      .import_func = import_func,
  };
  LITERT_RETURN_IF_ERROR(registry.RegisterHandlers(buffer_type, handlers));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferRegistry(LiteRtEnvironment env,
                                           void** registry) {
  *registry = reinterpret_cast<void*>(&env->GetTensorBufferRegistry());
  return kLiteRtStatusOk;
}
