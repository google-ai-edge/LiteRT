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

#include "litert/runtime/accelerators/legacy_compatibility_layer.h"

#include <cstring>
#include <memory>

#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/c/internal/litert_custom_tensor_buffer_handlers_def.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/runtime/accelerator.h"

namespace litert::internal {

namespace {

struct LiteRtAcceleratorDefV1 {
  int version;

  LiteRtStatus (*get_name)(LiteRtAccelerator accelerator, const char** name);
  LiteRtStatus (*get_version)(LiteRtAccelerator accelerator,
                              LiteRtApiVersion* version);
  LiteRtStatus (*get_hardware_support)(
      LiteRtAccelerator accelerator,
      LiteRtHwAcceleratorSet* supported_hardware);
  LiteRtStatus (*is_tflite_delegate_responsible_for_jit_compilation)(
      LiteRtAccelerator accelerator, bool* does_jit_compilation);
  LiteRtStatus (*create_delegate)(LiteRtAccelerator accelerator,
                                  LiteRtOptions options,
                                  LiteRtDelegateWrapper* delegate_wrapper);
  void (*destroy_delegate)(LiteRtDelegateWrapper delegate_wrapper);
  LiteRtStatus (*start_metrics_collection)(LiteRtDelegateWrapper delegate,
                                           int detail_level);
  LiteRtStatus (*stop_metrics_collection)(LiteRtDelegateWrapper delegate,
                                          LiteRtMetrics metrics);

  CreateCustomTensorBuffer create_func;
  DestroyCustomTensorBuffer destroy_func;
  LockCustomTensorBuffer lock_func;
  UnlockCustomTensorBuffer unlock_func;
  ClearCustomTensorBuffer clear_func;
  ImportCustomTensorBuffer import_func;

  LiteRtEnvOptionTag device_tag;
  LiteRtEnvOptionTag queue_tag;

  size_t num_supported_buffer_types;
  LiteRtTensorBufferType supported_buffer_types[16];
};

struct AcceleratorWrapperV1 {
  LiteRtStatus (*create_delegate)(LiteRtAccelerator, LiteRtOptions,
                                  LiteRtDelegateWrapper*);
};

LiteRtStatus CreateDelegateWrapperV1(LiteRtRuntimeContext* runtime_context,
                                     LiteRtEnvironment env,
                                     LiteRtAccelerator accelerator,
                                     LiteRtOptions options,
                                     LiteRtDelegateWrapper* delegate) {
  auto* wrapper = reinterpret_cast<AcceleratorWrapperV1*>(accelerator->data);
  if (!wrapper || !wrapper->create_delegate) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  return wrapper->create_delegate(accelerator, options, delegate);
}

void ReleaseWrapperDataV1(void* data) {
  delete reinterpret_cast<AcceleratorWrapperV1*>(data);
}

class V1ToActiveAdapter : public AcceleratorDefAdapter {
 public:
  LiteRtStatus Adapt(const LiteRtAcceleratorDef* legacy_def,
                     LiteRtAcceleratorDef* current_def, void** wrapper_data,
                     WrapperDeleter* wrapper_deleter) override {
    const auto* old_def =
        reinterpret_cast<const LiteRtAcceleratorDefV1*>(legacy_def);

    std::memset(current_def, 0, sizeof(LiteRtAcceleratorDef));
    current_def->version = LITERT_ACCELERATOR_DEF_CURRENT_VERSION;
    current_def->get_name = old_def->get_name;
    current_def->get_version = old_def->get_version;
    current_def->get_hardware_support = old_def->get_hardware_support;
    current_def->is_tflite_delegate_responsible_for_jit_compilation =
        old_def->is_tflite_delegate_responsible_for_jit_compilation;
    current_def->create_delegate = CreateDelegateWrapperV1;
    current_def->start_metrics_collection = nullptr;
    current_def->stop_metrics_collection = nullptr;

    current_def->buffer_handlers.create_func = old_def->create_func;
    current_def->buffer_handlers.destroy_func = old_def->destroy_func;
    current_def->buffer_handlers.lock_func = old_def->lock_func;
    current_def->buffer_handlers.unlock_func = old_def->unlock_func;
    current_def->buffer_handlers.clear_func = old_def->clear_func;
    current_def->buffer_handlers.import_func = old_def->import_func;
    current_def->buffer_handlers.device_tag = old_def->device_tag;
    current_def->buffer_handlers.queue_tag = old_def->queue_tag;

    size_t num_types = old_def->num_supported_buffer_types;
    if (num_types >
        LITERT_CUSTOM_BUFFER_HANDLERS_DEF_MAX_SUPPORTED_BUFFER_TYPES) {
      num_types = LITERT_CUSTOM_BUFFER_HANDLERS_DEF_MAX_SUPPORTED_BUFFER_TYPES;
    }
    current_def->buffer_handlers.num_supported_buffer_types = num_types;
    for (size_t i = 0; i < num_types; ++i) {
      current_def->buffer_handlers.supported_buffer_types[i] =
          old_def->supported_buffer_types[i];
    }

    *wrapper_data = new AcceleratorWrapperV1{old_def->create_delegate};
    *wrapper_deleter = ReleaseWrapperDataV1;

    return kLiteRtStatusOk;
  }
};

}  // namespace

std::unique_ptr<AcceleratorDefAdapter> AcceleratorDefAdapterFactory::Create(
    int version) {
  if (version == 1) {
    return std::make_unique<V1ToActiveAdapter>();
  }
  return nullptr;
}

}  // namespace litert::internal
