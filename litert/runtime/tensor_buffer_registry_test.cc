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

#include "litert/runtime/tensor_buffer_registry.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>

#include <gtest/gtest.h>
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_tensor_buffer_registry.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/tensor_buffer.h"
#include "litert/test/matchers.h"

namespace {

struct CustomHwMemoryInfo : public HwMemoryInfo {
  size_t bytes;
  void* mapped_ptr;
};

static int kDummyHandleStorage = 0;
constexpr char kMockBufferValue = 0xaa;

LiteRtStatus CreateMyCustomTensorBuffer(
    LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type, size_t bytes, size_t packed_bytes,
    HwMemoryInfoPtr* hw_memory_info) {
  auto memory_info =
      new CustomHwMemoryInfo{.bytes = bytes, .mapped_ptr = nullptr};
  memory_info->memory_handle =
      reinterpret_cast<HwMemoryHandle>(&kDummyHandleStorage);
  *hw_memory_info = memory_info;
  return kLiteRtStatusOk;
}

LiteRtStatus DestroyMyCustomTensorBuffer(LiteRtEnvironment env,
                                         HwMemoryInfoPtr hw_memory_info) {
  auto hw_info = reinterpret_cast<CustomHwMemoryInfo*>(hw_memory_info);
  if (hw_info->mapped_ptr) {
    free(hw_info->mapped_ptr);
  }
  delete hw_info;
  return kLiteRtStatusOk;
}

LiteRtStatus UnlockMyCustomTensorBuffer(LiteRtEnvironment env,
                                        HwMemoryInfoPtr hw_memory_info) {
  return kLiteRtStatusOk;
}

LiteRtStatus LockMyCustomTensorBuffer(LiteRtEnvironment env,
                                      HwMemoryInfoPtr hw_memory_info,
                                      LiteRtTensorBufferLockMode mode,
                                      void** host_memory_ptr) {
  auto hw_info = reinterpret_cast<CustomHwMemoryInfo*>(hw_memory_info);
  if (!hw_info->mapped_ptr) {
    LITERT_LOG(LITERT_INFO, "Allocating mapped ptr for %zu bytes",
               hw_info->bytes);
    hw_info->mapped_ptr = malloc(hw_info->bytes);
  }
  memset(hw_info->mapped_ptr, kMockBufferValue, hw_info->bytes);
  *host_memory_ptr = hw_info->mapped_ptr;
  return kLiteRtStatusOk;
}

TEST(TensorBufferRegistryTest, Basic) {
  LITERT_ASSIGN_OR_ABORT(litert::Environment env,
                         litert::Environment::Create({}));
  litert::internal::TensorBufferRegistry* registry = nullptr;
  LITERT_EXPECT_OK(LiteRtGetTensorBufferRegistry(
      env.Get(), reinterpret_cast<void**>(&registry)));
  litert::internal::CustomTensorBufferHandlers handlers = {
      .create_func = CreateMyCustomTensorBuffer,
      .destroy_func = DestroyMyCustomTensorBuffer,
      .lock_func = LockMyCustomTensorBuffer,
      .unlock_func = UnlockMyCustomTensorBuffer,
  };
  registry->RegisterHandlers(kLiteRtTensorBufferTypeWebGpuBuffer, handlers);

  auto registered_handlers =
      registry->GetCustomHandlers(kLiteRtTensorBufferTypeWebGpuBuffer);
  EXPECT_TRUE(registered_handlers);

  constexpr const LiteRtRankedTensorType kTensorType = {
      /*.element_type=*/kLiteRtElementTypeInt32, litert::BuildLayout({8})};
  LITERT_ASSIGN_OR_ABORT(
      auto tensor_buffer,
      LiteRtTensorBufferT::CreateManaged(
          env.Get(), kLiteRtTensorBufferTypeWebGpuBuffer, kTensorType, 32));

  LITERT_ASSIGN_OR_ABORT(auto custom_buffer, tensor_buffer->GetCustomBuffer());
  EXPECT_EQ(custom_buffer->hw_buffer_handle(), &kDummyHandleStorage);

  LITERT_ASSIGN_OR_ABORT(auto mapped_ptr,
                         tensor_buffer->Lock(kLiteRtTensorBufferLockModeRead));
  char* mapped_ptr_char = reinterpret_cast<char*>(mapped_ptr);
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(mapped_ptr_char[i], kMockBufferValue);
  }
  EXPECT_TRUE(tensor_buffer->Unlock());
}

TEST(TensorBufferRegistryTest, RegistryOwnership) {
  LITERT_ASSIGN_OR_ABORT(litert::Environment env,
                         litert::Environment::Create({}));
  litert::internal::TensorBufferRegistry* registry = nullptr;
  LITERT_EXPECT_OK(LiteRtGetTensorBufferRegistry(
      env.Get(), reinterpret_cast<void**>(&registry)));
  auto handlers =
      std::make_unique<litert::internal::CustomTensorBufferHandlers>();
  handlers->create_func = CreateMyCustomTensorBuffer;
  handlers->destroy_func = DestroyMyCustomTensorBuffer;
  handlers->lock_func = LockMyCustomTensorBuffer;
  handlers->unlock_func = UnlockMyCustomTensorBuffer;
  registry->RegisterHandlers(kLiteRtTensorBufferTypeWebGpuBuffer, *handlers);
  // Reset the handlers in the caller side to check the if registry copied them
  // properly.
  handlers.reset();

  auto registered_handlers =
      registry->GetCustomHandlers(kLiteRtTensorBufferTypeWebGpuBuffer);
  EXPECT_TRUE(registered_handlers);

  constexpr const LiteRtRankedTensorType kTensorType = {
      /*.element_type=*/kLiteRtElementTypeInt32, litert::BuildLayout({8})};
  LITERT_ASSIGN_OR_ABORT(
      auto tensor_buffer,
      LiteRtTensorBufferT::CreateManaged(
          env.Get(), kLiteRtTensorBufferTypeWebGpuBuffer, kTensorType, 32));

  LITERT_ASSIGN_OR_ABORT(auto custom_buffer, tensor_buffer->GetCustomBuffer());
  EXPECT_EQ(custom_buffer->hw_buffer_handle(), &kDummyHandleStorage);

  LITERT_ASSIGN_OR_ABORT(auto mapped_ptr,
                         tensor_buffer->Lock(kLiteRtTensorBufferLockModeRead));
  char* mapped_ptr_char = reinterpret_cast<char*>(mapped_ptr);
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(mapped_ptr_char[i], kMockBufferValue);
  }
  EXPECT_TRUE(tensor_buffer->Unlock());
}

}  // namespace
