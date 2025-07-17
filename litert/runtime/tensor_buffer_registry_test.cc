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
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/tensor_buffer.h"

namespace {

struct CustomHwMemoryInfo {
  size_t bytes;
  void* mapped_ptr;
};

LiteRtStatus CreateMyCustomTensorBuffer(
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type, size_t bytes,
    HwMemoryHandle* webgpu_memory, HwMemoryInfo* hw_memory_info) {
  auto memory_info =
      new CustomHwMemoryInfo{.bytes = bytes, .mapped_ptr = nullptr};
  *hw_memory_info = reinterpret_cast<HwMemoryInfo>(memory_info);
  return kLiteRtStatusOk;
}

LiteRtStatus DestroyMyCustomTensorBuffer(HwMemoryInfo hw_memory_info) {
  auto hw_info = reinterpret_cast<CustomHwMemoryInfo*>(hw_memory_info);
  if (hw_info->mapped_ptr) {
    free(hw_info->mapped_ptr);
  }
  delete hw_info;
  return kLiteRtStatusOk;
}

LiteRtStatus UnlockMyCustomTensorBuffer(LiteRtTensorBuffer tensor_buffer,
                                        HwMemoryInfo hw_memory_info) {
  return kLiteRtStatusOk;
}

LiteRtStatus LockMyCustomTensorBuffer(LiteRtTensorBuffer tensor_buffer,
                                      LiteRtTensorBufferLockMode mode,
                                      HwMemoryInfo hw_memory_info,
                                      void** host_memory_ptr) {
  auto hw_info = reinterpret_cast<CustomHwMemoryInfo*>(hw_memory_info);
  if (!hw_info->mapped_ptr) {
    LITERT_LOG(LITERT_INFO, "Allocating mapped ptr for %zu bytes",
               hw_info->bytes);
    hw_info->mapped_ptr = malloc(hw_info->bytes);
  }
  memset(hw_info->mapped_ptr, 0, hw_info->bytes);
  *host_memory_ptr = hw_info->mapped_ptr;
  return kLiteRtStatusOk;
}

TEST(TensorBufferRegistryTest, Basic) {
  auto& registry = litert::internal::TensorBufferRegistry::GetInstance();
  registry.RegisterHandlers(
      kLiteRtTensorBufferTypeWebGpuBuffer, CreateMyCustomTensorBuffer,
      DestroyMyCustomTensorBuffer, LockMyCustomTensorBuffer,
      UnlockMyCustomTensorBuffer);

  auto handlers =
      registry.GetCustomHandlers(kLiteRtTensorBufferTypeWebGpuBuffer);
  EXPECT_TRUE(handlers);

  LiteRtEnvironment env = nullptr;
  constexpr const LiteRtRankedTensorType kTensorType = {
      /*.element_type=*/kLiteRtElementTypeInt32, litert::BuildLayout({8})};
  LITERT_ASSIGN_OR_ABORT(
      auto tensor_buffer,
      LiteRtTensorBufferT::CreateManaged(
          env, kLiteRtTensorBufferTypeWebGpuBuffer, kTensorType, 32));
  LITERT_ASSIGN_OR_ABORT(auto mapped_ptr,
                         tensor_buffer->Lock(kLiteRtTensorBufferLockModeRead));
  int32_t* mapped_ptr_int32 = reinterpret_cast<int32_t*>(mapped_ptr);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(mapped_ptr_int32[i], 0);
  }
  EXPECT_TRUE(tensor_buffer->Unlock());
}

}  // namespace
