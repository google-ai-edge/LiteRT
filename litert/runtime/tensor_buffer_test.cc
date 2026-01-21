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

#include "litert/runtime/tensor_buffer.h"

#include <cstddef>
#include <cstdlib>
#include <memory>

#include <gtest/gtest.h>
#include "litert/c/internal/litert_tensor_buffer_registry.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/tensor_buffer_registry.h"
#include "litert/test/matchers.h"

namespace litert {
namespace {

struct CustomHwMemoryInfo : public HwMemoryInfo {
  size_t bytes;
  void* mapped_ptr;
};

static int kDummyHandleStorage = 0;

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
  return kLiteRtStatusOk;
}

TEST(TensorBufferTest, GetTensorBufferRegistry) {
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

  constexpr const LiteRtRankedTensorType kTensorType = {
      /*.element_type=*/kLiteRtElementTypeFloat32, litert::BuildLayout({4})};

  // Use a buffer type that keeps track of the environment.
  // WebGPU buffer is a good candidate as it uses CustomBuffer which requires
  // env.
  LITERT_ASSIGN_OR_ABORT(
      auto tensor_buffer,
      LiteRtTensorBufferT::CreateManaged(
          env.Get(), kLiteRtTensorBufferTypeWebGpuBuffer, kTensorType, 16));

  void* registry_from_buffer = tensor_buffer->GetTensorBufferRegistry();
  ASSERT_NE(registry_from_buffer, nullptr);

  void* registry_from_env = nullptr;
  LITERT_EXPECT_OK(
      LiteRtGetTensorBufferRegistry(env.Get(), &registry_from_env));

  EXPECT_EQ(registry_from_buffer, registry_from_env);
}

}  // namespace
}  // namespace litert
