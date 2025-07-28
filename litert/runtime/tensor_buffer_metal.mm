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

#import <Metal/Metal.h>

#include "litert/runtime/metal_memory.h"
#include "litert/runtime/tensor_buffer.h"

namespace litert::internal {

namespace {

class MetalMemoryWrapperImpl : public MetalMemoryWrapper {
 public:
  MetalMemoryWrapperImpl() : metal_memory_(MetalMemory()) {};
  MetalMemoryWrapperImpl(MetalMemoryWrapperImpl&& other)
      : metal_memory_(std::move(other.metal_memory_)) {};

  MetalMemoryWrapperImpl(GpuEnvironment* gpu_env, const LiteRtRankedTensorType& tensor_type,
                         LiteRtTensorBufferType buffer_type, void* metal_buffer, size_t buffer_size,
                         LiteRtMetalDeallocator deallocator)
      : metal_memory_(MetalMemory(gpu_env, tensor_type, buffer_type, metal_buffer, buffer_size,
                                  deallocator)) {};

  ~MetalMemoryWrapperImpl() override = default;

  template <typename T>
  Expected<T*> Lock(LiteRtTensorBufferLockMode mode) {
    return metal_memory_.Lock<T>(mode);
  }

  template <typename T>
  Expected<void> Unlock() {
    return metal_memory_.Unlock<T>();
  }

  void* GetMemoryPtr() { return metal_memory_.GetMemoryPtr(); }

  static Expected<MetalMemory> Alloc(GpuEnvironment* gpu_env,
                                     const LiteRtRankedTensorType& tensor_type,
                                     LiteRtTensorBufferType buffer_type, size_t bytes_size) {
    return MetalMemory::Alloc(gpu_env, tensor_type, buffer_type, bytes_size);
  }

 private:
  litert::internal::MetalMemory metal_memory_;
};

}  // namespace

MetalMemoryWrapperPtr MetalMemoryWrapper::Create() {
  return std::make_unique<MetalMemoryWrapperImpl>();
}

MetalMemoryWrapperPtr MetalMemoryWrapper::Create(GpuEnvironment* gpu_env,
                                                 const LiteRtRankedTensorType& tensor_type,
                                                 LiteRtTensorBufferType buffer_type,
                                                 void* metal_buffer, size_t buffer_size,
                                                 LiteRtMetalDeallocator deallocator) {
  return std::make_unique<MetalMemoryWrapperImpl>(gpu_env, tensor_type, buffer_type, metal_buffer,
                                                  buffer_size, deallocator);
}

}  // namespace litert::internal
