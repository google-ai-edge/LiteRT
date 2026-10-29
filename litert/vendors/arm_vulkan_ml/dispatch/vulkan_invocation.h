// Copyright 2026 The LiteRT Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_DISPATCH_VULKAN_INVOCATION_H_
#define ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_DISPATCH_VULKAN_INVOCATION_H_
#include "vulkan/vulkan.h"
#include "litert/vendors/arm_vulkan_ml/dispatch/external_buffer.h"
#include <memory>
#include <string>
#include <vector>
#include "litert/vendors/arm_vulkan_ml/common/arm_vulkan_ml_jit_executable.h"
namespace litert::arm_vulkan_ml {
// Synchronous execution with host staging or retained external-memory bindings. An invocation owns its Vulkan device and
// queue, avoiding shared queue synchronization and function-pointer lifetime.
class VulkanInvocation {
 public:
  static std::unique_ptr<VulkanInvocation> Create(
      std::shared_ptr<const CompiledPartition> artifact, bool allow_emulation,
      std::string& error);
  // Internal dependency-injection entry point; resolver must outlive invocation.
  static std::unique_ptr<VulkanInvocation> CreateWithResolver(
      std::shared_ptr<const CompiledPartition> artifact, bool allow_emulation,
      PFN_vkGetInstanceProcAddr resolver, std::string& error,
      std::shared_ptr<HardwareBufferApi> hardware_api = nullptr);
  ~VulkanInvocation();
  struct BufferRequirements { size_t bytes=0, alignment=1; bool dma_buf=false, ahb=false; };
  bool Requirements(size_t slot, BufferRequirements&, std::string& error);
  bool AttachExternal(size_t slot, std::shared_ptr<ExternalBuffer>, std::string& error);
  void DetachExternal(size_t slot);
  struct CopyStatistics { uint64_t input_bytes=0, output_bytes=0; size_t imported_slots=0; };
  CopyStatistics Statistics() const;
  bool Run(const std::vector<const void*>& inputs,
           const std::vector<void*>& outputs, std::string& error);
 private:
  struct Impl;
  explicit VulkanInvocation(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};
}  // namespace litert::arm_vulkan_ml
#endif
