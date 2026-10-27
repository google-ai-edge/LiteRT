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
#include <memory>
#include <string>
#include <vector>
#include "litert/vendors/arm_vulkan_ml/common/arm_vulkan_ml_jit_executable.h"
namespace litert::arm_vulkan_ml {
// Synchronous host-buffer execution. An invocation owns its Vulkan device and
// queue, avoiding shared queue synchronization and function-pointer lifetime.
class VulkanInvocation {
 public:
  static std::unique_ptr<VulkanInvocation> Create(
      std::shared_ptr<const CompiledPartition> artifact, bool allow_emulation,
      std::string& error);
  ~VulkanInvocation();
  bool Run(const std::vector<const void*>& inputs,
           const std::vector<void*>& outputs, std::string& error);
 private:
  struct Impl;
  explicit VulkanInvocation(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};
}  // namespace litert::arm_vulkan_ml
#endif
