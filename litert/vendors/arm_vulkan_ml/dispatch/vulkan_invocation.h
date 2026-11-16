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
#include "litert/vendors/arm_vulkan_ml/dispatch/external_buffer.h"
#include "vulkan/vulkan.h"
namespace litert::arm_vulkan_ml {
// Synchronous execution with host staging or retained external-memory bindings.
// An invocation owns its Vulkan device and queue, avoiding shared queue
// synchronization and function-pointer lifetime.
class VulkanInvocation {
 public:
  struct CreateOptions {
    bool allow_emulation = false;
    bool dump_timings = false;
  };
  static std::unique_ptr<VulkanInvocation> Create(
      std::shared_ptr<const CompiledPartition> artifact, bool allow_emulation,
      std::string& error);
  static std::unique_ptr<VulkanInvocation> Create(
      std::shared_ptr<const CompiledPartition> artifact,
      const CreateOptions& options, std::string& error);
  // Internal dependency-injection entry point; resolver must outlive
  // invocation.
  static std::unique_ptr<VulkanInvocation> CreateWithResolver(
      std::shared_ptr<const CompiledPartition> artifact, bool allow_emulation,
      PFN_vkGetInstanceProcAddr resolver, std::string& error,
      std::shared_ptr<HardwareBufferApi> hardware_api = nullptr);
  static std::unique_ptr<VulkanInvocation> CreateWithResolver(
      std::shared_ptr<const CompiledPartition> artifact,
      const CreateOptions& options, PFN_vkGetInstanceProcAddr resolver,
      std::string& error,
      std::shared_ptr<HardwareBufferApi> hardware_api = nullptr);
  ~VulkanInvocation();
  struct BufferRequirements {
    size_t bytes = 0, alignment = 1;
    bool dma_buf = false, ahb = false;
  };
  bool Requirements(size_t slot, BufferRequirements&, std::string& error);
  bool AttachExternal(size_t slot, std::shared_ptr<ExternalBuffer>,
                      std::string& error);
  void DetachExternal(size_t slot);
  struct CopyStatistics {
    uint64_t input_bytes = 0, output_bytes = 0;
    size_t imported_slots = 0;
  };
  // Wall-clock stage durations (in milliseconds) and session scratch memory
  // allocations recorded during Vulkan device setup and data-graph pipeline
  // construction.
  struct InitStatistics {
    // Dynamic library load, instance creation, and instance proc resolution.
    double dlopen_and_instance_ms = 0.0;
    // Physical device enumeration and VK_ARM_tensors / VK_ARM_data_graph
    // capability and TOSA profile queries.
    double query_capabilities_ms = 0.0;
    // Logical VkDevice creation, device proc resolution, and queue setup.
    double create_device_ms = 0.0;
    // Total wall-clock time for Vulkan device initialization prior to Build().
    double device_setup_total_ms = 0.0;
    // SPIR-V VkShaderModule creation duration.
    double create_shader_module_ms = 0.0;
    // Descriptor set layout/pool allocation and pipeline constant marshaling.
    double descriptors_and_constants_ms = 0.0;
    // vkCreateDataGraphPipelinesARM driver compilation duration.
    double create_data_graph_pipeline_ms = 0.0;
    // vkCreateDataGraphPipelineSessionARM duration.
    double create_session_ms = 0.0;
    // Session bind-point query, scratch memory allocation, and binding.
    double session_memory_alloc_bind_ms = 0.0;
    // Command pool, command buffer, and completion fence creation.
    double cmd_pool_and_fence_ms = 0.0;
    // Total wall-clock time spent inside Build().
    double build_total_ms = 0.0;
    // Number of VkDeviceMemory allocations bound to the data-graph session.
    size_t session_memory_objects = 0;
    // Total bytes requested across all bound session scratch memory objects.
    size_t session_memory_bytes = 0;
  };
  CopyStatistics Statistics() const;
  InitStatistics InitStats() const;
  bool Run(const std::vector<const void*>& inputs,
           const std::vector<void*>& outputs, std::string& error);

 private:
  struct Impl;
  explicit VulkanInvocation(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};
}  // namespace litert::arm_vulkan_ml
#endif
