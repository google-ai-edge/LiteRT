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

// Vulkan graph/tensor setup follows the public-API approach in Arm nxdelegate
// (src/vulkan/vulkan_compute.cc). This implementation owns all resources and
// function pointers per invocation and checks the advertised TOSA requirements.
#include "litert/vendors/arm_vulkan_ml/dispatch/vulkan_invocation.h"
#include "litert/vendors/arm_vulkan_ml/dispatch/vulkan_validation.h"
#include "litert/vendors/arm_vulkan_ml/dispatch/device_memory_allocator.h"
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#if defined(__ANDROID__)
#include <android/hardware_buffer.h>
#include "vulkan/vulkan_android.h"
#endif
#include <algorithm>
#include <cstring>
#include <utility>

#include "absl/synchronization/mutex.h"
#include "litert/vendors/arm_vulkan_ml/dispatch/tensor_capabilities.h"
#include "litert/vendors/arm_vulkan_ml/dispatch/device_capabilities.h"
#include "vulkan/vulkan.h"

namespace litert::arm_vulkan_ml {
static_assert(VK_ARM_TENSORS_SPEC_VERSION >= 2,
              "Arm Vulkan ML requires VK_ARM_tensors Revision 2 headers");
namespace {
size_t Bytes(const TensorDescriptor& tensor) {
  size_t result = tensor.element == TensorElement::kInt8 ? 1 : 4;
  for (auto dim : tensor.shape) {
    if (!dim || result > SIZE_MAX / dim) return 0;
    result *= dim;
  }
  return result;
}
}  // namespace

struct VulkanInvocation::Impl {
  VulkanValidation validation;
  std::shared_ptr<const CompiledPartition> artifact;
  absl::Mutex mutex;
  void* library = nullptr;
  PFN_vkGetInstanceProcAddr get = nullptr;
  PFN_vkDestroyInstance destroy_instance = nullptr;
  VkInstance instance = VK_NULL_HANDLE;
  PFN_vkGetDeviceProcAddr device_get = nullptr;
  PFN_vkGetPhysicalDeviceExternalTensorPropertiesARM external_properties = nullptr;
  PFN_vkGetMemoryFdPropertiesKHR fd_properties = nullptr;
  bool dma_enabled=false, ahb_enabled=false;
  std::shared_ptr<HardwareBufferApi> hardware_api;
  CopyStatistics statistics;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t max_allocations = 0;
  std::unique_ptr<DeviceMemoryAllocator> allocator;
  VkPhysicalDeviceMemoryProperties memory_properties{};
  uint32_t queue_family = 0;
  PFN_vkDestroyDevice DestroyDevice = nullptr;
  PFN_vkGetDeviceQueue GetDeviceQueue = nullptr;
  PFN_vkCreateShaderModule CreateShaderModule = nullptr;
  PFN_vkDestroyShaderModule DestroyShaderModule = nullptr;
  PFN_vkCreateTensorARM CreateTensorARM = nullptr;
  PFN_vkDestroyTensorARM DestroyTensorARM = nullptr;
  PFN_vkGetTensorMemoryRequirementsARM GetTensorMemoryRequirementsARM = nullptr;
  PFN_vkBindTensorMemoryARM BindTensorMemoryARM = nullptr;
  PFN_vkCreateTensorViewARM CreateTensorViewARM = nullptr;
  PFN_vkDestroyTensorViewARM DestroyTensorViewARM = nullptr;
  PFN_vkAllocateMemory AllocateMemory = nullptr;
  PFN_vkFreeMemory FreeMemory = nullptr;
  PFN_vkMapMemory MapMemory = nullptr;
  PFN_vkUnmapMemory UnmapMemory = nullptr;
  PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout = nullptr;
  PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout = nullptr;
  PFN_vkCreateDescriptorPool CreateDescriptorPool = nullptr;
  PFN_vkDestroyDescriptorPool DestroyDescriptorPool = nullptr;
  PFN_vkAllocateDescriptorSets AllocateDescriptorSets = nullptr;
  PFN_vkUpdateDescriptorSets UpdateDescriptorSets = nullptr;
  PFN_vkCreatePipelineLayout CreatePipelineLayout = nullptr;
  PFN_vkDestroyPipelineLayout DestroyPipelineLayout = nullptr;
  PFN_vkCreateDataGraphPipelinesARM CreateDataGraphPipelinesARM = nullptr;
  PFN_vkDestroyPipeline DestroyPipeline = nullptr;
  PFN_vkCreateDataGraphPipelineSessionARM CreateDataGraphPipelineSessionARM = nullptr;
  PFN_vkDestroyDataGraphPipelineSessionARM DestroyDataGraphPipelineSessionARM = nullptr;
  PFN_vkGetDataGraphPipelineSessionBindPointRequirementsARM GetDataGraphPipelineSessionBindPointRequirementsARM = nullptr;
  PFN_vkGetDataGraphPipelineSessionMemoryRequirementsARM GetDataGraphPipelineSessionMemoryRequirementsARM = nullptr;
  PFN_vkBindDataGraphPipelineSessionMemoryARM BindDataGraphPipelineSessionMemoryARM = nullptr;
  PFN_vkCreateCommandPool CreateCommandPool = nullptr;
  PFN_vkDestroyCommandPool DestroyCommandPool = nullptr;
  PFN_vkAllocateCommandBuffers AllocateCommandBuffers = nullptr;
  PFN_vkBeginCommandBuffer BeginCommandBuffer = nullptr;
  PFN_vkEndCommandBuffer EndCommandBuffer = nullptr;
  PFN_vkResetCommandBuffer ResetCommandBuffer = nullptr;
  PFN_vkCmdBindPipeline CmdBindPipeline = nullptr;
  PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets = nullptr;
  PFN_vkCmdDispatchDataGraphARM CmdDispatchDataGraphARM = nullptr;
  PFN_vkCmdPipelineBarrier2 CmdPipelineBarrier2 = nullptr;
  PFN_vkCreateFence CreateFence = nullptr;
  PFN_vkDestroyFence DestroyFence = nullptr;
  PFN_vkResetFences ResetFences = nullptr;
  PFN_vkQueueSubmit QueueSubmit = nullptr;
  PFN_vkWaitForFences WaitForFences = nullptr;
  PFN_vkDeviceWaitIdle DeviceWaitIdle = nullptr;
  VkShaderModule module = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkDataGraphPipelineSessionARM session = VK_NULL_HANDLE;
  std::vector<VkDeviceMemory> session_memory;
  VkCommandPool command_pool = VK_NULL_HANDLE;
  VkCommandBuffer command = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  struct TensorMemory {
    VkTensorARM tensor = VK_NULL_HANDLE;
    VkTensorViewARM view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    size_t bytes = 0;
    size_t binding_span = 0;
    std::shared_ptr<ExternalBuffer> external;
  };
  std::vector<TensorMemory> tensors;
  // Successful synchronous submissions leave this recording executable.
  bool command_recorded = false;
  bool command_needs_reset = false;
  bool failed_submission=false;
  std::string error;
  bool Check(VkResult result, const char* operation) {
    if (result == VK_SUCCESS) return true;
    error = std::string(operation) + " returned VkResult " + std::to_string(result);
    return false;
  }
#define VK_CHECK(call) do { if (!Check((call), #call)) return false; } while (false)

  ~Impl() {
    if (device && DeviceWaitIdle) DeviceWaitIdle(device);
    if (fence) DestroyFence(device, fence, nullptr);
    if (command_pool) DestroyCommandPool(device, command_pool, nullptr);
    if (session) DestroyDataGraphPipelineSessionARM(device, session, nullptr);
    for (auto memory : session_memory) FreeMemory(device, memory, nullptr);
    if (pipeline) DestroyPipeline(device, pipeline, nullptr);
    if (pipeline_layout) DestroyPipelineLayout(device, pipeline_layout, nullptr);
    if (descriptor_pool) DestroyDescriptorPool(device, descriptor_pool, nullptr);
    if (descriptor_layout) DestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
    for (auto& t : tensors) {
      if (t.mapped) UnmapMemory(device, t.memory);
      if (t.view) DestroyTensorViewARM(device, t.view, nullptr);
      if (t.tensor) DestroyTensorARM(device, t.tensor, nullptr);
      if (t.memory) FreeMemory(device, t.memory, nullptr);
    }
    if (module) DestroyShaderModule(device, module, nullptr);
    if (device && DestroyDevice) DestroyDevice(device, nullptr);
    validation.Close();
    if (instance && destroy_instance) destroy_instance(instance, nullptr);
    if (library) dlclose(library);
  }

  bool Initialize(bool allow_emulation, PFN_vkGetInstanceProcAddr resolver = nullptr) {
    get = resolver;
    if (!get) {
      const char* kVulkanLibrary = VulkanLibraryName();
      library = dlopen(kVulkanLibrary, RTLD_NOW | RTLD_LOCAL);
      if (!library) {
        const char* detail = dlerror();
        error = std::string("Cannot load ") + kVulkanLibrary + ": " +
            (detail ? detail : "unknown dynamic loader error");
        return false;
      }
      get = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"));
    }
    if (!get) { error = "Missing vkGetInstanceProcAddr"; return false; }
    auto create = reinterpret_cast<PFN_vkCreateInstance>(get(nullptr, "vkCreateInstance"));
    if (!create) { error = "Missing vkCreateInstance"; return false; }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "LiteRT Arm Vulkan ML";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    if (!validation.Prepare(get, ci, error)) return false;
    VK_CHECK(create(&ci, nullptr, &instance));
    destroy_instance = reinterpret_cast<PFN_vkDestroyInstance>(get(instance, "vkDestroyInstance"));
    if (!validation.Attach(get, instance, error)) return false;
#define INSTANCE_FN(name) auto name = reinterpret_cast<PFN_vk##name>(get(instance, "vk" #name)); if (!name) { error = "Missing vk" #name; return false; }
    INSTANCE_FN(EnumeratePhysicalDevices);
    INSTANCE_FN(GetPhysicalDeviceProperties);
    INSTANCE_FN(GetPhysicalDeviceProperties2);
    INSTANCE_FN(GetPhysicalDeviceFormatProperties2);
    INSTANCE_FN(GetPhysicalDeviceFeatures2);
    INSTANCE_FN(GetPhysicalDeviceMemoryProperties);
    INSTANCE_FN(CreateDevice);
    INSTANCE_FN(GetDeviceProcAddr);
    uint32_t count = 0;
    VK_CHECK(EnumeratePhysicalDevices(instance, &count, nullptr));
    std::vector<VkPhysicalDevice> candidates(count);
    VK_CHECK(EnumeratePhysicalDevices(instance, &count, candidates.data()));
    std::vector<const char*> extensions = {VK_ARM_TENSORS_EXTENSION_NAME,
      VK_ARM_DATA_GRAPH_EXTENSION_NAME, VK_ARM_DATA_GRAPH_INSTRUCTION_SET_TOSA_EXTENSION_NAME,
      VK_KHR_MAINTENANCE_5_EXTENSION_NAME, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
      VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME};
    for (auto candidate : candidates) {
      DeviceCapabilities caps;
      if (!QueryDeviceCapabilities(get, instance, candidate, caps, error) ||
          !CheckDeviceRequirements(caps, error)) continue;
      if (!QueryTensorCapabilities(candidate, *artifact, GetPhysicalDeviceProperties2,
                                   GetPhysicalDeviceFormatProperties2, error)) continue;
      for (const auto& engine : caps.engines) {
        if (CheckEngineRequirements(*artifact, caps, engine, allow_emulation, error)) {
          physical = candidate; queue_family = engine.queue_family;
          max_allocations = caps.properties.limits.maxMemoryAllocationCount; break;
        }
      }
      if (physical) break;
    }
    if (!physical) {
      error = "No compatible Vulkan graph device: " + (error.empty()
          ? std::string("required API revisions, graph engine or TOSA requirements unavailable")
          : error);
      return false;
    }
    error.clear();
    DeviceCapabilities selected;
    if(!QueryDeviceCapabilities(get,instance,physical,selected,error))return false;
    auto supported=[&](const char* name){return std::any_of(selected.extensions.begin(),selected.extensions.end(),[&](const auto& e){return !std::strcmp(e.extensionName,name);});};
    external_properties=reinterpret_cast<PFN_vkGetPhysicalDeviceExternalTensorPropertiesARM>(get(instance,"vkGetPhysicalDeviceExternalTensorPropertiesARM"));
    dma_enabled=external_properties && supported(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME) && supported(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    if(dma_enabled){extensions.push_back(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);}
    ahb_enabled=external_properties && hardware_api && supported("VK_ANDROID_external_memory_android_hardware_buffer") && supported(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME);
    if(ahb_enabled){extensions.push_back("VK_ANDROID_external_memory_android_hardware_buffer");extensions.push_back(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME);}
    VkPhysicalDeviceTensorFeaturesARM tensor_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM};
    VkPhysicalDeviceDataGraphFeaturesARM graph_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &f12; f12.pNext = &f13; f13.pNext = &tensor_features;
    VkPhysicalDeviceMaintenance5FeaturesKHR maintenance{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR};
    VkPhysicalDeviceShaderReplicatedCompositesFeaturesEXT replicated{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_REPLICATED_COMPOSITES_FEATURES_EXT};
    tensor_features.pNext = &graph_features;
    graph_features.pNext = &maintenance;maintenance.pNext = &replicated;
    GetPhysicalDeviceFeatures2(physical, &features);
    if (!tensor_features.tensors || !graph_features.dataGraph || !f13.synchronization2 || !maintenance.maintenance5 || !replicated.shaderReplicatedComposites) {
      error = "Required Vulkan graph/tensor/synchronization features unavailable"; return false;
    }
    float priority = 1;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = queue_family; queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.pNext = &features;
    device_info.queueCreateInfoCount = 1; device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size()); device_info.ppEnabledExtensionNames = extensions.data();
    VK_CHECK(CreateDevice(physical, &device_info, nullptr, &device));
#define DEVICE_FN(name) name = reinterpret_cast<PFN_vk##name>(GetDeviceProcAddr(device, "vk" #name)); if (!name) { error = "Missing vk" #name; return false; }
    DEVICE_FN(DestroyDevice);
    DEVICE_FN(GetDeviceQueue);
    DEVICE_FN(CreateShaderModule);
    DEVICE_FN(DestroyShaderModule);
    DEVICE_FN(CreateTensorARM);
    DEVICE_FN(DestroyTensorARM);
    DEVICE_FN(GetTensorMemoryRequirementsARM);
    DEVICE_FN(BindTensorMemoryARM);
    DEVICE_FN(CreateTensorViewARM);
    DEVICE_FN(DestroyTensorViewARM);
    DEVICE_FN(AllocateMemory);
    DEVICE_FN(FreeMemory);
    DEVICE_FN(MapMemory);
    DEVICE_FN(UnmapMemory);
    DEVICE_FN(CreateDescriptorSetLayout);
    DEVICE_FN(DestroyDescriptorSetLayout);
    DEVICE_FN(CreateDescriptorPool);
    DEVICE_FN(DestroyDescriptorPool);
    DEVICE_FN(AllocateDescriptorSets);
    DEVICE_FN(UpdateDescriptorSets);
    DEVICE_FN(CreatePipelineLayout);
    DEVICE_FN(DestroyPipelineLayout);
    DEVICE_FN(CreateDataGraphPipelinesARM);
    DEVICE_FN(DestroyPipeline);
    DEVICE_FN(CreateDataGraphPipelineSessionARM);
    DEVICE_FN(DestroyDataGraphPipelineSessionARM);
    DEVICE_FN(GetDataGraphPipelineSessionBindPointRequirementsARM);
    DEVICE_FN(GetDataGraphPipelineSessionMemoryRequirementsARM);
    DEVICE_FN(BindDataGraphPipelineSessionMemoryARM);
    DEVICE_FN(CreateCommandPool);
    DEVICE_FN(DestroyCommandPool);
    DEVICE_FN(AllocateCommandBuffers);
    DEVICE_FN(BeginCommandBuffer);
    DEVICE_FN(EndCommandBuffer);
    DEVICE_FN(ResetCommandBuffer);
    DEVICE_FN(CmdBindPipeline);
    DEVICE_FN(CmdBindDescriptorSets);
    DEVICE_FN(CmdDispatchDataGraphARM);
    DEVICE_FN(CmdPipelineBarrier2);
    DEVICE_FN(CreateFence);
    DEVICE_FN(DestroyFence);
    DEVICE_FN(ResetFences);
    DEVICE_FN(QueueSubmit);
    DEVICE_FN(WaitForFences);
    DEVICE_FN(DeviceWaitIdle);
#undef DEVICE_FN
#undef INSTANCE_FN
    device_get=GetDeviceProcAddr;
    if(dma_enabled){fd_properties=reinterpret_cast<PFN_vkGetMemoryFdPropertiesKHR>(device_get(device,"vkGetMemoryFdPropertiesKHR"));if(!fd_properties){error="Missing vkGetMemoryFdPropertiesKHR";return false;}}
    GetDeviceQueue(device, queue_family, 0, &queue);
    GetPhysicalDeviceMemoryProperties(physical, &memory_properties);
    allocator = std::make_unique<DeviceMemoryAllocator>(device, memory_properties, max_allocations, AllocateMemory);
    return Build();
  }

  bool Allocate(const VkMemoryRequirements& requirements, VkMemoryPropertyFlags flags,
                VkDeviceMemory& memory) {
    return allocator->Allocate(requirements, flags, memory, error);
  }

  TensorDescriptor Descriptor(size_t slot) const {
    return slot<artifact->inputs.size()?artifact->inputs[slot]:artifact->outputs[slot-artifact->inputs.size()];
  }
  void DestroyTensor(TensorMemory& t){
    if(t.mapped)UnmapMemory(device,t.memory);
    if(t.view)DestroyTensorViewARM(device,t.view,nullptr);
    if(t.tensor)DestroyTensorARM(device,t.tensor,nullptr);
    if(t.memory){FreeMemory(device,t.memory,nullptr);allocator->Released();}
    t=TensorMemory{};
  }
  bool Supports(size_t slot,VkExternalMemoryHandleTypeFlagBits handle){
    if(!external_properties || (handle==VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT?!dma_enabled:!ahb_enabled))return false;
    auto desc=Descriptor(slot);std::vector<int64_t> dims(desc.shape.begin(),desc.shape.end());
    VkTensorDescriptionARM d{VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM};d.tiling=VK_TENSOR_TILING_LINEAR_ARM;d.format=TensorFormat(desc.element);d.dimensionCount=static_cast<uint32_t>(dims.size());d.pDimensions=dims.data();d.usage=VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM;
    VkPhysicalDeviceExternalTensorInfoARM info{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_TENSOR_INFO_ARM};info.pDescription=&d;info.handleType=handle;
    VkExternalTensorPropertiesARM props{VK_STRUCTURE_TYPE_EXTERNAL_TENSOR_PROPERTIES_ARM};external_properties(physical,&info,&props);
    return (props.externalMemoryProperties.externalMemoryFeatures&VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) && (props.externalMemoryProperties.compatibleHandleTypes&handle);
  }
  bool CreateTensor(size_t slot,VkExternalMemoryHandleTypeFlags handles,TensorMemory& t,VkMemoryRequirements2& req,VkMemoryDedicatedRequirements& dedicated){
    auto desc=Descriptor(slot);std::vector<int64_t> dims(desc.shape.begin(),desc.shape.end());t.bytes=Bytes(desc);
    VkTensorDescriptionARM d{VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM};d.tiling=VK_TENSOR_TILING_LINEAR_ARM;d.format=TensorFormat(desc.element);d.dimensionCount=static_cast<uint32_t>(dims.size());d.pDimensions=dims.data();d.usage=VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM;
    VkExternalMemoryTensorCreateInfoARM external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_TENSOR_CREATE_INFO_ARM};external.handleTypes=handles;
    VkTensorCreateInfoARM info{VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_ARM};info.pNext=handles?&external:nullptr;info.pDescription=&d;info.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(CreateTensorARM(device,&info,nullptr,&t.tensor));
    VkTensorMemoryRequirementsInfoARM query{VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_ARM};query.tensor=t.tensor;req.pNext=&dedicated;GetTensorMemoryRequirementsARM(device,&query,&req);
    if(req.memoryRequirements.size<t.bytes || !req.memoryRequirements.alignment){error="Invalid tensor memory requirements";return false;}t.binding_span=static_cast<size_t>(req.memoryRequirements.size);return true;
  }
  bool Requirements(size_t slot,BufferRequirements& out){
    if(slot>=tensors.size()){error="Invalid tensor slot";return false;}
    out={};out.bytes=Bytes(Descriptor(slot));
    for(auto handle:{VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID}){
      if(!Supports(slot,handle))continue;
      TensorMemory t;VkMemoryRequirements2 req{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};VkMemoryDedicatedRequirements dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
      if(!CreateTensor(slot,handle,t,req,dedicated)){DestroyTensor(t);return false;}
      out.bytes=std::max(out.bytes,static_cast<size_t>(req.memoryRequirements.size));out.alignment=std::max(out.alignment,static_cast<size_t>(req.memoryRequirements.alignment));
      if(handle==VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT)out.dma_buf=true;else out.ahb=true;
      DestroyTensor(t);
    }return true;
  }
  bool PrepareTensor(size_t slot,std::shared_ptr<ExternalBuffer> external){
    if(slot>=tensors.size()){error="Invalid IO slot";return false;}
    if(tensors[slot].tensor && tensors[slot].external==external)return true;
    if(external){
      if(!Supports(slot,external->HandleType())){error="External tensor handle is not importable for this tensor";return false;}
    }
    TensorMemory t;VkMemoryRequirements2 req{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};VkMemoryDedicatedRequirements dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    auto prepare=[&]()->bool{
      if(!CreateTensor(slot,external?external->HandleType():0,t,req,dedicated))return false;
      if(external)for(size_t i=0;i<tensors.size();++i)if(i!=slot && tensors[i].external && (slot>=artifact->inputs.size() || i>=artifact->inputs.size()) && (external->kind!=tensors[i].external->kind || external->Overlaps(*tensors[i].external,t.binding_span,tensors[i].binding_span))){error="Overlapping writable views or mixed external handle kinds with writable IO";return false;}
      VkMemoryDedicatedAllocateInfoTensorARM di{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_TENSOR_ARM};di.tensor=t.tensor;
      bool needs_dedicated=dedicated.requiresDedicatedAllocation;
      if(external){
        VkPhysicalDeviceExternalTensorInfoARM info{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_TENSOR_INFO_ARM};
        // Query again using the same description for DEDICATED_ONLY.
        auto desc=Descriptor(slot);std::vector<int64_t> dims(desc.shape.begin(),desc.shape.end());VkTensorDescriptionARM description{VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM};description.tiling=VK_TENSOR_TILING_LINEAR_ARM;description.format=TensorFormat(desc.element);description.dimensionCount=static_cast<uint32_t>(dims.size());description.pDimensions=dims.data();description.usage=VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM;
        info.pDescription=&description;info.handleType=external->HandleType();VkExternalTensorPropertiesARM props{VK_STRUCTURE_TYPE_EXTERNAL_TENSOR_PROPERTIES_ARM};external_properties(physical,&info,&props);needs_dedicated|=(props.externalMemoryProperties.externalMemoryFeatures&VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT)!=0;
        if(external->offset>external->size || req.memoryRequirements.size>external->size-external->offset || external->offset%req.memoryRequirements.alignment || (needs_dedicated && external->offset)){error="External tensor allocation size, offset or alignment incompatible";return false;}
        VkMemoryRequirements imported=req.memoryRequirements;imported.size=needs_dedicated?req.memoryRequirements.size:external->size;
        if(external->kind==ExternalBuffer::Kind::kDmaBuf){
          VkMemoryFdPropertiesKHR props_fd{VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};VK_CHECK(fd_properties(device,external->HandleType(),external->fd,&props_fd));imported.memoryTypeBits&=props_fd.memoryTypeBits;
          VkImportMemoryFdInfoKHR fd{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};fd.handleType=external->HandleType();fd.fd=fcntl(external->fd,F_DUPFD_CLOEXEC,0);if(fd.fd<0){error="Cannot duplicate import FD";return false;}fd.pNext=needs_dedicated?&di:nullptr;
          if(!allocator->Allocate(imported,0,t.memory,error,&fd)){close(fd.fd);return false;}
        }else{
          uint32_t bits=0;VkDeviceSize size=0;if(!external->api->Properties(device,device_get,external->ahb,size,bits) || size<req.memoryRequirements.size){error="Incompatible AHardwareBuffer Vulkan properties";return false;}if(needs_dedicated && size!=req.memoryRequirements.size){error="Dedicated AHardwareBuffer allocation size must match tensor requirements";return false;}imported.size=size;imported.memoryTypeBits&=bits;
          // Platform-neutral ABI mirror; Android adapter is compiled against native declarations.
#if defined(__ANDROID__)
          VkImportAndroidHardwareBufferInfoANDROID ahb{VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID};ahb.pNext=needs_dedicated?&di:nullptr;ahb.buffer=static_cast<AHardwareBuffer*>(external->ahb);
#else
          struct AhbImport{VkStructureType sType;const void* pNext;void* buffer;} ahb{VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,needs_dedicated?&di:nullptr,external->ahb};
#endif
          if(!allocator->Allocate(imported,0,t.memory,error,&ahb))return false;
        }
      }else if(!allocator->Allocate(req.memoryRequirements,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,t.memory,error,needs_dedicated?&di:nullptr))return false;
      VkBindTensorMemoryInfoARM bind{VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_ARM};bind.tensor=t.tensor;bind.memory=t.memory;bind.memoryOffset=external?external->offset:0;VK_CHECK(BindTensorMemoryARM(device,1,&bind));
      VkTensorViewCreateInfoARM view{VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_ARM};view.tensor=t.tensor;view.format=TensorFormat(Descriptor(slot).element);VK_CHECK(CreateTensorViewARM(device,&view,nullptr,&t.view));
      if(!external)VK_CHECK(MapMemory(device,t.memory,0,VK_WHOLE_SIZE,0,&t.mapped));return true;
    };
    if(!prepare()){DestroyTensor(t);return false;}
    t.external=std::move(external);
    VkWriteDescriptorSetTensorARM tw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM};tw.tensorViewCount=1;tw.pTensorViews=&t.view;
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};write.pNext=&tw;write.dstSet=descriptor_set;write.dstBinding=Descriptor(slot).binding;write.descriptorCount=1;write.descriptorType=VK_DESCRIPTOR_TYPE_TENSOR_ARM;
    // Ordinary descriptor writes invalidate recordings referencing this set.
    command_recorded = false;
    UpdateDescriptorSets(device,1,&write,0,nullptr);
    DestroyTensor(tensors[slot]);tensors[slot]=std::move(t);return true;
  }

  bool Build() {
    VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader.codeSize = artifact->spirv.size() * sizeof(uint32_t);
    shader.pCode = artifact->spirv.data();
    VK_CHECK(CreateShaderModule(device, &shader, nullptr, &module));
    std::vector<TensorDescriptor> io = artifact->inputs;
    io.insert(io.end(), artifact->outputs.begin(), artifact->outputs.end());
    std::vector<std::vector<int64_t>> shapes;
    shapes.reserve(io.size() + artifact->constants.size());
    std::vector<VkTensorDescriptionARM> descriptions;
    descriptions.reserve(shapes.capacity());
    auto describe = [&](const TensorDescriptor& t) {
      shapes.emplace_back(t.shape.begin(), t.shape.end());
      VkTensorDescriptionARM d{VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM};
      d.tiling = VK_TENSOR_TILING_LINEAR_ARM; d.format = TensorFormat(t.element);
      d.dimensionCount = shapes.back().size(); d.pDimensions = shapes.back().data();
      d.usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM;
      descriptions.push_back(d);
    };
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    std::vector<VkDataGraphPipelineResourceInfoARM> resources;
    for (const auto& t : io) {
      describe(t);
      tensors.emplace_back(); auto& memory = tensors.back();
      memory.bytes = Bytes(t);
      if (!memory.bytes) { error = "Invalid tensor size"; return false; }
      bindings.push_back({t.binding, VK_DESCRIPTOR_TYPE_TENSOR_ARM, 1, VK_SHADER_STAGE_ALL, nullptr});
      VkDataGraphPipelineResourceInfoARM resource{VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM};
      resource.pNext = &descriptions.back(); resource.binding = t.binding;
      resources.push_back(resource);
    }
    VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = bindings.size(); layout_info.pBindings = bindings.data();
    VK_CHECK(CreateDescriptorSetLayout(device, &layout_info, nullptr, &descriptor_layout));
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_TENSOR_ARM, static_cast<uint32_t>(io.size())};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1; pool_info.poolSizeCount = 1; pool_info.pPoolSizes = &pool_size;
    VK_CHECK(CreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool));
    VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = descriptor_pool; set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &descriptor_layout;
    VK_CHECK(AllocateDescriptorSets(device, &set_info, &descriptor_set));
    VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_info.setLayoutCount = 1; pipeline_layout_info.pSetLayouts = &descriptor_layout;
    VK_CHECK(CreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout));
    std::vector<VkDataGraphPipelineConstantARM> constants;
    for (const auto& c : artifact->constants) {
      if (c.bytes.size() != Bytes(c.descriptor)) { error = "Invalid constant size"; return false; }
      describe(c.descriptor);
      VkDataGraphPipelineConstantARM constant{VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CONSTANT_ARM};
      constant.pNext = &descriptions.back(); constant.id = c.descriptor.binding;
      constant.pConstantData = c.bytes.data(); constants.push_back(constant);
    }
    VkDataGraphPipelineShaderModuleCreateInfoARM sm{VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO_ARM};
    sm.module = module; sm.pName = artifact->entry_point.c_str();
    sm.constantCount = constants.size(); sm.pConstants = constants.data();
    VkDataGraphPipelineCreateInfoARM pipeline_info{VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CREATE_INFO_ARM};
    pipeline_info.pNext = &sm; pipeline_info.layout = pipeline_layout;
    pipeline_info.resourceInfoCount = resources.size(); pipeline_info.pResourceInfos = resources.data();
    VK_CHECK(CreateDataGraphPipelinesARM(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1,
                                       &pipeline_info, nullptr, &pipeline));
    VkDataGraphPipelineSessionCreateInfoARM session_info{VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM};
    session_info.dataGraphPipeline = pipeline;
    VK_CHECK(CreateDataGraphPipelineSessionARM(device, &session_info, nullptr, &session));
    VkDataGraphPipelineSessionBindPointRequirementsInfoARM query{VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENTS_INFO_ARM};
    query.session = session;
    uint32_t count = 0;
    VK_CHECK(GetDataGraphPipelineSessionBindPointRequirementsARM(device, &query, &count, nullptr));
    std::vector<VkDataGraphPipelineSessionBindPointRequirementARM> requirements(count,
        {VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENT_ARM});
    VK_CHECK(GetDataGraphPipelineSessionBindPointRequirementsARM(device, &query, &count, requirements.data()));
    for (const auto& requirement : requirements) {
      if (requirement.bindPointType != VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TYPE_MEMORY_ARM) {
        error = "Unsupported Vulkan session bind point type"; return false;
      }
      for (uint32_t i = 0; i < requirement.numObjects; ++i) {
        VkDataGraphPipelineSessionMemoryRequirementsInfoARM info{VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_MEMORY_REQUIREMENTS_INFO_ARM};
        info.session = session; info.bindPoint = requirement.bindPoint; info.objectIndex = i;
        VkMemoryRequirements2 memory{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
        GetDataGraphPipelineSessionMemoryRequirementsARM(device, &info, &memory);
        session_memory.push_back(VK_NULL_HANDLE);
        if (!Allocate(memory.memoryRequirements, 0, session_memory.back())) return false;
        VkBindDataGraphPipelineSessionMemoryInfoARM bind{VK_STRUCTURE_TYPE_BIND_DATA_GRAPH_PIPELINE_SESSION_MEMORY_INFO_ARM};
        bind.session = session; bind.bindPoint = requirement.bindPoint; bind.objectIndex = i;
        bind.memory = session_memory.back();
        VK_CHECK(BindDataGraphPipelineSessionMemoryARM(device, 1, &bind));
      }
    }
    VkCommandPoolCreateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    command_info.queueFamilyIndex = queue_family;
    command_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(CreateCommandPool(device, &command_info, nullptr, &command_pool));
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = command_pool; allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; allocate.commandBufferCount = 1;
    VK_CHECK(AllocateCommandBuffers(device, &allocate, &command));
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(CreateFence(device, &fence_info, nullptr, &fence));
    return true;
  }

  bool RecordCommands() {
    if (command_recorded) return true;
    if (command_needs_reset) VK_CHECK(ResetCommandBuffer(command, 0));
    // Reset even a partially failed recording before retrying it.
    command_needs_reset = true;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = 0; // Reusable; the invocation never has concurrent submissions.
    VK_CHECK(BeginCommandBuffer(command, &begin));
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM;
    barrier.dstAccessMask = VK_ACCESS_2_DATA_GRAPH_READ_BIT_ARM;
    auto tensor_barrier = [&](size_t first, size_t count) {
      std::vector<VkTensorMemoryBarrierARM> tensor_barriers(
          count, {VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_ARM});
      for (size_t i = 0; i < count; ++i) {
        auto& tensor = tensor_barriers[i];
        tensor.srcStageMask = barrier.srcStageMask;
        tensor.srcAccessMask = barrier.srcAccessMask;
        tensor.dstStageMask = barrier.dstStageMask;
        tensor.dstAccessMask = barrier.dstAccessMask;
        tensor.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        tensor.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        const auto& t=tensors[first+i];
        if(t.external){
          bool acquire=barrier.dstStageMask==VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM;
          tensor.srcQueueFamilyIndex=acquire?t.external->ForeignQueue():queue_family;
          tensor.dstQueueFamilyIndex=acquire?queue_family:t.external->ForeignQueue();
          if(acquire){tensor.srcStageMask=VK_PIPELINE_STAGE_2_NONE;tensor.srcAccessMask=0;tensor.dstAccessMask=VK_ACCESS_2_DATA_GRAPH_READ_BIT_ARM|VK_ACCESS_2_DATA_GRAPH_WRITE_BIT_ARM;}
          else {tensor.dstStageMask=VK_PIPELINE_STAGE_2_NONE;tensor.dstAccessMask=0;tensor.srcAccessMask=VK_ACCESS_2_DATA_GRAPH_READ_BIT_ARM|VK_ACCESS_2_DATA_GRAPH_WRITE_BIT_ARM;}
        }
        tensor.tensor = t.tensor;
      }
      // Tensor barriers are an array in the tensor dependency wrapper, not
      // individual entries in VkDependencyInfo's pNext chain.
      VkTensorDependencyInfoARM tensor_dependency{
          VK_STRUCTURE_TYPE_TENSOR_DEPENDENCY_INFO_ARM};
      tensor_dependency.tensorMemoryBarrierCount = static_cast<uint32_t>(count);
      tensor_dependency.pTensorMemoryBarriers = tensor_barriers.data();
      VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      dependency.pNext = count ? &tensor_dependency : nullptr;
      // Preserve the global dependency for other graph/session memory too.
      dependency.memoryBarrierCount = 1;
      dependency.pMemoryBarriers = &barrier;
      CmdPipelineBarrier2(command, &dependency);
    };
    tensor_barrier(0, tensors.size());
    CmdBindPipeline(command, VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM, pipeline);
    CmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM, pipeline_layout,
                          0, 1, &descriptor_set, 0, nullptr);
    VkDataGraphPipelineDispatchInfoARM dispatch{VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_DISPATCH_INFO_ARM};
    CmdDispatchDataGraphARM(command, session, &dispatch);
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM;
    barrier.srcAccessMask = VK_ACCESS_2_DATA_GRAPH_WRITE_BIT_ARM;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    tensor_barrier(0, tensors.size());
    VK_CHECK(EndCommandBuffer(command));
    command_recorded = true;
    return true;
  }

  bool Run(const std::vector<const void*>& inputs, const std::vector<void*>& outputs) {
    if (inputs.size() != artifact->inputs.size() || outputs.size() != artifact->outputs.size()) {
      error = "Vulkan invocation input/output count mismatch"; return false;
    }
    statistics={};
    for(size_t slot=0;slot<tensors.size();++slot){if(!tensors[slot].tensor && !PrepareTensor(slot,nullptr))return false;if(tensors[slot].external)++statistics.imported_slots;}
    for (size_t i = 0; i < inputs.size(); ++i) {
      if(tensors[i].external)continue;
      if (!inputs[i]) { error = "Null input"; return false; }
      std::memcpy(tensors[i].mapped, inputs[i], tensors[i].bytes);
      statistics.input_bytes+=tensors[i].bytes;
    }
    for(size_t i=0;i<outputs.size();++i)if(!outputs[i] && !tensors[inputs.size()+i].external){error="Null output";return false;}
    if (!RecordCommands()) return false;
    VK_CHECK(ResetFences(device, 1, &fence));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
    if(!Check(QueueSubmit(queue,1,&submit,fence),"QueueSubmit")){failed_submission=true;return false;}
    // Completion is synchronous. A failed wait keeps resources owned until
    // device-idle teardown; callers cannot reuse in-flight allocations.
    if(!Check(WaitForFences(device,1,&fence,VK_TRUE,UINT64_MAX),"WaitForFences")){failed_submission=true;return false;}
    for (size_t i = 0; i < outputs.size(); ++i) {
      const auto& t = tensors[inputs.size() + i];
      if(t.external)continue;
      std::memcpy(outputs[i], t.mapped, t.bytes);
      statistics.output_bytes+=t.bytes;
    }
    return true;
  }
#undef VK_CHECK
};

VulkanInvocation::VulkanInvocation(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
VulkanInvocation::~VulkanInvocation() = default;
std::unique_ptr<VulkanInvocation> VulkanInvocation::Create(
    std::shared_ptr<const CompiledPartition> artifact, bool allow_emulation, std::string& error) {
  return CreateWithResolver(std::move(artifact), allow_emulation, nullptr, error);
}
std::unique_ptr<VulkanInvocation> VulkanInvocation::CreateWithResolver(
    std::shared_ptr<const CompiledPartition> artifact, bool allow_emulation,
    PFN_vkGetInstanceProcAddr resolver, std::string& error, std::shared_ptr<HardwareBufferApi> hardware_api) {
  if (!artifact || artifact->spirv.empty() || artifact->outputs.empty()) {
    error = "Empty compiled artifact"; return nullptr;
  }
  auto impl = std::make_unique<Impl>();impl->hardware_api=hardware_api?std::move(hardware_api):AndroidHardwareBufferApi(); impl->artifact = std::move(artifact);
  if (!impl->Initialize(allow_emulation, resolver)) { error = impl->error; return nullptr; }
  return std::unique_ptr<VulkanInvocation>(new VulkanInvocation(std::move(impl)));
}
bool VulkanInvocation::Requirements(size_t slot,BufferRequirements& r,std::string& error){absl::MutexLock lock(&impl_->mutex);if(!impl_->Requirements(slot,r)){error=impl_->error;return false;}return true;}
bool VulkanInvocation::AttachExternal(size_t slot,std::shared_ptr<ExternalBuffer> buffer,std::string& error){absl::MutexLock lock(&impl_->mutex);if(impl_->failed_submission || !impl_->PrepareTensor(slot,std::move(buffer))){error=impl_->error;return false;}return true;}
void VulkanInvocation::DetachExternal(size_t slot){absl::MutexLock lock(&impl_->mutex);if(slot<impl_->tensors.size()){if(impl_->failed_submission)impl_->DeviceWaitIdle(impl_->device);if(impl_->tensors[slot].tensor)impl_->command_recorded=false;impl_->DestroyTensor(impl_->tensors[slot]);}}
VulkanInvocation::CopyStatistics VulkanInvocation::Statistics() const {absl::MutexLock lock(&impl_->mutex);return impl_->statistics;}
bool VulkanInvocation::Run(const std::vector<const void*>& inputs,
                           const std::vector<void*>& outputs, std::string& error) {
  absl::MutexLock lock(&impl_->mutex);
  if(impl_->failed_submission){error="Invocation has an incomplete submission";return false;}
  if (!impl_->Run(inputs, outputs)) { error = impl_->error; return false; }
  return true;
}
}  // namespace litert::arm_vulkan_ml
