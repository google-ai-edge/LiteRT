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
#include <dlfcn.h>
#include <algorithm>
#include <cstring>
#include <utility>

#include "absl/synchronization/mutex.h"
#include "vulkan/vulkan.h"

namespace litert::arm_vulkan_ml {
namespace {
VkFormat Format(TensorElement element) {
  return element == TensorElement::kInt8 ? VK_FORMAT_R8_SINT :
      element == TensorElement::kFloat32 ? VK_FORMAT_R32_SFLOAT : VK_FORMAT_R32_SINT;
}
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
  std::shared_ptr<const CompiledPartition> artifact;
  absl::Mutex mutex;
  void* library = nullptr;
  PFN_vkGetInstanceProcAddr get = nullptr;
  PFN_vkDestroyInstance destroy_instance = nullptr;
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
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
  };
  std::vector<TensorMemory> tensors;
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
    if (instance && destroy_instance) destroy_instance(instance, nullptr);
    if (library) dlclose(library);
  }

  bool Initialize(bool allow_emulation) {
    library = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!library) { error = "Cannot load libvulkan.so.1"; return false; }
    get = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"));
    if (!get) { error = "Missing vkGetInstanceProcAddr"; return false; }
    auto create = reinterpret_cast<PFN_vkCreateInstance>(get(nullptr, "vkCreateInstance"));
    if (!create) { error = "Missing vkCreateInstance"; return false; }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "LiteRT Arm Vulkan ML";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    VK_CHECK(create(&ci, nullptr, &instance));
    destroy_instance = reinterpret_cast<PFN_vkDestroyInstance>(get(instance, "vkDestroyInstance"));
#define INSTANCE_FN(name) auto name = reinterpret_cast<PFN_vk##name>(get(instance, "vk" #name)); if (!name) { error = "Missing vk" #name; return false; }
    INSTANCE_FN(EnumeratePhysicalDevices);
    INSTANCE_FN(GetPhysicalDeviceProperties);
    INSTANCE_FN(EnumerateDeviceExtensionProperties);
    INSTANCE_FN(GetPhysicalDeviceQueueFamilyProperties);
    INSTANCE_FN(GetPhysicalDeviceFeatures2);
    INSTANCE_FN(GetPhysicalDeviceMemoryProperties);
    INSTANCE_FN(GetPhysicalDeviceQueueFamilyDataGraphPropertiesARM);
    INSTANCE_FN(GetPhysicalDeviceQueueFamilyDataGraphEngineOperationPropertiesARM);
    INSTANCE_FN(CreateDevice);
    INSTANCE_FN(GetDeviceProcAddr);
    uint32_t count = 0;
    VK_CHECK(EnumeratePhysicalDevices(instance, &count, nullptr));
    std::vector<VkPhysicalDevice> candidates(count);
    VK_CHECK(EnumeratePhysicalDevices(instance, &count, candidates.data()));
    const char* extensions[] = {VK_ARM_TENSORS_EXTENSION_NAME,
      VK_ARM_DATA_GRAPH_EXTENSION_NAME, VK_ARM_DATA_GRAPH_INSTRUCTION_SET_TOSA_EXTENSION_NAME,
      VK_KHR_MAINTENANCE_5_EXTENSION_NAME, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
      VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME};
    for (auto candidate : candidates) {
      VkPhysicalDeviceProperties props{};
      GetPhysicalDeviceProperties(candidate, &props);
      uint32_t extension_count = 0;
      VK_CHECK(EnumerateDeviceExtensionProperties(candidate, nullptr, &extension_count, nullptr));
      std::vector<VkExtensionProperties> advertised(extension_count);
      VK_CHECK(EnumerateDeviceExtensionProperties(candidate, nullptr, &extension_count, advertised.data()));
      bool all = true;
      for (auto required : extensions) {
        all &= std::any_of(advertised.begin(), advertised.end(), [&](const auto& e) {
          return !std::strcmp(required, e.extensionName);
        });
      }
      if (!all) continue;
      uint32_t queue_count = 0;
      GetPhysicalDeviceQueueFamilyProperties(candidate, &queue_count, nullptr);
      std::vector<VkQueueFamilyProperties> queues(queue_count);
      GetPhysicalDeviceQueueFamilyProperties(candidate, &queue_count, queues.data());
      for (uint32_t q = 0; q < queue_count && !physical; ++q) {
        if (!queues[q].queueCount || !(queues[q].queueFlags & VK_QUEUE_DATA_GRAPH_BIT_ARM)) continue;
        uint32_t n = 0;
        VK_CHECK(GetPhysicalDeviceQueueFamilyDataGraphPropertiesARM(candidate, q, &n, nullptr));
        std::vector<VkQueueFamilyDataGraphPropertiesARM> engines(n,
            {VK_STRUCTURE_TYPE_QUEUE_FAMILY_DATA_GRAPH_PROPERTIES_ARM});
        VK_CHECK(GetPhysicalDeviceQueueFamilyDataGraphPropertiesARM(candidate, q, &n, engines.data()));
        for (const auto& engine : engines) {
          if (engine.engine.isForeign || engine.engine.type != VK_PHYSICAL_DEVICE_DATA_GRAPH_PROCESSING_ENGINE_TYPE_DEFAULT_ARM ||
              engine.operation.operationType != VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_SPIRV_EXTENDED_INSTRUCTION_SET_ARM ||
              artifact->instruction_set != engine.operation.name) continue;
          VkQueueFamilyDataGraphTOSAPropertiesARM tosa{VK_STRUCTURE_TYPE_QUEUE_FAMILY_DATA_GRAPH_TOSA_PROPERTIES_ARM};
          VK_CHECK(GetPhysicalDeviceQueueFamilyDataGraphEngineOperationPropertiesARM(candidate, q, &engine,
                    reinterpret_cast<VkBaseOutStructure*>(&tosa)));
          auto has = [](uint32_t n, const VkDataGraphTOSANameQualityARM* names, const std::string& name) {
            for (uint32_t i = 0; i < n; ++i) if (name == names[i].name &&
                (names[i].qualityFlags & VK_DATA_GRAPH_TOSA_QUALITY_CONFORMANT_ARM)) return true;
            return false;
          };
          bool supported = true;
          // Explicit development opt-in for Arm's 0.11 emulator. Its profile
          // name is not PRO-INT; never treat it as hardware conformance.
          bool emulator = allow_emulation && props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU &&
              has(tosa.profileCount, tosa.pProfiles, "Emulation Layer");
          for (const auto& profile : artifact->profiles)
            supported &= has(tosa.profileCount, tosa.pProfiles, profile) ||
                         (emulator && (profile == "PRO-INT" || profile == "PRO-FP"));
          for (const auto& extension : artifact->extensions)
            supported &= has(tosa.extensionCount, tosa.pExtensions, extension);
          if (supported) { physical = candidate; queue_family = q; break; }
        }
      }
      if (physical) break;
    }
    if (!physical) { error = "No Vulkan device advertises the artifact's TOSA requirements"; return false; }
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
    device_info.enabledExtensionCount = sizeof(extensions)/sizeof(extensions[0]); device_info.ppEnabledExtensionNames = extensions;
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
    GetDeviceQueue(device, queue_family, 0, &queue);
    GetPhysicalDeviceMemoryProperties(physical, &memory_properties);
    return Build();
  }

  bool Allocate(const VkMemoryRequirements& requirements, VkMemoryPropertyFlags flags,
                VkDeviceMemory& memory) {
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
      if ((requirements.memoryTypeBits & (1u << i)) &&
          (memory_properties.memoryTypes[i].propertyFlags & flags) == flags) {
        VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        info.allocationSize = std::max(VkDeviceSize{1}, requirements.size);
        info.memoryTypeIndex = i;
        VK_CHECK(AllocateMemory(device, &info, nullptr, &memory));
        return true;
      }
    }
    error = "No compatible Vulkan memory type"; return false;
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
      d.tiling = VK_TENSOR_TILING_LINEAR_ARM; d.format = Format(t.element);
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
      VkTensorCreateInfoARM info{VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_ARM};
      info.pDescription = &descriptions.back(); info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      VK_CHECK(CreateTensorARM(device, &info, nullptr, &memory.tensor));
      VkTensorMemoryRequirementsInfoARM query{VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_ARM};
      query.tensor = memory.tensor;
      VkMemoryRequirements2 requirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
      GetTensorMemoryRequirementsARM(device, &query, &requirements);
      if (requirements.memoryRequirements.size < memory.bytes) { error = "Invalid tensor allocation size"; return false; }
      if (!Allocate(requirements.memoryRequirements,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory.memory)) return false;
      VkBindTensorMemoryInfoARM bind{VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_ARM};
      bind.tensor = memory.tensor; bind.memory = memory.memory;
      VK_CHECK(BindTensorMemoryARM(device, 1, &bind));
      VkTensorViewCreateInfoARM view{VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_ARM};
      view.tensor = memory.tensor; view.format = Format(t.element);
      VK_CHECK(CreateTensorViewARM(device, &view, nullptr, &memory.view));
      VK_CHECK(MapMemory(device, memory.memory, 0, VK_WHOLE_SIZE, 0, &memory.mapped));
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
    for (size_t i = 0; i < io.size(); ++i) {
      VkWriteDescriptorSetTensorARM tensor_write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM};
      tensor_write.tensorViewCount = 1; tensor_write.pTensorViews = &tensors[i].view;
      VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      write.pNext = &tensor_write; write.dstSet = descriptor_set; write.dstBinding = io[i].binding;
      write.descriptorCount = 1; write.descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM;
      UpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
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

  bool Run(const std::vector<const void*>& inputs, const std::vector<void*>& outputs) {
    if (inputs.size() != artifact->inputs.size() || outputs.size() != artifact->outputs.size()) {
      error = "Vulkan invocation input/output count mismatch"; return false;
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (!inputs[i]) { error = "Null input"; return false; }
      std::memcpy(tensors[i].mapped, inputs[i], tensors[i].bytes);
    }
    for (auto output : outputs) if (!output) { error = "Null output"; return false; }
    VK_CHECK(ResetCommandBuffer(command, 0));
    VK_CHECK(ResetFences(device, 1, &fence));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(BeginCommandBuffer(command, &begin));
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM;
    barrier.dstAccessMask = VK_ACCESS_2_DATA_GRAPH_READ_BIT_ARM;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1; dependency.pMemoryBarriers = &barrier;
    CmdPipelineBarrier2(command, &dependency);
    CmdBindPipeline(command, VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM, pipeline);
    CmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM, pipeline_layout,
                          0, 1, &descriptor_set, 0, nullptr);
    VkDataGraphPipelineDispatchInfoARM dispatch{VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_DISPATCH_INFO_ARM};
    CmdDispatchDataGraphARM(command, session, &dispatch);
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM;
    barrier.srcAccessMask = VK_ACCESS_2_DATA_GRAPH_WRITE_BIT_ARM;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    CmdPipelineBarrier2(command, &dependency);
    VK_CHECK(EndCommandBuffer(command));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
    VK_CHECK(QueueSubmit(queue, 1, &submit, fence));
    // Completion is synchronous. A failed wait keeps resources owned until
    // device-idle teardown; callers cannot reuse in-flight allocations.
    VK_CHECK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
    for (size_t i = 0; i < outputs.size(); ++i) {
      const auto& t = tensors[inputs.size() + i];
      std::memcpy(outputs[i], t.mapped, t.bytes);
    }
    return true;
  }
#undef VK_CHECK
};

VulkanInvocation::VulkanInvocation(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
VulkanInvocation::~VulkanInvocation() = default;
std::unique_ptr<VulkanInvocation> VulkanInvocation::Create(
    std::shared_ptr<const CompiledPartition> artifact, bool allow_emulation, std::string& error) {
  if (!artifact || artifact->spirv.empty() || artifact->outputs.empty()) {
    error = "Empty compiled artifact"; return nullptr;
  }
  auto impl = std::make_unique<Impl>(); impl->artifact = std::move(artifact);
  if (!impl->Initialize(allow_emulation)) { error = impl->error; return nullptr; }
  return std::unique_ptr<VulkanInvocation>(new VulkanInvocation(std::move(impl)));
}
bool VulkanInvocation::Run(const std::vector<const void*>& inputs,
                           const std::vector<void*>& outputs, std::string& error) {
  absl::MutexLock lock(&impl_->mutex);
  if (!impl_->Run(inputs, outputs)) { error = impl_->error; return false; }
  return true;
}
}  // namespace litert::arm_vulkan_ml
