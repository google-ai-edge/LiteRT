#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
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

#include "litert/vendors/arm_vulkan_ml/dispatch/vulkan_invocation.h"
#include "litert/vendors/arm_vulkan_ml/dispatch/device_memory_allocator.h"
#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <type_traits>
#include <vector>
#include "gtest/gtest.h"
#include "spirv/unified1/spirv.hpp11"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_arm_vulkan_ml_options.h"
#include "litert/core/model/model.h"
#include "litert/vendors/cc/litert_compiler_plugin.h"
#include "tflite/schema/schema_generated.h"
#include "litert/test/load_test_model.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/vendors/arm_vulkan_ml/dispatch/dispatch_api.cc"
namespace litert::arm_vulkan_ml {
namespace {
template<class T> T Handle(uintptr_t n) {
  if constexpr(std::is_pointer_v<T>)return reinterpret_cast<T>(n);
  else return static_cast<T>(n);
}
template<class T> uintptr_t Key(T h) {
  if constexpr(std::is_pointer_v<T>)return reinterpret_cast<uintptr_t>(h);
  else return static_cast<uintptr_t>(h);
}
struct Description {
  uint32_t id=0,set=0,array=0;
  VkFormat format=VK_FORMAT_UNDEFINED;
  std::vector<int64_t> dimensions;
  std::vector<uint8_t> bytes;
};
// Independent binary decoder: no production TensorFormat/descriptor helpers.
struct ModuleInterface {
  std::map<uint32_t,Description> constants;
  std::map<std::pair<uint32_t,uint32_t>,Description> resources;
};
ModuleInterface Decode(const std::vector<uint32_t>& binary) {
  std::map<uint32_t,std::vector<uint32_t>> types,shapes;
  std::map<uint32_t,uint32_t> integers,pointers,variables,sets,bindings;
  std::vector<std::pair<uint32_t,uint32_t>> constants;
  std::set<uint32_t> interface;
  for(size_t i=5;i<binary.size();) {
    auto words=binary[i]>>16;auto op=static_cast<spv::Op>(binary[i]&0xffff);
    if(!words || i+words>binary.size()){ADD_FAILURE()<<"Invalid SPIR-V";return {};}
    const auto a = binary.cbegin() + i + 1;
    switch(op) {
      case spv::Op::OpTypeInt:case spv::Op::OpTypeFloat:case spv::Op::OpTypeTensorARM:
        types[a[0]]=std::vector<uint32_t>(a+1,a+(words-1));break;
      case spv::Op::OpConstant:integers[a[1]]=a[2];break;
      case spv::Op::OpConstantComposite:shapes[a[1]]=std::vector<uint32_t>(a+2,a+(words-1));break;
      case spv::Op::OpTypePointer:pointers[a[0]]=a[2];break;
      case spv::Op::OpVariable:variables[a[1]]=pointers.at(a[0]);break;
      case spv::Op::OpDecorate:
        if(a[1]==static_cast<uint32_t>(spv::Decoration::Binding))bindings[a[0]]=a[2];
        if(a[1]==static_cast<uint32_t>(spv::Decoration::DescriptorSet))sets[a[0]]=a[2];break;
      case spv::Op::OpGraphConstantARM:constants.emplace_back(a[2],a[0]);break;
      case spv::Op::OpGraphEntryPointARM: {
        size_t start=1;bool end=false;
        while(start<words-1 && !end){uint32_t word=a[start++];for(int b=0;b<4;++b)if(((word>>(8*b))&255)==0)end=true;}
        for(;start<words-1;++start)interface.insert(a[start]);break;
      }
      default:break;
    }
    i+=words;
  }
  auto description=[&](uint32_t tensor) {
    Description d;const auto& t=types.at(tensor);const auto& element=types.at(t[0]);
    // TOSA integers are signless SPIR-V types; model semantics select SINT.
    d.format=element.size()==1?(element[0]==32?VK_FORMAT_R32_SFLOAT:VK_FORMAT_UNDEFINED):element[0]==8?VK_FORMAT_R8_SINT:element[0]==32?VK_FORMAT_R32_SINT:VK_FORMAT_UNDEFINED;
    EXPECT_EQ(integers.at(t[1]),shapes.at(t[2]).size());
    for(auto id:shapes.at(t[2]))d.dimensions.push_back(integers.at(id));return d;
  };
  ModuleInterface result;
  for(const auto& c:constants)result.constants.emplace(c.first,description(c.second));
  for(auto id:interface)result.resources.emplace(std::make_pair(sets.at(id),bindings.at(id)),description(variables.at(id)));
  return result;
}
Description Copy(const VkTensorDescriptionARM* d) {
  Description result;result.format=d->format;result.dimensions.assign(d->pDimensions,d->pDimensions+d->dimensionCount);return result;
}
bool Compatible(const Description& expected,const Description& actual) {
  return expected.format==actual.format && expected.dimensions==actual.dimensions;
}
struct Fake {
  enum class CommandState { kInitial, kRecording, kExecutable, kInvalid, kPending };
  std::map<VkCommandBuffer,CommandState> command_states;
  std::map<VkCommandBuffer,VkDescriptorSet> command_sets;
  std::map<VkCommandBuffer,VkDataGraphPipelineSessionARM> command_sessions;
  std::map<VkCommandBuffer,std::set<VkTensorARM>> command_tensors;
  int begins=0,ends=0,resets=0,submits=0,waits=0,fence_resets=0,recorded_dispatches=0;
  bool fail_begin=false,fail_end=false,fail_reset=false,fail_submit=false,fail_wait=false;
  bool external_dedicated=false;
  bool fail_tensor_bind=false;
  std::map<VkTensorARM,VkExternalMemoryHandleTypeFlags> tensor_handles;
  std::map<VkDeviceMemory,VkExternalMemoryHandleTypeFlags> memory_handles;
  std::map<VkDeviceMemory,VkTensorARM> dedicated_tensors;
  bool imports=false,importable=true,dedicated=false,host_memory=true;uint32_t import_bits=1;int maps=0,fd_imports=0,ahb_imports=0,ownership=0;
  uintptr_t next=10;uint32_t limit=1000;int allocations=0,dispatches=0;
  int fail_allocate=-1,fail_bind=-1,binds=0;bool fail_query=false;
  std::set<uintptr_t> live;
  std::map<VkDeviceMemory,std::vector<uint8_t>> memory;
  std::map<VkTensorARM,size_t> tensor_bytes;
  std::map<VkShaderModule,std::vector<uint32_t>> modules;
  std::map<VkPipeline,ModuleInterface> pipelines;
  std::map<VkDataGraphPipelineSessionARM,VkPipeline> sessions;
  std::map<VkCommandBuffer,VkPipeline> bound;
  std::map<VkDataGraphPipelineSessionARM,std::set<std::pair<uint32_t,uint32_t>>> bound_memory;
  std::vector<std::pair<uint32_t,uint32_t>> requirements={{VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM,2},{VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_OPTICAL_FLOW_CACHE_ARM,1}};
  std::vector<Description> captured_constants,captured_resources;
  std::vector<std::string> violations;
  std::vector<uint32_t> descriptor_bindings,written_bindings;
  std::string corrupt;
  template<class T> T New(){auto n=++next;live.insert(n);return Handle<T>(n);}
  template<class T> void Drop(T h){EXPECT_EQ(live.erase(Key(h)),1)<<"double/unowned destruction";}
  bool PipelineValid(const ModuleInterface& expected) {
    std::set<uint32_t> ids;std::set<std::pair<uint32_t,uint32_t>> locations;
    bool ok=true;
    for(const auto& c:captured_constants){auto e=expected.constants.find(c.id);ok &= ids.insert(c.id).second && e!=expected.constants.end() && (e==expected.constants.end() || Compatible(e->second,c));}
    ok &= ids.size()==expected.constants.size();if(!ok)violations.push_back("09921");
    bool resources_ok=true;
    for(const auto& r:captured_resources){auto key=std::make_pair(r.set,r.id);auto e=expected.resources.find(key);resources_ok &= locations.insert(key).second && r.array==0 && e!=expected.resources.end() && (e==expected.resources.end() || Compatible(e->second,r));}
    resources_ok &= locations.size()==expected.resources.size();if(!resources_ok)violations.push_back("09923");return ok && resources_ok;
  }
  bool DispatchValid(VkCommandBuffer command,VkDataGraphPipelineSessionARM session) {
    bool ok=true;
    for(auto [point,n]:requirements)for(uint32_t i=0;i<n;++i)if(!bound_memory[session].count({point,i})){violations.push_back("09796");ok=false;}
    if(bound[command]!=sessions[session]){violations.push_back("09951");ok=false;}return ok;
  }
};
thread_local Fake* state=nullptr;
PFN_vkVoidFunction VKAPI_CALL Resolve(VkInstance,const char*);
VKAPI_ATTR VkResult VKAPI_CALL AllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo* pAllocateInfo, VkCommandBuffer* pCommandBuffers) {
*pCommandBuffers=Handle<VkCommandBuffer>(++state->next);return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL ExternalTensorProperties(VkPhysicalDevice,const VkPhysicalDeviceExternalTensorInfoARM* info,VkExternalTensorPropertiesARM* props){props->externalMemoryProperties.compatibleHandleTypes=info->handleType;props->externalMemoryProperties.externalMemoryFeatures=(state->importable?VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT:0)|(state->external_dedicated?VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT:0);}
VKAPI_ATTR VkResult VKAPI_CALL MemoryFdProperties(VkDevice,VkExternalMemoryHandleTypeFlagBits,int,VkMemoryFdPropertiesKHR* p){p->memoryTypeBits=state->import_bits;return VK_SUCCESS;}
VKAPI_ATTR VkResult VKAPI_CALL AllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo, VkDescriptorSet* pDescriptorSets) {
*pDescriptorSets=Handle<VkDescriptorSet>(++state->next);return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL AllocateMemory(VkDevice device, const VkMemoryAllocateInfo* pAllocateInfo, const VkAllocationCallbacks* pAllocator, VkDeviceMemory* pMemory) {
++state->allocations;if(state->allocations==state->fail_allocate)return VK_ERROR_OUT_OF_DEVICE_MEMORY;
if(pAllocateInfo->pNext){const auto* head=static_cast<const VkBaseInStructure*>(pAllocateInfo->pNext);if(head->sType==VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR){auto* fd=static_cast<const VkImportMemoryFdInfoKHR*>(pAllocateInfo->pNext);EXPECT_GE(fcntl(fd->fd,F_GETFD),0);close(fd->fd);++state->fd_imports;if(state->dedicated){EXPECT_NE(fd->pNext,nullptr);}}else if(head->sType==VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID){++state->ahb_imports;}}
*pMemory=state->New<VkDeviceMemory>();
for(auto* h=static_cast<const VkBaseInStructure*>(pAllocateInfo->pNext);h;h=h->pNext){
  if(h->sType==VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR)state->memory_handles[*pMemory]=reinterpret_cast<const VkImportMemoryFdInfoKHR*>(h)->handleType;
  if(h->sType==VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID){state->memory_handles[*pMemory]=VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;EXPECT_EQ(pAllocateInfo->allocationSize,16);}
  if(h->sType==VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_TENSOR_ARM){auto tensor=reinterpret_cast<const VkMemoryDedicatedAllocateInfoTensorARM*>(h)->tensor;state->dedicated_tensors[*pMemory]=tensor;EXPECT_EQ(pAllocateInfo->allocationSize,(state->tensor_bytes[tensor]+15)/16*16);}
}
state->memory[*pMemory].resize(pAllocateInfo->allocationSize);return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL BeginCommandBuffer(VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo* pBeginInfo) {
++state->begins;EXPECT_EQ(pBeginInfo->flags,0);EXPECT_EQ(state->command_states[commandBuffer],Fake::CommandState::kInitial);
if(state->fail_begin)return VK_ERROR_OUT_OF_HOST_MEMORY;
state->command_states[commandBuffer]=Fake::CommandState::kRecording;return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL BindDataGraphPipelineSessionMemoryARM(VkDevice device, uint32_t bindInfoCount, const VkBindDataGraphPipelineSessionMemoryInfoARM* pBindInfos) {
++state->binds;if(state->binds==state->fail_bind)return VK_ERROR_OUT_OF_DEVICE_MEMORY;for(uint32_t i=0;i<bindInfoCount;++i){const auto& b=pBindInfos[i];EXPECT_TRUE(state->memory.count(b.memory));state->bound_memory[b.session].insert({b.bindPoint,b.objectIndex});}return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL BindTensorMemoryARM(VkDevice device, uint32_t bindInfoCount, const VkBindTensorMemoryInfoARM* pBindInfos) {
if(state->fail_tensor_bind)return VK_ERROR_OUT_OF_DEVICE_MEMORY;
for(uint32_t i=0;i<bindInfoCount;++i){const auto& b=pBindInfos[i];auto handle=state->memory_handles[b.memory];if(handle)EXPECT_TRUE(state->tensor_handles[b.tensor]&handle);if(state->dedicated || (handle && state->external_dedicated))EXPECT_EQ(state->dedicated_tensors[b.memory],b.tensor);}
return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL CmdBindDescriptorSets(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipelineLayout layout, uint32_t firstSet, uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets, uint32_t dynamicOffsetCount, const uint32_t* pDynamicOffsets) {
EXPECT_EQ(state->command_states[commandBuffer],Fake::CommandState::kRecording);ASSERT_EQ(descriptorSetCount,1);state->command_sets[commandBuffer]=pDescriptorSets[0];
}
VKAPI_ATTR void VKAPI_CALL CmdBindPipeline(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipeline pipeline) {
state->bound[commandBuffer]=pipeline;
}
VKAPI_ATTR void VKAPI_CALL CmdDispatchDataGraphARM(VkCommandBuffer commandBuffer, VkDataGraphPipelineSessionARM session, const VkDataGraphPipelineDispatchInfoARM* pInfo) {
++state->recorded_dispatches;EXPECT_EQ(state->command_states[commandBuffer],Fake::CommandState::kRecording);state->command_sessions[commandBuffer]=session;EXPECT_TRUE(state->DispatchValid(commandBuffer,session));
}
VKAPI_ATTR void VKAPI_CALL CmdPipelineBarrier2(VkCommandBuffer commandBuffer, const VkDependencyInfo* pDependencyInfo) {
  auto* d=static_cast<const VkTensorDependencyInfoARM*>(pDependencyInfo->pNext);ASSERT_NE(d,nullptr);EXPECT_EQ(d->sType,VK_STRUCTURE_TYPE_TENSOR_DEPENDENCY_INFO_ARM);
  for(uint32_t i=0;i<d->tensorMemoryBarrierCount;++i){const auto& b=d->pTensorMemoryBarriers[i];state->command_tensors[commandBuffer].insert(b.tensor);if(b.srcQueueFamilyIndex!=VK_QUEUE_FAMILY_IGNORED){++state->ownership;EXPECT_TRUE(b.srcQueueFamilyIndex==0 || b.dstQueueFamilyIndex==0);EXPECT_TRUE(b.srcQueueFamilyIndex==VK_QUEUE_FAMILY_EXTERNAL || b.dstQueueFamilyIndex==VK_QUEUE_FAMILY_EXTERNAL || b.srcQueueFamilyIndex==VK_QUEUE_FAMILY_FOREIGN_EXT || b.dstQueueFamilyIndex==VK_QUEUE_FAMILY_FOREIGN_EXT);}}
}
VKAPI_ATTR VkResult VKAPI_CALL CreateCommandPool(VkDevice device, const VkCommandPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkCommandPool* pCommandPool) {
*pCommandPool=state->New<VkCommandPool>();return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL CreateDataGraphPipelineSessionARM(VkDevice device, const VkDataGraphPipelineSessionCreateInfoARM* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDataGraphPipelineSessionARM* pSession) {
*pSession=state->New<VkDataGraphPipelineSessionARM>();state->sessions[*pSession]=pCreateInfo->dataGraphPipeline;return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL CreateDataGraphPipelinesARM(VkDevice device, VkDeferredOperationKHR deferredOperation, VkPipelineCache pipelineCache, uint32_t createInfoCount, const VkDataGraphPipelineCreateInfoARM* pCreateInfos, const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines) {
auto* sm=static_cast<const VkDataGraphPipelineShaderModuleCreateInfoARM*>(pCreateInfos->pNext);state->captured_constants.clear();state->captured_resources.clear();
for(uint32_t i=0;i<sm->constantCount;++i){const auto& c=sm->pConstants[i];auto d=Copy(static_cast<const VkTensorDescriptionARM*>(c.pNext));d.id=c.id;size_t bytes=d.format==VK_FORMAT_R8_SINT?1:4;for(auto n:d.dimensions)bytes*=n;d.bytes.assign(static_cast<const uint8_t*>(c.pConstantData),static_cast<const uint8_t*>(c.pConstantData)+bytes);state->captured_constants.push_back(d);}
for(uint32_t i=0;i<pCreateInfos->resourceInfoCount;++i){const auto& r=pCreateInfos->pResourceInfos[i];auto d=Copy(static_cast<const VkTensorDescriptionARM*>(r.pNext));d.id=r.binding;d.set=r.descriptorSet;d.array=r.arrayElement;state->captured_resources.push_back(d);}
auto& cs=state->captured_constants;auto& rs=state->captured_resources;
if(state->corrupt=="constant_missing" && !cs.empty())cs.pop_back();
if(state->corrupt=="constant_duplicate" && !cs.empty())cs.push_back(cs[0]);
if(state->corrupt=="constant_shape" && !cs.empty())++cs[0].dimensions[0];
if(state->corrupt=="constant_format" && !cs.empty())cs[0].format=VK_FORMAT_R16_SFLOAT;
if(state->corrupt=="resource_missing" && !rs.empty())rs.pop_back();
if(state->corrupt=="resource_duplicate" && !rs.empty())rs.push_back(rs[0]);
if(state->corrupt=="resource_array" && !rs.empty())rs[0].array=1;
if(state->corrupt=="resource_shape" && !rs.empty())++rs[0].dimensions[0];
if(state->corrupt=="resource_format" && !rs.empty())rs[0].format=VK_FORMAT_R16_SFLOAT;
if(state->corrupt=="resource_binding" && !rs.empty())rs[0].id=99;
auto expected=Decode(state->modules.at(sm->module));if(!state->PipelineValid(expected))return VK_ERROR_VALIDATION_FAILED_EXT;
*pPipelines=state->New<VkPipeline>();state->pipelines[*pPipelines]=expected;return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorPool(VkDevice device, const VkDescriptorPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDescriptorPool* pDescriptorPool) {
*pDescriptorPool=state->New<VkDescriptorPool>();return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorSetLayout(VkDevice device, const VkDescriptorSetLayoutCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDescriptorSetLayout* pSetLayout) {
for(uint32_t i=0;i<pCreateInfo->bindingCount;++i)state->descriptor_bindings.push_back(pCreateInfo->pBindings[i].binding);*pSetLayout=state->New<VkDescriptorSetLayout>();return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDevice* pDevice) {
*pDevice=state->New<VkDevice>();return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL CreateFence(VkDevice device, const VkFenceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkFence* pFence) {
*pFence=state->New<VkFence>();return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL CreateInstance(const VkInstanceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkInstance* pInstance) {
*pInstance=state->New<VkInstance>();return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL CreatePipelineLayout(VkDevice device, const VkPipelineLayoutCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkPipelineLayout* pPipelineLayout) {
*pPipelineLayout=state->New<VkPipelineLayout>();return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL CreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkShaderModule* pShaderModule) {
*pShaderModule=state->New<VkShaderModule>();state->modules[*pShaderModule]={pCreateInfo->pCode,pCreateInfo->pCode+pCreateInfo->codeSize/4};return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL CreateTensorARM(VkDevice device, const VkTensorCreateInfoARM* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkTensorARM* pTensor) {
*pTensor=state->New<VkTensorARM>();size_t bytes=pCreateInfo->pDescription->format==VK_FORMAT_R8_SINT?1:4;for(uint32_t i=0;i<pCreateInfo->pDescription->dimensionCount;++i)bytes*=pCreateInfo->pDescription->pDimensions[i];state->tensor_bytes[*pTensor]=bytes;if(pCreateInfo->pNext){auto* e=static_cast<const VkExternalMemoryTensorCreateInfoARM*>(pCreateInfo->pNext);EXPECT_EQ(e->sType,VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_TENSOR_CREATE_INFO_ARM);state->tensor_handles[*pTensor]=e->handleTypes;}return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL CreateTensorViewARM(VkDevice device, const VkTensorViewCreateInfoARM* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkTensorViewARM* pView) {
*pView=state->New<VkTensorViewARM>();return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroyCommandPool(VkDevice device, VkCommandPool commandPool, const VkAllocationCallbacks* pAllocator) {
state->Drop(commandPool);
}
VKAPI_ATTR void VKAPI_CALL DestroyDataGraphPipelineSessionARM(VkDevice device, VkDataGraphPipelineSessionARM session, const VkAllocationCallbacks* pAllocator) {
state->Drop(session);
}
VKAPI_ATTR void VKAPI_CALL DestroyDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool, const VkAllocationCallbacks* pAllocator) {
state->Drop(descriptorPool);
}
VKAPI_ATTR void VKAPI_CALL DestroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout, const VkAllocationCallbacks* pAllocator) {
state->Drop(descriptorSetLayout);
}
VKAPI_ATTR void VKAPI_CALL DestroyDevice(VkDevice device, const VkAllocationCallbacks* pAllocator) {
state->Drop(device);
}
VKAPI_ATTR void VKAPI_CALL DestroyFence(VkDevice device, VkFence fence, const VkAllocationCallbacks* pAllocator) {
state->Drop(fence);
}
VKAPI_ATTR void VKAPI_CALL DestroyInstance(VkInstance instance, const VkAllocationCallbacks* pAllocator) {
state->Drop(instance);
}
VKAPI_ATTR void VKAPI_CALL DestroyPipeline(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks* pAllocator) {
state->Drop(pipeline);
}
VKAPI_ATTR void VKAPI_CALL DestroyPipelineLayout(VkDevice device, VkPipelineLayout pipelineLayout, const VkAllocationCallbacks* pAllocator) {
state->Drop(pipelineLayout);
}
VKAPI_ATTR void VKAPI_CALL DestroyShaderModule(VkDevice device, VkShaderModule shaderModule, const VkAllocationCallbacks* pAllocator) {
state->Drop(shaderModule);
}
VKAPI_ATTR void VKAPI_CALL DestroyTensorARM(VkDevice device, VkTensorARM tensor, const VkAllocationCallbacks* pAllocator) {
state->Drop(tensor);
}
VKAPI_ATTR void VKAPI_CALL DestroyTensorViewARM(VkDevice device, VkTensorViewARM tensorView, const VkAllocationCallbacks* pAllocator) {
state->Drop(tensorView);
}
VKAPI_ATTR VkResult VKAPI_CALL DeviceWaitIdle(VkDevice device) {
for(auto& entry:state->command_states)if(entry.second==Fake::CommandState::kPending)entry.second=Fake::CommandState::kExecutable;return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL EndCommandBuffer(VkCommandBuffer commandBuffer) {
++state->ends;EXPECT_EQ(state->command_states[commandBuffer],Fake::CommandState::kRecording);
if(state->fail_end)return VK_ERROR_OUT_OF_HOST_MEMORY;
state->command_states[commandBuffer]=Fake::CommandState::kExecutable;return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL EnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice, const char* pLayerName, uint32_t* pPropertyCount, VkExtensionProperties* pProperties) {
const char* names[]={VK_ARM_TENSORS_EXTENSION_NAME,VK_ARM_DATA_GRAPH_EXTENSION_NAME,VK_ARM_DATA_GRAPH_INSTRUCTION_SET_TOSA_EXTENSION_NAME,VK_KHR_MAINTENANCE_5_EXTENSION_NAME,VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME,VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,"VK_ANDROID_external_memory_android_hardware_buffer",VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME};
int n=state->imports?10:6;
if(pProperties)for(int i=0;i<n;++i){std::strcpy(pProperties[i].extensionName,names[i]);pProperties[i].specVersion=2;}*pPropertyCount=n;return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL EnumeratePhysicalDevices(VkInstance instance, uint32_t* pPhysicalDeviceCount, VkPhysicalDevice* pPhysicalDevices) {
if(pPhysicalDevices)pPhysicalDevices[0]=Handle<VkPhysicalDevice>(1);*pPhysicalDeviceCount=1;return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL FreeMemory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* pAllocator) {
if(memory){state->memory.erase(memory);state->Drop(memory);}
}
VKAPI_ATTR VkResult VKAPI_CALL GetDataGraphPipelineSessionBindPointRequirementsARM(VkDevice device, const VkDataGraphPipelineSessionBindPointRequirementsInfoARM* pInfo, uint32_t* pBindPointRequirementCount, VkDataGraphPipelineSessionBindPointRequirementARM* pBindPointRequirements) {
if(state->fail_query)return VK_ERROR_UNKNOWN;if(pBindPointRequirements)for(size_t i=0;i<state->requirements.size();++i){pBindPointRequirements[i].bindPoint=static_cast<VkDataGraphPipelineSessionBindPointARM>(state->requirements[i].first);pBindPointRequirements[i].numObjects=state->requirements[i].second;pBindPointRequirements[i].bindPointType=VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TYPE_MEMORY_ARM;}*pBindPointRequirementCount=state->requirements.size();return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL GetDataGraphPipelineSessionMemoryRequirementsARM(VkDevice device, const VkDataGraphPipelineSessionMemoryRequirementsInfoARM* pInfo, VkMemoryRequirements2* pMemoryRequirements) {
pMemoryRequirements->memoryRequirements={64,16,1};
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice device, const char* pName) {
return Resolve(VK_NULL_HANDLE,pName);
}
VKAPI_ATTR void VKAPI_CALL GetDeviceQueue(VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex, VkQueue* pQueue) {
*pQueue=Handle<VkQueue>(2);
}
VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceFeatures2(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2* pFeatures) {
for(auto* node=static_cast<VkBaseOutStructure*>(pFeatures->pNext);node;node=node->pNext){switch(node->sType){
case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES:reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(node)->vulkanMemoryModel=VK_TRUE;break;
case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES:reinterpret_cast<VkPhysicalDeviceVulkan13Features*>(node)->synchronization2=VK_TRUE;break;
case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM:reinterpret_cast<VkPhysicalDeviceTensorFeaturesARM*>(node)->tensors=VK_TRUE;break;
case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM:reinterpret_cast<VkPhysicalDeviceDataGraphFeaturesARM*>(node)->dataGraph=VK_TRUE;break;
case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR:reinterpret_cast<VkPhysicalDeviceMaintenance5FeaturesKHR*>(node)->maintenance5=VK_TRUE;break;
case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_REPLICATED_COMPOSITES_FEATURES_EXT:reinterpret_cast<VkPhysicalDeviceShaderReplicatedCompositesFeaturesEXT*>(node)->shaderReplicatedComposites=VK_TRUE;break;
default:ADD_FAILURE()<<"Unexpected feature";}}

}
VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceFormatProperties2(VkPhysicalDevice physicalDevice, VkFormat format, VkFormatProperties2* pFormatProperties) {
static_cast<VkTensorFormatPropertiesARM*>(pFormatProperties->pNext)->linearTilingTensorFeatures=VK_FORMAT_FEATURE_2_TENSOR_DATA_GRAPH_BIT_ARM;
}
VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceMemoryProperties(VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties* pMemoryProperties) {
pMemoryProperties->memoryTypeCount=1;pMemoryProperties->memoryTypes[0].propertyFlags=state->host_memory?(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT):0;
}
VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceProperties(VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties* pProperties) {
pProperties->apiVersion=VK_API_VERSION_1_3;pProperties->limits.maxMemoryAllocationCount=state->limit;
}
VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceProperties2(VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties2* pProperties) {
auto* p=static_cast<VkPhysicalDeviceTensorPropertiesARM*>(pProperties->pNext);p->maxTensorDimensionCount=6;p->maxTensorElements=100000;p->maxPerDimensionTensorElements=8192;p->maxTensorSize=400000;p->maxDescriptorSetStorageTensors=100;p->maxPerStageDescriptorSetStorageTensors=100;
}
VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceQueueFamilyDataGraphEngineOperationPropertiesARM(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, const VkQueueFamilyDataGraphPropertiesARM* pQueueFamilyDataGraphProperties, VkBaseOutStructure* pProperties) {
static VkDataGraphTOSANameQualityARM profiles[2];std::strcpy(profiles[0].name,"PRO-INT");std::strcpy(profiles[1].name,"PRO-FP");profiles[0].qualityFlags=profiles[1].qualityFlags=VK_DATA_GRAPH_TOSA_QUALITY_CONFORMANT_ARM;auto* t=reinterpret_cast<VkQueueFamilyDataGraphTOSAPropertiesARM*>(pProperties);t->level=VK_DATA_GRAPH_TOSA_LEVEL_8K_ARM;t->profileCount=2;t->pProfiles=profiles;return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceQueueFamilyDataGraphPropertiesARM(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, uint32_t* pQueueFamilyDataGraphPropertyCount, VkQueueFamilyDataGraphPropertiesARM* pQueueFamilyDataGraphProperties) {
if(pQueueFamilyDataGraphProperties){auto& e=*pQueueFamilyDataGraphProperties;e.engine.type=VK_PHYSICAL_DEVICE_DATA_GRAPH_PROCESSING_ENGINE_TYPE_DEFAULT_ARM;e.operation.operationType=VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_SPIRV_EXTENDED_INSTRUCTION_SET_ARM;std::strcpy(e.operation.name,"TOSA.001000.1");}*pQueueFamilyDataGraphPropertyCount=1;return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice physicalDevice, uint32_t* pQueueFamilyPropertyCount, VkQueueFamilyProperties* pQueueFamilyProperties) {
if(pQueueFamilyProperties){pQueueFamilyProperties[0].queueCount=1;pQueueFamilyProperties[0].queueFlags=VK_QUEUE_DATA_GRAPH_BIT_ARM;}*pQueueFamilyPropertyCount=1;
}
VKAPI_ATTR void VKAPI_CALL GetTensorMemoryRequirementsARM(VkDevice device, const VkTensorMemoryRequirementsInfoARM* pInfo, VkMemoryRequirements2* pMemoryRequirements) {
if(pMemoryRequirements->pNext){auto* d=static_cast<VkMemoryDedicatedRequirements*>(pMemoryRequirements->pNext);d->requiresDedicatedAllocation=state->dedicated;}
pMemoryRequirements->memoryRequirements={(state->tensor_bytes.at(pInfo->tensor)+15)/16*16,16,1};
}
VKAPI_ATTR VkResult VKAPI_CALL MapMemory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, VkMemoryMapFlags flags, void** ppData) {
++state->maps;*ppData=state->memory.at(memory).data();return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence) {
++state->submits;
for(uint32_t i=0;i<submitCount;++i)for(uint32_t j=0;j<pSubmits[i].commandBufferCount;++j){auto c=pSubmits[i].pCommandBuffers[j];EXPECT_EQ(state->command_states[c],Fake::CommandState::kExecutable);EXPECT_TRUE(state->DispatchValid(c,state->command_sessions[c]));}
if(state->fail_submit)return VK_ERROR_DEVICE_LOST;
for(uint32_t i=0;i<submitCount;++i)for(uint32_t j=0;j<pSubmits[i].commandBufferCount;++j){auto c=pSubmits[i].pCommandBuffers[j];state->command_states[c]=Fake::CommandState::kPending;++state->dispatches;}return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL ResetCommandBuffer(VkCommandBuffer commandBuffer, VkCommandBufferResetFlags flags) {
++state->resets;EXPECT_NE(state->command_states[commandBuffer],Fake::CommandState::kPending);
if(state->fail_reset)return VK_ERROR_OUT_OF_HOST_MEMORY;
state->command_states[commandBuffer]=Fake::CommandState::kInitial;state->command_sets.erase(commandBuffer);state->command_sessions.erase(commandBuffer);state->command_tensors[commandBuffer].clear();state->bound.erase(commandBuffer);return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL ResetFences(VkDevice device, uint32_t fenceCount, const VkFence* pFences) {
++state->fence_resets;return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL UnmapMemory(VkDevice device, VkDeviceMemory memory) {

}
VKAPI_ATTR void VKAPI_CALL UpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount, const VkWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const VkCopyDescriptorSet* pDescriptorCopies) {
for(uint32_t i=0;i<descriptorWriteCount;++i){state->written_bindings.push_back(pDescriptorWrites[i].dstBinding);for(const auto& pair:state->command_sets)if(pair.second==pDescriptorWrites[i].dstSet)state->command_states[pair.first]=Fake::CommandState::kInvalid;}
}
VKAPI_ATTR VkResult VKAPI_CALL WaitForFences(VkDevice device, uint32_t fenceCount, const VkFence* pFences, VkBool32 waitAll, uint64_t timeout) {
++state->waits;if(state->fail_wait)return VK_ERROR_DEVICE_LOST;
for(auto& entry:state->command_states)if(entry.second==Fake::CommandState::kPending)entry.second=Fake::CommandState::kExecutable;return VK_SUCCESS;
}
PFN_vkVoidFunction VKAPI_CALL Resolve(VkInstance,const char* name) {
  if(!std::strcmp(name,"vkAllocateCommandBuffers"))return reinterpret_cast<PFN_vkVoidFunction>(AllocateCommandBuffers);
  if(!std::strcmp(name,"vkAllocateDescriptorSets"))return reinterpret_cast<PFN_vkVoidFunction>(AllocateDescriptorSets);
  if(!std::strcmp(name,"vkAllocateMemory"))return reinterpret_cast<PFN_vkVoidFunction>(AllocateMemory);
  if(!std::strcmp(name,"vkBeginCommandBuffer"))return reinterpret_cast<PFN_vkVoidFunction>(BeginCommandBuffer);
  if(!std::strcmp(name,"vkBindDataGraphPipelineSessionMemoryARM"))return reinterpret_cast<PFN_vkVoidFunction>(BindDataGraphPipelineSessionMemoryARM);
  if(!std::strcmp(name,"vkBindTensorMemoryARM"))return reinterpret_cast<PFN_vkVoidFunction>(BindTensorMemoryARM);
  if(!std::strcmp(name,"vkCmdBindDescriptorSets"))return reinterpret_cast<PFN_vkVoidFunction>(CmdBindDescriptorSets);
  if(!std::strcmp(name,"vkCmdBindPipeline"))return reinterpret_cast<PFN_vkVoidFunction>(CmdBindPipeline);
  if(!std::strcmp(name,"vkCmdDispatchDataGraphARM"))return reinterpret_cast<PFN_vkVoidFunction>(CmdDispatchDataGraphARM);
  if(!std::strcmp(name,"vkCmdPipelineBarrier2"))return reinterpret_cast<PFN_vkVoidFunction>(CmdPipelineBarrier2);
  if(!std::strcmp(name,"vkCreateCommandPool"))return reinterpret_cast<PFN_vkVoidFunction>(CreateCommandPool);
  if(!std::strcmp(name,"vkCreateDataGraphPipelineSessionARM"))return reinterpret_cast<PFN_vkVoidFunction>(CreateDataGraphPipelineSessionARM);
  if(!std::strcmp(name,"vkCreateDataGraphPipelinesARM"))return reinterpret_cast<PFN_vkVoidFunction>(CreateDataGraphPipelinesARM);
  if(!std::strcmp(name,"vkCreateDescriptorPool"))return reinterpret_cast<PFN_vkVoidFunction>(CreateDescriptorPool);
  if(!std::strcmp(name,"vkCreateDescriptorSetLayout"))return reinterpret_cast<PFN_vkVoidFunction>(CreateDescriptorSetLayout);
  if(!std::strcmp(name,"vkCreateDevice"))return reinterpret_cast<PFN_vkVoidFunction>(CreateDevice);
  if(!std::strcmp(name,"vkCreateFence"))return reinterpret_cast<PFN_vkVoidFunction>(CreateFence);
  if(!std::strcmp(name,"vkCreateInstance"))return reinterpret_cast<PFN_vkVoidFunction>(CreateInstance);
  if(!std::strcmp(name,"vkCreatePipelineLayout"))return reinterpret_cast<PFN_vkVoidFunction>(CreatePipelineLayout);
  if(!std::strcmp(name,"vkCreateShaderModule"))return reinterpret_cast<PFN_vkVoidFunction>(CreateShaderModule);
  if(!std::strcmp(name,"vkCreateTensorARM"))return reinterpret_cast<PFN_vkVoidFunction>(CreateTensorARM);
  if(!std::strcmp(name,"vkCreateTensorViewARM"))return reinterpret_cast<PFN_vkVoidFunction>(CreateTensorViewARM);
  if(!std::strcmp(name,"vkDestroyCommandPool"))return reinterpret_cast<PFN_vkVoidFunction>(DestroyCommandPool);
  if(!std::strcmp(name,"vkDestroyDataGraphPipelineSessionARM"))return reinterpret_cast<PFN_vkVoidFunction>(DestroyDataGraphPipelineSessionARM);
  if(!std::strcmp(name,"vkDestroyDescriptorPool"))return reinterpret_cast<PFN_vkVoidFunction>(DestroyDescriptorPool);
  if(!std::strcmp(name,"vkDestroyDescriptorSetLayout"))return reinterpret_cast<PFN_vkVoidFunction>(DestroyDescriptorSetLayout);
  if(!std::strcmp(name,"vkDestroyDevice"))return reinterpret_cast<PFN_vkVoidFunction>(DestroyDevice);
  if(!std::strcmp(name,"vkDestroyFence"))return reinterpret_cast<PFN_vkVoidFunction>(DestroyFence);
  if(!std::strcmp(name,"vkDestroyInstance"))return reinterpret_cast<PFN_vkVoidFunction>(DestroyInstance);
  if(!std::strcmp(name,"vkDestroyPipeline"))return reinterpret_cast<PFN_vkVoidFunction>(DestroyPipeline);
  if(!std::strcmp(name,"vkDestroyPipelineLayout"))return reinterpret_cast<PFN_vkVoidFunction>(DestroyPipelineLayout);
  if(!std::strcmp(name,"vkDestroyShaderModule"))return reinterpret_cast<PFN_vkVoidFunction>(DestroyShaderModule);
  if(!std::strcmp(name,"vkDestroyTensorARM"))return reinterpret_cast<PFN_vkVoidFunction>(DestroyTensorARM);
  if(!std::strcmp(name,"vkDestroyTensorViewARM"))return reinterpret_cast<PFN_vkVoidFunction>(DestroyTensorViewARM);
  if(!std::strcmp(name,"vkDeviceWaitIdle"))return reinterpret_cast<PFN_vkVoidFunction>(DeviceWaitIdle);
  if(!std::strcmp(name,"vkEndCommandBuffer"))return reinterpret_cast<PFN_vkVoidFunction>(EndCommandBuffer);
  if(!std::strcmp(name,"vkEnumerateDeviceExtensionProperties"))return reinterpret_cast<PFN_vkVoidFunction>(EnumerateDeviceExtensionProperties);
  if(!std::strcmp(name,"vkEnumeratePhysicalDevices"))return reinterpret_cast<PFN_vkVoidFunction>(EnumeratePhysicalDevices);
  if(!std::strcmp(name,"vkFreeMemory"))return reinterpret_cast<PFN_vkVoidFunction>(FreeMemory);
  if(!std::strcmp(name,"vkGetDataGraphPipelineSessionBindPointRequirementsARM"))return reinterpret_cast<PFN_vkVoidFunction>(GetDataGraphPipelineSessionBindPointRequirementsARM);
  if(!std::strcmp(name,"vkGetDataGraphPipelineSessionMemoryRequirementsARM"))return reinterpret_cast<PFN_vkVoidFunction>(GetDataGraphPipelineSessionMemoryRequirementsARM);
  if(!std::strcmp(name,"vkGetDeviceProcAddr"))return reinterpret_cast<PFN_vkVoidFunction>(GetDeviceProcAddr);
  if(!std::strcmp(name,"vkGetDeviceQueue"))return reinterpret_cast<PFN_vkVoidFunction>(GetDeviceQueue);
  if(!std::strcmp(name,"vkGetPhysicalDeviceFeatures2"))return reinterpret_cast<PFN_vkVoidFunction>(GetPhysicalDeviceFeatures2);
  if(!std::strcmp(name,"vkGetPhysicalDeviceFormatProperties2"))return reinterpret_cast<PFN_vkVoidFunction>(GetPhysicalDeviceFormatProperties2);
  if(!std::strcmp(name,"vkGetPhysicalDeviceMemoryProperties"))return reinterpret_cast<PFN_vkVoidFunction>(GetPhysicalDeviceMemoryProperties);
  if(!std::strcmp(name,"vkGetPhysicalDeviceProperties"))return reinterpret_cast<PFN_vkVoidFunction>(GetPhysicalDeviceProperties);
  if(!std::strcmp(name,"vkGetPhysicalDeviceProperties2"))return reinterpret_cast<PFN_vkVoidFunction>(GetPhysicalDeviceProperties2);
  if(!std::strcmp(name,"vkGetPhysicalDeviceQueueFamilyDataGraphEngineOperationPropertiesARM"))return reinterpret_cast<PFN_vkVoidFunction>(GetPhysicalDeviceQueueFamilyDataGraphEngineOperationPropertiesARM);
  if(!std::strcmp(name,"vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM"))return reinterpret_cast<PFN_vkVoidFunction>(GetPhysicalDeviceQueueFamilyDataGraphPropertiesARM);
  if(!std::strcmp(name,"vkGetPhysicalDeviceQueueFamilyProperties"))return reinterpret_cast<PFN_vkVoidFunction>(GetPhysicalDeviceQueueFamilyProperties);
  if(!std::strcmp(name,"vkGetPhysicalDeviceExternalTensorPropertiesARM"))return reinterpret_cast<PFN_vkVoidFunction>(ExternalTensorProperties);
  if(!std::strcmp(name,"vkGetMemoryFdPropertiesKHR"))return reinterpret_cast<PFN_vkVoidFunction>(MemoryFdProperties);
  if(!std::strcmp(name,"vkGetTensorMemoryRequirementsARM"))return reinterpret_cast<PFN_vkVoidFunction>(GetTensorMemoryRequirementsARM);
  if(!std::strcmp(name,"vkMapMemory"))return reinterpret_cast<PFN_vkVoidFunction>(MapMemory);
  if(!std::strcmp(name,"vkQueueSubmit"))return reinterpret_cast<PFN_vkVoidFunction>(QueueSubmit);
  if(!std::strcmp(name,"vkResetCommandBuffer"))return reinterpret_cast<PFN_vkVoidFunction>(ResetCommandBuffer);
  if(!std::strcmp(name,"vkResetFences"))return reinterpret_cast<PFN_vkVoidFunction>(ResetFences);
  if(!std::strcmp(name,"vkUnmapMemory"))return reinterpret_cast<PFN_vkVoidFunction>(UnmapMemory);
  if(!std::strcmp(name,"vkUpdateDescriptorSets"))return reinterpret_cast<PFN_vkVoidFunction>(UpdateDescriptorSets);
  if(!std::strcmp(name,"vkWaitForFences"))return reinterpret_cast<PFN_vkVoidFunction>(WaitForFences);
  ADD_FAILURE()<<"Unconfigured Vulkan entry point "<<name;return nullptr;
}
std::shared_ptr<const CompiledPartition> Compile(LiteRtModel model) {
  auto options=Options::Create();EXPECT_TRUE(options);if(!options)return nullptr;
  auto arm=options->GetOptions<ArmVulkanMLOptions>();EXPECT_TRUE(arm);if(!arm)return nullptr;
  EXPECT_TRUE(arm->SetEnableJustInTime(true));
  auto env=Environment::Create({});EXPECT_TRUE(env);if(!env)return nullptr;
  auto coptions=internal::LiteRtOptionsPtrBuilder::Build(*options,env->GetHolder());EXPECT_TRUE(coptions);if(!coptions)return nullptr;
  auto plugin=StaticallyLinkedPlugin::Create(LrtGetCompilerContext(),nullptr,coptions->get());EXPECT_TRUE(plugin);if(!plugin)return nullptr;
  LiteRtCompiledResult result=nullptr;
  EXPECT_EQ(plugin->Api()->compiler_plugin_compile(plugin->Get(),nullptr,model,&result),kLiteRtStatusOk);
  if(!result)return nullptr;
  LiteRtJitExecutable executable=nullptr;
  EXPECT_EQ(plugin->Api()->get_compiled_result_handle(result,0,&executable),kLiteRtStatusOk);
  auto artifact=executable?reinterpret_cast<ArmVulkanMLJitExecutable*>(executable)->partition:nullptr;
  plugin->Api()->destroy_compiled_result(result);return artifact;
}
std::shared_ptr<const CompiledPartition> FloatFixture() {
  tflite::ModelT model;model.version=3;model.buffers.push_back(std::make_unique<tflite::BufferT>());
  auto graph=std::make_unique<tflite::SubGraphT>();
  for(int i=0;i<11;++i){auto t=std::make_unique<tflite::TensorT>();t->type=tflite::TensorType_FLOAT32;t->shape={1,4};graph->tensors.push_back(std::move(t));}
  auto constant=[&](int id,const auto& values){auto b=std::make_unique<tflite::BufferT>();b->data.resize(values.size()*sizeof(values[0]));std::memcpy(b->data.data(),values.data(),b->data.size());graph->tensors[id]->buffer=model.buffers.size();model.buffers.push_back(std::move(b));};
  constant(2,std::vector<float>{1,2,3,4});constant(3,std::vector<float>{4,3,2,1});
  constant(7,std::vector<int32_t>{1,3});constant(8,std::vector<int32_t>{1});
  graph->tensors[7]->type=graph->tensors[8]->type=tflite::TensorType_INT32;
  graph->tensors[7]->shape={2};graph->tensors[8]->shape={1};
  graph->tensors[9]->shape={1,1};graph->tensors[10]->shape={1,3};
  for(auto code:{tflite::BuiltinOperator_ADD,tflite::BuiltinOperator_MUL,tflite::BuiltinOperator_SPLIT_V}){auto c=std::make_unique<tflite::OperatorCodeT>();c->builtin_code=code;c->deprecated_builtin_code=std::min(static_cast<int>(code),127);model.operator_codes.push_back(std::move(c));}
  for(int i=0;i<4;++i){auto op=std::make_unique<tflite::OperatorT>();op->opcode_index=i<2?0:i==2?1:2;
    op->inputs=i==0?std::vector<int32_t>{0,1}:i==1?std::vector<int32_t>{4,2}:i==2?std::vector<int32_t>{5,3}:std::vector<int32_t>{6,7,8};
    op->outputs=i==3?std::vector<int32_t>{9,10}:std::vector<int32_t>{4+i};
    if(i<2)op->builtin_options.Set(tflite::AddOptionsT{});
    else if(i==2)op->builtin_options.Set(tflite::MulOptionsT{});
    else {tflite::SplitVOptionsT opt;opt.num_splits=2;op->builtin_options.Set(opt);}
    graph->operators.push_back(std::move(op));
  }
  graph->inputs={0,1};graph->outputs={9,10};model.subgraphs.push_back(std::move(graph));
  flatbuffers::FlatBufferBuilder builder;tflite::FinishModelBuffer(builder,tflite::Model::Pack(builder,&model));
  auto loaded=ExtendedModel::CreateFromBuffer(BufferRef<uint8_t>(builder.GetBufferPointer(),builder.GetSize()));
  EXPECT_TRUE(loaded);if(!loaded)return nullptr;return Compile(loaded->Get());
}
class InvocationTest:public ::testing::Test {
 protected:
  Fake fake;std::string error, previous_validation;bool had_validation=false;
  void SetUp() override {
    state=&fake;const char* previous=std::getenv("LITERT_ARM_VULKAN_ML_VALIDATION");
    if(previous){had_validation=true;previous_validation=previous;}
    unsetenv("LITERT_ARM_VULKAN_ML_VALIDATION");
  }
  void TearDown() override {EXPECT_TRUE(fake.live.empty())<<"leaked Vulkan resource";state=nullptr;if(had_validation)setenv("LITERT_ARM_VULKAN_ML_VALIDATION",previous_validation.c_str(),1);else unsetenv("LITERT_ARM_VULKAN_ML_VALIDATION");}
  std::unique_ptr<VulkanInvocation> Create(std::shared_ptr<const CompiledPartition> a={}) {
    if(!a)a=FloatFixture();return VulkanInvocation::CreateWithResolver(a,false,Resolve,error);
  }
  bool Run(VulkanInvocation& invocation) {
    std::vector<float> x(4,1),y(4,2),a(1),b(3);
    return invocation.Run({x.data(),y.data()},{a.data(),b.data()},error);
  }
};

std::shared_ptr<const CompiledPartition> TypedFixture(tflite::BuiltinOperator code) {
  tflite::ModelT model;
  model.version = 3;
  model.buffers.push_back(std::make_unique<tflite::BufferT>());
  auto graph = std::make_unique<tflite::SubGraphT>();
  for (int i = 0; i < 3; ++i) {
    auto tensor = std::make_unique<tflite::TensorT>();
    tensor->name = "typed_tensor_" + std::to_string(i);
    tensor->shape = {2, 3};
    tensor->type = code == tflite::BuiltinOperator_RIGHT_SHIFT ? tflite::TensorType_UINT32
        : code == tflite::BuiltinOperator_EQUAL && i < 2 ? tflite::TensorType_FLOAT32
                                                      : tflite::TensorType_BOOL;
    graph->tensors.push_back(std::move(tensor));
  }
  auto opcode = std::make_unique<tflite::OperatorCodeT>();
  opcode->builtin_code = code;
  model.operator_codes.push_back(std::move(opcode));
  auto op = std::make_unique<tflite::OperatorT>();
  op->inputs = {0, 1};
  op->outputs = {2};
  graph->operators.push_back(std::move(op));
  graph->inputs = {0, 1};
  graph->outputs = {2};
  if (code == tflite::BuiltinOperator_RIGHT_SHIFT) {
    graph->inputs = {0};
    graph->tensors[1]->shape = {1};
    graph->tensors[1]->buffer = 1;
    auto buffer = std::make_unique<tflite::BufferT>();
    const uint32_t shift = 1;
    buffer->data.resize(sizeof(shift));
    std::memcpy(buffer->data.data(), &shift, sizeof(shift));
    model.buffers.push_back(std::move(buffer));
  }
  model.subgraphs.push_back(std::move(graph));
  flatbuffers::FlatBufferBuilder builder;
  tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, &model));
  auto loaded = ExtendedModel::CreateFromBuffer(BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
  EXPECT_TRUE(loaded);
  return loaded ? Compile(loaded->Get()) : nullptr;
}

TEST_F(InvocationTest, DispatchPreservesBoolAndUInt32ModelTypesAcrossBufferApis) {
  auto environment = Environment::Create({});
  ASSERT_TRUE(environment);
  const auto* runtime = LrtGetRuntimeContext();
  LiteRtInterface interface = nullptr;
  // The dispatch interface uses ABI version 1, independently of LiteRT's
  // project API version (currently 0.1).
  ASSERT_EQ(LiteRtDispatchQueryInterface(kLiteRtInterfaceBasic,
      {1, 0, 0},
      &interface), kLiteRtStatusOk);
  const auto* api = static_cast<const LiteRtDispatchInterface_V1*>(interface);
  for (auto code : {tflite::BuiltinOperator_LOGICAL_AND, tflite::BuiltinOperator_EQUAL,
                    tflite::BuiltinOperator_RIGHT_SHIFT}) {
    SCOPED_TRACE(static_cast<int>(code));
    auto artifact = TypedFixture(code);
    ASSERT_TRUE(artifact);
    LiteRtDispatchDeviceContext raw_device = nullptr;
    ASSERT_EQ(api->device_context_create(runtime, nullptr, &raw_device), kLiteRtStatusOk);
    std::unique_ptr<LiteRtDispatchDeviceContextT> device(raw_device);
    using Buffer = std::unique_ptr<LiteRtTensorBufferT, void (*)(LiteRtTensorBuffer)>;
    std::vector<Buffer> buffers;
    auto context = std::make_unique<LiteRtDispatchInvocationContextT>();
    context->state = device->state;
    context->artifact = artifact;
    context->invocation = Create(artifact);
    ASSERT_TRUE(context->invocation) << error;
    context->inputs.resize(artifact->inputs.size());
    context->outputs.resize(artifact->outputs.size());
    std::vector<std::pair<bool, LiteRtTensorBufferHandle>> attached;
    for (bool input : {true, false}) {
      const auto& descriptors = input ? artifact->inputs : artifact->outputs;
      for (size_t i = 0; i < descriptors.size(); ++i) {
        const auto& descriptor = descriptors[i];
        const auto model_element = code == tflite::BuiltinOperator_RIGHT_SHIFT
            ? kLiteRtElementTypeUInt32
            : code == tflite::BuiltinOperator_EQUAL && input
                ? kLiteRtElementTypeFloat32 : kLiteRtElementTypeBool;
        ASSERT_EQ(descriptor.logical_element, model_element);
        LiteRtRankedTensorType type{};
        type.element_type = model_element;
        type.layout.rank = static_cast<unsigned int>(descriptor.shape.size());
        for (size_t d = 0; d < descriptor.shape.size(); ++d)
          type.layout.dimensions[d] = static_cast<int32_t>(descriptor.shape[d]);
        auto requirements = input ? api->get_input_requirements : api->get_output_requirements;
        auto attach = input ? api->attach_input : api->attach_output;
        LiteRtTensorBufferRequirements raw_requirements = nullptr;
        ASSERT_EQ(requirements(context.get(), static_cast<int>(i), &type, &raw_requirements), kLiteRtStatusOk);
        std::unique_ptr<LiteRtTensorBufferRequirementsT, void (*)(LiteRtTensorBufferRequirements)>
            owned_requirements(raw_requirements, LiteRtDestroyTensorBufferRequirements);
        size_t bytes = 0;
        ASSERT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(raw_requirements, &bytes), kLiteRtStatusOk);
        auto make_buffer = [&](const LiteRtRankedTensorType& tensor_type, LiteRtTensorBufferHandle& handle) {
          LiteRtTensorBuffer buffer = nullptr;
          EXPECT_EQ(runtime->create_managed_tensor_buffer(environment->Get(),
              kLiteRtTensorBufferTypeHostMemory, &tensor_type, bytes, &buffer), kLiteRtStatusOk);
          if (!buffer) return;
          buffers.emplace_back(buffer, runtime->destroy_tensor_buffer);
          EXPECT_EQ(api->register_tensor_buffer(device.get(), buffer, &handle), kLiteRtStatusOk);
        };
        if (type.element_type == kLiteRtElementTypeBool || type.element_type == kLiteRtElementTypeUInt32) {
          auto storage_type = type;
          storage_type.element_type = type.element_type == kLiteRtElementTypeBool
              ? kLiteRtElementTypeInt8 : kLiteRtElementTypeInt32;
          LiteRtTensorBufferRequirements rejected = nullptr;
          EXPECT_EQ(requirements(context.get(), static_cast<int>(i), &storage_type, &rejected),
                    kLiteRtStatusErrorInvalidArgument);
          EXPECT_EQ(rejected, nullptr);
          LiteRtTensorBufferHandle wrong = 0;
          make_buffer(storage_type, wrong);
          ASSERT_NE(wrong, 0);
          EXPECT_EQ(attach(context.get(), static_cast<int>(i), wrong), kLiteRtStatusErrorInvalidArgument);
          EXPECT_EQ(api->unregister_tensor_buffer(device.get(), wrong), kLiteRtStatusOk);
        }
        LiteRtTensorBufferHandle handle = 0;
        make_buffer(type, handle);
        ASSERT_NE(handle, 0);
        ASSERT_EQ(attach(context.get(), static_cast<int>(i), handle), kLiteRtStatusOk);
        attached.emplace_back(input, handle);
      }
    }
    EXPECT_EQ(api->invoke(context.get()), kLiteRtStatusOk);
    int input_index = 0, output_index = 0;
    for (const auto& [input, handle] : attached) {
      EXPECT_EQ(input ? api->detach_input(context.get(), input_index++, handle)
                      : api->detach_output(context.get(), output_index++, handle), kLiteRtStatusOk);
      EXPECT_EQ(api->unregister_tensor_buffer(device.get(), handle), kLiteRtStatusOk);
    }
  }
}

TEST_F(InvocationTest, Vuid04101BoundaryFailureAndIndependentDevices) {
  VkPhysicalDeviceMemoryProperties properties{};properties.memoryTypeCount=1;
  properties.memoryTypes[0].propertyFlags=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  DeviceMemoryAllocator allocator(Handle<VkDevice>(1),properties,2,AllocateMemory);
  VkMemoryRequirements requirements{16,16,1};VkDeviceMemory first{},second{},third{};
  fake.fail_allocate=1;
  EXPECT_FALSE(allocator.Allocate(requirements,0,first,error));
  EXPECT_TRUE(allocator.Allocate(requirements,0,first,error));
  EXPECT_TRUE(allocator.Allocate(requirements,0,second,error));
  EXPECT_FALSE(allocator.Allocate(requirements,0,third,error));EXPECT_EQ(fake.allocations,3);
  DeviceMemoryAllocator other(Handle<VkDevice>(2),properties,2,AllocateMemory);
  EXPECT_TRUE(other.Allocate(requirements,0,third,error));
  FreeMemory(VK_NULL_HANDLE,first,nullptr);FreeMemory(VK_NULL_HANDLE,second,nullptr);FreeMemory(VK_NULL_HANDLE,third,nullptr);
}
TEST_F(InvocationTest, Vuid04101InvocationHonorsQueriedLimit) {
  fake.limit=2;
  EXPECT_FALSE(Create());
  EXPECT_EQ(fake.allocations,2);
  EXPECT_EQ(fake.dispatches,0);
  EXPECT_NE(error.find("count limit"),std::string::npos);
}
TEST_F(InvocationTest, Vuid04101IncompatibleMemoryDoesNotAllocate) {
  VkPhysicalDeviceMemoryProperties properties{};properties.memoryTypeCount=1;
  DeviceMemoryAllocator allocator(Handle<VkDevice>(1),properties,2,AllocateMemory);
  VkMemoryRequirements requirements{16,16,1};VkDeviceMemory memory{};
  EXPECT_FALSE(allocator.Allocate(requirements,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,memory,error));
  EXPECT_EQ(fake.allocations,0);
}
TEST_F(InvocationTest, Vuid09921And09923CompilerToPipelineContract) {
  auto artifact=FloatFixture();ASSERT_TRUE(artifact);ASSERT_EQ(artifact->constants.size(),2);
  auto invocation=Create(artifact);ASSERT_TRUE(invocation)<<error;
  ASSERT_EQ(fake.captured_constants.size(),2);EXPECT_NE(fake.captured_constants[0].bytes,fake.captured_constants[1].bytes);
  for(size_t i=0;i<artifact->constants.size();++i)EXPECT_EQ(fake.captured_constants[i].bytes,artifact->constants[i].bytes);
  EXPECT_EQ(fake.captured_resources.size(),4);
  EXPECT_NE(fake.captured_resources[2].dimensions,fake.captured_resources[3].dimensions);
  std::set<uint32_t> expected;for(const auto& r:fake.captured_resources){EXPECT_EQ(r.set,0);EXPECT_EQ(r.array,0);expected.insert(r.id);}
  EXPECT_EQ(std::set<uint32_t>(fake.descriptor_bindings.begin(),fake.descriptor_bindings.end()),expected);
  EXPECT_TRUE(Run(*invocation));
  EXPECT_EQ(std::set<uint32_t>(fake.written_bindings.begin(),fake.written_bindings.end()),expected);
  EXPECT_TRUE(Run(*invocation))<<error;EXPECT_TRUE(fake.violations.empty());
}
TEST_F(InvocationTest, Vuid09921And09923IntegerCompatibility) {
  auto model=litert::testing::LoadTestFileModel("arm_vulkan_ml_conv2d_int8.tflite");
  auto artifact=Compile(model.Get());ASSERT_TRUE(artifact);
  auto invocation=Create(artifact);ASSERT_TRUE(invocation)<<error;
  EXPECT_GE(fake.captured_constants.size(),4);
  EXPECT_EQ(fake.captured_resources.front().format,VK_FORMAT_R8_SINT);
  EXPECT_TRUE(fake.violations.empty());
}
TEST_F(InvocationTest, Vuid09921And09923MalformedArtifactFailurePropagates) {
  auto compiled=FloatFixture();ASSERT_TRUE(compiled);
  auto changed=std::make_shared<CompiledPartition>(*compiled);changed->constants[0].descriptor.binding=99;
  EXPECT_FALSE(Create(changed));EXPECT_EQ(fake.violations.back(),"09921");
  changed=std::make_shared<CompiledPartition>(*compiled);changed->outputs[0].binding=99;
  EXPECT_FALSE(Create(changed));EXPECT_EQ(fake.violations.back(),"09923");EXPECT_EQ(fake.dispatches,0);
}
TEST_F(InvocationTest, ContractCheckerRejectsSyntheticMalformedDescriptions) {
  for(const char* mutation:{"constant_missing","constant_duplicate","constant_shape","constant_format","resource_missing","resource_duplicate","resource_array","resource_binding","resource_shape","resource_format"}) {
    fake.corrupt=mutation;EXPECT_FALSE(Create())<<mutation;EXPECT_FALSE(fake.violations.empty());
  }
  EXPECT_EQ(fake.dispatches,0);
}
TEST_F(InvocationTest, Vuid09796BindsEveryObjectAndHandlesEmptyRequirements) {
  {auto invocation=Create();ASSERT_TRUE(invocation)<<error;EXPECT_EQ(fake.binds,3);EXPECT_TRUE(Run(*invocation));}
  fake.requirements={{VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM,1}};fake.binds=0;
  {auto invocation=Create();ASSERT_TRUE(invocation)<<error;EXPECT_EQ(fake.binds,1);EXPECT_TRUE(Run(*invocation));}
  fake.requirements.clear();fake.binds=0;
  {auto invocation=Create();ASSERT_TRUE(invocation)<<error;EXPECT_EQ(fake.binds,0);EXPECT_TRUE(Run(*invocation));}
}
TEST_F(InvocationTest, Vuid09796QueryAllocationAndBindingFailuresNeverDispatch) {
  fake.fail_query=true;EXPECT_FALSE(Create());fake.fail_query=false;
  // Session allocations precede lazy IO allocations.
  fake.fail_allocate=fake.allocations+1;EXPECT_FALSE(Create());fake.fail_allocate=-1;
  fake.fail_bind=fake.binds+2;EXPECT_FALSE(Create());
  EXPECT_EQ(fake.dispatches,0);
}
TEST_F(InvocationTest, Vuid09951RepeatedAndInterleavedInvocations) {
  auto first=Create(),second=Create();ASSERT_TRUE(first)<<error;ASSERT_TRUE(second)<<error;
  EXPECT_TRUE(Run(*first));EXPECT_TRUE(Run(*second));EXPECT_TRUE(Run(*first));
  EXPECT_EQ(fake.dispatches,3);EXPECT_TRUE(fake.violations.empty());
}
TEST_F(InvocationTest, SessionContractCheckerDetectsMissingBindingsAndWrongPipeline) {
  auto invocation=Create();ASSERT_TRUE(invocation)<<error;ASSERT_TRUE(Run(*invocation));
  auto session=fake.sessions.begin()->first;auto command=fake.bound.begin()->first;
  fake.bound_memory[session].erase({VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM,1});EXPECT_FALSE(fake.DispatchValid(command,session));EXPECT_EQ(fake.violations.back(),"09796");
  fake.bound_memory[session].insert({VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM,1});fake.bound[command]=VK_NULL_HANDLE;
  EXPECT_FALSE(fake.DispatchValid(command,session));EXPECT_EQ(fake.violations.back(),"09951");
}
class FakeHardware final:public HardwareBufferApi{
 public:
  uint64_t allocation_size=16;int refs=0;bool valid=true;
  bool Describe(void*,uint64_t& bytes,uint64_t& id)override{bytes=allocation_size;id=7;return valid;}
  void Acquire(void*)override{++refs;}
  void Release(void*)override{--refs;}
  bool Properties(VkDevice,PFN_vkGetDeviceProcAddr,void*,VkDeviceSize& bytes,uint32_t& bits)override{bytes=allocation_size;bits=1;return true;}
};
TEST_F(InvocationTest, ExternalImportsSkipMappingsAndCopiesAndReuseAllocation){
  fake.imports=true;auto invocation=Create();ASSERT_TRUE(invocation)<<error;
  VulkanInvocation::BufferRequirements req;ASSERT_TRUE(invocation->Requirements(0,req,error));EXPECT_TRUE(req.dma_buf);EXPECT_GE(req.bytes,16);
  auto* file=std::tmpfile();ASSERT_NE(file,nullptr);ASSERT_EQ(ftruncate(fileno(file),64),0);
  auto buffer=ExternalBuffer::DmaBuf(fileno(file),64,0,error);ASSERT_TRUE(buffer);fclose(file);
  ASSERT_TRUE(invocation->AttachExternal(0,buffer,error))<<error;
  int imported=fake.fd_imports;ASSERT_TRUE(Run(*invocation))<<error;
  EXPECT_EQ(fake.maps,3);EXPECT_EQ(invocation->Statistics().imported_slots,1);EXPECT_EQ(invocation->Statistics().input_bytes,16);EXPECT_EQ(invocation->Statistics().output_bytes,16);
  ASSERT_TRUE(Run(*invocation));EXPECT_EQ(fake.fd_imports,imported);
  EXPECT_FALSE(invocation->AttachExternal(2,buffer,error));
  invocation->DetachExternal(0);ASSERT_TRUE(Run(*invocation));EXPECT_EQ(invocation->Statistics().imported_slots,0);
}
TEST_F(InvocationTest, ExternalImportFailuresAreTransactional){
  fake.imports=true;auto invocation=Create();ASSERT_TRUE(invocation);
  auto* file=std::tmpfile();ASSERT_NE(file,nullptr);ASSERT_EQ(ftruncate(fileno(file),64),0);
  auto buffer=ExternalBuffer::DmaBuf(fileno(file),64,0,error);auto bad=ExternalBuffer::DmaBuf(fileno(file),64,1,error);fclose(file);
  ASSERT_TRUE(invocation->AttachExternal(0,buffer,error));
  EXPECT_FALSE(invocation->AttachExternal(0,bad,error));
  fake.import_bits=0;EXPECT_FALSE(invocation->AttachExternal(1,buffer,error));fake.import_bits=1;
  fake.importable=false;VulkanInvocation::BufferRequirements req;ASSERT_TRUE(invocation->Requirements(1,req,error));EXPECT_FALSE(req.dma_buf);fake.importable=true;
  ASSERT_TRUE(Run(*invocation));EXPECT_EQ(invocation->Statistics().imported_slots,1);
}
TEST_F(InvocationTest, HardwareBufferLifecycleDedicatedImportAndZeroStaging){
  fake.imports=true;fake.dedicated=true;auto api=std::make_shared<FakeHardware>();
  auto artifact=FloatFixture();auto invocation=VulkanInvocation::CreateWithResolver(artifact,false,Resolve,error,api);ASSERT_TRUE(invocation)<<error;
  int native=0;auto buffer=ExternalBuffer::Ahb(&native,16,0,api,error);ASSERT_TRUE(buffer);EXPECT_EQ(api->refs,1);
  EXPECT_FALSE(ExternalBuffer::Ahb(&native,64,1,api,error));
  ASSERT_TRUE(invocation->AttachExternal(0,buffer,error));ASSERT_TRUE(Run(*invocation));EXPECT_EQ(fake.ahb_imports,1);EXPECT_EQ(fake.maps,3);
  buffer.reset();EXPECT_EQ(api->refs,1);invocation->DetachExternal(0);EXPECT_EQ(api->refs,0);
  api->allocation_size=64;auto oversized=ExternalBuffer::Ahb(&native,64,0,api,error);ASSERT_TRUE(oversized);EXPECT_FALSE(invocation->AttachExternal(0,oversized,error));EXPECT_NE(error.find("Dedicated"),std::string::npos);
}
TEST_F(InvocationTest, FullyImportedRunUsesNoHostMemoryOrCopies){
  fake.imports=true;fake.external_dedicated=true;fake.host_memory=false;auto invocation=Create();ASSERT_TRUE(invocation)<<error;
  for(size_t slot=0;slot<4;++slot){auto* file=std::tmpfile();ASSERT_NE(file,nullptr);ASSERT_EQ(ftruncate(fileno(file),64),0);auto b=ExternalBuffer::DmaBuf(fileno(file),64,0,error);fclose(file);ASSERT_TRUE(b);ASSERT_TRUE(invocation->AttachExternal(slot,b,error))<<error;}
  ASSERT_TRUE(invocation->Run({nullptr,nullptr},{nullptr,nullptr},error))<<error;
  EXPECT_EQ(fake.maps,0);EXPECT_EQ(invocation->Statistics().input_bytes,0);EXPECT_EQ(invocation->Statistics().output_bytes,0);EXPECT_EQ(fake.ownership,8);
}
TEST_F(InvocationTest, ImportBudgetIsReclaimedOnDetach){
  fake.imports=true;fake.limit=4;auto invocation=Create();ASSERT_TRUE(invocation);
  auto* file=std::tmpfile();ASSERT_NE(file,nullptr);ASSERT_EQ(ftruncate(fileno(file),64),0);auto b=ExternalBuffer::DmaBuf(fileno(file),64,0,error);fclose(file);ASSERT_TRUE(b);
  ASSERT_TRUE(invocation->AttachExternal(0,b,error));EXPECT_FALSE(invocation->AttachExternal(1,b,error));invocation->DetachExternal(0);EXPECT_TRUE(invocation->AttachExternal(1,b,error));
}
TEST_F(InvocationTest, ImportAllocationFailureKeepsOriginalBinding){
  fake.imports=true;auto invocation=Create();ASSERT_TRUE(invocation);
  auto* file=std::tmpfile();ASSERT_NE(file,nullptr);ASSERT_EQ(ftruncate(fileno(file),64),0);auto a=ExternalBuffer::DmaBuf(fileno(file),64,0,error);auto b=ExternalBuffer::DmaBuf(fileno(file),64,16,error);fclose(file);ASSERT_TRUE(a);ASSERT_TRUE(b);
  ASSERT_TRUE(invocation->AttachExternal(0,a,error));fake.fail_allocate=fake.allocations+1;EXPECT_FALSE(invocation->AttachExternal(0,b,error));fake.fail_allocate=-1;fake.fail_tensor_bind=true;EXPECT_FALSE(invocation->AttachExternal(0,b,error));fake.fail_tensor_bind=false;ASSERT_TRUE(Run(*invocation));EXPECT_EQ(invocation->Statistics().imported_slots,1);EXPECT_GE(fcntl(b->fd,F_GETFD),0);
}
TEST_F(InvocationTest, ReusesRecordingAcrossChangedHostInputsAndIndependentInvocations){
  auto first=Create(),second=Create();ASSERT_TRUE(first);ASSERT_TRUE(second);
  ASSERT_TRUE(Run(*first));ASSERT_TRUE(Run(*first));ASSERT_TRUE(Run(*second));
  std::vector<float> x(4,9),y(4,-2),a(1),b(3);ASSERT_TRUE(first->Run({x.data(),y.data()},{a.data(),b.data()},error));
  EXPECT_EQ(fake.begins,2);EXPECT_EQ(fake.ends,2);EXPECT_EQ(fake.resets,0);EXPECT_EQ(fake.recorded_dispatches,2);EXPECT_EQ(fake.submits,4);EXPECT_EQ(fake.fence_resets,4);EXPECT_EQ(fake.waits,4);
}
TEST_F(InvocationTest, BindingChangesInvalidateButNoopAndFailedAttachmentsPreserveRecording){
  fake.imports=true;auto invocation=Create();ASSERT_TRUE(invocation);ASSERT_TRUE(Run(*invocation));
  ASSERT_TRUE(invocation->AttachExternal(0,nullptr,error));ASSERT_TRUE(Run(*invocation));EXPECT_EQ(fake.begins,1);
  auto* file=std::tmpfile();ASSERT_NE(file,nullptr);ASSERT_EQ(ftruncate(fileno(file),64),0);auto a=ExternalBuffer::DmaBuf(fileno(file),64,0,error),b=ExternalBuffer::DmaBuf(fileno(file),64,16,error),bad=ExternalBuffer::DmaBuf(fileno(file),64,1,error);fclose(file);
  ASSERT_TRUE(invocation->AttachExternal(0,a,error));ASSERT_TRUE(Run(*invocation));EXPECT_EQ(fake.begins,2);EXPECT_EQ(fake.resets,1);
  ASSERT_TRUE(invocation->AttachExternal(0,a,error));EXPECT_FALSE(invocation->AttachExternal(0,bad,error));ASSERT_TRUE(Run(*invocation));EXPECT_EQ(fake.begins,2);
  VulkanInvocation::BufferRequirements req;ASSERT_TRUE(invocation->Requirements(0,req,error));ASSERT_TRUE(Run(*invocation));EXPECT_EQ(fake.begins,2);
  ASSERT_TRUE(invocation->AttachExternal(0,b,error));ASSERT_TRUE(Run(*invocation));EXPECT_EQ(fake.begins,3);
  invocation->DetachExternal(0);ASSERT_TRUE(Run(*invocation));ASSERT_TRUE(Run(*invocation));EXPECT_EQ(fake.begins,4);EXPECT_EQ(fake.resets,3);
  ASSERT_TRUE(invocation->AttachExternal(2,a,error));ASSERT_TRUE(Run(*invocation));ASSERT_TRUE(Run(*invocation));EXPECT_EQ(fake.begins,5);EXPECT_EQ(fake.resets,4);
  invocation->DetachExternal(2);ASSERT_TRUE(Run(*invocation));EXPECT_EQ(fake.begins,6);EXPECT_EQ(fake.resets,5);
}
TEST_F(InvocationTest, RetriesPartialRecordingsAndPoisonsIncompleteSubmissions){
  for(int fault=0;fault<3;++fault){auto invocation=Create();ASSERT_TRUE(invocation);int begins=fake.begins;
    if(fault==0)fake.fail_begin=true;else if(fault==1)fake.fail_end=true;else {ASSERT_TRUE(Run(*invocation));invocation->DetachExternal(0);fake.fail_reset=true;begins=fake.begins;}
    EXPECT_FALSE(Run(*invocation));fake.fail_begin=false;fake.fail_end=false;fake.fail_reset=false;
    ASSERT_TRUE(Run(*invocation));ASSERT_TRUE(Run(*invocation));EXPECT_EQ(fake.begins,begins+(fault==2?1:2));
  }
  for(bool submit_fault:{true,false}){auto invocation=Create();ASSERT_TRUE(invocation);fake.fail_submit=submit_fault;fake.fail_wait=!submit_fault;EXPECT_FALSE(Run(*invocation));int begins=fake.begins,submits=fake.submits;fake.fail_submit=false;fake.fail_wait=false;EXPECT_FALSE(Run(*invocation));EXPECT_EQ(fake.begins,begins);EXPECT_EQ(fake.submits,submits);}
}
}  // namespace
}  // namespace litert::arm_vulkan_ml
