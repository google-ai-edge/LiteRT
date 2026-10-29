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

#include "litert/vendors/arm_vulkan_ml/tools/devtool/tool.h"
#include "litert/cc/litert_macros.h"
#include <filesystem>
#include <fstream>
#include <set>
#include <unistd.h>
#include "vgf/encoder.hpp"
#include "litert/vendors/arm_vulkan_ml/dispatch/tensor_capabilities.h"
namespace litert::arm_vulkan_ml::tool {
namespace {
Error InvalidVgf(const std::string& message) {
  return Error(kLiteRtStatusErrorInvalidArgument, "VGF export: " + message);
}
Expected<size_t> PackedBytes(const TensorDescriptor& tensor) {
  if(TensorFormat(tensor.element)==VK_FORMAT_UNDEFINED || tensor.shape.empty())
    return InvalidVgf("unsupported tensor format or scalar interface");
  size_t bytes=tensor.element==TensorElement::kInt8?1:4;
  for(auto d:tensor.shape){
    if(!d || bytes>SIZE_MAX/d)return InvalidVgf("invalid tensor dimensions");
    bytes*=d;
  }
  return bytes;
}
}
Expected<void> ExportVgf(const CompiledPartition& artifact, const std::string& name,
                         const std::string& path) {
  if(name.empty() || artifact.entry_point.empty() || artifact.spirv.size()<5 ||
     artifact.spirv[0]!=0x07230203 || artifact.outputs.empty())
    return InvalidVgf("missing graph module, entry point or outputs");
  std::set<uint32_t> bindings;
  for(const auto* tensors:{&artifact.inputs,&artifact.outputs})for(const auto& t:*tensors){
    LITERT_ASSIGN_OR_RETURN(auto bytes,PackedBytes(t));
    static_cast<void>(bytes);
    if(!bindings.insert(t.binding).second)return InvalidVgf("duplicate descriptor binding");
  }
  // Segment constant references follow OpGraphConstantARM indices, not IO bindings.
  for(size_t i=0;i<artifact.constants.size();++i){
    const auto& c=artifact.constants[i];
    LITERT_ASSIGN_OR_RETURN(auto bytes,PackedBytes(c.descriptor));
    if(c.descriptor.binding!=i || bytes!=c.bytes.size())
      return InvalidVgf("constant IDs must be consecutive with exact packed payload sizes");
  }
  namespace vgf=mlsdk::vgflib;
  auto encoder=vgf::CreateEncoder(VK_HEADER_VERSION);
  if(!encoder)return InvalidVgf("cannot create encoder");
  std::vector<vgf::BindingSlotRef> inputs,outputs,all;
  std::vector<std::string> input_names,output_names;
  auto add_io=[&](const auto& tensors,bool input){
    for(size_t i=0;i<tensors.size();++i){
      const auto& t=tensors[i];std::vector<int64_t> shape(t.shape.begin(),t.shape.end());
      auto resource=input?encoder->AddInputResource(static_cast<int32_t>(VK_DESCRIPTOR_TYPE_TENSOR_ARM),static_cast<int32_t>(TensorFormat(t.element)),shape,{}):encoder->AddOutputResource(static_cast<int32_t>(VK_DESCRIPTOR_TYPE_TENSOR_ARM),static_cast<int32_t>(TensorFormat(t.element)),shape,{});
      auto binding=encoder->AddBindingSlot(t.binding,resource);
      (input?inputs:outputs).push_back(binding);all.push_back(binding);
      (input?input_names:output_names).push_back(std::string(input?"input_":"output_")+std::to_string(i));
    }
  };
  add_io(artifact.inputs,true);add_io(artifact.outputs,false);
  std::vector<vgf::ConstantRef> constants;
  for(const auto& c:artifact.constants){
    std::vector<int64_t> shape(c.descriptor.shape.begin(),c.descriptor.shape.end());
    auto resource=encoder->AddConstantResource(static_cast<int32_t>(TensorFormat(c.descriptor.element)),shape,{});
    constants.push_back(encoder->AddConstant(resource,c.bytes.data(),c.bytes.size()));
  }
  auto set=encoder->AddDescriptorSetInfo(all); // The compiler uses descriptor set 0.
  auto module=encoder->AddModule(vgf::ModuleType::GRAPH,name,artifact.entry_point,artifact.spirv);
  encoder->AddSegmentInfo(module,name,{set},inputs,outputs,constants);
  encoder->AddModelSequenceInputsOutputs(inputs,input_names,outputs,output_names);
  encoder->Finish();
  std::filesystem::path temporary=path+".tmp."+std::to_string(getpid());
  std::ofstream output(temporary,std::ios::binary|std::ios::trunc);
  if(!output)return InvalidVgf("cannot open " + temporary.string());
  const bool encoded=encoder->WriteTo(output);output.flush();const bool written=static_cast<bool>(output);output.close();
  std::error_code error;
  if(!encoded || !written || !output){std::filesystem::remove(temporary,error);return InvalidVgf("cannot write " + path);}
  std::filesystem::rename(temporary,path,error);
  if(error){std::error_code ignored;std::filesystem::remove(temporary,ignored);return InvalidVgf("cannot publish " + path + ": " + error.message());}
  return {};
}
}
