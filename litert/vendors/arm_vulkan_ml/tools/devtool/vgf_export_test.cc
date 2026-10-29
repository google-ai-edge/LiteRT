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
#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include "vgf/decoder.hpp"
namespace litert::arm_vulkan_ml::tool {
namespace {
namespace vgf=mlsdk::vgflib;
class VgfExportTest:public testing::Test {
 protected:
  void SetUp()override{char name[]="/tmp/litert_vgf_XXXXXX";auto* p=mkdtemp(name);ASSERT_NE(p,nullptr);directory=p;}
  void TearDown()override{std::error_code error;std::filesystem::remove_all(directory,error);}
  std::filesystem::path directory;
  CompiledPartition Artifact(){
    CompiledPartition a;
    // Container tests preserve opaque SPIR-V bytes; CLI validates real modules.
    a.spirv={0x07230203,0x00010600,0,1,0};
    a.inputs={{TensorElement::kFloat32,{1,2},7},{TensorElement::kInt8,{2},3}};
    a.outputs={{TensorElement::kInt32,{1,2},11},{TensorElement::kFloat32,{1},4}};
    a.constants={{{TensorElement::kInt8,{2},0},{128,7}},{{TensorElement::kInt32,{1},1},{1,2,3,4}},{{TensorElement::kFloat32,{1},2},{0,0,128,62}}};
    return a;
  }
};
TEST_F(VgfExportTest, RoundTripsCodeBindingsAndAllConstantFormats){
  auto a=Artifact();auto path=(directory/"graph.vgf").string();ASSERT_TRUE(ExportVgf(a,"partition",path));
  std::ifstream input(path,std::ios::binary);std::vector<char> bytes{std::istreambuf_iterator<char>(input),{}};
  ASSERT_GT(bytes.size(),vgf::HeaderSize());auto h=vgf::CreateHeaderDecoder(bytes.data(),vgf::HeaderSize(),bytes.size());ASSERT_TRUE(h);ASSERT_TRUE(h->IsValid());EXPECT_EQ(h->GetEncoderVulkanHeadersVersion(),VK_HEADER_VERSION);
  auto module=vgf::CreateModuleTableDecoder(bytes.data()+h->GetModuleTableOffset(),h->GetModuleTableSize());ASSERT_TRUE(module);ASSERT_EQ(module->size(),1);EXPECT_EQ(module->getModuleType(0),vgf::ModuleType::GRAPH);EXPECT_EQ(module->getModuleEntryPoint(0),a.entry_point);auto code=module->getModuleCode(0);EXPECT_EQ(std::vector<uint32_t>(code.begin(),code.end()),a.spirv);
  auto resources=vgf::CreateModelResourceTableDecoder(bytes.data()+h->GetModelResourceTableOffset(),h->GetModelResourceTableSize());ASSERT_TRUE(resources);ASSERT_EQ(resources->size(),7);
  std::vector<TensorDescriptor> tensors=a.inputs;tensors.insert(tensors.end(),a.outputs.begin(),a.outputs.end());for(const auto& c:a.constants)tensors.push_back(c.descriptor);
  for(size_t i=0;i<tensors.size();++i){int expected=tensors[i].element==TensorElement::kInt8?VK_FORMAT_R8_SINT:tensors[i].element==TensorElement::kInt32?VK_FORMAT_R32_SINT:VK_FORMAT_R32_SFLOAT;EXPECT_EQ(resources->getVkFormat(static_cast<uint32_t>(i)),expected);auto shape=resources->getTensorShape(static_cast<uint32_t>(i));EXPECT_EQ(std::vector<int64_t>(shape.begin(),shape.end()),std::vector<int64_t>(tensors[i].shape.begin(),tensors[i].shape.end()));}
  auto constants=vgf::CreateConstantDecoder(bytes.data()+h->GetConstantsOffset(),h->GetConstantsSize());ASSERT_TRUE(constants);ASSERT_EQ(constants->size(),3);
  for(uint32_t i=0;i<3;++i){auto data=constants->getConstant(i);EXPECT_EQ(std::vector<uint8_t>(data.begin(),data.end()),a.constants[i].bytes);}
  auto sequence=vgf::CreateModelSequenceTableDecoder(bytes.data()+h->GetModelSequenceTableOffset(),h->GetModelSequenceTableSize());ASSERT_TRUE(sequence);ASSERT_EQ(sequence->modelSequenceTableSize(),1);auto bindings=sequence->getDescriptorBindingSlotsHandle(0,0);ASSERT_EQ(sequence->getBindingsSize(bindings),4);
  for(uint32_t i=0;i<4;++i)EXPECT_EQ(sequence->getBindingSlotBinding(bindings,i),tensors[i].binding);
  auto ids=sequence->getSegmentConstantIndexes(0);ASSERT_EQ(ids.size(),3);for(uint32_t i=0;i<3;++i)EXPECT_EQ(ids[i],i);
}
TEST_F(VgfExportTest, AcceptsNoConstantsAndRejectsBadMetadataBeforeWriting){
  auto a=Artifact();a.constants.clear();auto path=(directory/"empty.vgf").string();EXPECT_TRUE(ExportVgf(a,"empty",path));
  for(int variant=0;variant<4;++variant){a=Artifact();if(variant==0)a.constants[0].bytes.pop_back();if(variant==1)a.constants[1].descriptor.binding=9;if(variant==2)a.outputs[0].binding=a.inputs[0].binding;if(variant==3)a.inputs[0].shape={0};auto invalid=(directory/("bad"+std::to_string(variant)+".vgf")).string();EXPECT_FALSE(ExportVgf(a,"bad",invalid));EXPECT_FALSE(std::filesystem::exists(invalid));}
  EXPECT_FALSE(ExportVgf(Artifact(),"bad",(directory/"missing"/"x.vgf").string()));
}
TEST(VgfOptions, CompileOnlyBooleanFlag){
  auto parse=[](std::vector<std::string> args){std::vector<char*> argv;for(auto& a:args)argv.push_back(a.data());return ParseOptions(static_cast<int>(argv.size()),argv.data());};
  auto yes=parse({"tool","compile","--dump_vgf"});ASSERT_TRUE(yes);EXPECT_EQ(yes->Get("dump_vgf"),"1");
  EXPECT_TRUE(parse({"tool","compile","--dump_vgf=0"}));EXPECT_FALSE(parse({"tool","inspect","--dump_vgf"}));EXPECT_FALSE(parse({"tool","compile","--dump_vgf=2"}));
}
}
}
