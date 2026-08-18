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

#include "litert/vendors/nvidia/compiler/cache_update_plugin.h"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "NvInferRuntime.h"

namespace litert::nvidia {
namespace {

class CacheUpdatePluginTest : public testing::Test {
 protected:
  void SetUp() override {
    plugin_.reset(CreateCacheUpdatePlugin(true, true));
    ASSERT_NE(plugin_, nullptr);
    runtime_ = static_cast<nvinfer1::IPluginV3OneRuntime*>(
        plugin_->getCapabilityInterface(
            nvinfer1::PluginCapabilityType::kRUNTIME));
    inputs_[0].dims = nvinfer1::Dims4{1, 8, 1152, 256};
    inputs_[1].dims = inputs_[0].dims;
    inputs_[2].dims = nvinfer1::Dims4{1, 8, 1024, 256};
    inputs_[3].dims = inputs_[2].dims;
    inputs_[4].dims = nvinfer1::Dims4{1, 1, 1, 7};
    for (int i = 0; i < inputs_.size(); ++i) {
      inputs_[i].type =
          i == 4 ? nvinfer1::DataType::kINT32 : nvinfer1::DataType::kHALF;
      inputs_[i].format = nvinfer1::TensorFormat::kLINEAR;
    }
    outputs_ = {inputs_[0], inputs_[1]};
  }

  int ShapeStatus() {
    return runtime_->onShapeChange(inputs_.data(), inputs_.size(),
                                   outputs_.data(), outputs_.size());
  }

  std::unique_ptr<nvinfer1::IPluginV3> plugin_;
  nvinfer1::IPluginV3OneRuntime* runtime_ = nullptr;
  std::vector<nvinfer1::PluginTensorDesc> inputs_{5};
  std::vector<nvinfer1::PluginTensorDesc> outputs_{2};
};

TEST_F(CacheUpdatePluginTest,
       ProducesContiguousPatchesWithStandardBuildInterface) {
  auto* build = static_cast<nvinfer1::IPluginV3OneBuild*>(
      plugin_->getCapabilityInterface(nvinfer1::PluginCapabilityType::kBUILD));
  ASSERT_NE(build, nullptr);
  EXPECT_EQ(build->getInterfaceInfo().major, 1);
  EXPECT_EQ(build->getNbOutputs(), 2);
  for (bool ring : {false, true}) {
    for (bool transposed : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "ring=" << ring << " transposed=" << transposed);
      plugin_.reset(CreateCacheUpdatePlugin(ring, transposed));
      runtime_ = static_cast<nvinfer1::IPluginV3OneRuntime*>(
          plugin_->getCapabilityInterface(
              nvinfer1::PluginCapabilityType::kRUNTIME));
      inputs_[1].dims =
          transposed ? inputs_[0].dims : nvinfer1::Dims4{1, 2048, 1152, 1};
      outputs_ = {inputs_[0], inputs_[1]};
      outputs_[0].dims.d[2] = outputs_[1].dims.d[2] = ring ? 1152 : 1024;
      EXPECT_EQ(ShapeStatus(), 0);
      // A patch is neither an arbitrary prefix nor always the full cache.
      outputs_[1].dims.d[2] = ring ? 1024 : 1152;
      EXPECT_NE(ShapeStatus(), 0);
    }
  }
}

TEST_F(CacheUpdatePluginTest, RejectsMalformedDescriptorsAndCounts) {
  EXPECT_NE(runtime_->onShapeChange(nullptr, 5, outputs_.data(), 2), 0);
  EXPECT_NE(runtime_->onShapeChange(inputs_.data(), 5, nullptr, 2), 0);
  EXPECT_NE(runtime_->onShapeChange(inputs_.data(), 4, outputs_.data(), 2), 0);
  EXPECT_NE(runtime_->onShapeChange(inputs_.data(), 5, outputs_.data(), 1), 0);
  const auto original = inputs_;
  for (int i = 0; i < inputs_.size(); ++i) {
    inputs_ = original;
    inputs_[i].type = nvinfer1::DataType::kFLOAT;
    EXPECT_NE(ShapeStatus(), 0) << i;
    inputs_ = original;
    inputs_[i].dims.nbDims = 3;
    EXPECT_NE(ShapeStatus(), 0) << i;
  }
  inputs_ = original;
  inputs_[2].dims.d[2] = inputs_[3].dims.d[2] = 1153;
  EXPECT_NE(ShapeStatus(), 0);
  inputs_ = original;
  inputs_[3].dims.d[3] = 128;
  EXPECT_NE(ShapeStatus(), 0);
  inputs_ = original;
  inputs_[4].dims.d[3] = 4;
  EXPECT_NE(ShapeStatus(), 0);
  inputs_ = original;
  outputs_[1].dims.d[3] = 128;
  EXPECT_NE(ShapeStatus(), 0);
}

TEST_F(CacheUpdatePluginTest, RequiresFlattenedNativeValueCache) {
  plugin_.reset(CreateCacheUpdatePlugin(true, false));
  runtime_ = static_cast<nvinfer1::IPluginV3OneRuntime*>(
      plugin_->getCapabilityInterface(
          nvinfer1::PluginCapabilityType::kRUNTIME));
  inputs_[1].dims = nvinfer1::Dims4{1, 8, 256, 1152};
  outputs_[1] = inputs_[1];
  EXPECT_NE(ShapeStatus(), 0);
  inputs_[1].dims = nvinfer1::Dims4{1, 2048, 1152, 1};
  outputs_[1] = inputs_[1];
  EXPECT_EQ(ShapeStatus(), 0);
  --inputs_[1].dims.d[1];
  outputs_[1] = inputs_[1];
  EXPECT_NE(ShapeStatus(), 0);
}

TEST_F(CacheUpdatePluginTest, AcceptsOpaqueFloatingPointOrderingInputs) {
  const std::array<nvinfer1::DataType, 3> types = {nvinfer1::DataType::kFLOAT,
                                                   nvinfer1::DataType::kHALF,
                                                   nvinfer1::DataType::kBF16};
  const std::array<nvinfer1::Dims, 3> dims = {nvinfer1::Dims{0, {}},
                                              nvinfer1::Dims{1, {17}},
                                              nvinfer1::Dims4{1, 2, 7, 13}};
  for (int i = 0; i < types.size(); ++i) {
    nvinfer1::PluginTensorDesc dependency{};
    dependency.type = types[i];
    dependency.format = nvinfer1::TensorFormat::kLINEAR;
    dependency.dims = dims[i];
    inputs_.push_back(dependency);
  }
  EXPECT_EQ(ShapeStatus(), 0);
  auto* build = static_cast<nvinfer1::IPluginV3OneBuild*>(
      plugin_->getCapabilityInterface(nvinfer1::PluginCapabilityType::kBUILD));
  ASSERT_NE(build, nullptr);
  std::vector<nvinfer1::DynamicPluginTensorDesc> descriptors(inputs_.size() +
                                                             2);
  std::vector<nvinfer1::DataType> input_types;
  for (size_t i = 0; i < inputs_.size(); ++i) {
    descriptors[i].desc = inputs_[i];
    input_types.push_back(inputs_[i].type);
  }
  for (int i = 0; i < 2; ++i) {
    descriptors[inputs_.size() + i].desc = outputs_[i];
  }
  for (int i = 0; i < descriptors.size(); ++i) {
    EXPECT_TRUE(build->supportsFormatCombination(i, descriptors.data(),
                                                 inputs_.size(), 2))
        << i;
  }
  EXPECT_EQ(build->configurePlugin(descriptors.data(), inputs_.size(),
                                   descriptors.data() + inputs_.size(), 2),
            0);
  std::array<nvinfer1::DataType, 2> output_types;
  EXPECT_EQ(build->getOutputDataTypes(output_types.data(), 2,
                                      input_types.data(), input_types.size()),
            0);
  EXPECT_EQ(output_types[0], nvinfer1::DataType::kHALF);
  EXPECT_EQ(output_types[1], nvinfer1::DataType::kHALF);

  inputs_[5].type = nvinfer1::DataType::kINT32;
  EXPECT_NE(ShapeStatus(), 0);
  descriptors[5].desc = inputs_[5];
  EXPECT_FALSE(build->supportsFormatCombination(5, descriptors.data(),
                                                inputs_.size(), 2));
  EXPECT_NE(build->configurePlugin(descriptors.data(), inputs_.size(),
                                   descriptors.data() + inputs_.size(), 2),
            0);
  inputs_[5].type = nvinfer1::DataType::kFLOAT;
  inputs_[5].format = nvinfer1::TensorFormat::kCHW4;
  EXPECT_NE(ShapeStatus(), 0);
}

TEST_F(CacheUpdatePluginTest,
       RequiresOrderingBuffersBeforeLaunchAndAfterClone) {
  inputs_.push_back(inputs_[0]);
  inputs_.back().type = nvinfer1::DataType::kFLOAT;
  inputs_.back().dims = nvinfer1::Dims{0, {}};
  ASSERT_EQ(ShapeStatus(), 0);
  std::unique_ptr<nvinfer1::IPluginV3> cloned(plugin_->clone());
  ASSERT_NE(cloned, nullptr);
  auto* cloned_runtime = static_cast<nvinfer1::IPluginV3OneRuntime*>(
      cloned->getCapabilityInterface(nvinfer1::PluginCapabilityType::kRUNTIME));
  std::array<uint16_t, 8> sentinels{};
  std::array<const void*, 6> inputs;
  for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = &sentinels[i];
  std::array<void*, 2> outputs = {&sentinels[6], &sentinels[7]};
  for (auto* runtime : {runtime_, cloned_runtime}) {
    inputs[5] = nullptr;
    EXPECT_NE(runtime->enqueue(inputs_.data(), outputs_.data(), inputs.data(),
                               outputs.data(), nullptr, nullptr),
              0);
    inputs[5] = outputs[0];
    EXPECT_NE(runtime->enqueue(inputs_.data(), outputs_.data(), inputs.data(),
                               outputs.data(), nullptr, nullptr),
              0);
  }
}

TEST_F(CacheUpdatePluginTest,
       ForwardsFloatingPointShapeAndTypeWithExactCounts) {
  plugin_.reset(CreateCacheUpdatePlugin(true, true, /*forward_read=*/true));
  runtime_ = static_cast<nvinfer1::IPluginV3OneRuntime*>(
      plugin_->getCapabilityInterface(
          nvinfer1::PluginCapabilityType::kRUNTIME));
  auto* build = static_cast<nvinfer1::IPluginV3OneBuild*>(
      plugin_->getCapabilityInterface(nvinfer1::PluginCapabilityType::kBUILD));
  ASSERT_EQ(build->getNbOutputs(), 3);
  inputs_.push_back(inputs_[0]);
  outputs_.push_back(inputs_[5]);
  const std::array<nvinfer1::DataType, 3> types = {nvinfer1::DataType::kFLOAT,
                                                   nvinfer1::DataType::kHALF,
                                                   nvinfer1::DataType::kBF16};
  const std::array<nvinfer1::Dims, 3> shapes = {
      nvinfer1::Dims{0, {}}, nvinfer1::Dims{1, {17}},
      nvinfer1::Dims{5, {2, 3, 5, 7, 11}}};
  for (const auto type : types) {
    for (const auto& shape : shapes) {
      inputs_[5].type = type;
      inputs_[5].dims = shape;
      outputs_[2] = inputs_[5];
      ASSERT_EQ(ShapeStatus(), 0);
      std::array<nvinfer1::DynamicPluginTensorDesc, 9> descriptors;
      std::array<nvinfer1::DataType, 6> input_types;
      std::array<nvinfer1::DataType, 3> output_types;
      for (int i = 0; i < 6; ++i) {
        descriptors[i].desc = inputs_[i];
        input_types[i] = inputs_[i].type;
      }
      for (int i = 0; i < 3; ++i) descriptors[6 + i].desc = outputs_[i];
      for (int i = 0; i < 9; ++i) {
        EXPECT_TRUE(
            build->supportsFormatCombination(i, descriptors.data(), 6, 3));
      }
      EXPECT_EQ(build->configurePlugin(descriptors.data(), 6,
                                       descriptors.data() + 6, 3),
                0);
      EXPECT_EQ(build->getOutputDataTypes(output_types.data(), 3,
                                          input_types.data(), 6),
                0);
      EXPECT_EQ(output_types[2], type);
    }
  }
  EXPECT_NE(runtime_->onShapeChange(inputs_.data(), 5, outputs_.data(), 3), 0);
  EXPECT_NE(runtime_->onShapeChange(inputs_.data(), 6, outputs_.data(), 2), 0);
  inputs_.push_back(inputs_[5]);
  EXPECT_NE(ShapeStatus(), 0);
}

TEST_F(CacheUpdatePluginTest,
       RejectsMalformedAndOverflowingForwardDescriptors) {
  plugin_.reset(CreateCacheUpdatePlugin(true, true, /*forward_read=*/true));
  runtime_ = static_cast<nvinfer1::IPluginV3OneRuntime*>(
      plugin_->getCapabilityInterface(
          nvinfer1::PluginCapabilityType::kRUNTIME));
  inputs_.push_back(inputs_[0]);
  inputs_[5].dims = nvinfer1::Dims{1, {17}};
  outputs_.push_back(inputs_[5]);
  ASSERT_EQ(ShapeStatus(), 0);
  for (const auto shape :
       {nvinfer1::Dims{-1, {}}, nvinfer1::Dims{9, {}}, nvinfer1::Dims{1, {0}},
        nvinfer1::Dims{1, {-1}},
        nvinfer1::Dims{2, {std::numeric_limits<int64_t>::max(), 2}}}) {
    inputs_[5].dims = outputs_[2].dims = shape;
    EXPECT_NE(ShapeStatus(), 0);
  }
  inputs_[5].dims = outputs_[2].dims = nvinfer1::Dims{1, {17}};
  outputs_[2].dims.d[0] = 18;
  EXPECT_NE(ShapeStatus(), 0);
  outputs_[2] = inputs_[5];
  outputs_[2].type = nvinfer1::DataType::kFLOAT;
  EXPECT_NE(ShapeStatus(), 0);
  inputs_[5].type = outputs_[2].type = nvinfer1::DataType::kINT32;
  EXPECT_NE(ShapeStatus(), 0);
  inputs_[5].type = outputs_[2].type = nvinfer1::DataType::kHALF;
  outputs_[2].format = nvinfer1::TensorFormat::kCHW4;
  EXPECT_NE(ShapeStatus(), 0);
}

TEST_F(CacheUpdatePluginTest, RequiresForwardBuffersBeforeLaunchAndAfterClone) {
  plugin_.reset(CreateCacheUpdatePlugin(true, true, /*forward_read=*/true));
  runtime_ = static_cast<nvinfer1::IPluginV3OneRuntime*>(
      plugin_->getCapabilityInterface(
          nvinfer1::PluginCapabilityType::kRUNTIME));
  inputs_.push_back(inputs_[0]);
  inputs_[5].dims = nvinfer1::Dims{0, {}};
  outputs_.push_back(inputs_[5]);
  ASSERT_EQ(ShapeStatus(), 0);
  std::unique_ptr<nvinfer1::IPluginV3> clone(plugin_->clone());
  ASSERT_NE(clone, nullptr);
  auto* cloned_runtime = static_cast<nvinfer1::IPluginV3OneRuntime*>(
      clone->getCapabilityInterface(nvinfer1::PluginCapabilityType::kRUNTIME));
  std::array<uint16_t, 9> sentinels{};
  std::array<const void*, 6> inputs;
  std::array<void*, 3> outputs;
  for (int i = 0; i < 6; ++i) inputs[i] = &sentinels[i];
  for (int i = 0; i < 3; ++i) outputs[i] = &sentinels[6 + i];
  for (auto* runtime : {runtime_, cloned_runtime}) {
    EXPECT_EQ(runtime->onShapeChange(inputs_.data(), 6, outputs_.data(), 3), 0);
    inputs[5] = nullptr;
    EXPECT_NE(runtime->enqueue(inputs_.data(), outputs_.data(), inputs.data(),
                               outputs.data(), nullptr, nullptr),
              0);
    inputs[5] = &sentinels[5];
    outputs[2] = nullptr;
    EXPECT_NE(runtime->enqueue(inputs_.data(), outputs_.data(), inputs.data(),
                               outputs.data(), nullptr, nullptr),
              0);
    for (int i = 0; i < 8; ++i) {
      outputs[2] = &sentinels[i];
      EXPECT_NE(runtime->enqueue(inputs_.data(), outputs_.data(), inputs.data(),
                                 outputs.data(), nullptr, nullptr),
                0);
    }
    outputs[2] = &sentinels[8];
  }
}

TEST_F(CacheUpdatePluginTest, RejectsMissingOrAliasedBuffersBeforeLaunch) {
  std::array<uint16_t, 7> sentinels{};
  std::array<const void*, 5> inputs;
  for (int i = 0; i < inputs.size(); ++i) inputs[i] = &sentinels[i];
  std::array<void*, 2> outputs = {&sentinels[5], &sentinels[6]};
  EXPECT_NE(runtime_->enqueue(nullptr, outputs_.data(), inputs.data(),
                              outputs.data(), nullptr, nullptr),
            0);
  EXPECT_NE(runtime_->enqueue(inputs_.data(), nullptr, inputs.data(),
                              outputs.data(), nullptr, nullptr),
            0);
  EXPECT_NE(runtime_->enqueue(inputs_.data(), outputs_.data(), nullptr,
                              outputs.data(), nullptr, nullptr),
            0);
  EXPECT_NE(runtime_->enqueue(inputs_.data(), outputs_.data(), inputs.data(),
                              nullptr, nullptr, nullptr),
            0);
  for (int i = 0; i < inputs.size(); ++i) {
    const void* saved = inputs[i];
    inputs[i] = nullptr;
    EXPECT_NE(runtime_->enqueue(inputs_.data(), outputs_.data(), inputs.data(),
                                outputs.data(), nullptr, nullptr),
              0);
    inputs[i] = saved;
  }
  for (int i = 0; i < outputs.size(); ++i) {
    void* saved = outputs[i];
    outputs[i] = nullptr;
    EXPECT_NE(runtime_->enqueue(inputs_.data(), outputs_.data(), inputs.data(),
                                outputs.data(), nullptr, nullptr),
              0);
    for (int j = 0; j < inputs.size(); ++j) {
      outputs[i] = &sentinels[j];
      EXPECT_NE(
          runtime_->enqueue(inputs_.data(), outputs_.data(), inputs.data(),
                            outputs.data(), nullptr, nullptr),
          0);
    }
    outputs[i] = saved;
  }
  outputs[1] = outputs[0];
  EXPECT_NE(runtime_->enqueue(inputs_.data(), outputs_.data(), inputs.data(),
                              outputs.data(), nullptr, nullptr),
            0);
}

TEST_F(CacheUpdatePluginTest, RoundTripsFieldsAndRejectsInvalidSerialization) {
  auto* creator = static_cast<nvinfer1::IPluginCreatorV3One*>(
      getPluginRegistry()->getCreator("LiteRtNvidiaCacheUpdatePatch", "2", ""));
  ASSERT_NE(creator, nullptr);
  for (bool ring : {false, true}) {
    for (bool transposed : {false, true}) {
      for (bool forward : {false, true}) {
        std::unique_ptr<nvinfer1::IPluginV3> source(
            CreateCacheUpdatePlugin(ring, transposed, forward));
        auto* runtime = static_cast<nvinfer1::IPluginV3OneRuntime*>(
            source->getCapabilityInterface(
                nvinfer1::PluginCapabilityType::kRUNTIME));
        std::unique_ptr<nvinfer1::IPluginV3> copy(
            creator->createPlugin("copy", runtime->getFieldsToSerialize(),
                                  nvinfer1::TensorRTPhase::kRUNTIME));
        ASSERT_NE(copy, nullptr);
        source.reset();
        auto* copied_runtime = static_cast<nvinfer1::IPluginV3OneRuntime*>(
            copy->getCapabilityInterface(
                nvinfer1::PluginCapabilityType::kRUNTIME));
        const auto* fields = copied_runtime->getFieldsToSerialize();
        ASSERT_EQ(fields->nbFields, 3);
        EXPECT_EQ(*static_cast<const int32_t*>(fields->fields[0].data), ring);
        EXPECT_EQ(*static_cast<const int32_t*>(fields->fields[1].data),
                  transposed);
        EXPECT_EQ(*static_cast<const int32_t*>(fields->fields[2].data),
                  forward);
        auto* build = static_cast<nvinfer1::IPluginV3OneBuild*>(
            copy->getCapabilityInterface(
                nvinfer1::PluginCapabilityType::kBUILD));
        EXPECT_EQ(build->getNbOutputs(), forward ? 3 : 2);
      }
    }
  }
  EXPECT_EQ(
      creator->createPlugin("bad", nullptr, nvinfer1::TensorRTPhase::kRUNTIME),
      nullptr);
  int32_t invalid = 2;
  std::array<nvinfer1::PluginField, 3> fields = {
      {{"ring_buffer", &invalid, nvinfer1::PluginFieldType::kINT32, 1},
       {"transposed_value_cache", &invalid, nvinfer1::PluginFieldType::kINT32,
        1},
       {"forward_read", &invalid, nvinfer1::PluginFieldType::kINT32, 1}}};
  nvinfer1::PluginFieldCollection collection{3, fields.data()};
  EXPECT_EQ(creator->createPlugin("bad", &collection,
                                  nvinfer1::TensorRTPhase::kRUNTIME),
            nullptr);
  int32_t valid = 0;
  fields[0].data = fields[1].data = &valid;
  EXPECT_EQ(creator->createPlugin("bad_forward", &collection,
                                  nvinfer1::TensorRTPhase::kRUNTIME),
            nullptr);
  fields[2].data = &valid;
  collection.nbFields = 2;
  EXPECT_EQ(creator->createPlugin("missing_forward", &collection,
                                  nvinfer1::TensorRTPhase::kRUNTIME),
            nullptr);
}

}  // namespace
}  // namespace litert::nvidia
