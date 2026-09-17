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

#include "litert/vendors/nvidia/compiler/decode_attention_plugin.h"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>

#include <gtest/gtest.h>
#include "NvInferRuntime.h"

namespace litert::nvidia {
namespace {

// Descriptor-only tests: malformed buffers must be rejected before CUDA work.
class DecodeAttentionPluginTest : public testing::Test {
 protected:
  void SetUp() override {
    plugin_.reset(CreateDecodeAttentionPlugin(-1.0e9f));
    ASSERT_NE(plugin_, nullptr);
    runtime_ = static_cast<nvinfer1::IPluginV3OneRuntime*>(
        plugin_->getCapabilityInterface(
            nvinfer1::PluginCapabilityType::kRUNTIME));
    ASSERT_NE(runtime_, nullptr);
    inputs_[0].dims = nvinfer1::Dims4{1, 8, 4, 512};
    inputs_[0].type = nvinfer1::DataType::kBF16;
    inputs_[1].dims = nvinfer1::Dims4{1, 8, 32768, 512};
    inputs_[1].type = nvinfer1::DataType::kHALF;
    inputs_[2] = inputs_[1];
    inputs_[3].dims = nvinfer1::Dims4{1, 1, 1, 32768};
    inputs_[3].type = nvinfer1::DataType::kBOOL;
    for (auto& input : inputs_) input.format = nvinfer1::TensorFormat::kLINEAR;
    output_ = inputs_[0];
  }

  int ShapeStatus() {
    return runtime_->onShapeChange(inputs_.data(), inputs_.size(), &output_, 1);
  }

  std::unique_ptr<nvinfer1::IPluginV3> plugin_;
  nvinfer1::IPluginV3OneRuntime* runtime_ = nullptr;
  std::array<nvinfer1::PluginTensorDesc, 4> inputs_{};
  nvinfer1::PluginTensorDesc output_{};
};

TEST_F(DecodeAttentionPluginTest, AcceptsSupportedShapesAndTypes) {
  for (auto type : {nvinfer1::DataType::kHALF, nvinfer1::DataType::kBF16}) {
    for (int depth : {128, 256, 512}) {
      for (int rows : {1, 4, 16}) {
        inputs_[0].type = type;
        inputs_[0].dims.d[2] = rows;
        for (int i = 0; i < 3; ++i) inputs_[i].dims.d[3] = depth;
        output_ = inputs_[0];
        for (int mask_rows : {1, rows}) {
          inputs_[3].dims.d[2] = mask_rows;
          EXPECT_EQ(ShapeStatus(), 0);
        }
      }
    }
  }
}

TEST_F(DecodeAttentionPluginTest, RejectsInvalidShapeArguments) {
  EXPECT_NE(runtime_->onShapeChange(nullptr, 4, &output_, 1), 0);
  EXPECT_NE(runtime_->onShapeChange(inputs_.data(), 4, nullptr, 1), 0);
  for (int count : {0, 3, 5}) {
    EXPECT_NE(runtime_->onShapeChange(inputs_.data(), count, &output_, 1), 0);
  }
  for (int count : {0, 2}) {
    EXPECT_NE(runtime_->onShapeChange(inputs_.data(), 4, &output_, count), 0);
  }
}

TEST_F(DecodeAttentionPluginTest, RejectsMismatchedValueCache) {
  --inputs_[2].dims.d[2];
  EXPECT_NE(ShapeStatus(), 0);
  inputs_[2] = inputs_[1];
  inputs_[2].dims.nbDims = 3;
  EXPECT_NE(ShapeStatus(), 0);
}

TEST_F(DecodeAttentionPluginTest, RejectsUnsupportedMaskAndBatch) {
  inputs_[3].dims.d[1] = 8;
  EXPECT_NE(ShapeStatus(), 0);
  inputs_[3].dims.d[1] = 1;
  inputs_[3].dims.d[2] = 2;  // Neither broadcast nor the four query rows.
  EXPECT_NE(ShapeStatus(), 0);
  inputs_[3].dims.d[2] = 1;
  inputs_[1].dims.d[0] = 2;
  inputs_[2] = inputs_[1];
  EXPECT_NE(ShapeStatus(), 0);
}

TEST_F(DecodeAttentionPluginTest, RejectsInvalidOutputAndTypes) {
  output_.dims.d[3] = 256;
  EXPECT_NE(ShapeStatus(), 0);
  output_ = inputs_[0];
  output_.type = nvinfer1::DataType::kHALF;
  EXPECT_NE(ShapeStatus(), 0);
  output_ = inputs_[0];
  inputs_[2].type = nvinfer1::DataType::kBF16;
  EXPECT_NE(ShapeStatus(), 0);
  inputs_[2].type = nvinfer1::DataType::kHALF;
  inputs_[1].format = nvinfer1::TensorFormat::kCHW2;
  EXPECT_NE(ShapeStatus(), 0);
}

TEST_F(DecodeAttentionPluginTest, RejectsInvalidDimensions) {
  for (int64_t rows : {int64_t{-1}, int64_t{0}, int64_t{17},
                       int64_t{1} + std::numeric_limits<int32_t>::max()}) {
    inputs_[0].dims.d[2] = rows;
    output_ = inputs_[0];
    EXPECT_NE(ShapeStatus(), 0);
  }
}

TEST_F(DecodeAttentionPluginTest, RejectsUnsupportedDepth) {
  for (int depth : {64, 192, 1024}) {
    for (int i = 0; i < 3; ++i) inputs_[i].dims.d[3] = depth;
    output_ = inputs_[0];
    EXPECT_NE(ShapeStatus(), 0);
  }
}

TEST_F(DecodeAttentionPluginTest, EnqueueRejectsNullDescriptors) {
  EXPECT_NE(
      runtime_->enqueue(nullptr, &output_, nullptr, nullptr, nullptr, nullptr),
      0);
  EXPECT_NE(runtime_->enqueue(inputs_.data(), nullptr, nullptr, nullptr,
                              nullptr, nullptr),
            0);
}

TEST_F(DecodeAttentionPluginTest, EnqueueRejectsMissingBuffers) {
  // Host sentinels stand in for non-null pointers; each call has a missing
  // required argument and must return before launching a kernel.
  uint16_t sentinel = 0;
  std::array<const void*, 4> input_buffers;
  input_buffers.fill(&sentinel);
  void* output_buffer = &sentinel;
  EXPECT_NE(runtime_->enqueue(inputs_.data(), &output_, nullptr, &output_buffer,
                              &sentinel, nullptr),
            0);
  EXPECT_NE(runtime_->enqueue(inputs_.data(), &output_, input_buffers.data(),
                              nullptr, &sentinel, nullptr),
            0);
  EXPECT_NE(runtime_->enqueue(inputs_.data(), &output_, input_buffers.data(),
                              &output_buffer, nullptr, nullptr),
            0);
  output_buffer = nullptr;
  EXPECT_NE(runtime_->enqueue(inputs_.data(), &output_, input_buffers.data(),
                              &output_buffer, &sentinel, nullptr),
            0);
  output_buffer = &sentinel;
  for (int i = 0; i < 4; ++i) {
    input_buffers[i] = nullptr;
    EXPECT_NE(runtime_->enqueue(inputs_.data(), &output_, input_buffers.data(),
                                &output_buffer, &sentinel, nullptr),
              0)
        << "input=" << i;
    input_buffers[i] = &sentinel;
  }
}

}  // namespace
}  // namespace litert::nvidia
