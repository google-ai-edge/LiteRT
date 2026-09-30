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

TEST_F(DecodeAttentionPluginTest, AcceptsSingleHeadRowsInMultiplesOf16) {
  // Decode (16 query rows) and prefill (16 rows per prompt token) over a
  // single-head cache of depth 512; mask row r % mask_rows serves row r.
  inputs_[1].dims = nvinfer1::Dims4{1, 1, 32768, 512};
  inputs_[2] = inputs_[1];
  for (auto type : {nvinfer1::DataType::kHALF, nvinfer1::DataType::kBF16}) {
    for (int tokens : {1, 8, 128, 1024}) {
      inputs_[0].type = type;
      inputs_[0].dims = nvinfer1::Dims4{1, 1, 16 * tokens, 512};
      output_ = inputs_[0];
      for (int mask_rows : {1, tokens, 16 * tokens}) {
        inputs_[3].dims.d[2] = mask_rows;
        EXPECT_EQ(ShapeStatus(), 0)
            << "tokens=" << tokens << " mask_rows=" << mask_rows;
      }
    }
  }
}

TEST_F(DecodeAttentionPluginTest, RejectsUnsupportedSingleHeadShapes) {
  inputs_[1].dims = nvinfer1::Dims4{1, 1, 32768, 512};
  inputs_[2] = inputs_[1];
  inputs_[0].dims = nvinfer1::Dims4{1, 1, 48, 512};
  output_ = inputs_[0];
  inputs_[3].dims.d[2] = 5;  // Does not divide the rows.
  EXPECT_NE(ShapeStatus(), 0);
  inputs_[0].dims.d[2] = 24;  // More than 16 rows, not a multiple of 16.
  output_ = inputs_[0];
  inputs_[3].dims.d[2] = 1;
  EXPECT_NE(ShapeStatus(), 0);
  inputs_[0].dims = nvinfer1::Dims4{1, 1, 32, 256};  // Depth 256.
  output_ = inputs_[0];
  inputs_[1].dims.d[3] = 256;
  inputs_[2] = inputs_[1];
  EXPECT_NE(ShapeStatus(), 0);
  inputs_[0].dims = nvinfer1::Dims4{1, 2, 32, 512};  // Two cache heads.
  output_ = inputs_[0];
  inputs_[1].dims = nvinfer1::Dims4{1, 2, 32768, 512};
  inputs_[2] = inputs_[1];
  EXPECT_NE(ShapeStatus(), 0);
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

// Prefill over a cache followed by the keys of the chunk.
class NewKeysAttentionPluginTest : public testing::Test {
 protected:
  void SetUp() override {
    plugin_.reset(CreateDecodeAttentionPlugin(-45824.0f, /*new_keys=*/true));
    ASSERT_NE(plugin_, nullptr);
    runtime_ = static_cast<nvinfer1::IPluginV3OneRuntime*>(
        plugin_->getCapabilityInterface(
            nvinfer1::PluginCapabilityType::kRUNTIME));
    ASSERT_NE(runtime_, nullptr);
    SetShape(/*heads=*/8, /*tokens=*/1024, /*query_heads=*/2, /*depth=*/256,
             /*cache_len=*/1152);
  }

  void SetShape(int heads, int tokens, int query_heads, int depth,
                int cache_len) {
    inputs_[0].dims = nvinfer1::Dims4{1, heads, query_heads * tokens, depth};
    inputs_[0].type = nvinfer1::DataType::kBF16;
    inputs_[1].dims = nvinfer1::Dims4{1, heads, cache_len, depth};
    inputs_[1].type = nvinfer1::DataType::kHALF;
    inputs_[2] = inputs_[1];
    inputs_[3].dims = nvinfer1::Dims4{1, 1, tokens, cache_len + tokens};
    inputs_[3].type = nvinfer1::DataType::kBOOL;
    inputs_[4].dims = nvinfer1::Dims4{1, heads, tokens, depth};
    inputs_[4].type = nvinfer1::DataType::kHALF;
    inputs_[5] = inputs_[4];
    for (auto& input : inputs_) input.format = nvinfer1::TensorFormat::kLINEAR;
    output_ = inputs_[0];
  }

  int ShapeStatus() {
    return runtime_->onShapeChange(inputs_.data(), inputs_.size(), &output_, 1);
  }

  std::unique_ptr<nvinfer1::IPluginV3> plugin_;
  nvinfer1::IPluginV3OneRuntime* runtime_ = nullptr;
  std::array<nvinfer1::PluginTensorDesc, 6> inputs_{};
  nvinfer1::PluginTensorDesc output_{};
};

TEST_F(NewKeysAttentionPluginTest, AcceptsRingCachesAndSingleHeadCaches) {
  for (auto type : {nvinfer1::DataType::kHALF, nvinfer1::DataType::kBF16}) {
    for (int tokens : {128, 1024}) {
      SetShape(8, tokens, 2, 256, 1152);
      inputs_[0].type = type;
      output_ = inputs_[0];
      EXPECT_EQ(ShapeStatus(), 0) << "tokens=" << tokens;
      inputs_[3].dims.d[2] = 2 * tokens;  // A mask row per query row.
      EXPECT_EQ(ShapeStatus(), 0) << "tokens=" << tokens;
      SetShape(1, tokens, 16, 512, 4096);
      EXPECT_EQ(ShapeStatus(), 0) << "tokens=" << tokens;
    }
  }
}

TEST_F(NewKeysAttentionPluginTest, RejectsUnsupportedShapes) {
  SetShape(8, 32, 2, 256, 1152);  // Half a block of query rows.
  EXPECT_NE(ShapeStatus(), 0);
  SetShape(8, 1024, 2, 256, 1150);  // The new keys must start a tile.
  EXPECT_NE(ShapeStatus(), 0);
  SetShape(8, 1024, 2, 128, 1152);  // Depth 128.
  EXPECT_NE(ShapeStatus(), 0);
  SetShape(8, 1024, 2, 256, 1152);
  --inputs_[3].dims.d[3];  // The mask covers the cache and the new keys.
  EXPECT_NE(ShapeStatus(), 0);
  SetShape(8, 1024, 2, 256, 1152);
  inputs_[4].dims.d[1] = 4;
  inputs_[5] = inputs_[4];
  EXPECT_NE(ShapeStatus(), 0);
  SetShape(8, 1024, 2, 256, 1152);
  --inputs_[5].dims.d[2];
  EXPECT_NE(ShapeStatus(), 0);
}

TEST_F(NewKeysAttentionPluginTest, RejectsInvalidCountsAndTypes) {
  for (int count : {0, 4, 5, 7}) {
    EXPECT_NE(runtime_->onShapeChange(inputs_.data(), count, &output_, 1), 0);
  }
  inputs_[4].type = nvinfer1::DataType::kBF16;
  EXPECT_NE(ShapeStatus(), 0);
  inputs_[4].type = nvinfer1::DataType::kHALF;
  inputs_[5].format = nvinfer1::TensorFormat::kCHW2;
  EXPECT_NE(ShapeStatus(), 0);
}

TEST_F(NewKeysAttentionPluginTest, EnqueueRejectsMissingBuffers) {
  uint16_t sentinel = 0;
  std::array<const void*, 6> input_buffers;
  input_buffers.fill(&sentinel);
  input_buffers[5] = nullptr;
  void* output_buffer = &sentinel;
  EXPECT_NE(runtime_->enqueue(inputs_.data(), &output_, input_buffers.data(),
                              &output_buffer, &sentinel, nullptr),
            0);
}

TEST_F(DecodeAttentionPluginTest, RejectsNewKeyInputs) {
  std::array<nvinfer1::PluginTensorDesc, 6> inputs{};
  for (int i = 0; i < 4; ++i) inputs[i] = inputs_[i];
  inputs[4] = inputs[5] = inputs_[1];
  EXPECT_NE(runtime_->onShapeChange(inputs.data(), 6, &output_, 1), 0);
}

}  // namespace
}  // namespace litert::nvidia
