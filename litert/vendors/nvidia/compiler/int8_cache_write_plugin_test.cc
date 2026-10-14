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

#include "litert/vendors/nvidia/compiler/int8_cache_write_plugin.h"

#include <array>
#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "NvInferRuntime.h"

namespace litert::nvidia {
namespace {

// Descriptor-only tests: malformed buffers must be rejected before CUDA work.
class Int8CacheWritePluginTest : public testing::Test {
 protected:
  void SetUp() override {
    Reset(/*value_columns=*/false);
    inputs_[0].dims = nvinfer1::Dims4{1, 2, 2048, 256};
    inputs_[0].type = nvinfer1::DataType::kINT8;
    inputs_[1].dims = nvinfer1::Dims4{1, 2, 1, 256};
    inputs_[1].type = nvinfer1::DataType::kBF16;
    inputs_[2].dims.nbDims = 1;
    inputs_[2].dims.d[0] = 1;
    inputs_[2].type = nvinfer1::DataType::kINT32;
    for (auto& input : inputs_) input.format = nvinfer1::TensorFormat::kLINEAR;
    output_ = inputs_[0];
  }

  void Reset(bool value_columns) {
    plugin_.reset(CreateInt8CacheWritePlugin(value_columns, 0.05f, 3));
    ASSERT_NE(plugin_, nullptr);
    runtime_ = static_cast<nvinfer1::IPluginV3OneRuntime*>(
        plugin_->getCapabilityInterface(
            nvinfer1::PluginCapabilityType::kRUNTIME));
    ASSERT_NE(runtime_, nullptr);
  }

  int ShapeStatus() {
    return runtime_->onShapeChange(inputs_.data(), inputs_.size(), &output_, 1);
  }

  std::unique_ptr<nvinfer1::IPluginV3> plugin_;
  nvinfer1::IPluginV3OneRuntime* runtime_ = nullptr;
  std::array<nvinfer1::PluginTensorDesc, 3> inputs_{};
  nvinfer1::PluginTensorDesc output_{};
};

TEST_F(Int8CacheWritePluginTest, RejectsBadParameters) {
  EXPECT_EQ(CreateInt8CacheWritePlugin(false, 0.0f, 0), nullptr);
  EXPECT_EQ(CreateInt8CacheWritePlugin(false, -1.0f, 0), nullptr);
  EXPECT_EQ(CreateInt8CacheWritePlugin(false, 0.1f, 200), nullptr);
}

TEST_F(Int8CacheWritePluginTest, AcceptsKeyRows) {
  for (auto type : {nvinfer1::DataType::kHALF, nvinfer1::DataType::kBF16,
                    nvinfer1::DataType::kFLOAT}) {
    inputs_[1].type = type;
    for (int tokens : {1, 4, 1024, 2048}) {
      inputs_[1].dims.d[2] = tokens;
      EXPECT_EQ(ShapeStatus(), 0) << tokens;
    }
  }
  // More rows than the cache, a depth or head mismatch, an int8 update, a
  // non-int8 cache, a position that is not int32 [1], an output of another
  // shape: all rejected.
  inputs_[1].dims.d[2] = 2049;
  EXPECT_NE(ShapeStatus(), 0);
  inputs_[1].dims = nvinfer1::Dims4{1, 2, 1, 128};
  EXPECT_NE(ShapeStatus(), 0);
  inputs_[1].dims = nvinfer1::Dims4{1, 1, 1, 256};
  EXPECT_NE(ShapeStatus(), 0);
  inputs_[1].dims = nvinfer1::Dims4{1, 2, 1, 256};
  inputs_[1].type = nvinfer1::DataType::kINT8;
  EXPECT_NE(ShapeStatus(), 0);
  inputs_[1].type = nvinfer1::DataType::kBF16;
  inputs_[0].type = nvinfer1::DataType::kHALF;
  EXPECT_NE(ShapeStatus(), 0);
  inputs_[0].type = nvinfer1::DataType::kINT8;
  inputs_[2].dims.d[0] = 4;
  EXPECT_NE(ShapeStatus(), 0);
  inputs_[2].dims.d[0] = 1;
  inputs_[2].type = nvinfer1::DataType::kHALF;
  EXPECT_NE(ShapeStatus(), 0);
  inputs_[2].type = nvinfer1::DataType::kINT32;
  output_.dims.d[2] = 1024;
  EXPECT_NE(ShapeStatus(), 0);
  output_.dims.d[2] = 2048;
  EXPECT_EQ(ShapeStatus(), 0);
}

TEST_F(Int8CacheWritePluginTest, AcceptsValueColumns) {
  Reset(/*value_columns=*/true);
  // The value cache is [1, H, D, S]; the rows written are still [1, H, T, D].
  inputs_[0].dims = nvinfer1::Dims4{1, 2, 256, 2048};
  output_ = inputs_[0];
  EXPECT_EQ(ShapeStatus(), 0);
  inputs_[1].dims.d[2] = 1024;
  EXPECT_EQ(ShapeStatus(), 0);
  inputs_[1].dims.d[2] = 4096;
  EXPECT_NE(ShapeStatus(), 0);
  // A row-layout cache read as columns has the wrong depth.
  inputs_[1].dims.d[2] = 1;
  inputs_[0].dims = nvinfer1::Dims4{1, 2, 2048, 256};
  output_ = inputs_[0];
  EXPECT_NE(ShapeStatus(), 0);
}

}  // namespace
}  // namespace litert::nvidia
