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

// Host-only tests for wrapping existing memory as OpenVINO tensor buffers.
// They don't need an Intel NPU: wrapping builds an ov::Tensor over existing
// memory and never touches the OpenVINO device.

#include "litert/vendors/intel_openvino/dispatch/openvino_tensor_buffer.h"

#include <cstdint>

#include "openvino/core/shape.hpp"
#include <gtest/gtest.h>
#include "litert/c/internal/litert_custom_tensor_buffer_handlers_def.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_layout.h"
#include "litert/vendors/c/litert_dispatch_api.h"

namespace {

constexpr int32_t kDimensions[] = {2, 4};
constexpr LiteRtRankedTensorType kTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    ::litert::BuildLayout(kDimensions)};

TEST(OpenVinoTensorBufferTest, WrapUsesTheMemoryInPlace) {
  float data[8] = {};
  OpenVinoTensorBuffer buffer;
  ASSERT_TRUE(buffer.Wrap(kTensorType, data, sizeof(data)));

  auto tensor_data = buffer.GetTensorData();
  ASSERT_TRUE(tensor_data);
  EXPECT_EQ(*tensor_data, data);

  auto ov_tensor = buffer.GetOVTensor();
  ASSERT_TRUE(ov_tensor);
  EXPECT_EQ(ov_tensor->get_shape(), ov::Shape({2, 4}));
  EXPECT_EQ(ov_tensor->get_byte_size(), sizeof(data));
}

TEST(OpenVinoTensorBufferTest, WrapAcceptsLargerMemory) {
  // An alias onto the start of a larger buffer, like a smaller-context KV
  // cache onto the largest one.
  float data[16] = {};
  OpenVinoTensorBuffer buffer;
  ASSERT_TRUE(buffer.Wrap(kTensorType, data, sizeof(data)));
  auto ov_tensor = buffer.GetOVTensor();
  ASSERT_TRUE(ov_tensor);
  EXPECT_EQ(ov_tensor->get_byte_size(), 8 * sizeof(float));
}

TEST(OpenVinoTensorBufferTest, WrapRejectsNullMemory) {
  OpenVinoTensorBuffer buffer;
  EXPECT_FALSE(buffer.Wrap(kTensorType, nullptr, 8 * sizeof(float)));
  EXPECT_FALSE(buffer.GetTensorData());
}

TEST(OpenVinoTensorBufferTest, WrapRejectsMemorySmallerThanTensor) {
  float data[4] = {};
  OpenVinoTensorBuffer buffer;
  EXPECT_FALSE(buffer.Wrap(kTensorType, data, sizeof(data)));
  EXPECT_FALSE(buffer.GetTensorData());
}

TEST(OpenVinoTensorBufferTest, WrapRejectsRankAboveMax) {
  LiteRtRankedTensorType tensor_type = kTensorType;
  tensor_type.layout.rank = LITERT_TENSOR_MAX_RANK + 1;
  float data[8] = {};
  OpenVinoTensorBuffer buffer;
  EXPECT_FALSE(buffer.Wrap(tensor_type, data, sizeof(data)));
}

TEST(OpenVinoTensorBufferTest, WrapRejectsAlreadyInitializedBuffer) {
  float data[8] = {};
  OpenVinoTensorBuffer buffer;
  ASSERT_TRUE(buffer.Wrap(kTensorType, data, sizeof(data)));
  EXPECT_FALSE(buffer.Wrap(kTensorType, data, sizeof(data)));
}

// The import handler the dispatch registers for OpenVINO buffers, reached
// through the dispatch API like LiteRT's runtime does.
TEST(OpenVinoTensorBufferTest, ImportHandlerWrapsWithoutTakingOwnership) {
  LiteRtDispatchApi api;
  ASSERT_EQ(LiteRtDispatchGetApi(&api), kLiteRtStatusOk);
  const LiteRtCustomTensorBufferHandlersDef& handlers =
      *api.tensor_buffer_handlers_def;
  ASSERT_NE(handlers.import_func, nullptr);

  float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  HwMemoryInfoPtr hw_memory_info = nullptr;
  ASSERT_EQ(handlers.import_func(
                /*device_id=*/nullptr, /*queue_id=*/nullptr, &kTensorType,
                kLiteRtTensorBufferTypeOpenVINOTensorBuffer, data,
                sizeof(data), sizeof(data), &hw_memory_info),
            kLiteRtStatusOk);
  ASSERT_NE(hw_memory_info, nullptr);

  void* host_memory = nullptr;
  ASSERT_EQ(handlers.lock_func(hw_memory_info, kLiteRtTensorBufferLockModeRead,
                               &host_memory),
            kLiteRtStatusOk);
  EXPECT_EQ(host_memory, data);
  EXPECT_EQ(handlers.unlock_func(hw_memory_info), kLiteRtStatusOk);

  // Destroying the wrapper leaves the caller's memory intact.
  EXPECT_EQ(handlers.destroy_func(hw_memory_info), kLiteRtStatusOk);
  EXPECT_EQ(data[7], 8);
}

TEST(OpenVinoTensorBufferTest, ImportHandlerRejectsNullHandle) {
  LiteRtDispatchApi api;
  ASSERT_EQ(LiteRtDispatchGetApi(&api), kLiteRtStatusOk);
  HwMemoryInfoPtr hw_memory_info = nullptr;
  EXPECT_EQ(api.tensor_buffer_handlers_def->import_func(
                /*device_id=*/nullptr, /*queue_id=*/nullptr, &kTensorType,
                kLiteRtTensorBufferTypeOpenVINOTensorBuffer,
                /*hw_buffer_handle=*/nullptr, 8 * sizeof(float),
                8 * sizeof(float), &hw_memory_info),
            kLiteRtStatusErrorInvalidArgument);
}

}  // namespace
