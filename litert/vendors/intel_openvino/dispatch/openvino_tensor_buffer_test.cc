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

TEST(OpenVinoTensorBufferTest, WrapRejectsNegativeDimension) {
  // E.g. -1 for a dynamic dimension, which must not wrap to SIZE_MAX.
  constexpr int32_t kDynamicDimensions[] = {-1, 4};
  constexpr LiteRtRankedTensorType kDynamicTensorType = {
      kLiteRtElementTypeFloat32, ::litert::BuildLayout(kDynamicDimensions)};
  float data[8] = {};
  OpenVinoTensorBuffer buffer;
  EXPECT_FALSE(buffer.Wrap(kDynamicTensorType, data, sizeof(data)));
  EXPECT_FALSE(buffer.GetTensorData());
}

TEST(OpenVinoTensorBufferTest, WrapRejectsUnsupportedElementType) {
  // Maps to ov::element::dynamic, which ov::Tensor would throw on.
  LiteRtRankedTensorType tensor_type = kTensorType;
  tensor_type.element_type = kLiteRtElementTypeNone;
  float data[8] = {};
  OpenVinoTensorBuffer buffer;
  EXPECT_FALSE(buffer.Wrap(tensor_type, data, sizeof(data)));
  EXPECT_FALSE(buffer.GetTensorData());
}

const LiteRtCustomTensorBufferHandlersDef_V1* GetTensorBufferHandlers() {
  LiteRtInterface raw_interface = nullptr;
  constexpr LiteRtApiVersion kRuntimeVersion = {
      LITERT_DISPATCH_ABI_VERSION_MAJOR, LITERT_DISPATCH_ABI_VERSION_MINOR,
      LITERT_DISPATCH_ABI_VERSION_PATCH};
  if (LiteRtDispatchQueryInterface(kLiteRtInterfaceCustomTensorBufferHandlers,
                                   kRuntimeVersion,
                                   &raw_interface) != kLiteRtStatusOk) {
    return nullptr;
  }
  return static_cast<const LiteRtCustomTensorBufferHandlersDef_V1*>(
      raw_interface);
}

// The import handler the dispatch registers for OpenVINO buffers, reached
// through the dispatch API like LiteRT's runtime does. Its handle is the
// source's `OpenVinoTensorBuffer*`, the same handle
// LiteRtGetTensorBufferCustomTensorBufferHandle returns.
TEST(OpenVinoTensorBufferTest, ImportHandlerCreatesSmallerViewOfSource) {
  const LiteRtCustomTensorBufferHandlersDef_V1* handlers =
      GetTensorBufferHandlers();
  ASSERT_NE(handlers, nullptr);
  ASSERT_NE(handlers->import_func, nullptr);

  // Source: a [4, 4] buffer, standing in for the largest-context KV cache.
  constexpr int32_t kLargeDimensions[] = {4, 4};
  constexpr LiteRtRankedTensorType kLargeTensorType = {
      kLiteRtElementTypeFloat32, ::litert::BuildLayout(kLargeDimensions)};
  float data[16] = {1, 2, 3, 4, 5, 6, 7, 8};
  OpenVinoTensorBuffer source;
  ASSERT_TRUE(source.Wrap(kLargeTensorType, data, sizeof(data)));

  // View: [2, 4] onto the start of the source's memory.
  HwMemoryInfoPtr hw_memory_info = nullptr;
  ASSERT_EQ(handlers->import_func(
                /*device_id=*/nullptr, /*queue_id=*/nullptr, &kTensorType,
                kLiteRtTensorBufferTypeOpenVINOTensorBuffer, &source,
                8 * sizeof(float), 8 * sizeof(float), &hw_memory_info),
            kLiteRtStatusOk);
  ASSERT_NE(hw_memory_info, nullptr);
  auto* view =
      static_cast<OpenVinoTensorBuffer*>(hw_memory_info->memory_handle);
  auto view_tensor = view->GetOVTensor();
  ASSERT_TRUE(view_tensor);
  EXPECT_EQ(view_tensor->get_shape(), ov::Shape({2, 4}));

  void* host_memory = nullptr;
  ASSERT_EQ(handlers->lock_func(hw_memory_info, kLiteRtTensorBufferLockModeRead,
                                &host_memory),
            kLiteRtStatusOk);
  EXPECT_EQ(host_memory, data);
  EXPECT_EQ(handlers->unlock_func(hw_memory_info), kLiteRtStatusOk);

  // Destroying the view leaves the source and its memory intact.
  EXPECT_EQ(handlers->destroy_func(hw_memory_info), kLiteRtStatusOk);
  auto source_data = source.GetTensorData();
  ASSERT_TRUE(source_data);
  EXPECT_EQ(*source_data, data);
  EXPECT_EQ(data[7], 8);
}

TEST(OpenVinoTensorBufferTest, ImportHandlerRejectsNullHandle) {
  const LiteRtCustomTensorBufferHandlersDef_V1* handlers =
      GetTensorBufferHandlers();
  ASSERT_NE(handlers, nullptr);
  HwMemoryInfoPtr hw_memory_info = nullptr;
  EXPECT_EQ(handlers->import_func(
                /*device_id=*/nullptr, /*queue_id=*/nullptr, &kTensorType,
                kLiteRtTensorBufferTypeOpenVINOTensorBuffer,
                /*hw_buffer_handle=*/nullptr, 8 * sizeof(float),
                8 * sizeof(float), &hw_memory_info),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(OpenVinoTensorBufferTest, ImportHandlerRejectsUninitializedSource) {
  const LiteRtCustomTensorBufferHandlersDef_V1* handlers =
      GetTensorBufferHandlers();
  ASSERT_NE(handlers, nullptr);
  OpenVinoTensorBuffer source;  // Neither allocated nor wrapped.
  HwMemoryInfoPtr hw_memory_info = nullptr;
  EXPECT_EQ(handlers->import_func(
                /*device_id=*/nullptr, /*queue_id=*/nullptr, &kTensorType,
                kLiteRtTensorBufferTypeOpenVINOTensorBuffer, &source,
                8 * sizeof(float), 8 * sizeof(float), &hw_memory_info),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(OpenVinoTensorBufferTest, ImportHandlerRejectsViewLargerThanSource) {
  const LiteRtCustomTensorBufferHandlersDef_V1* handlers =
      GetTensorBufferHandlers();
  ASSERT_NE(handlers, nullptr);
  // Source: [2, 4] floats. Requesting 16 floats would read past its memory.
  float data[8] = {};
  OpenVinoTensorBuffer source;
  ASSERT_TRUE(source.Wrap(kTensorType, data, sizeof(data)));
  HwMemoryInfoPtr hw_memory_info = nullptr;
  EXPECT_EQ(handlers->import_func(
                /*device_id=*/nullptr, /*queue_id=*/nullptr, &kTensorType,
                kLiteRtTensorBufferTypeOpenVINOTensorBuffer, &source,
                16 * sizeof(float), 8 * sizeof(float), &hw_memory_info),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(hw_memory_info, nullptr);
}

}  // namespace
