// Copyright 2026 The LiteRT Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#include "litert/vendors/arm_vulkan_ml/dispatch/buffer_events.h"
#include <gtest/gtest.h>
namespace litert::arm_vulkan_ml {
namespace {
TEST(BufferEvents, WaitsAndPropagatesFailureWithoutConsumingEvent){
  LiteRtRuntimeContext api{};
  int object=0;auto buffer=reinterpret_cast<LiteRtTensorBuffer>(&object);
  EXPECT_TRUE(WaitForBufferEvent(&api,buffer));
  api.has_tensor_buffer_event=[](LiteRtTensorBuffer,bool* has){*has=true;return kLiteRtStatusOk;};
  EXPECT_FALSE(WaitForBufferEvent(&api,buffer));
  api.get_tensor_buffer_event=[](LiteRtTensorBuffer,LiteRtEvent* event){*event=reinterpret_cast<LiteRtEvent>(uintptr_t{1});return kLiteRtStatusOk;};
  api.wait_event=[](LiteRtEvent event,int64_t timeout){EXPECT_EQ(timeout,-1);EXPECT_NE(event,nullptr);return kLiteRtStatusOk;};
  EXPECT_TRUE(WaitForBufferEvent(&api,buffer));
  api.wait_event=[](LiteRtEvent,int64_t){return kLiteRtStatusErrorRuntimeFailure;};
  EXPECT_FALSE(WaitForBufferEvent(&api,buffer));
  api.has_tensor_buffer_event=[](LiteRtTensorBuffer,bool*){return kLiteRtStatusErrorRuntimeFailure;};
  EXPECT_FALSE(WaitForBufferEvent(&api,buffer));
}
}
}
