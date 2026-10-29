// Copyright 2026 The LiteRT Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef LITERT_ARM_VULKAN_ML_BUFFER_EVENTS_H_
#define LITERT_ARM_VULKAN_ML_BUFFER_EVENTS_H_
#include "litert/c/internal/litert_runtime_context.h"
namespace litert::arm_vulkan_ml {
inline bool WaitForBufferEvent(const LiteRtRuntimeContext* api, LiteRtTensorBuffer buffer) {
  if(!api || !buffer)return false;
  if(!api->has_tensor_buffer_event)return true;
  bool has=false;
  if(api->has_tensor_buffer_event(buffer,&has)!=kLiteRtStatusOk)return false;
  if(!has)return true;
  LiteRtEvent event=nullptr;
  return api->get_tensor_buffer_event && api->wait_event &&
      api->get_tensor_buffer_event(buffer,&event)==kLiteRtStatusOk && event &&
      api->wait_event(event,-1)==kLiteRtStatusOk;
}
}
#endif
