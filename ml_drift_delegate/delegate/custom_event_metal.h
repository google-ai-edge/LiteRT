// Copyright 2025 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_CUSTOM_EVENT_METAL_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_CUSTOM_EVENT_METAL_H_

#ifdef __cplusplus
#import <Metal/Metal.h>

#include "litert/c/litert_event.h"

namespace litert::ml_drift {

// Custom event for Metal wrapping a MTLSharedEvent for asynchronous execution.
class CustomEventMetal : public LiteRtCustomEventT {
 public:
  explicit CustomEventMetal(id<MTLDevice> device);
  ~CustomEventMetal() = default;

  // Encodes the signal command in the command buffer.
  void EncodeSignal(id<MTLCommandBuffer> command_buffer);
  // Encodes the wait command in the command buffer.
  void EncodeWait(id<MTLCommandBuffer> command_buffer);

  id<MTLSharedEvent> shared_event() const { return shared_event_; };

 private:
  // Callbacks of litert_custom_event_t.
  static void RetainStatic(LiteRtCustomEvent event);
  static void ReleaseStatic(LiteRtCustomEvent event);
  static void WaitStatic(LiteRtCustomEvent event, int64_t timeout_in_ms);
  static int IsSignaledStatic(LiteRtCustomEvent event);
  static void* GetNativeStatic(LiteRtCustomEvent event);

  int ref_count_;
  id<MTLSharedEvent> shared_event_;
  uint64_t value_to_wait_ = 0;
};

}  // namespace litert::ml_drift
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_CUSTOM_EVENT_METAL_H_
