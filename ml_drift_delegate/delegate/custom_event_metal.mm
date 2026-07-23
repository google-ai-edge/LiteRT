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

#include "ml_drift_delegate/delegate/custom_event_metal.h"

#import <Metal/Metal.h>

#include <cstdint>

#include "litert/c/litert_event.h"

namespace litert::ml_drift {

CustomEventMetal::CustomEventMetal(id<MTLDevice> device) : ref_count_(1) {
  Retain = RetainStatic;
  Release = ReleaseStatic;
  Wait = WaitStatic;
  IsSignaled = IsSignaledStatic;
  GetNative = GetNativeStatic;

  metal_shared_event_ = [device newSharedEvent];
}

void CustomEventMetal::EncodeSignal(id<MTLCommandBuffer> command_buffer) {
  ++value_to_wait_;
  [command_buffer encodeSignalEvent:metal_shared_event_ value:value_to_wait_];
}

// Encodes the wait command in the command buffer.
void CustomEventMetal::EncodeWait(id<MTLCommandBuffer> command_buffer) {
  [command_buffer encodeWaitForEvent:metal_shared_event_ value:value_to_wait_];
}

void CustomEventMetal::RetainStatic(LiteRtCustomEvent event) {
  auto* self = static_cast<CustomEventMetal*>(event);
  ++self->ref_count_;
}

void CustomEventMetal::ReleaseStatic(LiteRtCustomEvent event) {
  auto* self = static_cast<CustomEventMetal*>(event);
  if (--self->ref_count_ <= 0) {
    delete self;
  }
}

void CustomEventMetal::WaitStatic(LiteRtCustomEvent event, int64_t timeout_in_ms) {
  auto* self = static_cast<CustomEventMetal*>(event);
  // Mutex is not needed because delegate is running only on one thread and this is accessed only
  // by that single thread.
  if (self->value_to_wait_ > 0) {
    if (@available(iOS 15, *)) {
      [self->metal_shared_event_ waitUntilSignaledValue:self->value_to_wait_
                                              timeoutMS:timeout_in_ms];
    }
    self->value_to_wait_ = 0;
  }
}

int CustomEventMetal::IsSignaledStatic(LiteRtCustomEvent event) {
  auto* self = static_cast<CustomEventMetal*>(event);
  // If value_to_wait_ is 0, it means signaled. See WaitStatic() above.
  return static_cast<int>(self->value_to_wait_ == 0);
}

void* CustomEventMetal::GetNativeStatic(LiteRtCustomEvent event) {
  return static_cast<void*>(event);
}

}  // namespace litert::ml_drift
