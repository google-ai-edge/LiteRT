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

#ifndef ODML_LITERT_LITERT_C_LITERT_EVENT_TYPE_H_
#define ODML_LITERT_LITERT_C_LITERT_EVENT_TYPE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum {
  LiteRtEventTypeUnknown = 0,
  LiteRtEventTypeSyncFenceFd = 1,
  LiteRtEventTypeOpenCl = 2,
  LiteRtEventTypeEglSyncFence = 3,
  LiteRtEventTypeEglNativeSyncFence = 4,
  LiteRtEventTypeCustom = 5,
} LiteRtEventType;

// Custom events managed by the client.
typedef struct LiteRtCustomEventT* LiteRtCustomEvent;
struct LiteRtCustomEventT {
  // Retains the custom event, e.g. increases the reference count.
  void (*Retain)(LiteRtCustomEvent event);  // NOLINT
  // Releases the custom event, e.g. decreases the reference count.
  // If the reference count reaches 0, the custom event will be destroyed.
  void (*Release)(LiteRtCustomEvent event);  // NOLINT
  // Waits for the custom event to be signaled. How to signal the event is
  // backend dependent, e.g. emulating within Wait() or wrapping an actual GPU
  // event signaled by the device.
  void (*Wait)(LiteRtCustomEvent event, int64_t timeout_in_ms);  // NOLINT
  // Returns 1 if the custom event is signaled, 0 otherwise.
  int (*IsSignaled)(LiteRtCustomEvent event);  // NOLINT
  // Returns the native pointer of the custom event, or nullptr if not
  // supported.
  void* (*GetNative)(LiteRtCustomEvent event);  // NOLINT
};

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_EVENT_TYPE_H_
