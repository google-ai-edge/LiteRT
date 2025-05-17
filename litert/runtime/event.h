// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_RUNTIME_EVENT_H_
#define ODML_LITERT_LITERT_RUNTIME_EVENT_H_

#include <cstdint>

#include "litert/c/litert_common.h"
#include "litert/c/litert_event_type.h"
#include "litert/c/litert_gl_types.h"
#include "litert/cc/litert_expected.h"

#if LITERT_HAS_OPENCL_SUPPORT
extern "C" {
typedef struct _cl_event* cl_event;
}
#endif  // LITERT_HAS_OPENCL_SUPPORT

struct LiteRtEventT {
  LiteRtEnvironment env;
  LiteRtEventType type = LiteRtEventTypeUnknown;
#if LITERT_HAS_SYNC_FENCE_SUPPORT
  int fd = -1;
  bool owns_fd = false;
#endif  // LITERT_HAS_SYNC_FENCE_SUPPORT
#if LITERT_HAS_OPENCL_SUPPORT
  cl_event opencl_event;
#endif  // LITERT_HAS_OPENCL_SUPPORT
#if LITERT_HAS_OPENGL_SUPPORT
  EGLSyncKHR egl_sync;
#endif  // LITERT_HAS_OPENGL_SUPPORT
  ~LiteRtEventT();
  litert::Expected<void> Wait(int64_t timeout_in_ms);
  litert::Expected<int> GetSyncFenceFd();
  litert::Expected<void> Signal();
  litert::Expected<bool> IsSignaled() const;
  litert::Expected<int> DupFd() const;
  static litert::Expected<LiteRtEventT*> CreateManaged(LiteRtEnvironment env,
                                                       LiteRtEventType type);
};

litert::Expected<LiteRtEventType> GetEventTypeFromEglSync(LiteRtEnvironment env,
                                                          EGLSyncKHR egl_sync);

#endif  // ODML_LITERT_LITERT_RUNTIME_EVENT_H_
