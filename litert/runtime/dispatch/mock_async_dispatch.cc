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
#include <dlfcn.h>
#include <unistd.h>

#include <set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_event.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"

static LiteRtDispatchApi g_real_api;
static LiteRtDispatchInterface g_intercepted_interface;
static LiteRtDispatchAsyncInterface g_intercepted_async_interface;

static std::vector<std::pair<LiteRtEvent, int>>& GetPendingJobs() {
  static auto* pending_jobs = new std::vector<std::pair<LiteRtEvent, int>>();
  return *pending_jobs;
}

static std::set<LiteRtTensorBufferHandle>& GetUnregisteredHandles() {
  static auto* unregistered_handles = new std::set<LiteRtTensorBufferHandle>();
  return *unregistered_handles;
}

static absl::flat_hash_map<LiteRtTensorBuffer, LiteRtTensorBufferHandle>&
GetBufferToHandle() {
  static auto* buffer_to_handle =
      new absl::flat_hash_map<LiteRtTensorBuffer, LiteRtTensorBufferHandle>();
  return *buffer_to_handle;
}
static LiteRtEnvironment g_env = nullptr;

extern "C" {

void MockDispatchSetEnvironment(LiteRtEnvironment env) { g_env = env; }

void MockDispatchSignalNextJob() {
  if (!GetPendingJobs().empty()) {
    auto job = GetPendingJobs().front();
    char dummy = 1;
    write(job.second, &dummy, 1);
    close(job.second);
    GetPendingJobs().erase(GetPendingJobs().begin());
  }
}

bool MockDispatchIsBufferUnregistered(LiteRtTensorBufferHandle handle) {
  return GetUnregisteredHandles().find(handle) !=
         GetUnregisteredHandles().end();
}

LiteRtTensorBufferHandle MockDispatchGetHandle(LiteRtTensorBuffer buffer) {
  auto it = GetBufferToHandle().find(buffer);
  if (it != GetBufferToHandle().end()) return it->second;
  return 0;
}

LiteRtStatus LiteRtDispatchGetApi(LiteRtDispatchApi* api) {
  // Load real API
  void* handle = dlopen(
      "third_party/odml/litert/litert/vendors/examples/"
      "libLiteRtDispatch_Example.so",
      RTLD_NOW);
  if (!handle) {
    handle = dlopen("litert/vendors/examples/libLiteRtDispatch_Example.so",
                    RTLD_NOW);
  }
  if (!handle) return kLiteRtStatusErrorRuntimeFailure;

  auto get_api = reinterpret_cast<LiteRtStatus (*)(LiteRtDispatchApi*)>(
      dlsym(handle, "LiteRtDispatchGetApi"));
  if (!get_api) return kLiteRtStatusErrorRuntimeFailure;

  LiteRtStatus status = get_api(&g_real_api);
  if (status != kLiteRtStatusOk) return status;

  g_intercepted_interface = *g_real_api.interface;

  // Override get_capabilities to report async support
  g_intercepted_interface.get_capabilities = [](int* capabilities) {
    *capabilities =
        kLiteRtDispatchCapabilitiesBasic | kLiteRtDispatchCapabilitiesAsync;
    return kLiteRtStatusOk;
  };

  // Override Register to map buffers to handles
  g_intercepted_interface.register_tensor_buffer =
      [](LiteRtDispatchDeviceContext context, LiteRtTensorBuffer buffer,
         LiteRtTensorBufferHandle* handle) {
        LiteRtStatus s = g_real_api.interface->register_tensor_buffer(
            context, buffer, handle);
        if (s == kLiteRtStatusOk) {
          GetBufferToHandle()[buffer] = *handle;
        }
        return s;
      };

  // Override Unregister to track calls
  g_intercepted_interface.unregister_tensor_buffer =
      [](LiteRtDispatchDeviceContext context, LiteRtTensorBufferHandle handle) {
        GetUnregisteredHandles().insert(handle);
        return g_real_api.interface->unregister_tensor_buffer(context, handle);
      };

  // Provide Async interface
  g_intercepted_async_interface.invoke_async =
      [](LiteRtDispatchInvocationContext context, int num_events,
         LiteRtEvent* events) {
        if (!g_env) return kLiteRtStatusErrorRuntimeFailure;

        int pipefds[2];
        if (pipe(pipefds) != 0) return kLiteRtStatusErrorRuntimeFailure;

        LiteRtEvent event;
        LiteRtStatus s = LiteRtCreateEventFromSyncFenceFd(
            g_env, pipefds[0], /*owns_fd=*/true, &event);
        if (s != kLiteRtStatusOk) {
          close(pipefds[0]);
          close(pipefds[1]);
          return s;
        }

        GetPendingJobs().push_back({event, pipefds[1]});
        for (int i = 0; i < num_events; ++i) {
          events[i] = event;
        }
        return kLiteRtStatusOk;
      };
  g_intercepted_async_interface.attach_input_event =
      [](LiteRtDispatchInvocationContext context, int idx, LiteRtEvent event) {
        return kLiteRtStatusOk;
      };

  *api = g_real_api;
  api->interface = &g_intercepted_interface;
  api->async_interface = &g_intercepted_async_interface;

  return kLiteRtStatusOk;
}
}  // extern "C"
