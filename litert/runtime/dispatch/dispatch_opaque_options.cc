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

#include "litert/runtime/dispatch/dispatch_opaque_options.h"

#include <cstring>

#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_handle.h"
#include "litert/cc/litert_macros.h"

namespace litert::internal {

namespace {

struct Payload {
  const void* alloc_base = nullptr;
  int alloc_base_fd = -1;
};

}  // namespace

Expected<DispatchDelegateOptions> DispatchDelegateOptions::Create(
    LiteRtOpaqueOptions options) {
  const char* id;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpaqueOptionsIdentifier(options, &id));
  if (!id || strcmp(id, Discriminator()) != 0) {
    return Error(kLiteRtStatusErrorInvalidArgument);
  }
  return DispatchDelegateOptions(options, OwnHandle::kNo);
}

Expected<DispatchDelegateOptions> DispatchDelegateOptions::Create() {
  LiteRtOpaqueOptions opaque_options;
  LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
      Discriminator(), new Payload(),
      [](void* payload) { delete reinterpret_cast<Payload*>(payload); },
      &opaque_options));
  return DispatchDelegateOptions(opaque_options);
}

// TODO LUKE document
Expected<void> DispatchDelegateOptions::SetAllocBase(const void* alloc_base) {
  void* void_payload;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpaqueOptionsData(Get(), &void_payload));
  Payload* payload = reinterpret_cast<Payload*>(void_payload);
  payload->alloc_base = alloc_base;
  return {};
}
Expected<const void*> DispatchDelegateOptions::GetAllocBase() {
  void* void_payload;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpaqueOptionsData(Get(), &void_payload));
  Payload* payload = reinterpret_cast<Payload*>(void_payload);
  return payload->alloc_base;
}

// TODO LUKE document
Expected<void> DispatchDelegateOptions::SetAllocBaseFd(int alloc_base_fd) {
  void* void_payload;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpaqueOptionsData(Get(), &void_payload));
  Payload* payload = reinterpret_cast<Payload*>(void_payload);
  payload->alloc_base_fd = alloc_base_fd;
  return {};
}
Expected<int> DispatchDelegateOptions::GetAllocBaseFd() {
  void* void_payload;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpaqueOptionsData(Get(), &void_payload));
  Payload* payload = reinterpret_cast<Payload*>(void_payload);
  return payload->alloc_base_fd;
}

}  // namespace litert::internal
