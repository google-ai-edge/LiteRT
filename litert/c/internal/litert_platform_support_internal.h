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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_PLATFORM_SUPPORT_INTERNAL_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_PLATFORM_SUPPORT_INTERNAL_H_

namespace litert::internal {

// Dynamic-linker function table injected by unit tests to exercise runtime
// library probing without requiring vendor shared libraries on the host.
struct DlOps {
  void* (*dlopen)(const char* filename, int flags);
  void* (*dlsym)(void* handle, const char* symbol);
  int (*dlclose)(void* handle);
};

// Probes ION runtime availability using the provided dynamic-linker ops.
bool ProbeIonSupport(const DlOps& ops);

// Probes FastRPC runtime availability using the provided dynamic-linker ops.
bool ProbeFastRpcSupport(const DlOps& ops);

}  // namespace litert::internal

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_PLATFORM_SUPPORT_INTERNAL_H_
