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

#include "litert/c/litert_platform_support.h"

#if LITERT_HAS_ION_SUPPORT || LITERT_HAS_FASTRPC_SUPPORT
#include <dlfcn.h>
#endif  // LITERT_HAS_ION_SUPPORT || LITERT_HAS_FASTRPC_SUPPORT

#include "litert/c/internal/litert_platform_support_internal.h"
#include "litert/c/litert_common.h"

namespace litert::internal {

#if defined(RTLD_NOW) && defined(RTLD_LOCAL)
constexpr int kDlopenFlags = RTLD_NOW | RTLD_LOCAL;
#else
constexpr int kDlopenFlags = 0x2;  // RTLD_NOW | RTLD_LOCAL (Linux/Android)
#endif

// Probes ION runtime availability by opening libion.so, resolving the required
// symbols, and verifying that ion_open() returns a valid fd (then closing it).
// Requires LITERT_NO_CFI_CHECK because ion_open and ion_close are invoked via
// dlsym-resolved function pointers.
LITERT_NO_CFI_CHECK bool ProbeIonSupport(const DlOps& ops) {
  void* handle = ops.dlopen("libion.so", kDlopenFlags);
  if (!handle) {
    return false;
  }
  using IonOpenFn = int (*)();
  using IonCloseFn = int (*)(int);
  auto ion_open = reinterpret_cast<IonOpenFn>(ops.dlsym(handle, "ion_open"));
  auto ion_close = reinterpret_cast<IonCloseFn>(ops.dlsym(handle, "ion_close"));
  void* ion_alloc_fd = ops.dlsym(handle, "ion_alloc_fd");
  bool supported = false;
  if (ion_open && ion_close && ion_alloc_fd) {
    int fd = ion_open();
    if (fd >= 0) {
      supported = true;
      ion_close(fd);
    }
  }
  ops.dlclose(handle);
  return supported;
}

// Probes FastRPC runtime availability by opening libcdsprpc.so and verifying
// that rpcmem_alloc, rpcmem_free, and rpcmem_to_fd symbols are present.
bool ProbeFastRpcSupport(const DlOps& ops) {
  void* handle = ops.dlopen("libcdsprpc.so", kDlopenFlags);
  if (!handle) {
    return false;
  }
  bool supported = ops.dlsym(handle, "rpcmem_alloc") != nullptr &&
                   ops.dlsym(handle, "rpcmem_free") != nullptr &&
                   ops.dlsym(handle, "rpcmem_to_fd") != nullptr;
  ops.dlclose(handle);
  return supported;
}

}  // namespace litert::internal

extern "C" {

bool LiteRtHasIonSupport() {
#if LITERT_HAS_ION_SUPPORT
  static const bool supported =
      litert::internal::ProbeIonSupport({::dlopen, ::dlsym, ::dlclose});
  return supported;
#else
  return false;
#endif
}

bool LiteRtHasFastRpcSupport() {
#if LITERT_HAS_FASTRPC_SUPPORT
  static const bool supported =
      litert::internal::ProbeFastRpcSupport({::dlopen, ::dlsym, ::dlclose});
  return supported;
#else
  return false;
#endif
}

}  // extern "C"
