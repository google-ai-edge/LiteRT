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

#include "ml_drift_delegate/delegate/webgpu_dawn_loader.h"

#include "absl/strings/string_view.h"  // from @com_google_absl

#if defined(LITERT_WEBGPU_DAWN_DYNAMIC_LOADER) && !defined(__EMSCRIPTEN__)

#include <algorithm>
#include <atomic>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef NOGDI
#define NOGDI
#endif
#include <windows.h>
#ifdef ERROR
#undef ERROR
#endif
#else
#include <dlfcn.h>
#endif

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/base/const_init.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "dawn/dawn_proc.h"  // from @dawn
#include "dawn/dawn_proc_table.h"  // from @dawn
#include "dawn/webgpu.h"  // from @dawn
#include "ml_drift_delegate/delegate/webgpu_dawn_procs.inc"

namespace litert::ml_drift {
namespace {

ABSL_CONST_INIT std::atomic<bool> g_dawn_loaded{false};
ABSL_CONST_INIT absl::Mutex g_dawn_loader_mutex(absl::kConstInit);

WGPUInstance BootstrapCreateInstance(const WGPUInstanceDescriptor* descriptor) {
  if (!EnsureWebGpuDawnLoaded()) {
    return nullptr;
  }
  return wgpuCreateInstance(descriptor);
}

bool RegisterBootstrapProcTable() {
  DawnProcTable table = {};
  std::copy_n(dawnProcGetVersion(), sizeof(table.version), table.version);
  table.createInstance = &BootstrapCreateInstance;
  dawnProcSetProcs(&table);
  return true;
}

[[maybe_unused]] const bool kBootstrapRegistered = RegisterBootstrapProcTable();

#if defined(_WIN32)

void* ResolveWin32Symbol(const char* symbol_name, void* data) {
  HMODULE module = reinterpret_cast<HMODULE>(data);
  return reinterpret_cast<void*>(GetProcAddress(module, symbol_name));
}

bool TryLoadDawnFromWin32Module(HMODULE module, absl::string_view source) {
  if (module == nullptr) {
    return false;
  }
  const char* missing_symbol = nullptr;
  if (!internal_dawn_loader::PopulateDawnProcTable(
          &ResolveWin32Symbol, reinterpret_cast<void*>(module),
          &missing_symbol)) {
    ABSL_LOG(WARNING) << "Found " << source
                      << ", but required WebGPU symbol was missing: "
                      << (missing_symbol ? missing_symbol : "<unknown>");
    return false;
  }
  return true;
}

bool LoadWebGpuDawnLocked(absl::string_view runtime_library_dir) {
  HMODULE existing = nullptr;
  if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_PIN, "webgpu_dawn.dll",
                         &existing) &&
      TryLoadDawnFromWin32Module(existing, "loaded webgpu_dawn.dll")) {
    return true;
  }

  std::vector<std::string> candidates;
  if (!runtime_library_dir.empty()) {
    absl::string_view trimmed =
        absl::StripSuffix(absl::StripSuffix(runtime_library_dir, "\\"), "/");
    candidates.push_back(absl::StrCat(trimmed, "\\webgpu_dawn.dll"));
  }
  candidates.push_back("webgpu_dawn.dll");

  for (const std::string& candidate : candidates) {
    HMODULE handle = LoadLibraryA(candidate.c_str());
    if (handle == nullptr) {
      continue;
    }
    if (TryLoadDawnFromWin32Module(handle, candidate)) {
      ABSL_LOG(INFO) << "Loaded WebGPU Dawn runtime from " << candidate;
      return true;
    }
    FreeLibrary(handle);
  }
  return false;
}

#else  // !defined(_WIN32)

void* ResolvePosixHandleSymbol(const char* symbol_name, void* data) {
  return dlsym(data, symbol_name);
}

void* ResolvePosixDefaultSymbol(const char* symbol_name, void* /*data*/) {
  return dlsym(RTLD_DEFAULT, symbol_name);
}

bool IsDawnProcSymbolInSameModule(void* wgpu_symbol) {
  void* set_procs_symbol = dlsym(RTLD_DEFAULT, "dawnProcSetProcs");
  if (set_procs_symbol == nullptr) {
    return false;
  }
  Dl_info wgpu_info = {};
  Dl_info proc_info = {};
  return dladdr(wgpu_symbol, &wgpu_info) != 0 &&
         dladdr(set_procs_symbol, &proc_info) != 0 &&
         wgpu_info.dli_fbase == proc_info.dli_fbase;
}

bool LoadWebGpuDawnLocked(absl::string_view runtime_library_dir) {
  // First check if a real Dawn implementation is already present in global
  // symbol scope (for example preloaded by the host application). Real
  // `libwebgpu_dawn` exports `wgpu*` only (not `dawnProcSetProcs`), so skip
  // both our own `libdawn_proc` symbols and any other DSO's `libdawn_proc`.
  void* default_create_instance = dlsym(RTLD_DEFAULT, "wgpuCreateInstance");
  if (default_create_instance != nullptr &&
      default_create_instance != reinterpret_cast<void*>(&wgpuCreateInstance) &&
      !IsDawnProcSymbolInSameModule(default_create_instance)) {
    const char* missing_symbol = nullptr;
    if (internal_dawn_loader::PopulateDawnProcTable(&ResolvePosixDefaultSymbol,
                                                    nullptr, &missing_symbol)) {
      ABSL_LOG(INFO)
          << "Resolved WebGPU Dawn symbols from existing process scope.";
      return true;
    }
  }

#if defined(__APPLE__)
  constexpr absl::string_view kDawnLibName = "libwebgpu_dawn.dylib";
#else
  constexpr absl::string_view kDawnLibName = "libwebgpu_dawn.so";
#endif

  std::vector<std::string> candidates;
  if (!runtime_library_dir.empty()) {
    absl::string_view trimmed = absl::StripSuffix(runtime_library_dir, "/");
    candidates.push_back(absl::StrCat(trimmed, "/", kDawnLibName));
  }
  candidates.push_back(std::string(kDawnLibName));
#if defined(__APPLE__)
  candidates.push_back(absl::StrCat("@rpath/", kDawnLibName));
  candidates.push_back(absl::StrCat("@loader_path/", kDawnLibName));
#endif

  for (const std::string& candidate : candidates) {
    dlerror();  // Clear any prior error.
    void* handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      continue;
    }
    const char* missing_symbol = nullptr;
    if (internal_dawn_loader::PopulateDawnProcTable(&ResolvePosixHandleSymbol,
                                                    handle, &missing_symbol)) {
      ABSL_LOG(INFO) << "Loaded WebGPU Dawn runtime from " << candidate;
      return true;
    }
    ABSL_LOG(WARNING) << "Opened " << candidate
                      << ", but required WebGPU symbol was missing: "
                      << (missing_symbol ? missing_symbol : "<unknown>");
    dlclose(handle);
  }
  return false;
}

#endif  // defined(_WIN32)

}  // namespace

bool EnsureWebGpuDawnLoaded(absl::string_view runtime_library_dir) {
  if (g_dawn_loaded.load(std::memory_order_acquire)) {
    return true;
  }
  absl::MutexLock lock(g_dawn_loader_mutex);
  if (g_dawn_loaded.load(std::memory_order_relaxed)) {
    return true;
  }
  if (!LoadWebGpuDawnLocked(runtime_library_dir)) {
    return false;
  }
  g_dawn_loaded.store(true, std::memory_order_release);
  return true;
}

}  // namespace litert::ml_drift

#else  // Static Dawn or Emscripten

namespace litert::ml_drift {

bool EnsureWebGpuDawnLoaded(absl::string_view /*runtime_library_dir*/) {
  return true;
}

}  // namespace litert::ml_drift

#endif  // LITERT_WEBGPU_DAWN_DYNAMIC_LOADER && !__EMSCRIPTEN__
