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

#include "absl/status/status.h"  // from @com_google_absl
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
#include "absl/debugging/leak_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#if defined(_WIN32)
#include "absl/strings/ascii.h"  // from @com_google_absl
#else
#include "absl/strings/strip.h"  // from @com_google_absl
#endif
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "dawn/dawn_proc.h"  // from @dawn
#include "dawn/dawn_proc_table.h"  // from @dawn
#include "dawn/webgpu.h"  // from @dawn
#include "ml_drift_delegate/delegate/webgpu_dawn_procs.inc"

namespace litert::ml_drift {
namespace {

ABSL_CONST_INIT std::atomic<bool> dawn_loaded{false};
ABSL_CONST_INIT absl::Mutex dawn_loader_mutex(absl::kConstInit);

WGPUInstance BootstrapCreateInstance(const WGPUInstanceDescriptor* descriptor) {
  if (!EnsureWebGpuDawnLoaded().ok()) {
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

// Hooks `wgpuCreateInstance` in `libdawn_proc` for standalone `ml_drift/webgpu`
// callers (`ml_drift::webgpu::Instance::Get()`) that do not call
// `EnsureWebGpuDawnLoaded` directly. Safe during static initialization because
// `libdawn_proc`'s `DawnProcTable` and `kDawnVersion` are `constexpr` POD data
// with trivial destruction, and no `dlopen` runs until `wgpuCreateInstance` is
// actually called.
[[maybe_unused]] const bool kBootstrapRegistered = RegisterBootstrapProcTable();

#if defined(_WIN32)

void* ResolveWin32Symbol(const char* symbol_name, void* module_handle) {
  HMODULE module = reinterpret_cast<HMODULE>(module_handle);
  return reinterpret_cast<void*>(GetProcAddress(module, symbol_name));
}

absl::Status TryLoadDawnFromWin32Module(HMODULE module,
                                        absl::string_view source) {
  if (module == nullptr) {
    return absl::NotFoundError("Module handle is null.");
  }
  const char* missing_symbol = nullptr;
  if (!internal_dawn_loader::PopulateDawnProcTable(
          &ResolveWin32Symbol, reinterpret_cast<void*>(module),
          &missing_symbol)) {
    std::string error_message = absl::StrCat(
        "Found ", source, ", but required WebGPU symbol was missing: ",
        missing_symbol ? missing_symbol : "<unknown>");
    ABSL_LOG(WARNING) << error_message;
    return absl::FailedPreconditionError(error_message);
  }
  return absl::OkStatus();
}

bool IsAbsoluteWin32Path(absl::string_view path) {
  if (path.size() >= 3 &&
      absl::ascii_isalpha(static_cast<unsigned char>(path[0])) &&
      path[1] == ':' && path[2] == '\\') {
    return true;
  }
  return path.size() >= 2 && path[0] == '\\' && path[1] == '\\';
}

absl::Status LoadWebGpuDawnLocked(absl::string_view runtime_library_dir) {
  absl::Status last_symbol_error = absl::OkStatus();
  HMODULE existing = nullptr;
  if (GetModuleHandleExA(0, "webgpu_dawn.dll", &existing)) {
    absl::Status status =
        TryLoadDawnFromWin32Module(existing, "loaded webgpu_dawn.dll");
    if (status.ok()) {
      // Retain the reference acquired by `GetModuleHandleExA` so the DLL stays
      // loaded for the lifetime of the process.
      return absl::OkStatus();
    }
    last_symbol_error = status;
    FreeLibrary(existing);
  }

  std::vector<std::string> candidates;
  if (!runtime_library_dir.empty()) {
    // Avoids <filesystem> binary-size overhead in mobile LiteRT builds.
    std::string normalized_dir(runtime_library_dir);
    std::replace(normalized_dir.begin(), normalized_dir.end(), '/', '\\');
    while (!normalized_dir.empty() && normalized_dir.back() == '\\') {
      normalized_dir.pop_back();
    }
    candidates.push_back(absl::StrCat(normalized_dir, "\\webgpu_dawn.dll"));
  }
  candidates.push_back("webgpu_dawn.dll");

  for (const std::string& candidate : candidates) {
    HMODULE handle = nullptr;
    {
      absl::LeakCheckDisabler leak_disabler;
      handle = IsAbsoluteWin32Path(candidate)
                   ? LoadLibraryExA(candidate.c_str(), nullptr,
                                    LOAD_WITH_ALTERED_SEARCH_PATH)
                   : LoadLibraryA(candidate.c_str());
    }
    if (handle == nullptr) {
      continue;
    }
    absl::Status status = TryLoadDawnFromWin32Module(handle, candidate);
    if (status.ok()) {
      ABSL_LOG(INFO) << "Loaded WebGPU Dawn runtime from " << candidate;
      return absl::OkStatus();
    }
    last_symbol_error = status;
    FreeLibrary(handle);
  }
  if (!last_symbol_error.ok()) {
    return last_symbol_error;
  }
  return absl::NotFoundError(
      "WebGPU Dawn runtime library (webgpu_dawn.dll) not found.");
}

#else  // !defined(_WIN32)

void* ResolvePosixHandleSymbol(const char* symbol_name, void* library_handle) {
  return dlsym(library_handle, symbol_name);
}

void* ResolvePosixDefaultSymbol(const char* symbol_name,
                                void* /*unused_context*/) {
  return dlsym(RTLD_DEFAULT, symbol_name);
}

// Returns true if `wgpu_symbol` resides in the same loaded module as
// `dawnProcSetProcs` (identifying a `libdawn_proc` dispatch wrapper rather than
// a real `libwebgpu_dawn` implementation, so we do not bind `DawnProcTable`
// recursively to `libdawn_proc`).
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

absl::Status LoadWebGpuDawnLocked(absl::string_view runtime_library_dir) {
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
      return absl::OkStatus();
    }
  }

#if defined(__APPLE__)
  constexpr absl::string_view kDawnLibName = "libwebgpu_dawn.dylib";
#else
  constexpr absl::string_view kDawnLibName = "libwebgpu_dawn.so";
#endif

  std::vector<std::string> candidates;
  if (!runtime_library_dir.empty()) {
    // Avoids <filesystem> binary-size overhead in mobile LiteRT builds.
    absl::string_view trimmed = absl::StripSuffix(runtime_library_dir, "/");
    candidates.push_back(absl::StrCat(trimmed, "/", kDawnLibName));
  }
  candidates.push_back(std::string(kDawnLibName));
#if defined(__APPLE__)
  candidates.push_back(absl::StrCat("@rpath/", kDawnLibName));
  candidates.push_back(absl::StrCat("@loader_path/", kDawnLibName));
#endif

  absl::Status last_symbol_error = absl::OkStatus();
  for (const std::string& candidate : candidates) {
    dlerror();  // Clear any prior error.
    void* handle = nullptr;
    {
      absl::LeakCheckDisabler leak_disabler;
      handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
    }
    if (handle == nullptr) {
      continue;
    }
    const char* missing_symbol = nullptr;
    if (internal_dawn_loader::PopulateDawnProcTable(&ResolvePosixHandleSymbol,
                                                    handle, &missing_symbol)) {
      ABSL_LOG(INFO) << "Loaded WebGPU Dawn runtime from " << candidate;
      return absl::OkStatus();
    }
    std::string error_message = absl::StrCat(
        "Opened ", candidate, ", but required WebGPU symbol was missing: ",
        missing_symbol ? missing_symbol : "<unknown>");
    ABSL_LOG(WARNING) << error_message;
    last_symbol_error = absl::FailedPreconditionError(error_message);
    dlclose(handle);
  }
  if (!last_symbol_error.ok()) {
    return last_symbol_error;
  }
  return absl::NotFoundError(absl::StrCat("WebGPU Dawn runtime library (",
                                          kDawnLibName, ") not found."));
}

#endif  // defined(_WIN32)

}  // namespace

absl::Status EnsureWebGpuDawnLoaded(absl::string_view runtime_library_dir) {
  if (dawn_loaded.load(std::memory_order_acquire)) {
    return absl::OkStatus();
  }
  absl::MutexLock lock(dawn_loader_mutex);
  if (dawn_loaded.load(std::memory_order_relaxed)) {
    return absl::OkStatus();
  }
  absl::Status status = LoadWebGpuDawnLocked(runtime_library_dir);
  if (!status.ok()) {
    return status;
  }
  dawn_loaded.store(true, std::memory_order_release);
  return absl::OkStatus();
}

}  // namespace litert::ml_drift

#else  // Static Dawn or Emscripten

namespace litert::ml_drift {

absl::Status EnsureWebGpuDawnLoaded(absl::string_view /*runtime_library_dir*/) {
  return absl::OkStatus();
}

}  // namespace litert::ml_drift

#endif  // LITERT_WEBGPU_DAWN_DYNAMIC_LOADER && !__EMSCRIPTEN__
