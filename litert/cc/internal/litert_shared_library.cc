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

#include "litert/cc/internal/litert_shared_library.h"

#if defined(_GNU_SOURCE) && !defined(__ANDROID__) && !defined(__APPLE__)
#define LITERT_IMPLEMENT_SHARED_LIBRARY_INFO 1
#include <link.h>
#endif

#include <ostream>
#include <string>
#include <utility>

#include "absl/debugging/leak_check.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"  // IWYU pragma: keep
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"  // IWYU pragma: keep

#if !LITERT_WINDOWS_OS
#include <dlfcn.h>

// When using an address sanitizer, `RTLD_DEEPBIND` is not supported. When using
// one, we discard the flag and log an error.
#ifndef __has_feature       // Optional of course.
#define __has_feature(x) 0  // Compatibility with non-clang compilers.
#endif
#if defined(__SANITIZE_ADDRESS__) || \
    (__has_feature(address_sanitizer) || __has_feature(memory_sanitizer))
#define LITERT_SANITIZER_BUILD 1
#endif
#endif

#if LITERT_SANITIZER_BUILD && defined(RTLD_DEEPBIND)
namespace litert {
namespace {
RtldFlags SanitizeFlagsInCaseOfAsan(RtldFlags flags) {
  LITERT_LOG(
      LITERT_WARNING,
      "Trying to load a library using `RTLD_DEEPBIND` is not supported by "
      "address sanitizers. In an effort to enable testing we strip the flag. "
      "If this leads to unintended behaviour, either remove the "
      "`RTLD_DEEPBIND` flag or run without an address sanitizer. "
      "See https://github.com/google/sanitizers/issues/611 for more "
      "information.");
  flags.flags &= ~RTLD_DEEPBIND;
  return flags;
}
}  // namespace
}  // namespace litert
#else
#define SanitizeFlagsInCaseOfAsan(flags) (flags)
#endif

#if LITERT_WINDOWS_OS
#include <windows.h>

#include <cctype>
#include <string>

// Windows implementation of dlfcn.h functions
namespace {

// Thread-local storage for last error message
thread_local std::string g_last_error;

// Convert Windows error code to string
std::string GetWindowsErrorString(DWORD error_code) {
  if (error_code == 0) {
    return "No error";
  }

  LPSTR message_buffer = nullptr;
  size_t size = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL, error_code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      (LPSTR)&message_buffer, 0, NULL);

  std::string message(message_buffer, size);
  LocalFree(message_buffer);

  // Remove trailing whitespace
  while (!message.empty() && std::isspace(message.back())) {
    message.pop_back();
  }

  return message;
}

const char* dlerror() {
  return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

void* dlopen(const char* filename, int flags) {
  // Clear previous error
  g_last_error.clear();

  if (!filename) {
    // NULL filename means get handle to main executable
    return GetModuleHandle(NULL);
  }

  // Convert .so extension to .dll if present
  std::string dll_name(filename);
  size_t pos = dll_name.rfind(".so");
  if (pos != std::string::npos && pos == dll_name.length() - 3) {
    dll_name.replace(pos, 3, ".dll");
  }

  // Load the library
  HMODULE handle = LoadLibraryA(dll_name.c_str());
  if (!handle) {
    DWORD error = GetLastError();
    g_last_error = "Failed to load library '" + dll_name +
                   "': " + GetWindowsErrorString(error);
  }

  return handle;
}

void dlclose(void* handle) {
  if (handle && handle != GetModuleHandle(NULL)) {
    FreeLibrary(static_cast<HMODULE>(handle));
  }
}

void* dlsym(void* handle, const char* symbol) {
  // Clear previous error
  g_last_error.clear();

  if (!handle || !symbol) {
    g_last_error = "Invalid handle or symbol name";
    return nullptr;
  }

  // Handle special pseudo-handles
  if (handle == ((void*)-1) || handle == ((void*)0)) {
    // On Windows, we can't easily implement RTLD_NEXT/RTLD_DEFAULT
    // For now, just search in the main executable
    handle = GetModuleHandle(NULL);
  }

  void* address = GetProcAddress(static_cast<HMODULE>(handle), symbol);
  if (!address) {
    DWORD error = GetLastError();
    g_last_error = "Failed to find symbol '" + std::string(symbol) +
                   "': " + GetWindowsErrorString(error);
  }

  return address;
}
}  // namespace

// Define RTLD macros for Windows (outside anonymous namespace for visibility)
// These values are copied directly from POSIX/Linux dlfcn.h for API
// compatibility. Note: Most of these flags are defined for API compatibility
// but have no effect on Windows:
// - RTLD_LAZY/RTLD_NOW: Windows always resolves all symbols at load time
// - RTLD_GLOBAL/RTLD_LOCAL: Windows doesn't have equivalent symbol visibility
// control
// - RTLD_DEEPBIND: Linux-specific, no Windows equivalent
// - RTLD_NOLOAD: Could be simulated with GetModuleHandle but not implemented
// - RTLD_NODELETE: Windows uses reference counting but has no direct equivalent
// - RTLD_BINDING_MASK: Used to extract binding mode (LAZY/NOW) on POSIX, not
// used on Windows
#define RTLD_LAZY 0x00001  // No effect on Windows
#define RTLD_NOW 0x00002   // No effect on Windows (always immediate binding)
#define RTLD_BINDING_MASK 0x3  // Not used on Windows (would extract LAZY/NOW)
#define RTLD_NOLOAD 0x00004    // Not implemented on Windows
#define RTLD_DEEPBIND 0x00008  // No effect on Windows
#define RTLD_GLOBAL 0x00100    // No effect on Windows
#define RTLD_LOCAL 0           // No effect on Windows (default visibility)
#define RTLD_NODELETE 0x01000  // No effect on Windows

// Special pseudo-handles
#define RTLD_NEXT ((void*)-1)    // Simulated: searches main executable
#define RTLD_DEFAULT ((void*)0)  // Simulated: searches main executable

#endif  // LITERT_WINDOWS_OS

namespace litert {

SharedLibrary::~SharedLibrary() noexcept { Close(); }

SharedLibrary::SharedLibrary(SharedLibrary&& other) noexcept
    : handle_kind_(other.handle_kind_),
      path_(std::move(other.path_)),
      handle_(other.handle_) {
  other.handle_kind_ = HandleKind::kInvalid;
  other.handle_ = nullptr;
}

SharedLibrary& SharedLibrary::operator=(SharedLibrary&& other) noexcept {
  Close();
  handle_kind_ = other.handle_kind_;
  path_ = std::move(other.path_);
  handle_ = other.handle_;
  other.handle_kind_ = HandleKind::kInvalid;
  other.handle_ = nullptr;
  return *this;
}

void SharedLibrary::Close() noexcept {
  if (handle_kind_ == HandleKind::kPath) {
    dlclose(handle_);
  }
  handle_kind_ = HandleKind::kInvalid;
  path_.clear();
}

absl::string_view SharedLibrary::DlError() noexcept {
  const char* error = dlerror();
  if (!error) {
    return {};
  }
  return error;
}

Expected<SharedLibrary> SharedLibrary::LoadImpl(
    SharedLibrary::HandleKind handle_kind, absl::string_view path,
    RtldFlags flags) {
  SharedLibrary lib;
  switch (handle_kind) {
    case HandleKind::kInvalid:
      return Error(kLiteRtStatusErrorDynamicLoading,
                   "This is a logic error. LoadImpl should not be called with "
                   "HandleKind::kInvalid");
    case HandleKind::kPath:
      if (path.empty()) {
        return Error(kLiteRtStatusErrorDynamicLoading,
                     "Cannot not load shared library: empty path.");
      }
      lib.path_ = path;
      {
        absl::LeakCheckDisabler disabler;
        lib.handle_ =
            dlopen(lib.Path().c_str(), SanitizeFlagsInCaseOfAsan(flags));
      }
      if (!lib.handle_) {
        return Error(kLiteRtStatusErrorDynamicLoading,
                     absl::StrFormat("Could not load shared library %s: %s.",
                                     lib.path_, DlError()));
      }
      break;
    case HandleKind::kRtldNext:
      lib.handle_ = RTLD_NEXT;
      break;
    case HandleKind::kRtldDefault:
      lib.handle_ = RTLD_DEFAULT;
      break;
  }
  lib.handle_kind_ = handle_kind;
  return lib;
}

Expected<void*> SharedLibrary::LookupSymbolImpl(const char* symbol_name) const {
  void* symbol = dlsym(handle_, symbol_name);

  if (!symbol) {
    return Error(kLiteRtStatusErrorDynamicLoading,
                 absl::StrFormat("Could not load symbol %s: %s.", symbol_name,
                                 DlError()));
  }
  return symbol;
}

std::ostream& operator<<(std::ostream& os, const SharedLibrary& lib) {
  static constexpr absl::string_view kHeader = "/// DLL Info ///\n";
  static constexpr absl::string_view kFooter = "////////////////\n";

  if (lib.handle_ == nullptr) {
    os << kHeader << "Handle is nullptr.\n" << kFooter;
    return os;
  }

  os << kHeader;
#ifdef RTLD_DI_LMID
  if (Lmid_t dl_ns_idx; dlinfo(lib.handle_, RTLD_DI_LMID, &dl_ns_idx) != 0) {
    os << "Error getting lib namespace index: " << dlerror() << ".\n";
  } else {
    os << "LIB NAMESPACE INDEX: " << dl_ns_idx << "\n";
  }
#else
  os << "Cannot retrieve namespace index on this platform.\n";
#endif

#ifdef RTLD_DI_LINKMAP
  if (link_map* lm; dlinfo(lib.handle_, RTLD_DI_LINKMAP, &lm) != 0) {
    os << "Error getting linked objects: " << dlerror() << ".\n";
  } else {
    os << "LINKED OBJECTS:\n";
    // Rewind to the start of the linked list.
    const link_map* link = lm;
    while (link->l_prev) {
      link = link->l_prev;
    }
    // Print all list elements
    for (; link != nullptr; link = link->l_next) {
      os << (link != lm ? "   " : "***") << link->l_name << "\n";
    }
  }
#else
  os << "Cannot retrieve lib map on this platform.\n";
#endif
  return os << kFooter;
}

}  // namespace litert
