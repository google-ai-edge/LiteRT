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

#ifndef ODML_LITERT_LITERT_CC_LITERT_SHARED_LIBRARY_WINDOWS_H_
#define ODML_LITERT_LITERT_CC_LITERT_SHARED_LIBRARY_WINDOWS_H_

#if LITERT_WINDOWS_OS

#define RTLD_LAZY 0x00001
#define RTLD_NOW 0x00002
#define RTLD_BINDING_MASK 0x3
#define RTLD_NOLOAD 0x00004
#define RTLD_DEEPBIND 0x00008
#define RTLD_GLOBAL 0x00100
#define RTLD_LOCAL 0
#define RTLD_NODELETE 0x01000

#define RTLD_NEXT ((void*)-1)
#define RTLD_DEFAULT ((void*)0)

#include <windows.h>

#include <cctype>
#include <filesystem>  // NOLINT
#include <string>

namespace litert::internal::shared_library_windows {

inline thread_local std::string g_last_error;

inline std::string GetWindowsErrorString(DWORD error_code) {
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

  while (!message.empty() && std::isspace(message.back())) {
    message.pop_back();
  }

  return message;
}

inline const char* DlError() {
  return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

inline bool IsAbsolutePath(const std::string& path) {
  return std::filesystem::path(path).is_absolute();
}

inline void* DlOpen(const char* filename, int flags) {
  g_last_error.clear();

  if (!filename) {
    return GetModuleHandle(NULL);
  }

  std::filesystem::path requested_path(filename);
  requested_path.make_preferred();
  std::string requested_name = requested_path.string();
  std::string fallback_name = requested_name;
  size_t pos = fallback_name.rfind(".so");
  if (pos != std::string::npos && pos == fallback_name.length() - 3) {
    fallback_name.replace(pos, 3, ".dll");
  }

  const auto load_with_search_path = [](const std::string& path) {
    if (IsAbsolutePath(path)) {
      return LoadLibraryExA(path.c_str(), nullptr,
                            LOAD_WITH_ALTERED_SEARCH_PATH);
    }
    return LoadLibraryA(path.c_str());
  };

  HMODULE handle = load_with_search_path(requested_name);
  if (!handle && fallback_name != requested_name) {
    handle = load_with_search_path(fallback_name);
  }
  if (!handle) {
    DWORD error = GetLastError();
    g_last_error = "Failed to load library '" + requested_name + "'";
    if (fallback_name != requested_name) {
      g_last_error += " or fallback '" + fallback_name + "'";
    }
    g_last_error += ": " + GetWindowsErrorString(error);
  }

  return handle;
}

inline void DlClose(void* handle) {
  if (handle && handle != GetModuleHandle(NULL)) {
    FreeLibrary(static_cast<HMODULE>(handle));
  }
}

inline void* DlSym(void* handle, const char* symbol) {
  g_last_error.clear();

  if (!handle || !symbol) {
    g_last_error = "Invalid handle or symbol name";
    return nullptr;
  }

  if (handle == ((void*)-1) || handle == ((void*)0)) {
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

}  // namespace litert::internal::shared_library_windows

#endif  // LITERT_WINDOWS_OS

#endif  // ODML_LITERT_LITERT_CC_LITERT_SHARED_LIBRARY_WINDOWS_H_
