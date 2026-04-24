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

#ifndef ODML_LITERT_LITERT_CC_LITERT_SHARED_LIBRARY_H_
#define ODML_LITERT_LITERT_CC_LITERT_SHARED_LIBRARY_H_

#include <ostream>
#include <string>
#include <string_view>
#include <utility>

#include "absl/debugging/leak_check.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h" // IWYU pragma: keep
#include "litert/c/litert_common.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

#if !LITERT_WINDOWS_OS
#include "litert/cc/litert_common.h"
#include <dlfcn.h>
#endif

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if !LITERT_WINDOWS_OS &&                                                      \
    (defined(__SANITIZE_ADDRESS__) ||                                          \
     (__has_feature(address_sanitizer) || __has_feature(memory_sanitizer) ||   \
      __has_feature(thread_sanitizer)))
#define LITERT_SANITIZER_BUILD 1
#endif

#if LITERT_WINDOWS_OS
#include <windows.h>

#include <cctype>

namespace {

thread_local std::string g_last_error;

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

inline const char *dlerror() {
  return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

inline void *dlopen(const char *filename, int flags) {
  g_last_error.clear();

  if (!filename) {
    return GetModuleHandle(NULL);
  }

  std::string dll_name(filename);
  size_t pos = dll_name.rfind(".so");
  if (pos != std::string::npos && pos == dll_name.length() - 3) {
    dll_name.replace(pos, 3, ".dll");
  }

  HMODULE handle = LoadLibraryA(dll_name.c_str());
  if (!handle) {
    DWORD error = GetLastError();
    g_last_error = "Failed to load library '" + dll_name +
                   "': " + GetWindowsErrorString(error);
  }

  return handle;
}

inline void dlclose(void *handle) {
  if (handle && handle != GetModuleHandle(NULL)) {
    FreeLibrary(static_cast<HMODULE>(handle));
  }
}

inline void *dlsym(void *handle, const char *symbol) {
  g_last_error.clear();

  if (!handle || !symbol) {
    g_last_error = "Invalid handle or symbol name";
    return nullptr;
  }

  if (handle == ((void *)-1) || handle == ((void *)0)) {
    handle = GetModuleHandle(NULL);
  }

  void *address = GetProcAddress(static_cast<HMODULE>(handle), symbol);
  if (!address) {
    DWORD error = GetLastError();
    g_last_error = "Failed to find symbol '" + std::string(symbol) +
                   "': " + GetWindowsErrorString(error);
  }

  return address;
}

} // namespace

#define RTLD_LAZY 0x00001
#define RTLD_NOW 0x00002
#define RTLD_BINDING_MASK 0x3
#define RTLD_NOLOAD 0x00004
#define RTLD_DEEPBIND 0x00008
#define RTLD_GLOBAL 0x00100
#define RTLD_LOCAL 0
#define RTLD_NODELETE 0x01000

#define RTLD_NEXT ((void *)-1)
#define RTLD_DEFAULT ((void *)0)

#endif // LITERT_WINDOWS_OS

/// @file
/// @brief Defines a C++ wrapper for dynamically loaded shared libraries.

namespace litert {

struct RtldFlags {
  int flags;

  static constexpr struct NextTag {
  } kNext = {};
  static constexpr struct DefaultTag {
  } kDefault = {};

  // NOLINTNEXTLINE(*-explicit-constructor): we want this to be passed as flags.
  operator int() { return flags; }

  static constexpr RtldFlags Lazy() {
    return {
#if defined(RTLD_LAZY)
        RTLD_LAZY
#endif
    };
  }
  static constexpr RtldFlags Now() {
    return {
#if defined(RTLD_NOW)
        RTLD_NOW
#endif
    };
  }
  static constexpr RtldFlags Default() { return Lazy().Local().DeepBind(); }
  constexpr RtldFlags &Global() {
#if defined(RTLD_GLOBAL)
    flags |= RTLD_GLOBAL;
#endif
    return *this;
  }
  constexpr RtldFlags &Local() {
#if defined(RTLD_LOCAL)
    flags |= RTLD_LOCAL;
#endif
    return *this;
  }
  constexpr RtldFlags &NoDelete() {
#if defined(RTLD_NODELETE)
    flags |= RTLD_NODELETE;
#endif
    return *this;
  }
  constexpr RtldFlags &NoLoad() {
#if defined(RTLD_NOLOAD)
    flags |= RTLD_NOLOAD;
#endif
    return *this;
  }
  constexpr RtldFlags &DeepBind() {
#if defined(RTLD_DEEPBIND)
    flags |= RTLD_DEEPBIND;
#endif
    return *this;
  }
};

#if LITERT_SANITIZER_BUILD && defined(RTLD_DEEPBIND)
namespace {
inline RtldFlags SanitizeFlagsInCaseOfAsan(RtldFlags flags) {
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
} // namespace
#else
#define SanitizeFlagsInCaseOfAsan(flags) (flags)
#endif

/// @brief Wraps a dynamically loaded shared library to provide RAII semantics.
class SharedLibrary {
public:
  SharedLibrary() = default;
  SharedLibrary(const SharedLibrary &) = delete;
  SharedLibrary &operator=(const SharedLibrary &) = delete;
  SharedLibrary(SharedLibrary &&other) noexcept
      : handle_kind_(other.handle_kind_), path_(std::move(other.path_)),
        handle_(other.handle_) {
    other.handle_kind_ = HandleKind::kInvalid;
    other.handle_ = nullptr;
  }
  SharedLibrary &operator=(SharedLibrary &&other) noexcept {
    Close();
    handle_kind_ = other.handle_kind_;
    path_ = std::move(other.path_);
    handle_ = other.handle_;
    other.handle_kind_ = HandleKind::kInvalid;
    other.handle_ = nullptr;
    return *this;
  }
  ~SharedLibrary() noexcept { Close(); }

  /// @brief Loads the library at the given path.
  static Expected<SharedLibrary> Load(absl::string_view path,
                                      RtldFlags flags) noexcept {
    return LoadImpl(HandleKind::kPath, path, flags);
  }

#ifdef LITERT_NO_ABSL
  static Expected<SharedLibrary> Load(std::string_view path,
                                      RtldFlags flags) noexcept {
    return Load(internal::ToAbslStringView(path), flags);
  }
#endif // LITERT_NO_ABSL

  /// @brief Loads the library as the `RTLD_NEXT` special handle.
  static Expected<SharedLibrary> Load(RtldFlags::NextTag) noexcept {
    return LoadImpl(HandleKind::kRtldNext, "", RtldFlags{});
  }

  /// @brief Loads the library as the `RTLD_DEFAULT` special handle.
  static Expected<SharedLibrary> Load(RtldFlags::DefaultTag) noexcept {
    return LoadImpl(HandleKind::kRtldDefault, "", RtldFlags{});
  }

  /// @brief Gets the last shared library operation error, if any.
  ///
  /// If there was no error, returns an empty view.
#ifdef LITERT_NO_ABSL
  static std::string_view DlError() noexcept {
    return internal::ToStdStringView(DlErrorImpl());
  }
#else
  static absl::string_view DlError() noexcept { return DlErrorImpl(); }
#endif

  friend std::ostream &operator<<(std::ostream &os, const SharedLibrary &lib);

  bool Loaded() const noexcept { return handle_kind_ != HandleKind::kInvalid; }

  /// @brief Unloads the shared library.
  ///
  /// @note This is automatically called when the object is destroyed.
  void Close() noexcept {
    if (handle_kind_ == HandleKind::kPath) {
      dlclose(handle_);
    }
    handle_kind_ = HandleKind::kInvalid;
    path_.clear();
  }

  /// @brief Looks up a symbol in the shared library.
  ///
  /// @note This takes a `char*` because the underlying system call requires a
  /// null-terminated string, which a `string_view` does not guarantee.
  template <class T>
  Expected<T> LookupSymbol(const char *symbol) const noexcept {
    static_assert(std::is_pointer_v<T>,
                  "The template parameter should always be a pointer.");
    LITERT_ASSIGN_OR_RETURN(void *const raw_symbol, LookupSymbolImpl(symbol));
    return reinterpret_cast<T>(raw_symbol);
  }

  /// @brief Returns the loaded library path.
  const std::string &Path() const noexcept { return path_; }

  /// @brief Returns the underlying shared library handle.
  ///
  /// @warning Some special handle values may be `NULL`. Do not rely on this
  /// value to check whether a library is loaded.
  const void *Handle() const noexcept { return handle_; }
  void *Handle() noexcept { return handle_; }

private:
  enum class HandleKind { kInvalid, kPath, kRtldNext, kRtldDefault };
  static absl::string_view DlErrorImpl() noexcept {
    const char *error = dlerror();
    if (!error) {
      return {};
    }
    return error;
  }

  static Expected<SharedLibrary>
  LoadImpl(HandleKind handle_kind, absl::string_view path, RtldFlags flags) {
    SharedLibrary lib;
    switch (handle_kind) {
    case HandleKind::kInvalid:
      return Error(Status::kErrorDynamicLoading,
                   "This is a logic error. LoadImpl should not be called with "
                   "HandleKind::kInvalid");
    case HandleKind::kPath:
      if (path.empty()) {
        return Error(Status::kErrorDynamicLoading,
                     "Cannot not load shared library: empty path.");
      }
      lib.path_ = path;
      {
        absl::LeakCheckDisabler disabler;
        lib.handle_ =
            dlopen(lib.Path().c_str(), SanitizeFlagsInCaseOfAsan(flags));
      }
      if (!lib.handle_) {
        return Error(Status::kErrorDynamicLoading,
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

  Expected<void *> LookupSymbolImpl(const char *symbol_name) const {
    void *symbol = dlsym(handle_, symbol_name);

    if (!symbol) {
      return Error(Status::kErrorDynamicLoading,
                   absl::StrFormat("Could not load symbol %s: %s.", symbol_name,
                                   DlError()));
    }
    return symbol;
  }

  HandleKind handle_kind_ = HandleKind::kInvalid;
  std::string path_;
  void *handle_ = nullptr;
};

inline std::ostream &operator<<(std::ostream &os, const SharedLibrary &lib) {
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
  if (link_map *lm; dlinfo(lib.handle_, RTLD_DI_LINKMAP, &lm) != 0) {
    os << "Error getting linked objects: " << dlerror() << ".\n";
  } else {
    os << "LINKED OBJECTS:\n";
    const link_map *link = lm;
    while (link->l_prev) {
      link = link->l_prev;
    }
    for (; link != nullptr; link = link->l_next) {
      os << (link != lm ? "   " : "***") << link->l_name << "\n";
    }
  }
#else
  os << "Cannot retrieve lib map on this platform.\n";
#endif
  return os << kFooter;
}

} // namespace litert

#endif // ODML_LITERT_LITERT_CC_LITERT_SHARED_LIBRARY_H_
