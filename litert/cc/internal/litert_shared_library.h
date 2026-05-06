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
#include "litert/c/internal/litert_logging.h"  // IWYU pragma: keep
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

#if !LITERT_WINDOWS_OS
#include <dlfcn.h>

#if defined(_GNU_SOURCE) && !defined(__ANDROID__) && !defined(__APPLE__)
#define LITERT_IMPLEMENT_SHARED_LIBRARY_INFO 1
#include <link.h>
#endif

#include "litert/cc/litert_common.h"
#endif

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if !LITERT_WINDOWS_OS &&                                                    \
    (defined(__SANITIZE_ADDRESS__) ||                                        \
     (__has_feature(address_sanitizer) || __has_feature(memory_sanitizer) || \
      __has_feature(thread_sanitizer)))
#define LITERT_SANITIZER_BUILD 1
#endif

#if LITERT_WINDOWS_OS
#include "litert/cc/internal/litert_shared_library_windows.h"
#endif  // LITERT_WINDOWS_OS

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
  constexpr RtldFlags& Global() {
#if defined(RTLD_GLOBAL)
    flags |= RTLD_GLOBAL;
#endif
    return *this;
  }
  constexpr RtldFlags& Local() {
#if defined(RTLD_LOCAL)
    flags |= RTLD_LOCAL;
#endif
    return *this;
  }
  constexpr RtldFlags& NoDelete() {
#if defined(RTLD_NODELETE)
    flags |= RTLD_NODELETE;
#endif
    return *this;
  }
  constexpr RtldFlags& NoLoad() {
#if defined(RTLD_NOLOAD)
    flags |= RTLD_NOLOAD;
#endif
    return *this;
  }
  constexpr RtldFlags& DeepBind() {
#if defined(RTLD_DEEPBIND)
    flags |= RTLD_DEEPBIND;
#endif
    return *this;
  }
};

#if LITERT_SANITIZER_BUILD && defined(RTLD_DEEPBIND)
namespace internal::shared_library_detail {
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
}  // namespace internal::shared_library_detail
#else
namespace internal::shared_library_detail {
inline RtldFlags SanitizeFlagsInCaseOfAsan(RtldFlags flags) { return flags; }
}  // namespace internal::shared_library_detail
#endif

namespace internal::shared_library_detail {

inline const char* DlError() {
#if LITERT_WINDOWS_OS
  return shared_library_windows::DlError();
#else
  return dlerror();
#endif
}

inline void* DlOpen(const char* filename, int flags) {
#if LITERT_WINDOWS_OS
  return shared_library_windows::DlOpen(filename, flags);
#else
  return dlopen(filename, flags);
#endif
}

inline void DlClose(void* handle) {
#if LITERT_WINDOWS_OS
  shared_library_windows::DlClose(handle);
#else
  dlclose(handle);
#endif
}

inline void* DlSym(void* handle, const char* symbol) {
#if LITERT_WINDOWS_OS
  return shared_library_windows::DlSym(handle, symbol);
#else
  return dlsym(handle, symbol);
#endif
}

}  // namespace internal::shared_library_detail

/// @brief Wraps a dynamically loaded shared library to provide RAII semantics.
class SharedLibrary {
 public:
  SharedLibrary() = default;
  SharedLibrary(const SharedLibrary&) = delete;
  SharedLibrary& operator=(const SharedLibrary&) = delete;
  SharedLibrary(SharedLibrary&& other) noexcept
      : handle_kind_(other.handle_kind_),
        path_(std::move(other.path_)),
        handle_(other.handle_) {
    other.handle_kind_ = HandleKind::kInvalid;
    other.handle_ = nullptr;
  }
  SharedLibrary& operator=(SharedLibrary&& other) noexcept {
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
  static absl::string_view DlError() noexcept { return DlErrorImpl(); }

  friend std::ostream& operator<<(std::ostream& os, const SharedLibrary& lib);

  bool Loaded() const noexcept { return handle_kind_ != HandleKind::kInvalid; }

  /// @brief Unloads the shared library.
  ///
  /// @note This is automatically called when the object is destroyed.
  void Close() noexcept {
    if (handle_kind_ == HandleKind::kPath) {
      internal::shared_library_detail::DlClose(handle_);
    }
    handle_kind_ = HandleKind::kInvalid;
    path_.clear();
  }

  /// @brief Looks up a symbol in the shared library.
  ///
  /// @note This takes a `char*` because the underlying system call requires a
  /// null-terminated string, which a `string_view` does not guarantee.
  template <class T>
  Expected<T> LookupSymbol(const char* symbol) const noexcept {
    static_assert(std::is_pointer_v<T>,
                  "The template parameter should always be a pointer.");
    LITERT_ASSIGN_OR_RETURN(void* const raw_symbol, LookupSymbolImpl(symbol));
    return reinterpret_cast<T>(raw_symbol);
  }

  /// @brief Returns the loaded library path.
  const std::string& Path() const noexcept { return path_; }

  /// @brief Returns the underlying shared library handle.
  ///
  /// @warning Some special handle values may be `NULL`. Do not rely on this
  /// value to check whether a library is loaded.
  const void* Handle() const noexcept { return handle_; }
  void* Handle() noexcept { return handle_; }

 private:
  enum class HandleKind { kInvalid, kPath, kRtldNext, kRtldDefault };
  static absl::string_view DlErrorImpl() noexcept {
    const char* error = internal::shared_library_detail::DlError();
    if (!error) {
      return {};
    }
    return error;
  }

  static Expected<SharedLibrary> LoadImpl(HandleKind handle_kind,
                                          absl::string_view path,
                                          RtldFlags flags) {
    SharedLibrary lib;
    switch (handle_kind) {
      case HandleKind::kInvalid:
        return Error(
            Status::kErrorDynamicLoading,
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
          lib.handle_ = internal::shared_library_detail::DlOpen(
              lib.Path().c_str(),
              internal::shared_library_detail::SanitizeFlagsInCaseOfAsan(
                  flags));
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

  Expected<void*> LookupSymbolImpl(const char* symbol_name) const {
    void* symbol = internal::shared_library_detail::DlSym(handle_, symbol_name);

    if (!symbol) {
      return Error(Status::kErrorDynamicLoading,
                   absl::StrFormat("Could not load symbol %s: %s.", symbol_name,
                                   DlError()));
    }
    return symbol;
  }

  HandleKind handle_kind_ = HandleKind::kInvalid;
  std::string path_;
  void* handle_ = nullptr;
};

inline std::ostream& operator<<(std::ostream& os, const SharedLibrary& lib) {
  static constexpr absl::string_view kHeader = "/// DLL Info ///\n";
  static constexpr absl::string_view kFooter = "////////////////\n";

  if (lib.handle_ == nullptr) {
    os << kHeader << "Handle is nullptr.\n" << kFooter;
    return os;
  }

  os << kHeader;
#ifdef RTLD_DI_LMID
  if (Lmid_t dl_ns_idx; dlinfo(lib.handle_, RTLD_DI_LMID, &dl_ns_idx) != 0) {
    os << "Error getting lib namespace index: "
       << internal::shared_library_detail::DlError() << ".\n";
  } else {
    os << "LIB NAMESPACE INDEX: " << dl_ns_idx << "\n";
  }
#else
  os << "Cannot retrieve namespace index on this platform.\n";
#endif

#ifdef RTLD_DI_LINKMAP
  if (link_map* lm; dlinfo(lib.handle_, RTLD_DI_LINKMAP, &lm) != 0) {
    os << "Error getting linked objects: "
       << internal::shared_library_detail::DlError() << ".\n";
  } else {
    os << "LINKED OBJECTS:\n";
    const link_map* link = lm;
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

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_SHARED_LIBRARY_H_
