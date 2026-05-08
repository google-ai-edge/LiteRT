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

#include "litert/core/dynamic_loading.h"

#if LITERT_WINDOWS_OS
#include <windows.h>
#endif

#include <dlfcn.h>

#if !LITERT_WINDOWS_OS
#include <unistd.h>
#endif

// clang-format off
#ifndef __ANDROID__
#if !LITERT_WINDOWS_OS && __has_include(<link.h>)
#include <link.h>
#endif
#endif
// clang-format on

#include <cstdlib>
#include <filesystem>  // NOLINT
#include <string>
#include <vector>

#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/filesystem.h"

namespace litert::internal {

namespace {

static constexpr absl::string_view kLdLibraryPath = "LD_LIBRARY_PATH";

bool EnvPathContains(absl::string_view path, absl::string_view var_value) {
  return absl::EndsWith(var_value, path) ||
         absl::StrContains(var_value, absl::StrCat(path, ":"));
}

}  // namespace

#if LITERT_WINDOWS_OS
static constexpr absl::string_view kSharedLibExt = ".dll";
static constexpr absl::string_view kSharedLibExtFallback = ".so";
#else
static constexpr absl::string_view kSharedLibExt = ".so";
#endif

LiteRtStatus FindLiteRtSharedLibsHelper(const std::string& search_path,
                                        const std::string& lib_pattern,
                                        bool full_match,
                                        std::vector<std::string>& results) {
  if (!Exists(search_path)) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  // TODO implement path glob in core/filesystem.h and remove filesystem
  // include from this file.
  for (const auto& entry : std::filesystem::directory_iterator(
           search_path,
           std::filesystem::directory_options::skip_permission_denied)) {
    const auto& path = entry.path();
    if (entry.is_regular_file()) {
      if (full_match) {
        if (path.string().find(lib_pattern) != -1) {
          LITERT_LOG(LITERT_VERBOSE, "Found shared library: %s", path.c_str());
          results.push_back(path);
        }
      } else {
        const auto stem = path.stem().string();
        const auto ext = path.extension().string();
        bool match = (stem.find(lib_pattern) == 0);
#if LITERT_WINDOWS_OS
        match &= (ext == kSharedLibExt || ext == kSharedLibExtFallback);
#else
        match &= (ext == kSharedLibExt);
#endif
        if (match) {
          LITERT_LOG(LITERT_VERBOSE, "Found shared library: %s", path.c_str());
          results.push_back(path);
        }
      }
    } else if (entry.is_directory()) {
      FindLiteRtSharedLibsHelper(path, lib_pattern, full_match, results);
    }
  }

  return kLiteRtStatusOk;
}

static const char kCompilerPluginLibPatternFmt[] = "CompilerPlugin";

LiteRtStatus FindLiteRtCompilerPluginSharedLibs(
    absl::string_view search_path, std::vector<std::string>& results) {
  std::string root(search_path);
  const std::string lib_pattern =
      absl::StrCat(kLiteRtSharedLibPrefix, kCompilerPluginLibPatternFmt);
  return FindLiteRtSharedLibsHelper(root, lib_pattern, /*full_match=*/false,
                                    results);
}

static const char kDispatchLibPatternFmt[] = "Dispatch";

LiteRtStatus FindLiteRtDispatchSharedLibs(absl::string_view search_path,
                                          std::vector<std::string>& results) {
  std::string root(search_path.data(), search_path.size());
  const std::string lib_pattern =
      absl::StrCat(kLiteRtSharedLibPrefix, kDispatchLibPatternFmt);
  return FindLiteRtSharedLibsHelper(root, lib_pattern, /*full_match=*/false,
                                    results);
}

LiteRtStatus PutLibOnLdPath(absl::string_view search_path,
                            absl::string_view lib_pattern) {
  std::vector<std::string> results;
  LITERT_RETURN_IF_ERROR(FindLiteRtSharedLibsHelper(
      std::string(search_path), std::string(lib_pattern), true, results));
  if (results.empty()) {
    LITERT_LOG(LITERT_INFO, "No match found in %s", search_path.data());
    return kLiteRtStatusOk;
  }

  const auto lib_dir = std::filesystem::path(results[0]).parent_path().string();
  absl::string_view ld =
      absl::NullSafeStringView(getenv(kLdLibraryPath.data()));

  if (EnvPathContains(lib_dir, ld)) {
    LITERT_LOG(LITERT_INFO, "dir already in LD_LIBRARY_PATH");
    return kLiteRtStatusOk;
  }

  std::string new_ld;
  if (ld.empty()) {
    new_ld = lib_dir;
  } else {
    new_ld = absl::StrCat(ld, ":", lib_dir);
  }

  LITERT_LOG(LITERT_INFO, "Adding %s to LD_LIBRARY_PATH", new_ld.c_str());
#if LITERT_WINDOWS_OS
  _putenv_s(kLdLibraryPath.data(), new_ld.c_str());
#else
  setenv(kLdLibraryPath.data(), new_ld.c_str(), /*overwrite=*/1);
#endif

  return kLiteRtStatusOk;
}

}  // namespace litert::internal
