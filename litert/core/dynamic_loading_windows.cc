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

#include <windows.h>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/core/dynamic_loading.h"
#include "litert/core/filesystem.h"

namespace litert::internal {

namespace {

// Convert forward slashes to backslashes for Windows paths
std::string NormalizePath(absl::string_view path) {
  return absl::StrReplaceAll(path, {{"/", "\\"}});
}

// Convert library name to Windows format
std::string ToWindowsLibName(absl::string_view lib_name) {
  std::string name(lib_name);
  // Remove "lib" prefix if present
  if (absl::StartsWith(name, "lib")) {
    name = name.substr(3);
  }
  // Replace .so with .dll
  if (absl::EndsWith(name, ".so")) {
    name = name.substr(0, name.length() - 3) + ".dll";
  }
  return name;
}

// Get directory name from a path
std::string Dirname(const std::string& path) {
  size_t pos = path.find_last_of("\\/");
  if (pos == std::string::npos) {
    return ".";
  }
  if (pos == 0) {
    return path.substr(0, 1);
  }
  return path.substr(0, pos);
}

}  // namespace

LiteRtStatus FindLiteRtCompilerPluginSharedLibs(
    absl::string_view search_path, std::vector<std::string>& results) {
  const std::string lib_pattern =
      absl::StrCat(ToWindowsLibName(kLiteRtSharedLibPrefix), "CompilerPlugin");
  return FindLiteRtSharedLibsHelper(std::string(search_path), lib_pattern,
                                    /*full_match=*/false, results);
}

LiteRtStatus FindLiteRtDispatchSharedLibs(absl::string_view search_path,
                                          std::vector<std::string>& results) {
  const std::string lib_pattern =
      absl::StrCat(ToWindowsLibName(kLiteRtSharedLibPrefix), "Dispatch");
  return FindLiteRtSharedLibsHelper(std::string(search_path), lib_pattern,
                                    /*full_match=*/false, results);
}

LiteRtStatus FindLiteRtSharedLibsHelper(const std::string& search_path,
                                        const std::string& lib_pattern,
                                        bool full_match,
                                        std::vector<std::string>& results) {
  results.clear();

  std::string normalized_path = NormalizePath(search_path);
  if (!Exists(normalized_path)) {
    LITERT_LOG(LITERT_ERROR, "Search path doesn't exist: %s",
               normalized_path.c_str());
    return kLiteRtStatusErrorInvalidArgument;
  }

  // Prepare search pattern
  std::string search_pattern = normalized_path;
  if (!search_pattern.empty() && search_pattern.back() != '\\') {
    search_pattern += '\\';
  }
  search_pattern += full_match ? lib_pattern : ("*" + lib_pattern + "*.dll");

  WIN32_FIND_DATAA find_data;
  HANDLE find_handle = FindFirstFileA(search_pattern.c_str(), &find_data);

  if (find_handle == INVALID_HANDLE_VALUE) {
    // No matching files found
    return kLiteRtStatusOk;
  }

  do {
    // Skip directories
    if (!(find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
      std::string filename(find_data.cFileName);

      // Apply pattern matching if not using Windows wildcard search
      if (full_match) {
        if (filename == lib_pattern || filename == (lib_pattern + ".dll")) {
          results.push_back(Join({normalized_path, filename}));
        }
      } else {
        // Windows wildcard already filtered for us
        results.push_back(Join({normalized_path, filename}));
      }
    }
  } while (FindNextFileA(find_handle, &find_data));

  FindClose(find_handle);

  std::sort(results.begin(), results.end());
  return kLiteRtStatusOk;
}

LiteRtStatus PutLibOnLdPath(absl::string_view search_path,
                            absl::string_view lib_pattern) {
  std::vector<std::string> results;
  auto status = FindLiteRtSharedLibsHelper(std::string(search_path),
                                           std::string(lib_pattern),
                                           /*full_match=*/true, results);

  if (status != kLiteRtStatusOk) {
    return status;
  }

  if (results.empty()) {
    LITERT_LOG(LITERT_ERROR, "No libraries found matching pattern: %s",
               std::string(lib_pattern).c_str());
    return kLiteRtStatusErrorNotFound;
  }

  // Get the directory of the first matching library
  std::string lib_dir = Dirname(results[0]);

  // Get current PATH
  char* current_path = std::getenv("PATH");
  std::string new_path;

  if (current_path != nullptr) {
    new_path = absl::StrCat(lib_dir, ";", current_path);
  } else {
    new_path = lib_dir;
  }

  // Update PATH environment variable
  if (_putenv_s("PATH", new_path.c_str()) != 0) {
    LITERT_LOG(LITERT_ERROR, "Failed to update PATH environment variable");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  LITERT_LOG(LITERT_INFO, "Added to PATH: %s", lib_dir.c_str());
  return kLiteRtStatusOk;
}

}  // namespace litert::internal
