// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/backend_utils.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "absl/strings/string_view.h"  // from @com_google_absl

namespace qnn {

std::optional<std::string> WriteTemporaryFile(absl::string_view contents) {
#if defined(_WIN32)
  char temp_dir[MAX_PATH];
  const DWORD temp_dir_length = GetTempPathA(MAX_PATH, temp_dir);
  if (temp_dir_length == 0 || temp_dir_length >= MAX_PATH) return std::nullopt;
  char path[MAX_PATH];
  if (GetTempFileNameA(temp_dir, "lrt", 0, path) == 0) return std::nullopt;
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) {
    std::error_code error;
    std::filesystem::remove(path, error);
    return std::nullopt;
  }
  file.write(contents.data(), contents.size());
  file.close();
  if (!file) {
    std::error_code error;
    std::filesystem::remove(path, error);
    return std::nullopt;
  }
  return std::string(path);
#else
  std::error_code error;
  const auto temp_dir = std::filesystem::temp_directory_path(error);
  if (error) return std::nullopt;
  std::string path_template =
      (temp_dir / "litert_backend_utils_XXXXXX").string();
  std::vector<char> path(path_template.begin(), path_template.end());
  path.push_back('\0');
  const int fd = mkstemp(path.data());
  if (fd < 0) return std::nullopt;
  std::FILE* file = fdopen(fd, "wb");
  if (file == nullptr) {
    close(fd);
    std::filesystem::remove(path.data(), error);
    return std::nullopt;
  }
  const size_t written = std::fwrite(contents.data(), 1, contents.size(), file);
  if (std::fclose(file) != 0 || written != contents.size()) {
    std::filesystem::remove(path.data(), error);
    return std::nullopt;
  }
  return std::string(path.data());
#endif
}

}  // namespace qnn
