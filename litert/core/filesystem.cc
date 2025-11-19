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

#include "litert/core/filesystem.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <string>
#include <system_error> // NOLINT
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"

namespace litert::internal {

namespace {

using StdPath = std::filesystem::path;

StdPath MakeStdPath(absl::string_view path) {
  return StdPath(std::string(path.begin(), path.end()));
}

bool StdExists(const StdPath& std_path) {
  return std::filesystem::exists(std_path);
}

size_t StdSize(const StdPath& std_path) {
  return std::filesystem::file_size(std_path);
}

LiteRtStatus StdIFRead(const StdPath& std_path, char* data, size_t size) {
  std::ifstream in_file_stream(std_path, std::ifstream::binary);
  if (!in_file_stream) {
    return kLiteRtStatusErrorFileIO;
  }

  in_file_stream.read(data, size);
  if (!in_file_stream) {
    return kLiteRtStatusErrorFileIO;
  }

  in_file_stream.close();
  return kLiteRtStatusOk;
}

}  // namespace

void Touch(absl::string_view path) { std::ofstream(MakeStdPath(path)); }

std::string Join(const std::vector<absl::string_view>& paths) {
  StdPath std_path;
  for (auto subpath : paths) {
    std_path /= MakeStdPath(subpath);
  }
  return std_path.generic_string();
}

std::string Stem(absl::string_view path) {
  return MakeStdPath(path).stem().generic_string();
}

bool Exists(absl::string_view path) { return StdExists(MakeStdPath(path)); }

Expected<size_t> Size(absl::string_view path) {
  auto std_path = MakeStdPath(path);
  if (!StdExists(std_path)) {
    return Error(kLiteRtStatusErrorNotFound,
                 absl::StrFormat("File not found: %s", std_path.c_str()));
  }
  return StdSize(std_path);
}

Expected<OwningBufferRef<uint8_t>> LoadBinaryFile(absl::string_view path) {
  auto std_path = MakeStdPath(path);

  if (!StdExists(std_path)) {
    return Error(kLiteRtStatusErrorNotFound,
                 absl::StrFormat("File not found: %s", std_path.c_str()));
  }

  OwningBufferRef<uint8_t> buf(StdSize(std_path));
  if (auto status = StdIFRead(std_path, buf.StrData(), buf.Size());
      status != kLiteRtStatusOk) {
    return Error(status,
                 absl::StrFormat("Failed to read: %s", std_path.c_str()));
  }

  return buf;
}

Expected<std::vector<std::string>> ListDir(absl::string_view path) {
  auto std_path = MakeStdPath(path);
  if (!StdExists(std_path)) {
    return Error(kLiteRtStatusErrorNotFound,
                 absl::StrFormat("Directory not found: %s", std_path.c_str()));
  }
  std::vector<std::string> res;
  for (const auto& entry : std::filesystem::directory_iterator(std_path)) {
    if (std::filesystem::is_regular_file(entry)) {
      res.push_back(entry.path().generic_string());
    }
  }
  return res;
}

Expected<std::string> Filename(absl::string_view path) {
  auto std_path = MakeStdPath(path);
  if (!StdExists(std_path)) {
    return Error(kLiteRtStatusErrorNotFound,
                 absl::StrFormat("File not found: %s", std_path.c_str()));
  }
  return std_path.filename().generic_string();
}

bool IsDir(absl::string_view path) {
  auto std_path = MakeStdPath(path);
  return std::filesystem::is_directory(std_path);
}

Expected<void> MkDir(absl::string_view path) {
  if (IsDir(path)) {
    return {};
  }
  if (Exists(path)) {
    return Error(
        kLiteRtStatusErrorAlreadyExists,
        absl::StrFormat("Path exists and is not a directory: %s", path.data()));
  }
  auto std_path = MakeStdPath(path);
  const auto stat = std::filesystem::create_directories(std_path);
  if (!stat) {
    return Error(
        kLiteRtStatusErrorFileIO,
        absl::StrFormat("Failed to create directory: %s", std_path.c_str()));
  }
  return {};
}

Expected<std::string> Parent(absl::string_view path) {
  auto std_path = MakeStdPath(path);
  return std_path.parent_path().generic_string();
}

Expected<void> RmDir(std::string path_to_remove) {
  std::error_code error_code;
  std::uintmax_t count =
      std::filesystem::remove_all(path_to_remove, error_code);
  if (error_code) {
    return Error(kLiteRtStatusErrorFileIO,
                 absl::StrFormat("Could not remove: %s, error: %s",
                                 path_to_remove.c_str(), error_code.message()));
  }

  if (!Exists(path_to_remove)) {
    if (count > 0) {
      LITERT_LOG(LITERT_INFO,
                 "Successfully removed directory and its contents: %s (%ju "
                 "items deleted)",
                 path_to_remove.c_str(), count);
    }
    // If count == 0 and it doesn't exist, it means it never existed. Still Ok.
    return {};
  } else {
    return Error(kLiteRtStatusErrorFileIO,
                 absl::StrFormat("Could not fully remove: %s",
                                 path_to_remove.c_str()));
  }
}

}  // namespace litert::internal
