// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/tools/vision_eval/run_lock.h"

#include <filesystem>  // NOLINT(build/c++17)

#if defined(_WIN32)
#include <cstdint>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifdef ERROR
#undef ERROR
#endif
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"

namespace litert::vision {

RunLock::~RunLock() { Release(); }

Expected<void> RunLock::Acquire(const std::filesystem::path& path) {
  Release();
#if defined(_WIN32)
  HANDLE handle =
      CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                  OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Cannot lock output directory");
  }
  handle_ = reinterpret_cast<intptr_t>(handle);
#else
  const int fd = open(path.c_str(), O_CREAT | O_RDWR, 0600);
  if (fd < 0) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Cannot open output directory lock");
  }
  if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
    close(fd);
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Another evaluator owns the output directory");
  }
  handle_ = fd;
#endif
  return {};
}

void RunLock::Release() {
#if defined(_WIN32)
  if (handle_ != -1) {
    CloseHandle(reinterpret_cast<HANDLE>(handle_));
    handle_ = -1;
  }
#else
  if (handle_ >= 0) {
    close(static_cast<int>(handle_));
    handle_ = -1;
  }
#endif
}

}  // namespace litert::vision
