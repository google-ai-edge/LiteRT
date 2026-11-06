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

#ifndef LITERT_TOOLS_VISION_EVAL_RUN_LOCK_H_
#define LITERT_TOOLS_VISION_EVAL_RUN_LOCK_H_

#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)

#include "litert/cc/litert_expected.h"

namespace litert::vision {

// Holds a non-blocking exclusive OS file lock for the lifetime of an evaluation
// run so concurrent evaluators cannot corrupt the same output directory.
class RunLock {
 public:
  RunLock() = default;
  RunLock(const RunLock&) = delete;
  RunLock& operator=(const RunLock&) = delete;

  // Releases any held OS file lock descriptor or handle.
  ~RunLock();

  // Creates or opens `path` and acquires an exclusive non-blocking file lock,
  // releasing any previously acquired lock first.
  Expected<void> Acquire(const std::filesystem::path& path);

  // Releases the lock if currently held.
  void Release();

 private:
  intptr_t handle_ = -1;
};

}  // namespace litert::vision

#endif  // LITERT_TOOLS_VISION_EVAL_RUN_LOCK_H_
