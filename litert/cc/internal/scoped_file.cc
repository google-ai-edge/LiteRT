// Copyright 2025 The ODML Authors.
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

#include "litert/cc/internal/scoped_file.h"

#include <cstddef>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl

namespace litert {

namespace {

bool IsFileValid(ScopedFile::PlatformFile file) {
  return file != ScopedFile::kInvalidPlatformFile;
}

}  // namespace

// static
absl::StatusOr<size_t> ScopedFile::GetSize(PlatformFile file) {
  if (!IsFileValid(file)) {
    return absl::FailedPreconditionError("Scoped file is not valid");
  }
  return GetSizeImpl(file);
}

}  // namespace litert
