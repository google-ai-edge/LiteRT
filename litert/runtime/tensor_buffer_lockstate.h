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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_TENSOR_BUFFER_LOCKSTATE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_TENSOR_BUFFER_LOCKSTATE_H_

#include "litert/c/litert_common.h"

namespace litert::internal {

enum class LockState {
  kUnlocked = 0,
  kReadLocked = 1,
  kWriteLocked = 2,
  kReadWriteLocked = 3,
};

inline LockState ToLockState(LiteRtTensorBufferLockMode mode) {
  switch (mode) {
    case kLiteRtTensorBufferLockModeRead:
      return LockState::kReadLocked;
    case kLiteRtTensorBufferLockModeWrite:
      return LockState::kWriteLocked;
    case kLiteRtTensorBufferLockModeReadWrite:
      return LockState::kReadWriteLocked;
  }
}

}  // namespace litert::internal

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_TENSOR_BUFFER_LOCKSTATE_H_
