// Copyright 2026 Google LLC.
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

#include "ml_drift_delegate/delegate/incrementable_blocking_counter.h"

#include "absl/synchronization/mutex.h"  // from @com_google_absl

namespace litert::ml_drift {

IncrementableBlockingCounter::IncrementableBlockingCounter(int initial_counter)
    : counter_(initial_counter) {}

void IncrementableBlockingCounter::Increment() {
  absl::MutexLock lock(mutex_);
  ++counter_;
}

void IncrementableBlockingCounter::Decrement() {
  absl::MutexLock lock(mutex_);
  --counter_;
}

void IncrementableBlockingCounter::Wait() {
  absl::MutexLock lock(mutex_);
  auto is_counter_zero = [this]() {
    mutex_.AssertHeld();
    return counter_ == 0;
  };
  mutex_.Await(absl::Condition(&is_counter_zero));
}

}  // namespace litert::ml_drift
