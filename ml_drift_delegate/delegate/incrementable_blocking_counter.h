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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_INCREMENTABLE_BLOCKING_COUNTER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_INCREMENTABLE_BLOCKING_COUNTER_H_

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl

namespace litert::ml_drift {

// Similar to absl::BlockingCounter, but allows to increase the counter
// value.
class IncrementableBlockingCounter {
 public:
  explicit IncrementableBlockingCounter(int initial_counter);
  ~IncrementableBlockingCounter() = default;

  // Increments the counter value by 1.
  void Increment();
  // Decrements the counter value by 1.
  void Decrement();
  // Waits for the counter value to be 0.
  void Wait();

 private:
  absl::Mutex mutex_;
  int counter_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_INCREMENTABLE_BLOCKING_COUNTER_H_
