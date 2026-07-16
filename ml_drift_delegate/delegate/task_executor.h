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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_TASK_EXECUTOR_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_TASK_EXECUTOR_H_

#include <deque>
#include <functional>
#include <string>
#include <thread>  // NOLINT (Open source code)
#include <vector>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "ml_drift/common/executor.h"  // from @ml_drift

namespace litert::ml_drift {

// ML Drift Executor implementation that schedules tasks on a thread pool of
// |num_threads| threads. Task will be executed in the order of Schedule()
// calls on any thread idle first.
class TaskExecutor : public ::ml_drift::Executor {
 public:
  explicit TaskExecutor(absl::string_view name, int num_threads = 1);
  ~TaskExecutor() override;

  // ::ml_drift::Executor implementation.
  void Schedule(std::function<void()> fn) override;

 private:
  bool StopOrTask() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(task_mutex_);

  const std::string name_;
  std::vector<std::thread> threads_;

  absl::Mutex task_mutex_;
  std::deque<std::function<void()>> task_queue_ ABSL_GUARDED_BY(task_mutex_);
  bool stop_ ABSL_GUARDED_BY(task_mutex_);
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_TASK_EXECUTOR_H_
