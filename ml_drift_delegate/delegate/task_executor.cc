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

#include "ml_drift_delegate/delegate/task_executor.h"

#include <functional>
#include <thread>  // NOLINT (Open source code)
#include <utility>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"

namespace litert::ml_drift {

TaskExecutor::TaskExecutor(absl::string_view name, int num_threads)
    : name_(name), stop_(false) {
  for (int i = 0; i < num_threads; ++i) {
    threads_.push_back(std::thread([this, i]() {
      LITERT_LOG(LITERT_DEBUG, "TaskExecutor %s:%d started.", name_.c_str(), i);
      std::function<void()> fn;
      while (true) {
        {
          absl::MutexLock lock(task_mutex_);
          task_mutex_.Await(absl::Condition(this, &TaskExecutor::StopOrTask));
          if (stop_) {
            break;
          }
          fn = std::move(task_queue_.front());
          task_queue_.pop_front();
        }

        LITERT_LOG(LITERT_DEBUG, "TaskExecutor %s:%d executing a task.",
                   name_.c_str(), i);
        std::move(fn)();  // std::move() to release the state after execution.
        LITERT_LOG(LITERT_DEBUG, "TaskExecutor %s:%d finished a task.",
                   name_.c_str(), i);
      }
      LITERT_LOG(LITERT_DEBUG, "TaskExecutor %s:%d stopped.", name_.c_str(), i);
    }));
  }
}

TaskExecutor::~TaskExecutor() {
  {
    absl::MutexLock lock(task_mutex_);
    stop_ = true;
  }
  for (auto& thread : threads_) {
    thread.join();
  }
}

void TaskExecutor::Schedule(std::function<void()> fn) {
  absl::MutexLock lock(task_mutex_);
  task_queue_.push_back(std::move(fn));
}

bool TaskExecutor::StopOrTask() const {
  return stop_ || !task_queue_.empty();
}

}  // namespace litert::ml_drift
