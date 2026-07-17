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

#include "testing/base/public/gunit.h"
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"

namespace litert::ml_drift {
namespace {

TEST(TaskExecutorTest, Basic) {
  LiteRtSetMinLoggerSeverity(LiteRtGetDefaultLogger(), LITERT_DEBUG);

  TaskExecutor executor("test");
  absl::Notification done1;
  executor.Schedule([&done1]() { done1.Notify(); });
  absl::Notification done2;
  executor.Schedule([&done2]() { done2.Notify(); });
  absl::Notification done3;
  executor.Schedule([&done3]() { done3.Notify(); });
  absl::Notification done4;
  executor.Schedule([&done4]() { done4.Notify(); });
  done1.WaitForNotification();
  done2.WaitForNotification();
  done3.WaitForNotification();
  done4.WaitForNotification();
}

TEST(TaskExecutorTest, MultiThreads) {
  LiteRtSetMinLoggerSeverity(LiteRtGetDefaultLogger(), LITERT_DEBUG);

  TaskExecutor executor("test", /*num_threads=*/6);
  // Wait for the threads to be ready.
  absl::SleepFor(absl::Milliseconds(10));

  absl::Notification done1;
  executor.Schedule([&done1]() {
    absl::SleepFor(absl::Milliseconds(10));
    done1.Notify();
  });
  absl::Notification done2;
  executor.Schedule([&done2]() {
    absl::SleepFor(absl::Milliseconds(10));
    done2.Notify();
  });
  absl::Notification done3;
  executor.Schedule([&done3]() {
    absl::SleepFor(absl::Milliseconds(10));
    done3.Notify(); });
  absl::Notification done4;
  executor.Schedule([&done4]() {
    absl::SleepFor(absl::Milliseconds(10));
    done4.Notify(); });
  done1.WaitForNotification();
  done2.WaitForNotification();
  done3.WaitForNotification();
  done4.WaitForNotification();
}

}  // namespace
}  // namespace litert::ml_drift
