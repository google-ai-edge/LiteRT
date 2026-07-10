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

#include "third_party/odml/litert/ml_drift/delegate/incrementable_blocking_counter.h"

#include "testing/base/public/gunit.h"
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "third_party/odml/litert/ml_drift/delegate/task_executor.h"

namespace litert::ml_drift {
namespace {

TEST(IncrementableBlockingCounterTest, Basic) {
  IncrementableBlockingCounter counter(0);
  counter.Wait();

  counter.Increment();
  counter.Decrement();
  counter.Wait();
}

TEST(IncrementableBlockingCounterTest, InitialValue) {
  IncrementableBlockingCounter counter(3);
  counter.Decrement();
  counter.Decrement();
  counter.Decrement();
  counter.Wait();
}

TEST(IncrementableBlockingCounterTest, MultiThreaded) {
  const int kNumThreads = 10;
  IncrementableBlockingCounter counter(kNumThreads);
  TaskExecutor executor("test", kNumThreads);

  for (int i = 0; i < kNumThreads; ++i) {
    executor.Schedule([&counter]() {
      absl::SleepFor(absl::Milliseconds(50));
      counter.Decrement();
    });
  }

  counter.Wait();
}

TEST(IncrementableBlockingCounterTest, IncrementWhileWaiting) {
  IncrementableBlockingCounter counter(1);
  TaskExecutor executor("test", 2);

  absl::Notification wait_started;
  absl::Notification increment_done;

  executor.Schedule([&counter, &wait_started, &increment_done]() {
    wait_started.Notify();
    counter.Wait();
    increment_done.Notify();
  });

  wait_started.WaitForNotification();
  // Small sleep to ensure the other thread is likely in Wait().
  absl::SleepFor(absl::Milliseconds(50));

  EXPECT_FALSE(increment_done.HasBeenNotified());

  counter.Increment();
  counter.Decrement();
  EXPECT_FALSE(increment_done.HasBeenNotified());

  counter.Decrement();
  increment_done.WaitForNotification();
}

}  // namespace
}  // namespace litert::ml_drift
