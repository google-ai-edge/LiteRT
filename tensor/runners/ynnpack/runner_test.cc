/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/runners/ynnpack/runner.h"

#if defined(__unix__) || defined(__APPLE__)
#include <limits.h>
#include <pthread.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/backends/ynnpack/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/runners/common_nnpack/runner_test_suite.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor {

struct YnnpackTestTraits {
  using Tag = YnnpackMixinTag;
  using Runner = YnnpackRunner;

  static constexpr uint32_t kFlagExternalInput = YNN_VALUE_FLAG_EXTERNAL_INPUT;
  static constexpr uint32_t kFlagExternalOutput =
      YNN_VALUE_FLAG_EXTERNAL_OUTPUT;

  // YNNPACK composes 2D convolutions and pooling out of stencil copies,
  // transposes, dots, and reductions; transpose-conv and resize remain
  // unimplemented.
  static constexpr bool kSupportsConv2D = true;
  static constexpr bool kSupportsDepthwiseConv2D = true;
  static constexpr bool kSupportsTransposeConv2D = false;
  static constexpr bool kSupportsResize = false;
  static constexpr bool kSupportsSoftmaxBeta = true;
  static constexpr bool kSupportsSqueezeOfReshapedInput = true;
  static constexpr bool kRejectsInvalidSqueeze = true;
};

INSTANTIATE_TYPED_TEST_SUITE_P(Ynnpack, NnpackRunnerTest, YnnpackTestTraits);

namespace {

using YnnTensor = Tensor<YnnpackMixinTag>;

// Counts the tasks that ran, and whether any of them ran on the thread that
// scheduled them.
struct TaskCounter {
  std::atomic<int> count{0};
  std::atomic<bool> ran_on_caller{false};
  std::thread::id caller = std::this_thread::get_id();

  static void Run(void* context) {
    auto* counter = static_cast<TaskCounter*>(context);
    if (std::this_thread::get_id() == counter->caller) {
      counter->ran_on_caller.store(true, std::memory_order_relaxed);
    }
    counter->count.fetch_add(1, std::memory_order_release);
  }

  // Returns false if `n` tasks didn't run within a generous timeout.
  bool WaitFor(int n) const {
    const absl::Time deadline = absl::Now() + absl::Seconds(30);
    while (count.load(std::memory_order_acquire) < n) {
      if (absl::Now() > deadline) return false;
      absl::SleepFor(absl::Microseconds(100));
    }
    return true;
  }
};

TEST(YnnpackSchedulerTest, RunsTasksOnBackgroundThreads) {
  for (const absl::Duration spin_duration :
       {absl::ZeroDuration(), YnnpackScheduler::kDefaultSpinDuration}) {
    SCOPED_TRACE(spin_duration);
    YnnpackScheduler scheduler(/*num_background_threads=*/3, spin_duration);
    const ynn_scheduler* vtable = YnnpackScheduler::Vtable();
    EXPECT_EQ(vtable->num_threads(&scheduler), 3);

    TaskCounter counter;
    constexpr int kTasks = 1000;
    for (int i = 0; i < kTasks; ++i) {
      vtable->schedule(&scheduler, &counter, TaskCounter::Run);
    }
    ASSERT_TRUE(counter.WaitFor(kTasks));
    EXPECT_FALSE(counter.ran_on_caller.load(std::memory_order_relaxed));

    // Threads that went to sleep after running out of tasks wake up for new
    // ones.
    absl::SleepFor(absl::Milliseconds(5));
    vtable->schedule(&scheduler, &counter, TaskCounter::Run);
    EXPECT_TRUE(counter.WaitFor(kTasks + 1));
  }
}

TEST(YnnpackSchedulerTest, DestructorRunsPendingTasks) {
  TaskCounter counter;
  absl::Notification release;
  constexpr int kTasks = 100;
  {
    YnnpackScheduler scheduler(/*num_background_threads=*/1);
    const ynn_scheduler* vtable = YnnpackScheduler::Vtable();
    // Keep the only thread busy so the other tasks stay queued.
    vtable->schedule(&scheduler, &release, [](void* context) {
      static_cast<absl::Notification*>(context)->WaitForNotification();
    });
    for (int i = 0; i < kTasks; ++i) {
      vtable->schedule(&scheduler, &counter, TaskCounter::Run);
    }
    release.Notify();
  }
  EXPECT_EQ(counter.count.load(std::memory_order_relaxed), kTasks);
}

TEST(YnnpackSchedulerTest, HasNoThreadsWithoutBackgroundThreads) {
  YnnpackScheduler scheduler(/*num_background_threads=*/0);
  EXPECT_EQ(YnnpackScheduler::Vtable()->num_threads(&scheduler), 0);
}

TEST(YnnpackRunnerTest, RejectsConstantWithTooSmallBuffer) {
  YnnTensor x({.name = "x",
               .type = Type::kFP32,
               .shape = {4},
               .buffer = std::vector<float>{1.f, 2.f}});
  YnnTensor output = Add(x, x);
  EXPECT_EQ(YnnpackRunner::Create({output}).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(YnnpackRunnerTest, SharesExternalThreadpoolAcrossRunners) {
  YnnpackScheduler scheduler(/*num_background_threads=*/3);
  ynn_threadpool_t raw_pool = nullptr;
  ASSERT_EQ(ynn_create_threadpool(YnnpackScheduler::Vtable(), &scheduler,
                                  /*flags=*/0, &raw_pool),
            ynn_status_success);
  YnnpackRunner::ThreadpoolPtr shared_pool(raw_pool);
  ASSERT_NE(shared_pool.get(), nullptr);

  std::vector<float> input(16, 1.5f);
  {
    YnnTensor x1({.name = "x1", .type = Type::kFP32, .shape = {16}});
    YnnTensor y1 = Add(x1, x1);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(YnnpackRunner r1,
                                    YnnpackRunner::Create({y1}));
    ASSERT_THAT(r1.SetThreadpool(shared_pool.get()), ::testing::tensor::IsOk());
    EXPECT_EQ(r1.threadpool(), shared_pool.get());
    ASSERT_THAT(r1.PrepareRuntime(), ::testing::tensor::IsOk());
    EXPECT_EQ(r1.SetThreadpool(shared_pool.get()).code(),
              absl::StatusCode::kFailedPrecondition);

    YnnTensor x2({.name = "x2", .type = Type::kFP32, .shape = {16}});
    YnnTensor y2 = Mul(x2, x2);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(YnnpackRunner r2,
                                    YnnpackRunner::Create({y2}));
    ASSERT_THAT(r2.SetThreadpool(shared_pool.get()), ::testing::tensor::IsOk());
    ASSERT_THAT(r2.SetNumThreads(2), ::testing::tensor::IsOk());
    EXPECT_EQ(r2.threadpool(), nullptr);
    ASSERT_THAT(r2.SetThreadpool(shared_pool.get()), ::testing::tensor::IsOk());
    EXPECT_EQ(r2.threadpool(), shared_pool.get());
    ASSERT_THAT(r2.PrepareRuntime(), ::testing::tensor::IsOk());

    ASSERT_THAT(r1.SetInput(x1, input), ::testing::tensor::IsOk());
    ASSERT_THAT(r1.Run(), ::testing::tensor::IsOk());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const float> out1,
                                    r1.ReadOutputAs<float>(y1));
    EXPECT_FLOAT_EQ(out1.data()[0], 3.0f);

    ASSERT_THAT(r2.SetInput(x2, input), ::testing::tensor::IsOk());
    ASSERT_THAT(r2.Run(), ::testing::tensor::IsOk());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const float> out2,
                                    r2.ReadOutputAs<float>(y2));
    EXPECT_FLOAT_EQ(out2.data()[0], 2.25f);
  }
}

#if defined(__unix__) || defined(__APPLE__)
TEST(YnnpackRunnerTest, CompilesDeepGraphOnConstrainedCallerStack) {
  float value = 0.0f;
  const size_t constrained_stack =
      std::max<size_t>(PTHREAD_STACK_MIN, 256 * 1024);
  ASSERT_OK(RunWithLargeStack(
      [&]() -> absl::Status {
        YnnTensor x({.name = "x", .type = Type::kFP32, .shape = {1, 32}});
        YnnTensor bias({.name = "b",
                        .type = Type::kFP32,
                        .shape = {1, 32},
                        .buffer = std::vector<float>(32, 0.5f)});
        YnnTensor cur = x;
        // 512 unfused reductions + elementwise stages create a deep chain
        // of nested Slinky `allocate` scopes during runtime compilation.
        for (int i = 0; i < 512; ++i) {
          YnnTensor mean = Mean(cur, /*axes=*/{1}, /*keep_dims=*/true);
          cur = Add(Sub(cur, mean), bias);
        }
        absl::StatusOr<YnnpackRunner> runner = YnnpackRunner::Create({cur});
        if (!runner.ok()) return runner.status();
        std::vector<float> in(32, 1.0f);
        if (absl::Status s = runner->SetInput(x, in); !s.ok()) {
          return s;
        }
        if (absl::Status s = runner->Run(); !s.ok()) {
          return s;
        }
        absl::StatusOr<absl::Span<const float>> out =
            runner->ReadOutputAs<float>(cur);
        if (!out.ok()) return out.status();
        value = out->data()[0];
        return absl::OkStatus();
      },
      constrained_stack));
  EXPECT_NEAR(value, 0.5f, 1e-4f);
}
#endif

}  // namespace
}  // namespace litert::tensor
