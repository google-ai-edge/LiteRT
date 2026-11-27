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

#include "tensor/examples/utils/ring_buffer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

#if defined(__linux__)
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/syscall.h>

#include <cerrno>
#include <cstdlib>
#endif  // defined(__linux__)

namespace litert::tensor::examples {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::IsEmpty;

using Mode = RingBuffer<float>::Mode;
using ::litert::tensor::examples::internal::RingBufferStorage;

class RingBufferTest : public ::testing::TestWithParam<Mode> {
 protected:
  void SetUp() override {
    if (GetParam() == Mode::kMirrored &&
        !RingBuffer<float>::IsMirroredModeSupported()) {
      GTEST_SKIP() << "The mirrored mode isn't supported on this platform.";
    }
  }

  // Creates a buffer of `capacity` elements in the tested mode.
  template <class T = float>
  static RingBuffer<T> Create(size_t capacity) {
    absl::StatusOr<RingBuffer<T>> buffer =
        RingBuffer<T>::Create(capacity, GetParam());
    ABSL_CHECK_OK(buffer);
    return *std::move(buffer);
  }
};

INSTANTIATE_TEST_SUITE_P(Modes, RingBufferTest,
                         ::testing::Values(Mode::kMirrored, Mode::kCompacting),
                         [](const ::testing::TestParamInfo<Mode>& info) {
                           return info.param == Mode::kMirrored
                                      ? std::string("Mirrored")
                                      : std::string("Compacting");
                         });

TEST_P(RingBufferTest, CreatesAnEmptyBuffer) {
  RingBuffer<float> buffer = Create(5);

  EXPECT_EQ(buffer.mode(), GetParam());
  EXPECT_EQ(buffer.capacity(), 5);
  EXPECT_EQ(buffer.size(), 0);
  EXPECT_TRUE(buffer.empty());
  EXPECT_EQ(buffer.begin(), buffer.end());
  EXPECT_THAT(buffer, IsEmpty());
}

TEST_P(RingBufferTest, PushBackBelowCapacityKeepsEverything) {
  RingBuffer<float> buffer = Create(5);

  buffer.append_range({1, 2});
  buffer.push_back(3);

  EXPECT_THAT(buffer, ElementsAre(1, 2, 3));
}

TEST_P(RingBufferTest, PushBackOverCapacityDropsTheFront) {
  RingBuffer<float> buffer = Create(3);

  buffer.append_range({1, 2, 3});
  buffer.push_back(4);
  EXPECT_THAT(buffer, ElementsAre(2, 3, 4));

  buffer.append_range({5, 6});
  EXPECT_THAT(buffer, ElementsAre(4, 5, 6));
}

TEST_P(RingBufferTest, PushBackLargerThanCapacityKeepsItsTail) {
  RingBuffer<float> buffer = Create(3);
  buffer.push_back(1);

  buffer.append_range({2, 3, 4, 5, 6});

  EXPECT_THAT(buffer, ElementsAre(4, 5, 6));
}

TEST_P(RingBufferTest, ResizeGrowsAndShrinksAtTheBack) {
  RingBuffer<float> buffer = Create(4);
  buffer.append_range({1, 2});

  ASSERT_THAT(buffer.resize(3), IsOk());
  EXPECT_THAT(buffer, ElementsAre(1, 2, 0));

  ASSERT_THAT(buffer.resize(4, 5), IsOk());
  EXPECT_THAT(buffer, ElementsAre(1, 2, 0, 5));

  // Resizing to the capacity never drops front elements.
  ASSERT_THAT(buffer.resize(4, 9), IsOk());
  EXPECT_THAT(buffer, ElementsAre(1, 2, 0, 5));

  ASSERT_THAT(buffer.resize(1), IsOk());
  EXPECT_THAT(buffer, ElementsAre(1));

  ASSERT_THAT(buffer.resize(0), IsOk());
  EXPECT_TRUE(buffer.empty());
}

TEST_P(RingBufferTest, ResizeCopiesAnElementOfTheBuffer) {
  RingBuffer<float> buffer = Create(4);
  // Moves the elements near the end of the compacting buffer, so that growing
  // moves them back to its start.
  buffer.append_range({1, 2, 3, 4});
  ASSERT_THAT(buffer.pop_front(3), IsOk());
  buffer.append_range({5, 6, 7});
  ASSERT_THAT(buffer.pop_front(3), IsOk());
  ASSERT_THAT(buffer, ElementsAre(7));

  ASSERT_THAT(buffer.resize(4, buffer.front()), IsOk());

  EXPECT_THAT(buffer, ElementsAre(7, 7, 7, 7));
}

TEST_P(RingBufferTest, PopFrontAndClear) {
  RingBuffer<float> buffer = Create(4);
  buffer.append_range({1, 2, 3, 4});

  ASSERT_THAT(buffer.pop_front(), IsOk());
  EXPECT_THAT(buffer, ElementsAre(2, 3, 4));

  ASSERT_THAT(buffer.pop_front(2), IsOk());
  EXPECT_THAT(buffer, ElementsAre(4));

  buffer.clear();
  EXPECT_TRUE(buffer.empty());

  buffer.append_range({5, 6});
  EXPECT_THAT(buffer, ElementsAre(5, 6));
}

TEST_P(RingBufferTest, ResizeWithoutInitKeepsTheElements) {
  RingBuffer<float> buffer = Create(4);
  buffer.append_range({1, 2});

  ASSERT_THAT(buffer.resize(4, RingBuffer<float>::kNoInit), IsOk());
  ASSERT_EQ(buffer.size(), 4);
  buffer[2] = 3;
  buffer[3] = 4;
  EXPECT_THAT(buffer, ElementsAre(1, 2, 3, 4));

  ASSERT_THAT(buffer.resize(1, RingBuffer<float>::kNoInit), IsOk());
  EXPECT_THAT(buffer, ElementsAre(1));
}

TEST_P(RingBufferTest, InvalidOperationsFailWithoutChangingIt) {
  constexpr size_t kHuge = std::numeric_limits<size_t>::max();
  RingBuffer<float> buffer = Create(4);
  EXPECT_THAT(buffer.pop_front(), StatusIs(absl::StatusCode::kOutOfRange));
  buffer.append_range({1, 2, 3});

  EXPECT_THAT(buffer.pop_front(4), StatusIs(absl::StatusCode::kOutOfRange));
  // Would overflow when converted to bytes.
  EXPECT_THAT(buffer.pop_front(kHuge), StatusIs(absl::StatusCode::kOutOfRange));
  EXPECT_THAT(buffer.resize(5), StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(buffer.resize(5, 9),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(buffer.resize(5, RingBuffer<float>::kNoInit),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(buffer.resize(kHuge),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(buffer, ElementsAre(1, 2, 3));

  buffer.push_back(4);
  EXPECT_THAT(buffer, ElementsAre(1, 2, 3, 4));
}

// Pops a full buffer from every start position. In the mirrored mode, this
// includes the positions where the popped bytes end exactly at or past the end
// of the first copy of the memory.
TEST_P(RingBufferTest, PopsAFullBufferFromEveryPosition) {
  // 4 KiB: a multiple of the page size, so the window of the mirrored mode can
  // start at the last element of the memory.
  constexpr size_t kCapacity = 1024;
  RingBuffer<float> buffer = Create(kCapacity);
  std::vector<float> values(kCapacity);
  std::iota(values.begin(), values.end(), 0.0f);

  // Goes through every position with up to 64 KiB pages.
  for (int position = 0; position < 16384; ++position) {
    buffer.append_range(values);
    ASSERT_TRUE(
        std::equal(values.begin(), values.end(), buffer.begin(), buffer.end()))
        << "position " << position;
    ASSERT_THAT(buffer.pop_front(kCapacity), IsOk());
    ASSERT_TRUE(buffer.empty());

    // Moves the start by one element.
    buffer.push_back(0);
    ASSERT_THAT(buffer.pop_front(), IsOk());
  }
}

TEST_P(RingBufferTest, WorksLikeAStandardContainer) {
  RingBuffer<float> buffer = Create(4);
  buffer.append_range({5, 6, 7, 8, 9});  // Holds {6, 7, 8, 9}.

  float sum = 0;
  for (float value : buffer) {
    sum += value;
  }
  EXPECT_EQ(sum, 30);
  EXPECT_EQ(std::accumulate(buffer.cbegin(), buffer.cend(), 0.0f), 30);
  EXPECT_EQ(buffer.front(), 6);
  EXPECT_EQ(buffer.back(), 9);
  EXPECT_THAT(std::vector<float>(buffer.rbegin(), buffer.rend()),
              ElementsAre(9, 8, 7, 6));

  // The elements are writable in place.
  std::reverse(buffer.begin(), buffer.end());
  EXPECT_THAT(buffer, ElementsAre(9, 8, 7, 6));
  buffer.front() = 1;
  buffer[1] = 2;
  buffer.data()[2] = 3;
  buffer.back() = 4;
  EXPECT_THAT(absl::MakeConstSpan(buffer), ElementsAre(1, 2, 3, 4));

  const RingBuffer<float>& const_buffer = buffer;
  EXPECT_EQ(*std::max_element(const_buffer.begin(), const_buffer.end()), 4);
}

TEST_P(RingBufferTest, WrapsAroundManyTimes) {
  // Not a multiple of the page size, so the mirrored mode has more memory
  // than the capacity and the window wraps at a different place.
  constexpr size_t kCapacity = 1000;
  RingBuffer<float> buffer = Create(kCapacity);
  std::deque<float> expected;
  float next = 0;

  for (int push = 0; push < 10000; ++push) {
    const size_t count = 1 + push % 37;
    std::vector<float> values(count);
    for (float& value : values) {
      value = next++;
    }
    buffer.append_range(values);
    expected.insert(expected.end(), values.begin(), values.end());
    while (expected.size() > kCapacity) {
      expected.pop_front();
    }

    ASSERT_THAT(buffer, ElementsAreArray(expected)) << "push " << push;
  }
}

TEST_P(RingBufferTest, MatchesADequeOnRandomOperations) {
  constexpr size_t kCapacity = 777;
  RingBuffer<int32_t> buffer = Create<int32_t>(kCapacity);
  std::deque<int32_t> expected;
  std::mt19937 rng(42);
  int32_t next = 0;

  for (int op = 0; op < 100000; ++op) {
    switch (rng() % 10) {
      case 0:
      case 1:
      case 2: {
        std::vector<int32_t> values(rng() % 64);
        std::iota(values.begin(), values.end(), next);
        next += values.size();
        buffer.append_range(values);
        expected.insert(expected.end(), values.begin(), values.end());
        break;
      }
      case 3:
        buffer.push_back(next);
        expected.push_back(next++);
        break;
      case 4:
      case 5: {
        // Moves the window: drops front elements if needed, then grows the
        // back and fills the new elements in place.
        const size_t count = 1 + rng() % 64;
        if (buffer.size() + count > kCapacity) {
          const size_t dropped = buffer.size() + count - kCapacity;
          ASSERT_THAT(buffer.pop_front(dropped), IsOk());
          expected.erase(expected.begin(), expected.begin() + dropped);
        }
        const size_t old_size = buffer.size();
        ASSERT_THAT(
            buffer.resize(old_size + count, RingBuffer<int32_t>::kNoInit),
            IsOk());
        for (size_t i = old_size; i < buffer.size(); ++i) {
          buffer[i] = next;
          expected.push_back(next++);
        }
        break;
      }
      case 6: {
        const size_t count = expected.empty() ? 0 : rng() % expected.size();
        ASSERT_THAT(buffer.pop_front(count), IsOk());
        expected.erase(expected.begin(), expected.begin() + count);
        break;
      }
      case 7:
        if (!expected.empty()) {
          ASSERT_THAT(buffer.pop_front(), IsOk());
          expected.pop_front();
        }
        break;
      case 8: {
        const size_t new_size =
            expected.empty() ? 0 : rng() % (expected.size() + 1);
        ASSERT_THAT(buffer.resize(new_size), IsOk());
        expected.resize(new_size);
        break;
      }
      default:
        if (rng() % 64 == 0) {
          buffer.clear();
          expected.clear();
        }
        break;
    }
    while (expected.size() > kCapacity) {
      expected.pop_front();
    }

    ASSERT_EQ(buffer.size(), expected.size()) << "operation " << op;
    ASSERT_TRUE(std::equal(expected.begin(), expected.end(), buffer.begin()))
        << "operation " << op;
  }
}

TEST_P(RingBufferTest, PaddingAfterTheWindowIsReadable) {
  // With 4 KiB pages, the mirrored mode maps an extra padding page for the
  // first two capacities (0 and 56 free bytes in the memory) but not for the
  // last two (64 and 96 free bytes).
  for (size_t capacity : {1024, 1010, 1008, 1000}) {
    RingBuffer<float> buffer = Create(capacity);
    buffer.append_range(std::vector<float>(capacity, 1.0f));

    // Pushing one element at a time moves the full window through every
    // position, including the ones that end right before the end of the memory
    // (with up to 16 KiB pages).
    for (int push = 0; push < 8192; ++push) {
      buffer.push_back(static_cast<float>(push));
      // Reads the padding bytes, which must not crash (ASan or a segfault).
      const volatile unsigned char* padding =
          reinterpret_cast<const unsigned char*>(buffer.data() + buffer.size());
      unsigned sum = 0;
      for (size_t i = 0; i < RingBuffer<float>::kReadablePaddingBytes; ++i) {
        sum += padding[i];
      }
      volatile unsigned sink = sum;
      (void)sink;
    }
    EXPECT_EQ(buffer.back(), 8191.0f) << "capacity " << capacity;
  }
}

// The pattern that a sliding window K/V cache of `kHeads` heads uses once it's
// full: the window holds `[kHeads, kWindow, kDim]` and each new token moves it
// by one token (`pop_front` + `resize`), then writes the new token of head `h`
// at `h * kWindow + kWindow - 1`. That slot held the dropped token of head
// `h + 1`.
//
// The window moves by 8 bytes per token: 10000 tokens go past the end of the
// mirrored memory with pages up to 64 KiB.
TEST_P(RingBufferTest, SlidesAMultiHeadWindowInPlace) {
  constexpr int kHeads = 2;
  constexpr int kWindow = 3;
  constexpr int kDim = 2;
  RingBuffer<float> buffer = Create(kHeads * kWindow * kDim);
  // Value of element `d` of token `t` of head `h`.
  auto value = [](int h, int t, int d) { return 1000 * h + 10 * t + d; };
  std::vector<float> initial;
  for (int h = 0; h < kHeads; ++h) {
    for (int t = 0; t < kWindow; ++t) {
      for (int d = 0; d < kDim; ++d) {
        initial.push_back(value(h, t, d));
      }
    }
  }
  buffer.append_range(initial);

  for (int t = kWindow; t < kWindow + 10000; ++t) {
    ASSERT_THAT(buffer.pop_front(kDim), IsOk());
    ASSERT_THAT(buffer.resize(buffer.size() + kDim, RingBuffer<float>::kNoInit),
                IsOk());
    for (int h = 0; h < kHeads; ++h) {
      for (int d = 0; d < kDim; ++d) {
        buffer[(h * kWindow + kWindow - 1) * kDim + d] = value(h, t, d);
      }
    }

    std::vector<float> expected;
    for (int h = 0; h < kHeads; ++h) {
      for (int w = t - kWindow + 1; w <= t; ++w) {
        for (int d = 0; d < kDim; ++d) {
          expected.push_back(value(h, w, d));
        }
      }
    }
    ASSERT_THAT(buffer, ElementsAreArray(expected)) << "token " << t;
  }
}

TEST_P(RingBufferTest, MovesKeepTheElements) {
  RingBuffer<float> buffer = Create(3);
  buffer.append_range({1, 2, 3, 4});

  RingBuffer<float> moved = std::move(buffer);
  EXPECT_THAT(moved, ElementsAre(2, 3, 4));

  RingBuffer<float> assigned = Create(1);
  assigned.push_back(9);
  assigned = std::move(moved);
  EXPECT_THAT(assigned, ElementsAre(2, 3, 4));

  assigned.push_back(5);
  EXPECT_THAT(assigned, ElementsAre(3, 4, 5));
}

// Elements larger than the floats of the other tests.
struct Element16 {
  uint32_t values[4];

  friend bool operator==(const Element16& a, const Element16& b) {
    return std::equal(std::begin(a.values), std::end(a.values),
                      std::begin(b.values));
  }
};

// Appends batches of increasing values to `buffer` and checks it against a
// deque of the last `capacity()` values. `make_value(i)` returns value `i`.
template <class T, class MakeValue>
void ExpectWrapsAroundLikeADeque(RingBuffer<T>& buffer, MakeValue make_value) {
  std::deque<T> expected;
  int next = 0;
  for (int push = 0; push < 5000; ++push) {
    std::vector<T> values;
    for (int i = 0; i < 1 + push % 37; ++i) {
      values.push_back(make_value(next++));
    }
    buffer.append_range(values);
    expected.insert(expected.end(), values.begin(), values.end());
    while (expected.size() > buffer.capacity()) {
      expected.pop_front();
    }

    ASSERT_EQ(buffer.size(), expected.size()) << "push " << push;
    ASSERT_TRUE(std::equal(expected.begin(), expected.end(), buffer.begin()))
        << "push " << push;
  }
}

TEST_P(RingBufferTest, WorksWithOtherElementSizes) {
  RingBuffer<uint8_t> bytes = Create<uint8_t>(1000);
  ASSERT_NO_FATAL_FAILURE(ExpectWrapsAroundLikeADeque(
      bytes, [](int i) { return static_cast<uint8_t>(i); }));

  RingBuffer<Element16> elements = Create<Element16>(1000);
  ASSERT_NO_FATAL_FAILURE(ExpectWrapsAroundLikeADeque(elements, [](int i) {
    const uint32_t value = static_cast<uint32_t>(i);
    return Element16{{value, value + 1, value + 2, value + 3}};
  }));
}

TEST(RingBufferCreateTest, RejectsAZeroCapacity) {
  EXPECT_THAT(RingBuffer<float>::Create(0),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(RingBufferCreateTest, RejectsATooLargeCapacity) {
  constexpr size_t kMax = std::numeric_limits<size_t>::max();
  EXPECT_THAT(RingBuffer<float>::Create(kMax / 4 / sizeof(float) + 1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(RingBuffer<float>::Create(kMax),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(RingBufferCreateTest, AutoModePrefersTheMirroredMode) {
  absl::StatusOr<RingBuffer<float>> buffer = RingBuffer<float>::Create(16);

  ASSERT_THAT(buffer, IsOk());
  EXPECT_EQ(buffer->mode(), RingBuffer<float>::IsMirroredModeSupported()
                                ? Mode::kMirrored
                                : Mode::kCompacting);
}

TEST(RingBufferStorageTest, CreateDefaultsToTheAutoMode) {
  absl::StatusOr<RingBufferStorage> storage = RingBufferStorage::Create(64);

  ASSERT_THAT(storage, IsOk());
  EXPECT_EQ(storage->mode(), RingBufferStorage::IsMirroredModeSupported()
                                 ? Mode::kMirrored
                                 : Mode::kCompacting);
}

TEST(RingBufferStorageTest, RejectsATooLargeCapacity) {
  constexpr size_t kMax = std::numeric_limits<size_t>::max();
  for (Mode mode : {Mode::kAuto, Mode::kMirrored, Mode::kCompacting}) {
    // Would overflow the allocation or mapping size.
    EXPECT_THAT(RingBufferStorage::Create(kMax / 4 + 1, mode),
                StatusIs(absl::StatusCode::kInvalidArgument));
    EXPECT_THAT(RingBufferStorage::Create(kMax, mode),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(RingBufferStorageTest, InvalidOperationsFailWithoutChangingIt) {
  std::vector<Mode> modes = {Mode::kCompacting};
  if (RingBufferStorage::IsMirroredModeSupported()) {
    modes.push_back(Mode::kMirrored);
  }
  for (Mode mode : modes) {
    absl::StatusOr<RingBufferStorage> storage =
        RingBufferStorage::Create(64, mode);
    ASSERT_THAT(storage, IsOk());

    EXPECT_THAT(storage->ExtendBack(65),
                StatusIs(absl::StatusCode::kInvalidArgument));
    EXPECT_THAT(storage->Resize(65),
                StatusIs(absl::StatusCode::kInvalidArgument));
    EXPECT_THAT(storage->PopFront(1), StatusIs(absl::StatusCode::kOutOfRange));
    EXPECT_EQ(storage->size(), 0);

    ASSERT_THAT(storage->ExtendBack(16), IsOk());
    EXPECT_THAT(storage->PopFront(17), StatusIs(absl::StatusCode::kOutOfRange));
    EXPECT_EQ(storage->size(), 16);
  }
}

#if defined(__linux__) && defined(SYS_memfd_create)

// Makes `memfd_create` fail with `EPERM` in this process, like a seccomp
// sandbox that forbids it.
bool ForbidMemfdCreate() {
  sock_filter filter[] = {
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_memfd_create, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | (EPERM & SECCOMP_RET_DATA)),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
  };
  sock_fprog program = {static_cast<uint16_t>(std::size(filter)), filter};
  return prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == 0 &&
         prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) == 0;
}

TEST(RingBufferCreateTest, AutoModeFallsBackInASandbox) {
  // Runs the statement in a fresh process, where `IsMirroredModeSupported`
  // hasn't cached its result yet. Nothing before `EXPECT_EXIT` may call it.
  const std::string death_test_style = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");

  EXPECT_EXIT(
      {
        if (!ForbidMemfdCreate()) {
          std::_Exit(2);
        }
        if (RingBuffer<float>::IsMirroredModeSupported()) {
          std::_Exit(3);
        }
        absl::StatusOr<RingBuffer<float>> buffer =
            RingBuffer<float>::Create(16);
        if (!buffer.ok() || buffer->mode() != Mode::kCompacting) {
          std::_Exit(4);
        }
        if (RingBuffer<float>::Create(16, Mode::kMirrored).ok()) {
          std::_Exit(5);
        }
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");

  // Keeps the default style for the other tests.
  GTEST_FLAG_SET(death_test_style, death_test_style);
}

#endif  // defined(__linux__) && defined(SYS_memfd_create)

}  // namespace
}  // namespace litert::tensor::examples
