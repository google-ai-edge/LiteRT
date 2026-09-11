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

#include "tensor/runners/common_nnpack/runner.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/common_nnpack/graph.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Not;
using ::testing::SizeIs;
using ::testing::ValuesIn;

// Value flags used by `TestRunner` to mark the graph external values.
inline constexpr uint32_t kExternalInput = 1 << 0;
inline constexpr uint32_t kExternalOutput = 1 << 1;

// Registers `tensor` in `graph` with the given `flags` and returns its index.
//
// The value id is the index of the value in the graph, which keeps ids stable
// and readable in test failures.
size_t AddValue(NnpackGraph& graph, const TensorHandle& tensor,
                uint32_t flags) {
  const size_t index = graph.mutable_values().size();
  graph.mutable_values().push_back(NnpackValue{
      .info = {.name = std::string(tensor.GetName()),
               .type = tensor.GetType(),
               .shape = tensor.GetShape()},
      .id = static_cast<uint32_t>(index),
      .flags = flags,
  });
  graph.mutable_tensor_index()[tensor.GetRaw()] = index;
  return index;
}

// The backend hooks that a `TestRunner` can be told to fail on.
enum class Hook {
  kCreateRuntime,
  kSetExternalValueShape,
  kReshapeRuntime,
  kGetExternalValueShape,
  kSetupExternalValues,
  kInvokeRuntime,
  kNumHooks,
};

// A minimal `NnpackRunner` whose runtime implements the identity function: on
// invocation it copies the external input bytes into the external output.
//
// The output shape follows the input shape, which mimics an elementwise
// operation and is enough to exercise the base class output resizing.
class TestRunner : public NnpackRunner {
 public:
  using NnpackRunner::NnpackRunner;

  // The external input registered in the graph.
  const TensorHandle& Input() const { return input_; }
  // The external output registered in the graph.
  const TensorHandle& Output() const { return output_; }

  void SetTensors(TensorHandle input, TensorHandle output) {
    input_ = std::move(input);
    output_ = std::move(output);
  }

  // Makes `hook` report `status` instead of doing its work.
  void FailOn(Hook hook, absl::Status status) {
    hook_statuses_[static_cast<size_t>(hook)] = std::move(status);
  }

  // Returns the graph information associated to `tensor`.
  const graph::TensorInformation& InfoOf(const TensorHandle& tensor) const {
    const absl::StatusOr<size_t> index = graph().Lookup(tensor);
    ABSL_CHECK_OK(index);
    return graph().values()[*index].info;
  }

 protected:
  uint32_t FlagExternalInput() const override { return kExternalInput; }
  uint32_t FlagExternalOutput() const override { return kExternalOutput; }

  absl::Status CreateRuntime(size_t num_threads) override {
    return StatusOf(Hook::kCreateRuntime);
  }

  absl::Status SetExternalValueShape(uint32_t id,
                                     absl::Span<const size_t> dims) override {
    LRT_TENSOR_RETURN_IF_ERROR(StatusOf(Hook::kSetExternalValueShape));
    input_dims_.assign(dims.begin(), dims.end());
    return absl::OkStatus();
  }

  absl::Status ReshapeRuntime() override {
    return StatusOf(Hook::kReshapeRuntime);
  }

  absl::Status GetExternalValueShape(uint32_t id,
                                     std::vector<size_t>& dims) override {
    LRT_TENSOR_RETURN_IF_ERROR(StatusOf(Hook::kGetExternalValueShape));
    // The identity operation produces an output shaped like its input.
    dims = input_dims_;
    return absl::OkStatus();
  }

  absl::Status SetupExternalValues(
      absl::Span<NnpackValue> values,
      const absl::flat_hash_map<uint32_t, std::shared_ptr<Buffer>>&
          external_buffers,
      std::vector<LockedBufferSpan<const std::byte>>& locks) override {
    LRT_TENSOR_RETURN_IF_ERROR(StatusOf(Hook::kSetupExternalValues));
    for (const NnpackValue& value : values) {
      const auto it = external_buffers.find(value.id);
      if (it == external_buffers.end() || it->second == nullptr) {
        continue;
      }
      if ((value.flags & kExternalInput) != 0) {
        LockedBufferSpan<const std::byte> lock = it->second->Lock();
        in_ = lock;
        locks.push_back(std::move(lock));
      } else if ((value.flags & kExternalOutput) != 0) {
        out_ = it->second->LockMutable();
      }
    }
    return absl::OkStatus();
  }

  absl::Status InvokeRuntime() override {
    LRT_TENSOR_RETURN_IF_ERROR(StatusOf(Hook::kInvokeRuntime));
    if (in_.data() != nullptr && out_.data() != nullptr) {
      std::memcpy(out_.data(), in_.data(), std::min(in_.size(), out_.size()));
    }
    return absl::OkStatus();
  }

 private:
  // Returns the status that `hook` must report, `absl::OkStatus()` by default.
  const absl::Status& StatusOf(Hook hook) const {
    return hook_statuses_[static_cast<size_t>(hook)];
  }

  std::array<absl::Status, static_cast<size_t>(Hook::kNumHooks)> hook_statuses_;
  TensorHandle input_ = TensorHandle::Invalid();
  TensorHandle output_ = TensorHandle::Invalid();
  std::vector<size_t> input_dims_;
  LockedBufferSpan<const std::byte> in_ =
      LockedBufferSpan<const std::byte>::Empty();
  LockedBufferSpan<std::byte> out_ = LockedBufferSpan<std::byte>::Empty();
};

// Builds a runner over a graph holding a single external input and a single
// external output, both of the given `type` and `shape`.
//
// The tensors are reachable as `runner.Input()` and `runner.Output()`.
TestRunner MakeTestRunner(Type type = Type::kFP32, Shape shape = {4}) {
  const TensorHandle input({.name = "input", .type = type, .shape = shape});
  const TensorHandle output({.name = "output", .type = type, .shape = shape});
  auto graph = std::make_unique<NnpackGraph>();
  AddValue(*graph, input, kExternalInput);
  AddValue(*graph, output, kExternalOutput);
  TestRunner runner(std::move(graph));
  runner.SetTensors(input, output);
  return runner;
}

// Returns a tensor that is not registered in any graph.
TensorHandle UnknownTensor() {
  return TensorHandle({.name = "unknown", .type = Type::kFP32, .shape = {4}});
}

// Returns a read-only byte view over `values`.
template <class T>
absl::Span<const std::byte> AsConstSpan(const std::vector<T>& values) {
  return absl::Span<const std::byte>(
      reinterpret_cast<const std::byte*>(values.data()),
      values.size() * sizeof(T));
}

// Returns a mutable byte view over `values`.
template <class T>
absl::Span<std::byte> AsSpan(std::vector<T>& values) {
  return absl::Span<std::byte>(reinterpret_cast<std::byte*>(values.data()),
                               values.size() * sizeof(T));
}

TEST(SetInputTest, RunReadsBackTheInputData) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f};

  ASSERT_THAT(runner.SetInput(runner.Input(), AsConstSpan(data)), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  EXPECT_THAT(runner.ReadOutputAs<float>(runner.Output()),
              IsOkAndHolds(ElementsAre(1.f, 2.f, 3.f, 4.f)));
}

TEST(SetInputTest, KeepsAViewOnTheDataByDefault) {
  TestRunner runner = MakeTestRunner();
  std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  ASSERT_THAT(runner.SetInput(runner.Input(), AsConstSpan(data)), IsOk());

  // The runner only holds a view, so this update is picked up by `Run()`.
  data[0] = 10.f;
  ASSERT_THAT(runner.Run(), IsOk());

  EXPECT_THAT(runner.ReadOutputAs<float>(runner.Output()),
              IsOkAndHolds(ElementsAre(10.f, 2.f, 3.f, 4.f)));
}

TEST(SetInputTest, CopiesTheDataWhenRequested) {
  TestRunner runner = MakeTestRunner();
  std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  ASSERT_THAT(
      runner.SetInput(runner.Input(), AsConstSpan(data), /*copy_data=*/true),
      IsOk());

  // The runner owns a copy, so this update is not picked up by `Run()`.
  data[0] = 10.f;
  ASSERT_THAT(runner.Run(), IsOk());

  EXPECT_THAT(runner.ReadOutputAs<float>(runner.Output()),
              IsOkAndHolds(ElementsAre(1.f, 2.f, 3.f, 4.f)));
}

TEST(SetInputTest, AcceptsMutableByteSpan) {
  TestRunner runner = MakeTestRunner();
  std::vector<float> data = {1.f, 2.f, 3.f, 4.f};

  ASSERT_THAT(runner.SetInput(runner.Input(), AsSpan(data)), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  EXPECT_THAT(runner.ReadOutputAs<float>(runner.Output()),
              IsOkAndHolds(ElementsAre(1.f, 2.f, 3.f, 4.f)));
}

TEST(SetInputTest, RejectsOversizedData) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f, 5.f};

  EXPECT_THAT(runner.SetInput(runner.Input(), AsConstSpan(data)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SetInputTest, RejectsUndersizedData) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f};

  EXPECT_THAT(runner.SetInput(runner.Input(), AsConstSpan(data)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SetInputTest, RejectsNonExternalInputs) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f};

  EXPECT_THAT(runner.SetInput(runner.Output(), AsConstSpan(data)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SetInputTest, RejectsUnknownTensors) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f};

  EXPECT_THAT(runner.SetInput(UnknownTensor(), AsConstSpan(data)),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(SetInputTest, ReplacesAPreviouslySetInput) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> first = {1.f, 2.f, 3.f, 4.f};
  const std::vector<float> second = {5.f, 6.f, 7.f, 8.f};

  ASSERT_THAT(runner.SetInput(runner.Input(), first), IsOk());
  ASSERT_THAT(runner.SetInput(runner.Input(), second), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  EXPECT_THAT(runner.ReadOutputAs<float>(runner.Output()),
              IsOkAndHolds(ElementsAre(5.f, 6.f, 7.f, 8.f)));
}

TEST(SetInputFromTensorTest, AdoptsTheShapeAndDataOfTheExternalTensor) {
  TestRunner runner = MakeTestRunner();
  const TensorHandle external(
      {.name = "external",
       .type = Type::kFP32,
       .shape = {8},
       .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f}});

  ASSERT_THAT(runner.SetInput(runner.Input(), external), IsOk());
  EXPECT_THAT(runner.InfoOf(runner.Input()).shape, ElementsAre(8));

  ASSERT_THAT(runner.Run(), IsOk());
  EXPECT_THAT(
      runner.ReadOutputAs<float>(runner.Output()),
      IsOkAndHolds(ElementsAre(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f)));
}

TEST(SetInputFromTensorTest, RejectsATypeMismatch) {
  TestRunner runner = MakeTestRunner();
  const TensorHandle external({.name = "external",
                               .type = Type::kI32,
                               .shape = {4},
                               .buffer = std::vector<int32_t>{1, 2, 3, 4}});

  EXPECT_THAT(runner.SetInput(runner.Input(), external),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SetInputFromTensorTest, RejectsATensorWithoutABuffer) {
  TestRunner runner = MakeTestRunner();
  // The shape differs from the input shape to let us check that failure will
  // not mutate the internal shape.
  const TensorHandle external(
      {.name = "external", .type = Type::kFP32, .shape = {8}});
  ASSERT_THAT(runner.InfoOf(runner.Input()).shape,
              Not(ElementsAreArray(external.GetShape())));

  EXPECT_THAT(runner.SetInput(runner.Input(), external),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(runner.InfoOf(runner.Input()).shape, ElementsAre(4));
}

TEST(SetInputFromTensorTest, GrowsAnUndersizedOwningBuffer) {
  TestRunner runner = MakeTestRunner();
  // The external tensor declares 8 elements but only carries 4.
  const TensorHandle external(
      {.name = "external",
       .type = Type::kFP32,
       .shape = {8},
       .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f}});

  ASSERT_THAT(runner.SetInput(runner.Input(), external), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const LockedBufferSpan<const float> out,
                                  runner.ReadOutputAs<float>(runner.Output()));
  // The buffer was reallocated to hold 8 elements, preserving the original
  // data. The 4 extra elements are uninitialized.
  EXPECT_THAT(out, SizeIs(8));
  EXPECT_THAT(std::vector<float>(out.begin(), out.begin() + 4),
              ElementsAre(1.f, 2.f, 3.f, 4.f));
}

TEST(SetInputFromTensorTest, RejectsAnUndersizedNonOwningBuffer) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  const std::shared_ptr<Buffer> view = std::make_shared<SpanCpuBuffer>(
      reinterpret_cast<const std::byte*>(data.data()),
      data.size() * sizeof(float));
  // The external tensor declares 8 elements but only views 4, and a view
  // cannot be resized.
  const TensorHandle external(
      {.name = "external", .type = Type::kFP32, .shape = {8}, .buffer = view});

  EXPECT_THAT(runner.SetInput(runner.Input(), external),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // A failed call must not leave the graph shape mutated.
  EXPECT_THAT(runner.InfoOf(runner.Input()).shape, ElementsAre(4));
}

TEST(SetInputFromTensorTest, RejectsNonExternalInputs) {
  TestRunner runner = MakeTestRunner();
  const TensorHandle external(
      {.name = "external",
       .type = Type::kFP32,
       .shape = {4},
       .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f}});

  EXPECT_THAT(runner.SetInput(runner.Output(), external),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SetInputFromTensorTest, RejectsUnknownTensors) {
  TestRunner runner = MakeTestRunner();
  const TensorHandle external(
      {.name = "external",
       .type = Type::kFP32,
       .shape = {4},
       .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f}});

  EXPECT_THAT(runner.SetInput(UnknownTensor(), external),
              StatusIs(absl::StatusCode::kNotFound));
}

// Detects whether `TestRunner::SetInput` can be called with a `Sequence`.
template <class Sequence, class = void>
struct CanSetInput : std::false_type {};

template <class Sequence>
struct CanSetInput<
    Sequence,
    std::void_t<decltype(std::declval<TestRunner&>().SetInput(
        std::declval<const TensorHandle&>(), std::declval<Sequence>()))>>
    : std::true_type {};

// The sequence overload keeps a view on its argument, which is only safe if the
// sequence outlives the runner. The rvalue overload is therefore deleted.
static_assert(CanSetInput<const std::vector<float>&>::value);
static_assert(!CanSetInput<std::vector<float>&&>::value);

TEST(SetInputSequenceTest, KeepsAViewOnTheSequence) {
  TestRunner runner = MakeTestRunner();
  std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  ASSERT_THAT(runner.SetInput(runner.Input(), data), IsOk());

  data[0] = 10.f;
  ASSERT_THAT(runner.Run(), IsOk());

  EXPECT_THAT(runner.ReadOutputAs<float>(runner.Output()),
              IsOkAndHolds(ElementsAre(10.f, 2.f, 3.f, 4.f)));
}

TEST(SetInputSequenceTest, RejectsATypeMismatch) {
  TestRunner runner = MakeTestRunner();
  const std::vector<int32_t> data = {1, 2, 3, 4};

  EXPECT_THAT(runner.SetInput(runner.Input(), data),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SetInputSequenceTest, RejectsASizeMismatch) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f};

  EXPECT_THAT(runner.SetInput(runner.Input(), data),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SetInputAsCopyTest, CopiesTheSequence) {
  TestRunner runner = MakeTestRunner();
  std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  ASSERT_THAT(runner.SetInputAsCopy(runner.Input(), data), IsOk());

  data.assign(4, 10.f);
  ASSERT_THAT(runner.Run(), IsOk());

  EXPECT_THAT(runner.ReadOutputAs<float>(runner.Output()),
              IsOkAndHolds(ElementsAre(1.f, 2.f, 3.f, 4.f)));
}

TEST(SetInputAsCopyTest, RejectsATypeMismatch) {
  TestRunner runner = MakeTestRunner();
  std::vector<int32_t> data = {1, 2, 3, 4};

  EXPECT_THAT(runner.SetInputAsCopy(runner.Input(), data),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SetOutputTest, TheRuntimeWritesIntoTheCallerStorage) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  ASSERT_THAT(runner.SetInput(runner.Input(), data), IsOk());

  std::vector<float> out(4, 0.f);
  ASSERT_THAT(runner.SetOutput(runner.Output(), AsSpan(out)), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  EXPECT_THAT(out, ElementsAre(1.f, 2.f, 3.f, 4.f));
}

TEST(SetOutputTest, RejectsAnOversizedBuffer) {
  TestRunner runner = MakeTestRunner();
  std::vector<float> out(5, 0.f);

  EXPECT_THAT(runner.SetOutput(runner.Output(), AsSpan(out)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SetOutputTest, RejectsAnUndersizedBuffer) {
  TestRunner runner = MakeTestRunner();
  std::vector<float> out(3, 0.f);

  EXPECT_THAT(runner.SetOutput(runner.Output(), AsSpan(out)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SetOutputTest, RejectsNonOutputTensors) {
  TestRunner runner = MakeTestRunner();
  std::vector<float> out(4, 0.f);

  EXPECT_THAT(runner.SetOutput(runner.Input(), AsSpan(out)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SetOutputTest, RejectsUnknownTensors) {
  TestRunner runner = MakeTestRunner();
  std::vector<float> out(4, 0.f);

  EXPECT_THAT(runner.SetOutput(UnknownTensor(), AsSpan(out)),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(ReshapeInputTest, UpdatesTheGraphShape) {
  TestRunner runner = MakeTestRunner();
  const std::vector<int32_t> shape = {2, 4};

  ASSERT_THAT(runner.ReshapeInput(runner.Input(), shape), IsOk());

  EXPECT_THAT(runner.InfoOf(runner.Input()).shape, ElementsAre(2, 4));
}

TEST(ReshapeInputTest, NewShapeIsUsedByRun) {
  TestRunner runner = MakeTestRunner();
  const std::vector<int32_t> shape = {8};
  ASSERT_THAT(runner.ReshapeInput(runner.Input(), shape), IsOk());

  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
  ASSERT_THAT(runner.SetInput(runner.Input(), data), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  EXPECT_THAT(runner.InfoOf(runner.Output()).shape, ElementsAre(8));
  EXPECT_THAT(
      runner.ReadOutputAs<float>(runner.Output()),
      IsOkAndHolds(ElementsAre(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f)));
}

TEST(ReshapeInputTest, GrowsAnOwningInputBuffer) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  ASSERT_THAT(runner.SetInputAsCopy(runner.Input(), data), IsOk());

  const std::vector<int32_t> shape = {8};
  ASSERT_THAT(runner.ReshapeInput(runner.Input(), shape), IsOk());

  // The owning buffer was reallocated, so the second half is now writable.
  const std::vector<float> tail = {5.f, 6.f, 7.f, 8.f};
  EXPECT_THAT(runner.WriteInput(runner.Input(), 4 * sizeof(float), tail),
              IsOk());
}

TEST(ReshapeInputTest, ShrinkingKeepsTheRunnerUsable) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  ASSERT_THAT(runner.SetInputAsCopy(runner.Input(), data), IsOk());

  const std::vector<int32_t> shape = {2};
  ASSERT_THAT(runner.ReshapeInput(runner.Input(), shape), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  EXPECT_THAT(runner.ReadOutputAs<float>(runner.Output()),
              IsOkAndHolds(ElementsAre(1.f, 2.f)));
}

TEST(ReshapeInputTest, DoesNotEagerlyFailOnNonOwningBuffers) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  // Keeps a view, which cannot be resized to hold the new shape.
  ASSERT_THAT(runner.SetInput(runner.Input(), data), IsOk());

  const std::vector<int32_t> shape = {8};

  EXPECT_THAT(runner.ReshapeInput(runner.Input(), shape), IsOk());
}

TEST(ReshapeInputTest, RejectsNonExternalInputs) {
  TestRunner runner = MakeTestRunner();
  const std::vector<int32_t> shape = {8};

  EXPECT_THAT(runner.ReshapeInput(runner.Output(), shape),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ReshapeInputTest, RejectsUnknownTensors) {
  TestRunner runner = MakeTestRunner();
  const std::vector<int32_t> shape = {8};

  EXPECT_THAT(runner.ReshapeInput(UnknownTensor(), shape),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(WriteInputTest, WritesAtTheGivenOffset) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  ASSERT_THAT(runner.SetInputAsCopy(runner.Input(), data), IsOk());

  const std::vector<float> patch = {30.f};
  ASSERT_THAT(runner.WriteInput(runner.Input(), 2 * sizeof(float), patch),
              IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  EXPECT_THAT(runner.ReadOutputAs<float>(runner.Output()),
              IsOkAndHolds(ElementsAre(1.f, 2.f, 30.f, 4.f)));
}

TEST(WriteInputTest, WritesByteSpan) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  ASSERT_THAT(runner.SetInputAsCopy(runner.Input(), data), IsOk());

  const std::vector<float> patch = {30.f};
  ASSERT_THAT(
      runner.WriteInput(runner.Input(), 2 * sizeof(float), AsConstSpan(patch)),
      IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  EXPECT_THAT(runner.ReadOutputAs<float>(runner.Output()),
              IsOkAndHolds(ElementsAre(1.f, 2.f, 30.f, 4.f)));
}

TEST(WriteInputTest, WritesTypedSpan) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  ASSERT_THAT(runner.SetInputAsCopy(runner.Input(), data), IsOk());

  const std::vector<float> patch = {30.f};
  ASSERT_THAT(runner.WriteInput(runner.Input(), 2 * sizeof(float),
                                absl::Span<const float>(patch)),
              IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  EXPECT_THAT(runner.ReadOutputAs<float>(runner.Output()),
              IsOkAndHolds(ElementsAre(1.f, 2.f, 30.f, 4.f)));
}

TEST(WriteInputTest, WritesThroughAMutableView) {
  TestRunner runner = MakeTestRunner();
  std::vector<float> storage = {1.f, 2.f, 3.f, 4.f};
  const std::shared_ptr<Buffer> view = std::make_shared<MutableSpanCpuBuffer>(
      reinterpret_cast<std::byte*>(storage.data()),
      storage.size() * sizeof(float));
  const TensorHandle external(
      {.name = "external", .type = Type::kFP32, .shape = {4}, .buffer = view});
  ASSERT_THAT(runner.SetInput(runner.Input(), external), IsOk());

  const std::vector<float> patch = {30.f};
  ASSERT_THAT(runner.WriteInput(runner.Input(), 2 * sizeof(float), patch),
              IsOk());

  EXPECT_THAT(storage, ElementsAre(1.f, 2.f, 30.f, 4.f));
}

TEST(WriteInputTest, RejectsAMissingBuffer) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> patch = {30.f};

  EXPECT_THAT(runner.WriteInput(runner.Input(), 0, patch),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(WriteInputTest, RejectsAnImmutableBuffer) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  // Keeps a read-only view on `data`.
  ASSERT_THAT(runner.SetInput(runner.Input(), data), IsOk());

  const std::vector<float> patch = {30.f};

  EXPECT_THAT(runner.WriteInput(runner.Input(), 0, patch),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(WriteInputTest, RejectsAnOutOfBoundsWrite) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  ASSERT_THAT(runner.SetInputAsCopy(runner.Input(), data), IsOk());

  const std::vector<float> patch = {30.f};

  EXPECT_THAT(runner.WriteInput(runner.Input(), 4 * sizeof(float), patch),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(WriteInputTest, RejectsNonExternalInputs) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> patch = {30.f};

  EXPECT_THAT(runner.WriteInput(runner.Output(), 0, patch),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(WriteInputTest, RejectsUnknownTensors) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> patch = {30.f};

  EXPECT_THAT(runner.WriteInput(UnknownTensor(), 0, patch),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(PrepareRuntimeTest, PreparingBeforeRunningWorks) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  ASSERT_THAT(runner.SetInput(runner.Input(), data), IsOk());

  ASSERT_THAT(runner.PrepareRuntime(), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  EXPECT_THAT(runner.ReadOutputAs<float>(runner.Output()),
              IsOkAndHolds(ElementsAre(1.f, 2.f, 3.f, 4.f)));
}

TEST(PrepareRuntimeTest, ReportsBackendFailures) {
  TestRunner runner = MakeTestRunner();
  runner.FailOn(Hook::kCreateRuntime,
                absl::InternalError("cannot create runtime"));

  EXPECT_THAT(runner.PrepareRuntime(), StatusIs(absl::StatusCode::kInternal));
}

TEST(RunTest, AllocatesTheOutputBufferWhenAbsent) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  ASSERT_THAT(runner.SetInput(runner.Input(), data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  EXPECT_THAT(runner.ReadOutputAs<float>(runner.Output()),
              IsOkAndHolds(SizeIs(4)));
}

TEST(RunTest, AdoptsTheOutputShapeReportedByTheRuntime) {
  TestRunner runner = MakeTestRunner();
  const std::vector<int32_t> shape = {8};
  ASSERT_THAT(runner.ReshapeInput(runner.Input(), shape), IsOk());
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
  ASSERT_THAT(runner.SetInput(runner.Input(), data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  EXPECT_THAT(runner.InfoOf(runner.Output()).shape, ElementsAre(8));
  EXPECT_THAT(runner.ReadOutputAs<float>(runner.Output()),
              IsOkAndHolds(SizeIs(8)));
}

TEST(RunTest, ReflectsInputUpdatesBetweenRuns) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  ASSERT_THAT(runner.SetInputAsCopy(runner.Input(), data), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());
  ASSERT_THAT(runner.ReadOutputAs<float>(runner.Output()),
              IsOkAndHolds(ElementsAre(1.f, 2.f, 3.f, 4.f)));

  const std::vector<float> patch = {30.f};
  ASSERT_THAT(runner.WriteInput(runner.Input(), 2 * sizeof(float), patch),
              IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  EXPECT_THAT(runner.ReadOutputAs<float>(runner.Output()),
              IsOkAndHolds(ElementsAre(1.f, 2.f, 30.f, 4.f)));
}

TEST(RunTest, RejectsNegativeInputDimensions) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  ASSERT_THAT(runner.SetInput(runner.Input(), data), IsOk());
  const std::vector<int32_t> shape = {-1};
  ASSERT_THAT(runner.ReshapeInput(runner.Input(), shape), IsOk());

  EXPECT_THAT(runner.Run(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(RunTest, RejectsAMissingInputBuffer) {
  TestRunner runner = MakeTestRunner();

  EXPECT_THAT(runner.Run(), StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(RunTest, RejectsAnUndersizedNonOwningInputBuffer) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  // Keeps a view, which cannot be resized to hold the new shape.
  ASSERT_THAT(runner.SetInput(runner.Input(), data), IsOk());
  const std::vector<int32_t> shape = {8};
  ASSERT_THAT(runner.ReshapeInput(runner.Input(), shape), IsOk());

  EXPECT_THAT(runner.Run(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(RunTest, RejectsAnUndersizedOutputView) {
  TestRunner runner = MakeTestRunner();
  std::vector<float> out(4, 0.f);
  ASSERT_THAT(runner.SetOutput(runner.Output(), AsSpan(out)), IsOk());

  // The runtime will report an output twice as large as the view.
  const std::vector<int32_t> shape = {8};
  ASSERT_THAT(runner.ReshapeInput(runner.Input(), shape), IsOk());
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
  ASSERT_THAT(runner.SetInput(runner.Input(), data), IsOk());

  EXPECT_THAT(runner.Run(), StatusIs(absl::StatusCode::kInvalidArgument));
}

// Names a backend hook for the failure propagation tests.
struct BackendFailure {
  std::string name;
  Hook hook;
};

class RunBackendFailureTest : public ::testing::TestWithParam<BackendFailure> {
};

TEST_P(RunBackendFailureTest, ReportsBackendFailures) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  ASSERT_THAT(runner.SetInput(runner.Input(), data), IsOk());
  runner.FailOn(GetParam().hook, absl::InternalError("backend failure"));

  EXPECT_THAT(runner.Run(), StatusIs(absl::StatusCode::kInternal));
}

INSTANTIATE_TEST_SUITE_P(
    Hooks, RunBackendFailureTest,
    ValuesIn(std::vector<BackendFailure>{
        {"CreateRuntime", Hook::kCreateRuntime},
        {"SetExternalValueShape", Hook::kSetExternalValueShape},
        {"ReshapeRuntime", Hook::kReshapeRuntime},
        {"GetExternalValueShape", Hook::kGetExternalValueShape},
        {"SetupExternalValues", Hook::kSetupExternalValues},
        {"InvokeRuntime", Hook::kInvokeRuntime},
    }),
    [](const ::testing::TestParamInfo<BackendFailure>& info) {
      return info.param.name;
    });

TEST(ReadOutputTest, ReturnsTheOutputBytes) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  ASSERT_THAT(runner.SetInput(runner.Input(), data), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const LockedBufferSpan<const std::byte> bytes,
                                  runner.ReadOutput(runner.Output()));

  EXPECT_EQ(bytes.size(), 4 * sizeof(float));
}

TEST(ReadOutputTest, RejectsAMissingBuffer) {
  TestRunner runner = MakeTestRunner();

  EXPECT_THAT(runner.ReadOutput(runner.Output()),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(ReadOutputTest, RejectsNonOutputTensors) {
  TestRunner runner = MakeTestRunner();

  EXPECT_THAT(runner.ReadOutput(runner.Input()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ReadOutputTest, RejectsUnknownTensors) {
  TestRunner runner = MakeTestRunner();

  EXPECT_THAT(runner.ReadOutput(UnknownTensor()),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(ReadOutputAsTest, ReturnsTypedValues) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  ASSERT_THAT(runner.SetInput(runner.Input(), data), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  EXPECT_THAT(runner.ReadOutputAs<float>(runner.Output()),
              IsOkAndHolds(ElementsAre(1.f, 2.f, 3.f, 4.f)));
}

TEST(ReadOutputAsTest, RejectsATypeMismatch) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  ASSERT_THAT(runner.SetInput(runner.Input(), data), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  EXPECT_THAT(runner.ReadOutputAs<int32_t>(runner.Output()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(NnpackRunnerTest, MovingTransfersTheRunnerState) {
  TestRunner runner = MakeTestRunner();
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  ASSERT_THAT(runner.SetInput(runner.Input(), data), IsOk());

  TestRunner moved = std::move(runner);
  ASSERT_THAT(moved.Run(), IsOk());

  EXPECT_THAT(moved.ReadOutputAs<float>(moved.Output()),
              IsOkAndHolds(ElementsAre(1.f, 2.f, 3.f, 4.f)));
}

}  // namespace
}  // namespace litert::tensor
