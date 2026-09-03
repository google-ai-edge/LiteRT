/* Copyright 2026 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "flatbuffers/flatbuffer_builder.h"
#include "fuzztest/fuzztest.h"
#include "tflite/c/common.h"
#include "tflite/core/kernels/builtin_op_kernels.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/testing/fuzzing/fuzzing_util.h"
#include "tflite/testing/fuzzing/one_op_fuzz_model.h"

namespace tflite {
namespace ops {
namespace builtin {

TfLiteRegistration* Register_SEGMENT_SUM();
TfLiteRegistration* Register_UNSORTED_SEGMENT_PROD();
TfLiteRegistration* Register_UNSORTED_SEGMENT_MAX();
TfLiteRegistration* Register_UNSORTED_SEGMENT_SUM();
TfLiteRegistration* Register_UNSORTED_SEGMENT_MIN();

}  // namespace builtin
}  // namespace ops

namespace {

using fuzzing::RunResult;

enum class UnsortedSegmentKind { kSum, kProd, kMin, kMax };

struct SegmentSumCase {
  std::vector<int32_t> data_shape;
  std::vector<int64_t> segment_id_values;
  std::vector<uint8_t> data_bytes;
  TensorType data_type;
  bool invoke;
};

struct UnsortedSegmentCase {
  std::vector<int32_t> data_shape;
  std::vector<int32_t> segment_ids_shape;
  std::vector<int64_t> segment_id_values;
  std::vector<uint8_t> data_bytes;
  int32_t num_segments;
  TensorType data_type;
  UnsortedSegmentKind segment_kind;
  bool invoke;
};

constexpr size_t kMaxDataElements = 256;
constexpr size_t kMaxSegmentIdElements = 256;
constexpr size_t kMaxLiveAllocationBytes = 64 * 1024 * 1024;

bool IsSupportedSegmentDataType(TensorType type) {
  return type == TensorType_FLOAT32 || type == TensorType_INT32;
}

RunResult RunSegmentSumCase(const SegmentSumCase& test_case) {
  if (!IsSupportedSegmentDataType(test_case.data_type)) {
    return RunResult::kRejected;
  }
  size_t data_elements = 0;
  if (!fuzzing::CheckedShapeElementCount(test_case.data_shape,
                                         &data_elements) ||
      data_elements > kMaxDataElements) {
    return RunResult::kRejected;
  }
  const int32_t segment_id_count =
      test_case.data_shape.empty()
          ? 0
          : std::max<int32_t>(0, test_case.data_shape[0]);
  if (static_cast<size_t>(segment_id_count) > kMaxSegmentIdElements) {
    return RunResult::kRejected;
  }

  std::vector<uint8_t> data_bytes =
      fuzzing::MakeValues(test_case.data_type, data_elements, 17);
  fuzzing::OverlayBytes(test_case.data_bytes, &data_bytes);
  std::vector<uint8_t> segment_id_bytes = fuzzing::MakeIntegerValues(
      TensorType_INT32, fuzzing::MaterializeValues(test_case.segment_id_values,
                                                   segment_id_count));

  flatbuffers::FlatBufferBuilder builder;
  const auto data_shape = builder.CreateVector(test_case.data_shape);
  const auto segment_ids_shape =
      builder.CreateVector(std::vector<int32_t>{segment_id_count});
  const auto data_tensor =
      CreateTensor(builder, data_shape, test_case.data_type);
  const auto segment_ids_tensor =
      CreateTensor(builder, segment_ids_shape, TensorType_INT32);
  const auto output_tensor =
      CreateTensor(builder, data_shape, test_case.data_type);

  fuzzing::OneOpModelSpec model_spec;
  model_spec.description = "segment_sum_fuzz";
  model_spec.builtin_operator = BuiltinOperator_SEGMENT_SUM;
  model_spec.version = 1;
  model_spec.tensors = {data_tensor, segment_ids_tensor, output_tensor};
  model_spec.buffers = {
      fuzzing::CreateAlignedBuffer(&builder, std::vector<uint8_t>{})};
  model_spec.model_inputs = {0, 1};
  model_spec.model_outputs = {2};
  model_spec.op_inputs = {0, 1};
  model_spec.op_outputs = {2};

  fuzzing::OneOpRunSpec run_spec;
  run_spec.registration = ops::builtin::Register_SEGMENT_SUM();
  run_spec.min_version = 1;
  run_spec.max_version = 1;
  run_spec.max_live_allocation_bytes = kMaxLiveAllocationBytes;
  run_spec.invoke = test_case.invoke;
  run_spec.runtime_tensors.push_back(
      {/*tensor_index=*/0, test_case.data_shape, std::move(data_bytes)});
  run_spec.runtime_tensors.push_back({/*tensor_index=*/1,
                                      std::vector<int32_t>{segment_id_count},
                                      std::move(segment_id_bytes)});

  return fuzzing::BuildAndRunOneOpModel(&builder, model_spec, run_spec);
}

BuiltinOperator BuiltinOperatorForUnsortedSegmentKind(
    UnsortedSegmentKind kind) {
  switch (kind) {
    case UnsortedSegmentKind::kSum:
      return BuiltinOperator_UNSORTED_SEGMENT_SUM;
    case UnsortedSegmentKind::kProd:
      return BuiltinOperator_UNSORTED_SEGMENT_PROD;
    case UnsortedSegmentKind::kMin:
      return BuiltinOperator_UNSORTED_SEGMENT_MIN;
    case UnsortedSegmentKind::kMax:
      return BuiltinOperator_UNSORTED_SEGMENT_MAX;
  }
}

TfLiteRegistration* RegistrationForUnsortedSegmentKind(
    UnsortedSegmentKind kind) {
  switch (kind) {
    case UnsortedSegmentKind::kSum:
      return ops::builtin::Register_UNSORTED_SEGMENT_SUM();
    case UnsortedSegmentKind::kProd:
      return ops::builtin::Register_UNSORTED_SEGMENT_PROD();
    case UnsortedSegmentKind::kMin:
      return ops::builtin::Register_UNSORTED_SEGMENT_MIN();
    case UnsortedSegmentKind::kMax:
      return ops::builtin::Register_UNSORTED_SEGMENT_MAX();
  }
}

RunResult RunUnsortedSegmentCase(const UnsortedSegmentCase& test_case) {
  if (!IsSupportedSegmentDataType(test_case.data_type)) {
    return RunResult::kRejected;
  }
  size_t data_elements = 0;
  size_t segment_id_elements = 0;
  if (!fuzzing::CheckedShapeElementCount(test_case.data_shape,
                                         &data_elements) ||
      !fuzzing::CheckedShapeElementCount(test_case.segment_ids_shape,
                                         &segment_id_elements) ||
      data_elements > kMaxDataElements ||
      segment_id_elements > kMaxSegmentIdElements) {
    return RunResult::kRejected;
  }

  std::vector<uint8_t> data_bytes =
      fuzzing::MakeValues(test_case.data_type, data_elements, 23);
  fuzzing::OverlayBytes(test_case.data_bytes, &data_bytes);
  std::vector<uint8_t> segment_id_bytes = fuzzing::MakeIntegerValues(
      TensorType_INT32, fuzzing::MaterializeValues(test_case.segment_id_values,
                                                   segment_id_elements));
  std::vector<uint8_t> num_segments_bytes = fuzzing::MakeIntegerValues(
      TensorType_INT32, std::vector<int64_t>{test_case.num_segments});

  flatbuffers::FlatBufferBuilder builder;
  const auto data_shape = builder.CreateVector(test_case.data_shape);
  const auto segment_ids_shape =
      builder.CreateVector(test_case.segment_ids_shape);
  const auto num_segments_shape = builder.CreateVector(std::vector<int32_t>{1});
  const auto data_tensor =
      CreateTensor(builder, data_shape, test_case.data_type);
  const auto segment_ids_tensor =
      CreateTensor(builder, segment_ids_shape, TensorType_INT32);
  const auto num_segments_tensor =
      CreateTensor(builder, num_segments_shape, TensorType_INT32);
  const auto output_tensor =
      CreateTensor(builder, data_shape, test_case.data_type);

  fuzzing::OneOpModelSpec model_spec;
  model_spec.description = "unsorted_segment_fuzz";
  model_spec.builtin_operator =
      BuiltinOperatorForUnsortedSegmentKind(test_case.segment_kind);
  model_spec.version = 1;
  model_spec.tensors = {data_tensor, segment_ids_tensor, num_segments_tensor,
                        output_tensor};
  model_spec.buffers = {
      fuzzing::CreateAlignedBuffer(&builder, std::vector<uint8_t>{})};
  model_spec.model_inputs = {0, 1, 2};
  model_spec.model_outputs = {3};
  model_spec.op_inputs = {0, 1, 2};
  model_spec.op_outputs = {3};

  fuzzing::OneOpRunSpec run_spec;
  run_spec.registration =
      RegistrationForUnsortedSegmentKind(test_case.segment_kind);
  run_spec.min_version = 1;
  run_spec.max_version = 1;
  run_spec.max_live_allocation_bytes = kMaxLiveAllocationBytes;
  run_spec.invoke = test_case.invoke;
  run_spec.runtime_tensors.push_back(
      {/*tensor_index=*/0, test_case.data_shape, std::move(data_bytes)});
  run_spec.runtime_tensors.push_back({/*tensor_index=*/1,
                                      test_case.segment_ids_shape,
                                      std::move(segment_id_bytes)});
  run_spec.runtime_tensors.push_back({/*tensor_index=*/2,
                                      std::vector<int32_t>{1},
                                      std::move(num_segments_bytes)});

  return fuzzing::BuildAndRunOneOpModel(&builder, model_spec, run_spec);
}

auto ValidSegmentDataShapeDomain() {
  return fuzztest::VectorOf(fuzztest::InRange<int32_t>(1, 3))
      .WithMinSize(1)
      .WithMaxSize(4);
}

auto ValidSegmentSumCaseDomain() {
  return fuzztest::Map(
      [](std::vector<int32_t> data_shape, uint32_t segment_count_seed,
         std::vector<uint8_t> data_bytes, TensorType data_type) {
        const int32_t row_count = data_shape.front();
        const int32_t segment_count =
            1 + static_cast<int32_t>(segment_count_seed % row_count);
        std::vector<int64_t> segment_ids(row_count);
        for (int32_t i = 0; i < row_count; ++i) {
          segment_ids[i] = static_cast<int64_t>(i) * segment_count / row_count;
        }
        return SegmentSumCase{std::move(data_shape), std::move(segment_ids),
                              std::move(data_bytes), data_type,
                              /*invoke=*/true};
      },
      ValidSegmentDataShapeDomain(), fuzztest::Arbitrary<uint32_t>(),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(64),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_INT32}));
}

auto InvalidSegmentSumCaseDomain() {
  return fuzztest::Map(
      [](std::vector<int32_t> data_shape, TensorType data_type) {
        // SEGMENT_SUM requires the first segment id to be zero.
        std::vector<int64_t> segment_ids(data_shape.front(), 1);
        return SegmentSumCase{std::move(data_shape), std::move(segment_ids),
                              /*data_bytes=*/{}, data_type, /*invoke=*/true};
      },
      ValidSegmentDataShapeDomain(),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_INT32}));
}

auto ValidUnsortedSegmentCaseDomain() {
  return fuzztest::Map(
      [](std::vector<int32_t> data_shape, uint8_t segment_count_seed,
         uint8_t id_offset, std::vector<uint8_t> data_bytes,
         TensorType data_type, UnsortedSegmentKind segment_kind) {
        const int32_t num_segments = 1 + segment_count_seed % 8;
        const int32_t id_count = data_shape.front();
        std::vector<int64_t> segment_ids(id_count);
        for (int32_t i = 0; i < id_count; ++i) {
          segment_ids[i] = (i + id_offset) % num_segments;
        }
        return UnsortedSegmentCase{
            std::move(data_shape), {id_count},   std::move(segment_ids),
            std::move(data_bytes), num_segments, data_type,
            segment_kind,
            /*invoke=*/true};
      },
      ValidSegmentDataShapeDomain(), fuzztest::Arbitrary<uint8_t>(),
      fuzztest::Arbitrary<uint8_t>(),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(64),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_INT32}),
      fuzztest::ElementOf<UnsortedSegmentKind>(
          {UnsortedSegmentKind::kSum, UnsortedSegmentKind::kProd,
           UnsortedSegmentKind::kMin, UnsortedSegmentKind::kMax}));
}

auto InvalidUnsortedSegmentShapeCaseDomain() {
  return fuzztest::Map(
      [](std::vector<int32_t> data_shape, TensorType data_type,
         UnsortedSegmentKind segment_kind) {
        const int32_t invalid_id_count = data_shape.front() + 1;
        return UnsortedSegmentCase{std::move(data_shape),
                                   {invalid_id_count},
                                   std::vector<int64_t>(invalid_id_count, 0),
                                   /*data_bytes=*/{},
                                   /*num_segments=*/1,
                                   data_type,
                                   segment_kind,
                                   /*invoke=*/true};
      },
      ValidSegmentDataShapeDomain(),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_INT32}),
      fuzztest::ElementOf<UnsortedSegmentKind>(
          {UnsortedSegmentKind::kSum, UnsortedSegmentKind::kProd,
           UnsortedSegmentKind::kMin, UnsortedSegmentKind::kMax}));
}

void SegmentSumExecutesValidCases(const SegmentSumCase& test_case) {
  SCOPED_TRACE(::testing::Message()
               << "shape=" << ::testing::PrintToString(test_case.data_shape)
               << ", ids="
               << ::testing::PrintToString(test_case.segment_id_values)
               << ", type=" << static_cast<int>(test_case.data_type));
  ASSERT_EQ(RunSegmentSumCase(test_case), RunResult::kSuccess);
}

void SegmentSumRejectsInvalidIds(const SegmentSumCase& test_case) {
  ASSERT_EQ(RunSegmentSumCase(test_case), RunResult::kRejected);
}

void UnsortedSegmentExecutesValidCases(const UnsortedSegmentCase& test_case) {
  SCOPED_TRACE(
      ::testing::Message()
      << "data_shape=" << ::testing::PrintToString(test_case.data_shape)
      << ", ids_shape=" << ::testing::PrintToString(test_case.segment_ids_shape)
      << ", ids=" << ::testing::PrintToString(test_case.segment_id_values)
      << ", num_segments=" << test_case.num_segments
      << ", type=" << static_cast<int>(test_case.data_type)
      << ", kind=" << static_cast<int>(test_case.segment_kind));
  ASSERT_EQ(RunUnsortedSegmentCase(test_case), RunResult::kSuccess);
}

void UnsortedSegmentRejectsInvalidShape(const UnsortedSegmentCase& test_case) {
  ASSERT_EQ(RunUnsortedSegmentCase(test_case), RunResult::kRejected);
}

FUZZ_TEST(SegmentFuzzTest, SegmentSumExecutesValidCases)
    .WithDomains(ValidSegmentSumCaseDomain());
FUZZ_TEST(SegmentFuzzTest, SegmentSumRejectsInvalidIds)
    .WithDomains(InvalidSegmentSumCaseDomain());
FUZZ_TEST(SegmentFuzzTest, UnsortedSegmentExecutesValidCases)
    .WithDomains(ValidUnsortedSegmentCaseDomain());
FUZZ_TEST(SegmentFuzzTest, UnsortedSegmentRejectsInvalidShape)
    .WithDomains(InvalidUnsortedSegmentShapeCaseDomain());

}  // namespace
}  // namespace tflite
