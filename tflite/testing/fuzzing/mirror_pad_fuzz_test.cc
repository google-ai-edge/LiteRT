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
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"
#include "fuzztest/fuzztest.h"
#include "tflite/core/kernels/builtin_op_kernels.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/testing/fuzzing/fuzzing_util.h"
#include "tflite/testing/fuzzing/one_op_fuzz_model.h"

namespace tflite {
namespace {

enum class PaddingShapeKind { kValid, kWrongRows, kWrongColumns, kRankOne };

struct MirrorPadCase {
  std::vector<int32_t> input_shape;
  std::vector<int64_t> padding_values;
  std::vector<uint8_t> input_data;
  std::vector<uint8_t> padding_data;
  PaddingShapeKind padding_shape_kind;
  TensorType input_type;
  TensorType output_type;
  TensorType padding_type;
  MirrorPadMode mode;
  bool dynamic_paddings;
  bool quantized_int8;
  bool invoke;
};

struct MirrorPadStressSpec {
  int32_t rank;
  std::vector<int32_t> input_dims;
  std::vector<uint8_t> padding_split;
  MirrorPadMode mode;
  TensorType input_type;
  TensorType padding_type;
  bool quantized_int8;
};

constexpr size_t kMaxInputElementsToPopulate = 4096;
constexpr size_t kMaxFuzzerLiveAllocationBytes = 16 * 1024 * 1024;
constexpr int32_t kMaxStressRank = 8;

std::vector<int32_t> PaddingShape(const MirrorPadCase& test_case, size_t rank,
                                  size_t* rows, size_t* columns) {
  *rows = rank;
  *columns = 2;
  switch (test_case.padding_shape_kind) {
    case PaddingShapeKind::kWrongRows:
      *rows = rank + 1;
      break;
    case PaddingShapeKind::kWrongColumns:
      *columns = 1 + (rank % 3);
      if (*columns == 2) *columns = 3;
      break;
    case PaddingShapeKind::kRankOne:
      return {static_cast<int32_t>(rank * 2)};
    case PaddingShapeKind::kValid:
      break;
  }
  return {static_cast<int32_t>(*rows), static_cast<int32_t>(*columns)};
}

std::vector<uint8_t> MaterializePaddingBytes(const MirrorPadCase& test_case,
                                             size_t padding_count) {
  const std::vector<int64_t> padding_values =
      fuzzing::MaterializeValues(test_case.padding_values, padding_count);
  std::vector<uint8_t> bytes =
      fuzzing::MakeIntegerValues(test_case.padding_type, padding_values);
  fuzzing::OverlayBytes(test_case.padding_data, &bytes);
  fuzzing::ApplyCentralTensorInputInvariants(test_case.padding_type, &bytes);
  return bytes;
}

template <typename T>
std::vector<int64_t> DecodePaddingValues(const std::vector<uint8_t>& bytes) {
  if (bytes.size() % sizeof(T) != 0) {
    return {};
  }
  std::vector<int64_t> values(bytes.size() / sizeof(T));
  for (size_t i = 0; i < values.size(); ++i) {
    T value;
    std::memcpy(&value, bytes.data() + i * sizeof(T), sizeof(T));
    values[i] = static_cast<int64_t>(value);
  }
  return values;
}

std::vector<int64_t> DecodePaddingValues(TensorType type,
                                         const std::vector<uint8_t>& bytes) {
  switch (type) {
    case TensorType_INT32:
      return DecodePaddingValues<int32_t>(bytes);
    case TensorType_INT64:
      return DecodePaddingValues<int64_t>(bytes);
    default:
      return {};
  }
}

bool MirrorPadOutputIsWithinFuzzerBudget(const MirrorPadCase& test_case) {
  if (test_case.padding_shape_kind != PaddingShapeKind::kValid ||
      test_case.input_type != test_case.output_type ||
      (test_case.padding_type != TensorType_INT32 &&
       test_case.padding_type != TensorType_INT64) ||
      fuzzing::TypeSize(test_case.input_type) == 0 ||
      (test_case.mode != MirrorPadMode_REFLECT &&
       test_case.mode != MirrorPadMode_SYMMETRIC)) {
    return false;
  }

  const size_t rank = test_case.input_shape.size();
  const int64_t mode_offset = test_case.mode == MirrorPadMode_REFLECT ? 1 : 0;
  const std::vector<int64_t> padding_values = DecodePaddingValues(
      test_case.padding_type, MaterializePaddingBytes(test_case, rank * 2));
  if (padding_values.size() != rank * 2) {
    return false;
  }

  size_t output_elements = 1;
  for (size_t i = 0; i < rank; ++i) {
    const int64_t input_dim = test_case.input_shape[i];
    const int64_t left_padding = padding_values[i * 2];
    const int64_t right_padding = padding_values[i * 2 + 1];
    if (input_dim < 0 || left_padding < 0 || right_padding < 0) {
      return false;
    }
    const int64_t max_padding = input_dim - mode_offset;
    if (left_padding > max_padding || right_padding > max_padding) {
      return false;
    }
    int64_t output_dim = 0;
    int64_t partial = 0;
    if (!fuzzing::CheckedAddInt64(input_dim, left_padding, &partial) ||
        !fuzzing::CheckedAddInt64(partial, right_padding, &output_dim) ||
        output_dim < 0 || output_dim > std::numeric_limits<int32_t>::max()) {
      return false;
    }
    if (!fuzzing::CheckedMultiply(output_elements,
                                  static_cast<size_t>(output_dim),
                                  &output_elements)) {
      return false;
    }
  }

  size_t output_bytes = 0;
  if (!fuzzing::CheckedMultiply(output_elements,
                                fuzzing::TypeSize(test_case.input_type),
                                &output_bytes)) {
    return false;
  }
  return output_bytes <= kMaxFuzzerLiveAllocationBytes;
}

MirrorPadCase MakeMirrorPadStressCase(const MirrorPadStressSpec& spec) {
  MirrorPadCase test_case;
  const int32_t rank = std::clamp(spec.rank, int32_t{1}, kMaxStressRank);
  const int64_t mode_offset = spec.mode == MirrorPadMode_REFLECT ? 1 : 0;
  test_case.input_shape.reserve(rank);
  test_case.padding_values.reserve(static_cast<size_t>(rank) * 2);
  for (int32_t i = 0; i < rank; ++i) {
    const int32_t input_dim =
        std::max<int32_t>(1, spec.input_dims[static_cast<size_t>(i)]);
    const int64_t max_padding =
        std::max<int64_t>(0, static_cast<int64_t>(input_dim) - mode_offset);
    const uint8_t split = spec.padding_split[static_cast<size_t>(i)];
    const int64_t left_padding =
        max_padding == 0 ? 0 : (max_padding * split) / UINT8_MAX;
    const int64_t right_padding = max_padding - left_padding;
    test_case.input_shape.push_back(input_dim);
    test_case.padding_values.push_back(left_padding);
    test_case.padding_values.push_back(right_padding);
  }
  test_case.input_data = {};
  test_case.padding_data = {};
  test_case.padding_shape_kind = PaddingShapeKind::kValid;
  test_case.input_type = spec.input_type;
  test_case.output_type = spec.input_type;
  test_case.padding_type = spec.padding_type;
  test_case.mode = spec.mode;
  test_case.dynamic_paddings = false;
  test_case.quantized_int8 = spec.quantized_int8;
  test_case.invoke = true;
  return test_case;
}

fuzzing::RunResult RunMirrorPadCase(const MirrorPadCase& test_case) {
  if (fuzzing::TypeSize(test_case.input_type) == 0 ||
      fuzzing::TypeSize(test_case.output_type) == 0 ||
      fuzzing::TypeSize(test_case.padding_type) == 0) {
    return fuzzing::RunResult::kRejected;
  }

  const size_t rank = test_case.input_shape.size();
  size_t padding_rows = 0;
  size_t padding_columns = 0;
  const std::vector<int32_t> padding_shape =
      PaddingShape(test_case, rank, &padding_rows, &padding_columns);
  size_t input_elements = 0;
  if (!fuzzing::CheckedShapeElementCount(test_case.input_shape,
                                         &input_elements) ||
      input_elements > kMaxInputElementsToPopulate) {
    return fuzzing::RunResult::kRejected;
  }
  std::vector<uint8_t> input_bytes =
      fuzzing::MakeValues(test_case.input_type, input_elements, 1);
  fuzzing::OverlayBytes(test_case.input_data, &input_bytes);
  fuzzing::ApplyCentralTensorInputInvariants(test_case.input_type,
                                             &input_bytes);
  std::vector<uint8_t> padding_bytes =
      MaterializePaddingBytes(test_case, padding_rows * padding_columns);

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<Buffer>> buffers = {
      fuzzing::CreateAlignedBuffer(&builder, std::vector<uint8_t>{})};
  if (!test_case.dynamic_paddings) {
    buffers.push_back(fuzzing::CreateAlignedBuffer(&builder, padding_bytes));
  }

  const bool quantized =
      test_case.quantized_int8 && (test_case.input_type == TensorType_INT8 ||
                                   test_case.input_type == TensorType_UINT8 ||
                                   test_case.input_type == TensorType_INT16);
  flatbuffers::Offset<QuantizationParameters> quantization = 0;
  if (quantized) {
    quantization = CreateQuantizationParameters(
        builder, 0, 0, builder.CreateVector<float>({0.25f}),
        builder.CreateVector<int64_t>({0}));
  }

  const auto input_tensor =
      CreateTensor(builder, builder.CreateVector(test_case.input_shape),
                   test_case.input_type, 0, 0, quantization);
  const auto padding_tensor =
      CreateTensor(builder, builder.CreateVector(padding_shape),
                   test_case.padding_type, test_case.dynamic_paddings ? 0 : 1);
  const auto output_tensor =
      CreateTensor(builder, builder.CreateVector(std::vector<int32_t>{}),
                   test_case.output_type, 0, 0, quantization);

  std::vector<flatbuffers::Offset<Tensor>> tensors = {
      input_tensor, padding_tensor, output_tensor};
  const std::vector<int32_t> op_inputs = {0, 1};
  const std::vector<int32_t> op_outputs = {2};
  const std::vector<int32_t> subgraph_inputs = test_case.dynamic_paddings
                                                   ? std::vector<int32_t>{0, 1}
                                                   : std::vector<int32_t>{0};
  const auto options = CreateMirrorPadOptions(builder, test_case.mode).Union();
  fuzzing::OneOpModelSpec model_spec;
  model_spec.description = "mirror_pad_fuzz";
  model_spec.builtin_operator = BuiltinOperator_MIRROR_PAD;
  model_spec.builtin_options_type = BuiltinOptions_MirrorPadOptions;
  model_spec.builtin_options = options;
  model_spec.tensors = std::move(tensors);
  model_spec.buffers = std::move(buffers);
  model_spec.model_inputs = subgraph_inputs;
  model_spec.model_outputs = op_outputs;
  model_spec.op_inputs = op_inputs;
  model_spec.op_outputs = op_outputs;

  fuzzing::OneOpRunSpec run_spec;
  run_spec.registration = ops::builtin::Register_MIRROR_PAD();
  run_spec.max_live_allocation_bytes = kMaxFuzzerLiveAllocationBytes;
  run_spec.invoke = test_case.invoke;
  run_spec.runtime_tensors.push_back(
      {/*tensor_index=*/0, test_case.input_shape, std::move(input_bytes)});
  if (test_case.dynamic_paddings) {
    run_spec.runtime_tensors.push_back(
        {/*tensor_index=*/1, padding_shape, std::move(padding_bytes)});
  }
  return fuzzing::BuildAndRunOneOpModel(&builder, model_spec, run_spec);
}

auto ValidMirrorPadInputShapeDomain() {
  return fuzztest::VectorOf(fuzztest::InRange<int32_t>(2, 5))
      .WithMinSize(1)
      .WithMaxSize(5);
}

auto MirrorPadValueSeedDomain() {
  return fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>())
      .WithMinSize(10)
      .WithMaxSize(10);
}

template <typename PaddingShapeKindDomain>
auto StructuredMirrorPadCaseDomain(
    PaddingShapeKindDomain padding_shape_kind_domain) {
  return fuzztest::Map(
      [](std::vector<int32_t> input_shape, std::vector<uint8_t> padding_seeds,
         std::vector<uint8_t> input_data, PaddingShapeKind padding_shape_kind,
         TensorType input_type, TensorType padding_type, MirrorPadMode mode,
         bool dynamic_paddings, bool quantized_int8) {
        std::vector<int64_t> padding_values(input_shape.size() * 2);
        const int32_t mode_offset = mode == MirrorPadMode_REFLECT ? 1 : 0;
        for (size_t i = 0; i < input_shape.size(); ++i) {
          const int32_t max_padding = input_shape[i] - mode_offset;
          padding_values[2 * i] = padding_seeds[2 * i] % (max_padding + 1);
          padding_values[2 * i + 1] =
              padding_seeds[2 * i + 1] % (max_padding + 1);
        }
        return MirrorPadCase{std::move(input_shape),
                             std::move(padding_values),
                             std::move(input_data),
                             /*padding_data=*/{},
                             padding_shape_kind,
                             input_type,
                             input_type,
                             padding_type,
                             mode,
                             dynamic_paddings,
                             quantized_int8 && input_type == TensorType_INT8,
                             /*invoke=*/true};
      },
      ValidMirrorPadInputShapeDomain(), MirrorPadValueSeedDomain(),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(64),
      std::move(padding_shape_kind_domain),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_UINT8,
                                       TensorType_INT8, TensorType_INT16,
                                       TensorType_INT32, TensorType_INT64,
                                       TensorType_BOOL}),
      fuzztest::ElementOf<TensorType>({TensorType_INT32, TensorType_INT64}),
      fuzztest::ElementOf<MirrorPadMode>(
          {MirrorPadMode_REFLECT, MirrorPadMode_SYMMETRIC}),
      fuzztest::Arbitrary<bool>(), fuzztest::Arbitrary<bool>());
}

auto ValidMirrorPadCaseDomain() {
  return StructuredMirrorPadCaseDomain(
      fuzztest::Just(PaddingShapeKind::kValid));
}

auto MalformedMirrorPadCaseDomain() {
  return StructuredMirrorPadCaseDomain(fuzztest::ElementOf<PaddingShapeKind>(
      {PaddingShapeKind::kWrongRows, PaddingShapeKind::kWrongColumns,
       PaddingShapeKind::kRankOne}));
}

auto StressInputDimensionDomain() {
  return fuzztest::OneOf(
      fuzztest::InRange<int32_t>(1, 4), fuzztest::Just<int32_t>(32767),
      fuzztest::Just<int32_t>(32768), fuzztest::Just<int32_t>(46340),
      fuzztest::Just<int32_t>(65536), fuzztest::Just<int32_t>(715827882),
      fuzztest::Just<int32_t>(715827883),
      fuzztest::Just<int32_t>(std::numeric_limits<int32_t>::max()));
}

auto MirrorPadStressSpecDomain() {
  return fuzztest::StructOf<MirrorPadStressSpec>(
      fuzztest::InRange<int32_t>(1, kMaxStressRank),
      fuzztest::VectorOf(StressInputDimensionDomain())
          .WithMinSize(kMaxStressRank)
          .WithMaxSize(kMaxStressRank),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>())
          .WithMinSize(kMaxStressRank)
          .WithMaxSize(kMaxStressRank),
      fuzztest::ElementOf<MirrorPadMode>(
          {MirrorPadMode_REFLECT, MirrorPadMode_SYMMETRIC}),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_INT8,
                                       TensorType_INT32, TensorType_BOOL}),
      fuzztest::ElementOf<TensorType>({TensorType_INT32, TensorType_INT64}),
      fuzztest::Arbitrary<bool>());
}

void MirrorPadExecutesValidCases(const MirrorPadCase& test_case) {
  ASSERT_EQ(RunMirrorPadCase(test_case), fuzzing::RunResult::kSuccess);
}

void MirrorPadRejectsMalformedPaddingShape(const MirrorPadCase& test_case) {
  ASSERT_EQ(RunMirrorPadCase(test_case), fuzzing::RunResult::kRejected);
}

TEST(MirrorPadFuzzTest, MirrorPadSmokeInvokes) {
  MirrorPadCase test_case;
  test_case.input_shape = {2, 3};
  test_case.padding_values = {1, 1, 1, 1};
  test_case.input_data = {};
  test_case.padding_data = {};
  test_case.padding_shape_kind = PaddingShapeKind::kValid;
  test_case.input_type = TensorType_INT32;
  test_case.output_type = TensorType_INT32;
  test_case.padding_type = TensorType_INT32;
  test_case.mode = MirrorPadMode_REFLECT;
  test_case.dynamic_paddings = false;
  test_case.quantized_int8 = false;
  test_case.invoke = true;

  EXPECT_EQ(RunMirrorPadCase(test_case), fuzzing::RunResult::kSuccess);
}

TEST(MirrorPadFuzzTest, PaddingDataOverlayControlsValidation) {
  MirrorPadCase test_case;
  test_case.input_shape = {1};
  test_case.padding_values = {
      static_cast<int64_t>(std::numeric_limits<int32_t>::min()) - 1};
  test_case.input_data = {};
  test_case.padding_data = {1, 0, 0, 0, 0, 0, 0, 0};
  test_case.padding_shape_kind = PaddingShapeKind::kValid;
  test_case.input_type = TensorType_INT32;
  test_case.output_type = TensorType_INT32;
  test_case.padding_type = TensorType_INT32;
  test_case.mode = MirrorPadMode_SYMMETRIC;
  test_case.dynamic_paddings = false;
  test_case.quantized_int8 = false;
  test_case.invoke = true;

  EXPECT_TRUE(MirrorPadOutputIsWithinFuzzerBudget(test_case));
  EXPECT_EQ(RunMirrorPadCase(test_case), fuzzing::RunResult::kSuccess);
}

void MirrorPadRejectsProductStressOverflow(const MirrorPadStressSpec& spec) {
  const MirrorPadCase test_case = MakeMirrorPadStressCase(spec);
  const bool must_reject = !MirrorPadOutputIsWithinFuzzerBudget(test_case);
  const fuzzing::RunResult result = RunMirrorPadCase(test_case);
  EXPECT_NE(result, fuzzing::RunResult::kHarnessFailure);
  if (must_reject) {
    EXPECT_EQ(result, fuzzing::RunResult::kRejected);
  }
}

FUZZ_TEST(MirrorPadFuzzTest, MirrorPadExecutesValidCases)
    .WithDomains(ValidMirrorPadCaseDomain());
FUZZ_TEST(MirrorPadFuzzTest, MirrorPadRejectsMalformedPaddingShape)
    .WithDomains(MalformedMirrorPadCaseDomain());
FUZZ_TEST(MirrorPadFuzzTest, MirrorPadRejectsProductStressOverflow)
    .WithDomains(MirrorPadStressSpecDomain());

}  // namespace
}  // namespace tflite
