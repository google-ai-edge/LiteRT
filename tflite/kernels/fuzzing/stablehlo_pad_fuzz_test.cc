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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "flatbuffers/flatbuffer_builder.h"
#include "fuzztest/fuzztest.h"
#include "gtest/gtest.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/core/kernels/builtin_op_kernels.h"
#include "tflite/kernels/fuzzing/fuzzing_util.h"
#include "tflite/kernels/fuzzing/one_op_fuzz_model.h"
#include "tflite/schema/schema_generated.h"

namespace tflite {
namespace {

enum class StablehloOptionsKind {
  kRank,
  kRankMinusOne,
  kRankPlusOne,
  kTooLong,
  kInconsistent,
  kMissingLow,
  kMissingHigh,
  kMissingInterior,
  kMissingBuiltinOptions,
};

enum class PaddingValueShapeKind { kScalar, kOneElementVector, kTwoElements };

struct StablehloPadCase {
  std::vector<int32_t> input_shape;
  std::vector<int64_t> edge_padding_low;
  std::vector<int64_t> edge_padding_high;
  std::vector<int64_t> interior_padding;
  std::vector<uint8_t> input_data;
  std::vector<uint8_t> padding_value_data;
  StablehloOptionsKind options_kind;
  PaddingValueShapeKind padding_value_shape_kind;
  TensorType input_type;
  TensorType padding_value_type;
  TensorType output_type;
  bool invoke;
};

struct StablehloPadStressSpec {
  int32_t rank;
  std::vector<int32_t> input_dims;
  std::vector<int64_t> target_output_dims;
  std::vector<int64_t> interior_padding;
  std::vector<uint8_t> low_padding_split;
  TensorType input_type;
};

struct StablehloOptions {
  std::vector<int64_t> low;
  std::vector<int64_t> high;
  std::vector<int64_t> interior;
  bool has_low = true;
  bool has_high = true;
  bool has_interior = true;
  bool has_builtin_options = true;
};

constexpr int kMaxStablehloRank =
    TFLITE_STABLEHLO_PAD_PARAMS_MAX_DIMENSION_COUNT;
constexpr size_t kMaxInputElementsToPopulate = 4096;
constexpr size_t kMaxFuzzerLiveAllocationBytes = 16 * 1024 * 1024;

size_t OptionSizeForKind(StablehloOptionsKind kind, size_t rank) {
  switch (kind) {
    case StablehloOptionsKind::kRank:
    case StablehloOptionsKind::kMissingLow:
    case StablehloOptionsKind::kMissingHigh:
    case StablehloOptionsKind::kMissingInterior:
    case StablehloOptionsKind::kMissingBuiltinOptions:
      return rank;
    case StablehloOptionsKind::kRankMinusOne:
      return rank == 0 ? 0 : rank - 1;
    case StablehloOptionsKind::kRankPlusOne:
      return rank + 1;
    case StablehloOptionsKind::kTooLong:
      return kMaxStablehloRank + 1;
    case StablehloOptionsKind::kInconsistent:
      return rank;
  }
  return rank;
}

StablehloOptions MakeStablehloOptions(const StablehloPadCase& test_case) {
  const size_t rank = test_case.input_shape.size();
  const size_t option_size = OptionSizeForKind(test_case.options_kind, rank);
  StablehloOptions options;
  options.low = fuzzing::MaterializeValues(test_case.edge_padding_low,
                                           option_size);
  options.high = fuzzing::MaterializeValues(test_case.edge_padding_high,
                                            option_size);
  options.interior = fuzzing::MaterializeValues(test_case.interior_padding,
                                                option_size);
  if (test_case.options_kind == StablehloOptionsKind::kInconsistent) {
    options.interior.push_back(test_case.interior_padding.empty()
                                   ? 0
                                   : test_case.interior_padding[0]);
  }
  options.has_low =
      test_case.options_kind != StablehloOptionsKind::kMissingLow;
  options.has_high =
      test_case.options_kind != StablehloOptionsKind::kMissingHigh;
  options.has_interior =
      test_case.options_kind != StablehloOptionsKind::kMissingInterior;
  options.has_builtin_options =
      test_case.options_kind != StablehloOptionsKind::kMissingBuiltinOptions;
  return options;
}

std::vector<int32_t> PaddingValueShape(PaddingValueShapeKind kind) {
  switch (kind) {
    case PaddingValueShapeKind::kScalar:
      return {};
    case PaddingValueShapeKind::kOneElementVector:
      return {1};
    case PaddingValueShapeKind::kTwoElements:
      return {2};
  }
  return {1};
}

bool PaddingValueHasOneElement(PaddingValueShapeKind kind) {
  return kind == PaddingValueShapeKind::kScalar ||
         kind == PaddingValueShapeKind::kOneElementVector;
}

int64_t OptionValueForDimension(const std::vector<int64_t>& values,
                                size_t index) {
  return index < values.size() ? values[index] : 0;
}

bool DivNegRoundAwayOrZero(int64_t numerator, int64_t denominator,
                           int64_t* result) {
  if (denominator <= 0 || result == nullptr) {
    return false;
  }
  if (numerator >= 0) {
    *result = 0;
    return true;
  }
  int64_t adjusted = 0;
  return fuzzing::CheckedSubInt64(numerator, denominator, &adjusted) &&
         fuzzing::CheckedAddInt64(adjusted, 1, &adjusted) &&
         (*result = adjusted / denominator, true);
}

bool StablehloPadOutputIsWithinFuzzerBudget(
    const StablehloPadCase& test_case) {
  const StablehloOptions options = MakeStablehloOptions(test_case);
  if (!options.has_builtin_options || !options.has_low || !options.has_high ||
      !options.has_interior || options.low.size() != options.high.size() ||
      options.low.size() != options.interior.size() ||
      options.low.size() > kMaxStablehloRank ||
      fuzzing::TypeSize(test_case.input_type) == 0 ||
      test_case.input_type != test_case.padding_value_type ||
      test_case.input_type != test_case.output_type ||
      !PaddingValueHasOneElement(test_case.padding_value_shape_kind)) {
    return false;
  }

  const size_t rank = test_case.input_shape.size();
  if (rank == 0 || rank > kMaxStablehloRank) {
    return false;
  }

  const int64_t element_size =
      static_cast<int64_t>(fuzzing::TypeSize(test_case.input_type));
  int64_t output_shape[kMaxStablehloRank] = {};
  int64_t input_shape[kMaxStablehloRank] = {};
  int64_t interior_step[kMaxStablehloRank] = {};
  int64_t output_dimension_sizes[kMaxStablehloRank] = {};
  int64_t output_strides[kMaxStablehloRank] = {};
  int64_t input_strides[kMaxStablehloRank] = {};

  bool has_non_positive_output_dimension = false;
  for (size_t i = 0; i < rank; ++i) {
    const int64_t input_dim = test_case.input_shape[i];
    const int64_t low = OptionValueForDimension(options.low, i);
    const int64_t high = OptionValueForDimension(options.high, i);
    const int64_t interior = OptionValueForDimension(options.interior, i);
    if (input_dim < 0 || interior < 0 ||
        !fuzzing::CheckedAddInt64(interior, 1, &interior_step[i])) {
      return false;
    }
    const int64_t interior_gap_count = input_dim > 0 ? input_dim - 1 : 0;
    int64_t interior_elements = 0;
    int64_t output_dim = 0;
    if (!fuzzing::CheckedMulInt64(interior_gap_count, interior,
                                  &interior_elements) ||
        !fuzzing::CheckedAddInt64(input_dim, interior_elements, &output_dim) ||
        !fuzzing::CheckedAddInt64(output_dim, low, &output_dim) ||
        !fuzzing::CheckedAddInt64(output_dim, high, &output_dim)) {
      return false;
    }
    output_shape[i] = output_dim;
    if (output_dim <= 0) {
      has_non_positive_output_dimension = true;
    } else if (output_dim > std::numeric_limits<int32_t>::max()) {
      return false;
    }
  }

  size_t input_elements = 0;
  if (!fuzzing::CheckedShapeElementCount(test_case.input_shape,
                                         &input_elements)) {
    return false;
  }
  size_t input_bytes = 0;
  if (!fuzzing::CheckedMultiply(input_elements,
                                fuzzing::TypeSize(test_case.input_type),
                                &input_bytes)) {
    return false;
  }

  if (has_non_positive_output_dimension) {
    return true;
  }

  size_t output_elements = 1;
  for (size_t i = 0; i < rank; ++i) {
    if (!fuzzing::CheckedMultiply(
            output_elements, static_cast<size_t>(output_shape[i]),
            &output_elements)) {
      return false;
    }
  }
  size_t output_bytes = 0;
  if (!fuzzing::CheckedMultiply(output_elements,
                                fuzzing::TypeSize(test_case.input_type),
                                &output_bytes) ||
      output_bytes > kMaxFuzzerLiveAllocationBytes ||
      output_bytes >
          static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    return false;
  }

  output_dimension_sizes[rank - 1] = element_size;
  for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
    if (!fuzzing::CheckedMulInt64(output_shape[i + 1],
                                  output_dimension_sizes[i + 1],
                                  &output_dimension_sizes[i])) {
      return false;
    }
  }

  if (!fuzzing::CheckedMulInt64(element_size, interior_step[rank - 1],
                                &output_strides[rank - 1])) {
    return false;
  }
  for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
    if (!fuzzing::CheckedMulInt64(output_dimension_sizes[i],
                                  interior_step[i], &output_strides[i])) {
      return false;
    }
  }

  int64_t output_offset = 0;
  for (size_t i = 0; i < rank; ++i) {
    int64_t offset_term = 0;
    if (!fuzzing::CheckedMulInt64(
            std::max<int64_t>(OptionValueForDimension(options.low, i), 0),
            output_dimension_sizes[i], &offset_term) ||
        !fuzzing::CheckedAddInt64(output_offset, offset_term,
                                  &output_offset)) {
      return false;
    }
  }

  input_strides[rank - 1] = element_size;
  for (int i = static_cast<int>(rank) - 1; i >= 1; --i) {
    if (!fuzzing::CheckedMulInt64(test_case.input_shape[i],
                                  input_strides[i], &input_strides[i - 1])) {
      return false;
    }
  }

  bool has_input_copy = true;
  for (size_t i = 0; i < rank; ++i) {
    int64_t low_crop = 0;
    int64_t high_crop = 0;
    if (!DivNegRoundAwayOrZero(OptionValueForDimension(options.low, i),
                               interior_step[i], &low_crop) ||
        !DivNegRoundAwayOrZero(OptionValueForDimension(options.high, i),
                               interior_step[i], &high_crop) ||
        !fuzzing::CheckedAddInt64(test_case.input_shape[i], low_crop,
                                  &input_shape[i]) ||
        !fuzzing::CheckedAddInt64(input_shape[i], high_crop,
                                  &input_shape[i])) {
      return false;
    }
    if (input_shape[i] <= 0) {
      has_input_copy = false;
    }
  }
  if (!has_input_copy) {
    return true;
  }

  int64_t input_offset = 0;
  for (size_t i = 0; i < rank; ++i) {
    int64_t low_crop = 0;
    int64_t input_offset_term = 0;
    if (!DivNegRoundAwayOrZero(OptionValueForDimension(options.low, i),
                               interior_step[i], &low_crop) ||
        !fuzzing::CheckedMulInt64(low_crop, input_strides[i],
                                  &input_offset_term) ||
        !fuzzing::CheckedSubInt64(input_offset, input_offset_term,
                                  &input_offset)) {
      return false;
    }
    const int64_t low = OptionValueForDimension(options.low, i);
    if (low < 0) {
      int64_t tmp_offset = (interior_step[i] + low) % interior_step[i];
      if (tmp_offset < 0) tmp_offset += interior_step[i];
      int64_t output_offset_term = 0;
      if (!fuzzing::CheckedMulInt64(tmp_offset, output_dimension_sizes[i],
                                    &output_offset_term) ||
          !fuzzing::CheckedAddInt64(output_offset, output_offset_term,
                                    &output_offset)) {
        return false;
      }
    }
  }
  return input_offset >= 0 && static_cast<size_t>(input_offset) < input_bytes &&
         output_offset >= 0 &&
         static_cast<size_t>(output_offset) < output_bytes;
}

StablehloPadCase MakeStablehloPadStressCase(
    const StablehloPadStressSpec& spec) {
  StablehloPadCase test_case;
  const int32_t rank = std::clamp(spec.rank, int32_t{1}, kMaxStablehloRank);
  test_case.input_shape.reserve(rank);
  test_case.edge_padding_low.reserve(rank);
  test_case.edge_padding_high.reserve(rank);
  test_case.interior_padding.reserve(rank);
  for (int32_t i = 0; i < rank; ++i) {
    const int64_t input_dim =
        std::max<int32_t>(1, spec.input_dims[static_cast<size_t>(i)]);
    const int64_t interior =
        std::max<int64_t>(0, spec.interior_padding[static_cast<size_t>(i)]);
    int64_t dilated_input = input_dim;
    if (input_dim > 0) {
      int64_t step = 0;
      int64_t input_minus_one = 0;
      if (fuzzing::CheckedAddInt64(interior, 1, &step) &&
          fuzzing::CheckedSubInt64(input_dim, 1, &input_minus_one) &&
          fuzzing::CheckedMulInt64(input_minus_one, step, &dilated_input) &&
          fuzzing::CheckedAddInt64(dilated_input, 1, &dilated_input)) {
      } else {
        dilated_input = input_dim;
      }
    }
    int64_t target_output_dim =
        spec.target_output_dims[static_cast<size_t>(i)];
    if (target_output_dim < dilated_input) {
      target_output_dim = dilated_input;
    }
    const int64_t total_padding = target_output_dim - dilated_input;
    const uint8_t split = spec.low_padding_split[static_cast<size_t>(i)];
    const int64_t low =
        total_padding == 0
            ? 0
            : (total_padding / UINT8_MAX) * split +
                  ((total_padding % UINT8_MAX) * split) / UINT8_MAX;
    const int64_t high = total_padding - low;
    test_case.input_shape.push_back(static_cast<int32_t>(input_dim));
    test_case.edge_padding_low.push_back(low);
    test_case.edge_padding_high.push_back(high);
    test_case.interior_padding.push_back(interior);
  }
  test_case.input_data = {};
  test_case.padding_value_data = {};
  test_case.options_kind = StablehloOptionsKind::kRank;
  test_case.padding_value_shape_kind = PaddingValueShapeKind::kOneElementVector;
  test_case.input_type = spec.input_type;
  test_case.padding_value_type = spec.input_type;
  test_case.output_type = spec.input_type;
  test_case.invoke = true;
  return test_case;
}

fuzzing::RunResult RunStablehloPadCase(const StablehloPadCase& test_case) {
  if (fuzzing::TypeSize(test_case.input_type) == 0 ||
      fuzzing::TypeSize(test_case.padding_value_type) == 0 ||
      fuzzing::TypeSize(test_case.output_type) == 0) {
    return fuzzing::RunResult::kRejected;
  }
  const StablehloOptions options = MakeStablehloOptions(test_case);
  const std::vector<int32_t> padding_value_shape =
      PaddingValueShape(test_case.padding_value_shape_kind);
  size_t padding_value_elements = 0;
  if (!fuzzing::CheckedShapeElementCount(padding_value_shape,
                                         &padding_value_elements)) {
    return fuzzing::RunResult::kRejected;
  }
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

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<Buffer>> buffers = {
      fuzzing::CreateAlignedBuffer(&builder, std::vector<uint8_t>{})};
  std::vector<uint8_t> padding_value_bytes = fuzzing::MakeValues(
      test_case.padding_value_type, padding_value_elements, 3);
  fuzzing::OverlayBytes(test_case.padding_value_data, &padding_value_bytes);
  fuzzing::ApplyCentralTensorInputInvariants(test_case.padding_value_type,
                                             &padding_value_bytes);
  buffers.push_back(
      fuzzing::CreateAlignedBuffer(&builder, padding_value_bytes));

  const auto input_tensor =
      CreateTensor(builder, builder.CreateVector(test_case.input_shape),
                   test_case.input_type, 0);
  const auto padding_value_tensor = CreateTensor(
      builder, builder.CreateVector(padding_value_shape),
      test_case.padding_value_type, 1);
  const auto output_tensor =
      CreateTensor(builder, builder.CreateVector(std::vector<int32_t>{}),
                   test_case.output_type, 0);
  std::vector<flatbuffers::Offset<Tensor>> tensors = {
      input_tensor, padding_value_tensor, output_tensor};
  const std::vector<int32_t> op_inputs = {0, 1};
  const std::vector<int32_t> op_outputs = {2};
  const std::vector<int32_t> subgraph_inputs = {0};

  flatbuffers::Offset<void> builtin_options_2 = 0;
  BuiltinOptions2 builtin_options_2_type = BuiltinOptions2_NONE;
  if (options.has_builtin_options) {
    const auto low_offset = options.has_low
                                ? builder.CreateVector(options.low)
                                : flatbuffers::Offset<
                                      flatbuffers::Vector<int64_t>>();
    const auto high_offset = options.has_high
                                 ? builder.CreateVector(options.high)
                                 : flatbuffers::Offset<
                                       flatbuffers::Vector<int64_t>>();
    const auto interior_offset = options.has_interior
                                     ? builder.CreateVector(options.interior)
                                     : flatbuffers::Offset<
                                           flatbuffers::Vector<int64_t>>();
    builtin_options_2 =
        CreateStablehloPadOptions(builder, low_offset, high_offset,
                                  interior_offset)
            .Union();
    builtin_options_2_type = BuiltinOptions2_StablehloPadOptions;
  }

  fuzzing::OneOpModelSpec model_spec;
  model_spec.description = "stablehlo_pad_fuzz";
  model_spec.builtin_operator = BuiltinOperator_STABLEHLO_PAD;
  model_spec.builtin_options_2_type = builtin_options_2_type;
  model_spec.builtin_options_2 = builtin_options_2;
  model_spec.tensors = std::move(tensors);
  model_spec.buffers = std::move(buffers);
  model_spec.model_inputs = subgraph_inputs;
  model_spec.model_outputs = op_outputs;
  model_spec.op_inputs = op_inputs;
  model_spec.op_outputs = op_outputs;

  fuzzing::OneOpRunSpec run_spec;
  run_spec.registration = ops::builtin::Register_STABLEHLO_PAD();
  run_spec.max_live_allocation_bytes = kMaxFuzzerLiveAllocationBytes;
  run_spec.invoke = test_case.invoke;
  run_spec.runtime_tensors.push_back(
      {/*tensor_index=*/0, test_case.input_shape, std::move(input_bytes)});
  return fuzzing::BuildAndRunOneOpModel(&builder, model_spec, run_spec);
}

auto StablehloPaddingValueDomain() {
  return fuzztest::OneOf(
      fuzztest::InRange<int64_t>(-4, 4),
      fuzztest::Just<int64_t>(std::numeric_limits<int64_t>::min()),
      fuzztest::Just<int64_t>(std::numeric_limits<int64_t>::max()),
      fuzztest::Just<int64_t>(INT32_MAX),
      fuzztest::Just<int64_t>(static_cast<int64_t>(INT32_MAX) + 1));
}

template <typename InvokeDomain>
auto StablehloPadCaseDomain(InvokeDomain invoke_domain) {
  return fuzztest::StructOf<StablehloPadCase>(
      fuzztest::VectorOf(fuzztest::InRange<int32_t>(0, 4))
          .WithMinSize(0)
          .WithMaxSize(kMaxStablehloRank + 1),
      fuzztest::VectorOf(StablehloPaddingValueDomain())
          .WithMinSize(0)
          .WithMaxSize(kMaxStablehloRank + 1),
      fuzztest::VectorOf(StablehloPaddingValueDomain())
          .WithMinSize(0)
          .WithMaxSize(kMaxStablehloRank + 1),
      fuzztest::VectorOf(StablehloPaddingValueDomain())
          .WithMinSize(0)
          .WithMaxSize(kMaxStablehloRank + 1),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(64),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(64),
      fuzztest::ElementOf<StablehloOptionsKind>(
          {StablehloOptionsKind::kRank,
           StablehloOptionsKind::kRankMinusOne,
           StablehloOptionsKind::kRankPlusOne,
           StablehloOptionsKind::kTooLong,
           StablehloOptionsKind::kInconsistent,
           StablehloOptionsKind::kMissingLow,
           StablehloOptionsKind::kMissingHigh,
           StablehloOptionsKind::kMissingInterior,
           StablehloOptionsKind::kMissingBuiltinOptions}),
      fuzztest::ElementOf<PaddingValueShapeKind>(
          {PaddingValueShapeKind::kScalar,
           PaddingValueShapeKind::kOneElementVector,
           PaddingValueShapeKind::kTwoElements}),
      fuzztest::ElementOf<TensorType>(
          {TensorType_FLOAT32, TensorType_UINT8, TensorType_INT8,
           TensorType_INT16, TensorType_INT32, TensorType_INT64,
           TensorType_BOOL}),
      fuzztest::ElementOf<TensorType>(
          {TensorType_FLOAT32, TensorType_UINT8, TensorType_INT8,
           TensorType_INT16, TensorType_INT32, TensorType_INT64,
           TensorType_BOOL}),
      fuzztest::ElementOf<TensorType>(
          {TensorType_FLOAT32, TensorType_UINT8, TensorType_INT8,
           TensorType_INT16, TensorType_INT32, TensorType_INT64,
           TensorType_BOOL}),
      std::move(invoke_domain));
}

auto StablehloTargetOutputDimensionDomain() {
  return fuzztest::OneOf(
      fuzztest::InRange<int64_t>(1, 4), fuzztest::Just<int64_t>(32767),
      fuzztest::Just<int64_t>(32768), fuzztest::Just<int64_t>(46340),
      fuzztest::Just<int64_t>(46341), fuzztest::Just<int64_t>(65535),
      fuzztest::Just<int64_t>(65536),
      fuzztest::Just<int64_t>(static_cast<int64_t>(INT32_MAX)),
      fuzztest::Just<int64_t>(static_cast<int64_t>(INT32_MAX) + 1),
      fuzztest::Just<int64_t>(std::numeric_limits<int64_t>::max()));
}

auto StablehloPadStressSpecDomain() {
  return fuzztest::StructOf<StablehloPadStressSpec>(
      fuzztest::InRange<int32_t>(1, kMaxStablehloRank),
      fuzztest::VectorOf(fuzztest::InRange<int32_t>(1, 2))
          .WithMinSize(kMaxStablehloRank)
          .WithMaxSize(kMaxStablehloRank),
      fuzztest::VectorOf(StablehloTargetOutputDimensionDomain())
          .WithMinSize(kMaxStablehloRank)
          .WithMaxSize(kMaxStablehloRank),
      fuzztest::VectorOf(fuzztest::OneOf(fuzztest::InRange<int64_t>(0, 3),
                                         fuzztest::Just<int64_t>(INT32_MAX),
                                         fuzztest::Just<int64_t>(
                                             std::numeric_limits<int64_t>::max())))
          .WithMinSize(kMaxStablehloRank)
          .WithMaxSize(kMaxStablehloRank),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>())
          .WithMinSize(kMaxStablehloRank)
          .WithMaxSize(kMaxStablehloRank),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_INT8,
                                       TensorType_INT32, TensorType_BOOL}));
}

void StablehloPadNeverCrashes(const StablehloPadCase& test_case) {
  EXPECT_NE(RunStablehloPadCase(test_case),
            fuzzing::RunResult::kHarnessFailure);
}

TEST(StablehloPadFuzzTest, StablehloPadSmokeInvokes) {
  StablehloPadCase test_case;
  test_case.input_shape = {2, 3};
  test_case.edge_padding_low = {1, 0};
  test_case.edge_padding_high = {0, 1};
  test_case.interior_padding = {0, 0};
  test_case.input_data = {};
  test_case.padding_value_data = {};
  test_case.options_kind = StablehloOptionsKind::kRank;
  test_case.padding_value_shape_kind = PaddingValueShapeKind::kOneElementVector;
  test_case.input_type = TensorType_INT32;
  test_case.padding_value_type = TensorType_INT32;
  test_case.output_type = TensorType_INT32;
  test_case.invoke = true;

  EXPECT_EQ(RunStablehloPadCase(test_case), fuzzing::RunResult::kSuccess);
}

void StablehloPadRejectsInvalidOrOversizedOutput(
    const StablehloPadCase& test_case) {
  const bool must_reject = !StablehloPadOutputIsWithinFuzzerBudget(test_case);
  const fuzzing::RunResult result = RunStablehloPadCase(test_case);
  EXPECT_NE(result, fuzzing::RunResult::kHarnessFailure);
  if (must_reject) {
    EXPECT_EQ(result, fuzzing::RunResult::kRejected);
  }
}

void StablehloPadRejectsProductStressOverflow(
    const StablehloPadStressSpec& spec) {
  const StablehloPadCase test_case = MakeStablehloPadStressCase(spec);
  const bool must_reject = !StablehloPadOutputIsWithinFuzzerBudget(test_case);
  const fuzzing::RunResult result = RunStablehloPadCase(test_case);
  EXPECT_NE(result, fuzzing::RunResult::kHarnessFailure);
  if (must_reject) {
    EXPECT_EQ(result, fuzzing::RunResult::kRejected);
  }
}

FUZZ_TEST(StablehloPadFuzzTest, StablehloPadNeverCrashes)
    .WithDomains(StablehloPadCaseDomain(fuzztest::Arbitrary<bool>()));
FUZZ_TEST(StablehloPadFuzzTest, StablehloPadRejectsInvalidOrOversizedOutput)
    .WithDomains(StablehloPadCaseDomain(fuzztest::Just(true)));
FUZZ_TEST(StablehloPadFuzzTest, StablehloPadRejectsProductStressOverflow)
    .WithDomains(StablehloPadStressSpecDomain());

}  // namespace
}  // namespace tflite
