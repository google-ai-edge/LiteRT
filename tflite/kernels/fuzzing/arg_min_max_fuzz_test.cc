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

#include <cstdint>
#include <limits>
#include <vector>

#include "flatbuffers/flatbuffer_builder.h"
#include "fuzztest/fuzztest.h"
#include "gtest/gtest.h"
#include "tflite/core/kernels/builtin_op_kernels.h"
#include "tflite/kernels/fuzzing/fuzzing_util.h"
#include "tflite/kernels/fuzzing/one_op_fuzz_model.h"
#include "tflite/schema/schema_generated.h"

namespace tflite {
namespace ops {
namespace builtin {

TfLiteRegistration* Register_ARG_MAX();
TfLiteRegistration* Register_ARG_MIN();

}  // namespace builtin
}  // namespace ops

namespace {

using fuzzing::RunResult;

struct ArgMinMaxCase {
  std::vector<int32_t> input_shape;
  std::vector<uint8_t> input_data;
  int64_t axis_value;
  TensorType input_type;
  TensorType axis_type;
  TensorType output_type;
  bool is_arg_max;
  bool dynamic_axis;
  bool invoke;
};

constexpr size_t kMaxInputElements = 256;
constexpr size_t kMaxLiveAllocationBytes = 64 * 1024 * 1024;

bool IsSupportedInputType(TensorType type) {
  return type == TensorType_FLOAT32 || type == TensorType_UINT8 ||
         type == TensorType_INT8 || type == TensorType_INT32 ||
         type == TensorType_BOOL;
}

RunResult RunArgMinMaxCase(const ArgMinMaxCase& test_case) {
  if (!IsSupportedInputType(test_case.input_type) ||
      (test_case.axis_type != TensorType_INT32 &&
       test_case.axis_type != TensorType_INT64) ||
      (test_case.output_type != TensorType_INT32 &&
       test_case.output_type != TensorType_INT64)) {
    return RunResult::kRejected;
  }

  size_t input_elements = 0;
  if (fuzzing::TypeSize(test_case.input_type) == 0 ||
      !fuzzing::CheckedShapeElementCount(test_case.input_shape,
                                         &input_elements) ||
      input_elements > kMaxInputElements) {
    return RunResult::kRejected;
  }

  std::vector<uint8_t> input_bytes =
      fuzzing::MakeValues(test_case.input_type, input_elements, 5);
  fuzzing::OverlayBytes(test_case.input_data, &input_bytes);
  fuzzing::ApplyCentralTensorInputInvariants(test_case.input_type,
                                             &input_bytes);

  std::vector<uint8_t> axis_bytes = fuzzing::MakeIntegerValues(
      test_case.axis_type, std::vector<int64_t>{test_case.axis_value});

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<Buffer>> buffers = {
      fuzzing::CreateAlignedBuffer(&builder, std::vector<uint8_t>{})};
  if (!test_case.dynamic_axis) {
    buffers.push_back(fuzzing::CreateAlignedBuffer(&builder, axis_bytes));
  }

  const auto input_shape = builder.CreateVector(test_case.input_shape);
  const auto axis_shape = builder.CreateVector(std::vector<int32_t>{1});
  const auto output_shape = builder.CreateVector(test_case.input_shape);
  const auto input_tensor =
      CreateTensor(builder, input_shape, test_case.input_type);
  const auto axis_tensor =
      CreateTensor(builder, axis_shape, test_case.axis_type,
                   test_case.dynamic_axis ? 0 : 1);
  const auto output_tensor =
      CreateTensor(builder, output_shape, test_case.output_type);

  const BuiltinOperator op = test_case.is_arg_max ? BuiltinOperator_ARG_MAX
                                                  : BuiltinOperator_ARG_MIN;
  const BuiltinOptions options_type = test_case.is_arg_max
                                          ? BuiltinOptions_ArgMaxOptions
                                          : BuiltinOptions_ArgMinOptions;
  const auto options =
      test_case.is_arg_max
          ? CreateArgMaxOptions(builder, test_case.output_type).Union()
          : CreateArgMinOptions(builder, test_case.output_type).Union();

  fuzzing::OneOpModelSpec model_spec;
  model_spec.description = "arg_min_max_fuzz";
  model_spec.builtin_operator = op;
  model_spec.version = 1;
  model_spec.builtin_options_type = options_type;
  model_spec.builtin_options = options;
  model_spec.tensors = {input_tensor, axis_tensor, output_tensor};
  model_spec.buffers = std::move(buffers);
  model_spec.model_inputs = test_case.dynamic_axis ? std::vector<int32_t>{0, 1}
                                                   : std::vector<int32_t>{0};
  model_spec.model_outputs = {2};
  model_spec.op_inputs = {0, 1};
  model_spec.op_outputs = {2};

  fuzzing::OneOpRunSpec run_spec;
  run_spec.registration = test_case.is_arg_max
                              ? ops::builtin::Register_ARG_MAX()
                              : ops::builtin::Register_ARG_MIN();
  run_spec.min_version = 1;
  run_spec.max_version = 2;
  run_spec.max_live_allocation_bytes = kMaxLiveAllocationBytes;
  run_spec.invoke = test_case.invoke;
  run_spec.runtime_tensors.push_back(
      {/*tensor_index=*/0, test_case.input_shape, std::move(input_bytes)});
  if (test_case.dynamic_axis) {
    run_spec.runtime_tensors.push_back(
        {/*tensor_index=*/1, std::vector<int32_t>{1}, std::move(axis_bytes)});
  }

  return fuzzing::BuildAndRunOneOpModel(&builder, model_spec, run_spec);
}

auto AxisValueDomain() {
  return fuzztest::OneOf(
      fuzztest::InRange<int64_t>(-80, 81),
      fuzztest::Just<int64_t>(std::numeric_limits<int32_t>::max()),
      fuzztest::Just<int64_t>(std::numeric_limits<int32_t>::min()),
      fuzztest::Just<int64_t>(std::numeric_limits<int64_t>::max()),
      fuzztest::Just<int64_t>(std::numeric_limits<int64_t>::min()));
}

auto ArgMinMaxCaseDomain() {
  return fuzztest::StructOf<ArgMinMaxCase>(
      fuzztest::VectorOf(fuzztest::InRange<int32_t>(0, 4))
          .WithMinSize(0)
          .WithMaxSize(8),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(64),
      AxisValueDomain(),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_UINT8,
                                       TensorType_INT8, TensorType_INT32,
                                       TensorType_BOOL}),
      fuzztest::ElementOf<TensorType>({TensorType_INT32, TensorType_INT64}),
      fuzztest::ElementOf<TensorType>({TensorType_INT32, TensorType_INT64}),
      fuzztest::Arbitrary<bool>(), fuzztest::Arbitrary<bool>(),
      fuzztest::Arbitrary<bool>());
}

void ArgMinMaxNeverCrashes(const ArgMinMaxCase& test_case) {
  EXPECT_NE(RunArgMinMaxCase(test_case), RunResult::kHarnessFailure);
}

TEST(ArgMinMaxFuzzTest, Rank64Smoke) {
  ArgMinMaxCase test_case;
  test_case.input_shape.assign(64, 1);
  test_case.input_shape.back() = 2;
  test_case.axis_value = -1;
  test_case.input_type = TensorType_INT32;
  test_case.axis_type = TensorType_INT64;
  test_case.output_type = TensorType_INT32;
  test_case.is_arg_max = true;
  test_case.dynamic_axis = true;
  test_case.invoke = true;
  EXPECT_EQ(RunArgMinMaxCase(test_case), RunResult::kSuccess);
}

FUZZ_TEST(ArgMinMaxFuzzTest, ArgMinMaxNeverCrashes)
    .WithDomains(ArgMinMaxCaseDomain());

}  // namespace
}  // namespace tflite
