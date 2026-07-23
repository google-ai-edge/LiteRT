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

TfLiteRegistration* Register_CUMSUM();

}  // namespace builtin
}  // namespace ops

namespace {

using fuzzing::RunResult;

struct CumsumCase {
  std::vector<int32_t> input_shape;
  std::vector<uint8_t> input_data;
  int32_t axis_value;
  TensorType input_type;
  bool exclusive;
  bool reverse;
  bool invoke;
};

constexpr size_t kMaxInputElements = 256;
constexpr size_t kMaxLiveAllocationBytes = 64 * 1024 * 1024;

RunResult RunCumsumCase(const CumsumCase& test_case) {
  if (test_case.input_type != TensorType_FLOAT32 &&
      test_case.input_type != TensorType_INT32 &&
      test_case.input_type != TensorType_INT64) {
    return RunResult::kRejected;
  }

  size_t input_elements = 0;
  if (!fuzzing::CheckedShapeElementCount(test_case.input_shape,
                                         &input_elements) ||
      input_elements > kMaxInputElements) {
    return RunResult::kRejected;
  }

  std::vector<uint8_t> input_bytes =
      fuzzing::MakeValues(test_case.input_type, input_elements, 11);
  fuzzing::OverlayBytes(test_case.input_data, &input_bytes);
  std::vector<uint8_t> axis_bytes = fuzzing::MakeIntegerValues(
      TensorType_INT32, std::vector<int64_t>{test_case.axis_value});

  flatbuffers::FlatBufferBuilder builder;
  const auto input_shape = builder.CreateVector(test_case.input_shape);
  const auto axis_shape = builder.CreateVector(std::vector<int32_t>{1});
  const auto input_tensor =
      CreateTensor(builder, input_shape, test_case.input_type);
  const auto axis_tensor = CreateTensor(builder, axis_shape, TensorType_INT32);
  const auto output_tensor =
      CreateTensor(builder, input_shape, test_case.input_type);

  fuzzing::OneOpModelSpec model_spec;
  model_spec.description = "cumsum_fuzz";
  model_spec.builtin_operator = BuiltinOperator_CUMSUM;
  model_spec.version = 1;
  model_spec.builtin_options_type = BuiltinOptions_CumsumOptions;
  model_spec.builtin_options =
      CreateCumsumOptions(builder, test_case.exclusive, test_case.reverse)
          .Union();
  model_spec.tensors = {input_tensor, axis_tensor, output_tensor};
  model_spec.buffers = {
      fuzzing::CreateAlignedBuffer(&builder, std::vector<uint8_t>{})};
  model_spec.model_inputs = {0, 1};
  model_spec.model_outputs = {2};
  model_spec.op_inputs = {0, 1};
  model_spec.op_outputs = {2};

  fuzzing::OneOpRunSpec run_spec;
  run_spec.registration = ops::builtin::Register_CUMSUM();
  run_spec.min_version = 1;
  run_spec.max_version = 1;
  run_spec.max_live_allocation_bytes = kMaxLiveAllocationBytes;
  run_spec.invoke = test_case.invoke;
  run_spec.runtime_tensors.push_back(
      {/*tensor_index=*/0, test_case.input_shape, std::move(input_bytes)});
  run_spec.runtime_tensors.push_back(
      {/*tensor_index=*/1, std::vector<int32_t>{1}, std::move(axis_bytes)});

  return fuzzing::BuildAndRunOneOpModel(&builder, model_spec, run_spec);
}

auto CumsumCaseDomain() {
  return fuzztest::StructOf<CumsumCase>(
      fuzztest::VectorOf(fuzztest::InRange<int32_t>(0, 4))
          .WithMinSize(0)
          .WithMaxSize(8),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(64),
      fuzztest::OneOf(
          fuzztest::InRange<int32_t>(-80, 81),
          fuzztest::Just<int32_t>(std::numeric_limits<int32_t>::max()),
          fuzztest::Just<int32_t>(std::numeric_limits<int32_t>::min())),
      fuzztest::ElementOf<TensorType>(
          {TensorType_FLOAT32, TensorType_INT32, TensorType_INT64}),
      fuzztest::Arbitrary<bool>(), fuzztest::Arbitrary<bool>(),
      fuzztest::Arbitrary<bool>());
}

void CumsumNeverCrashes(const CumsumCase& test_case) {
  EXPECT_NE(RunCumsumCase(test_case), RunResult::kHarnessFailure);
}

TEST(CumsumFuzzTest, Rank64Smoke) {
  CumsumCase test_case;
  test_case.input_shape.assign(64, 1);
  test_case.input_shape.back() = 2;
  test_case.axis_value = -1;
  test_case.input_type = TensorType_INT32;
  test_case.exclusive = false;
  test_case.reverse = false;
  test_case.invoke = true;
  EXPECT_EQ(RunCumsumCase(test_case), RunResult::kSuccess);
}

FUZZ_TEST(CumsumFuzzTest, CumsumNeverCrashes)
    .WithDomains(CumsumCaseDomain());

}  // namespace
}  // namespace tflite
