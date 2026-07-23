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
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "flatbuffers/flatbuffer_builder.h"
#include "fuzztest/fuzztest.h"
#include "gtest/gtest.h"
#include "tflite/core/interpreter.h"
#include "tflite/core/kernels/builtin_op_kernels.h"
#include "tflite/kernels/fuzzing/fuzzing_util.h"
#include "tflite/kernels/fuzzing/one_op_fuzz_model.h"
#include "tflite/schema/schema_generated.h"
#if defined(TFLITE_REDUCE_FUZZ_ENABLE_XNNPACK)
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"
#endif

namespace tflite {
namespace ops {
namespace builtin {

TfLiteRegistration* Register_MEAN_REF();
TfLiteRegistration* Register_MEAN_OPT();
TfLiteRegistration* Register_SUM_REF();
TfLiteRegistration* Register_SUM_OPT();
TfLiteRegistration* Register_REDUCE_PROD_REF();
TfLiteRegistration* Register_REDUCE_PROD_OPT();
TfLiteRegistration* Register_REDUCE_MAX_REF();
TfLiteRegistration* Register_REDUCE_MAX_OPT();
TfLiteRegistration* Register_REDUCE_MIN_REF();
TfLiteRegistration* Register_REDUCE_MIN_OPT();
TfLiteRegistration* Register_REDUCE_ANY_REF();
TfLiteRegistration* Register_REDUCE_ANY_OPT();
TfLiteRegistration* Register_REDUCE_ALL_REF();
TfLiteRegistration* Register_REDUCE_ALL_OPT();

}  // namespace builtin
}  // namespace ops

namespace {

using fuzzing::RunResult;

enum class ReduceKind { kMean, kSum, kProd, kMax, kMin, kAny, kAll };
enum class KernelVariant { kReference, kGenericOptimized };
enum class ExecutionMode { kBuiltin, kXnnpack };
enum class AxisShapeKind { kVector, kScalar, kMatrix, kEmpty };

struct ReduceCase {
  std::vector<int32_t> input_shape;
  std::vector<int32_t> axis_values;
  std::vector<uint8_t> input_data;
  std::vector<uint8_t> axis_data;
  ReduceKind reduce_kind;
  AxisShapeKind axis_shape_kind;
  TensorType input_type;
  bool dynamic_axis;
  bool keep_dims;
  bool quantized;
  bool invoke;
};

struct HighRankReduceSpec {
  int32_t rank;
  ReduceKind reduce_kind;
  bool keep_dims;
  bool negative_axes;
  bool reverse_axes;
  bool duplicate_axes;
  bool dynamic_axis;
};

struct XnnpackReduceSpec {
  std::vector<int32_t> input_shape;
  std::vector<int32_t> axis_values;
  std::vector<uint8_t> input_data;
  ReduceKind reduce_kind;
  TensorType input_type;
  bool keep_dims;
};

constexpr size_t kMaxGeneralInputElements = 256;
constexpr size_t kMaxFuzzerLiveAllocationBytes = 64 * 1024 * 1024;
constexpr int32_t kMinHighRank = 31;
constexpr int32_t kMaxHighRank = 40;
constexpr int32_t kMaxXnnpackReduceRank = 6;

bool IsQuantizable(TensorType type) {
  return type == TensorType_UINT8 || type == TensorType_INT8 ||
         type == TensorType_INT16;
}

bool IsMeanOrSum(ReduceKind reduce_kind) {
  return reduce_kind == ReduceKind::kMean || reduce_kind == ReduceKind::kSum;
}

TensorType EffectiveInputType(const ReduceCase& test_case) {
  if (test_case.reduce_kind == ReduceKind::kAny ||
      test_case.reduce_kind == ReduceKind::kAll) {
    return TensorType_BOOL;
  }
  return test_case.input_type;
}

BuiltinOperator BuiltinOperatorForReduceKind(ReduceKind reduce_kind) {
  switch (reduce_kind) {
    case ReduceKind::kMean:
      return BuiltinOperator_MEAN;
    case ReduceKind::kSum:
      return BuiltinOperator_SUM;
    case ReduceKind::kProd:
      return BuiltinOperator_REDUCE_PROD;
    case ReduceKind::kMax:
      return BuiltinOperator_REDUCE_MAX;
    case ReduceKind::kMin:
      return BuiltinOperator_REDUCE_MIN;
    case ReduceKind::kAny:
      return BuiltinOperator_REDUCE_ANY;
    case ReduceKind::kAll:
      return BuiltinOperator_REDUCE_ALL;
  }
}

TfLiteRegistration* RegistrationForReduceKind(ReduceKind reduce_kind,
                                              KernelVariant kernel_variant) {
  switch (reduce_kind) {
    case ReduceKind::kMean:
      return kernel_variant == KernelVariant::kReference
                 ? ops::builtin::Register_MEAN_REF()
                 : ops::builtin::Register_MEAN_OPT();
    case ReduceKind::kSum:
      return kernel_variant == KernelVariant::kReference
                 ? ops::builtin::Register_SUM_REF()
                 : ops::builtin::Register_SUM_OPT();
    case ReduceKind::kProd:
      return kernel_variant == KernelVariant::kReference
                 ? ops::builtin::Register_REDUCE_PROD_REF()
                 : ops::builtin::Register_REDUCE_PROD_OPT();
    case ReduceKind::kMax:
      return kernel_variant == KernelVariant::kReference
                 ? ops::builtin::Register_REDUCE_MAX_REF()
                 : ops::builtin::Register_REDUCE_MAX_OPT();
    case ReduceKind::kMin:
      return kernel_variant == KernelVariant::kReference
                 ? ops::builtin::Register_REDUCE_MIN_REF()
                 : ops::builtin::Register_REDUCE_MIN_OPT();
    case ReduceKind::kAny:
      return kernel_variant == KernelVariant::kReference
                 ? ops::builtin::Register_REDUCE_ANY_REF()
                 : ops::builtin::Register_REDUCE_ANY_OPT();
    case ReduceKind::kAll:
      return kernel_variant == KernelVariant::kReference
                 ? ops::builtin::Register_REDUCE_ALL_REF()
                 : ops::builtin::Register_REDUCE_ALL_OPT();
  }
}

TfLiteStatus ApplyXnnpackDelegate(Interpreter* interpreter) {
#if defined(TFLITE_REDUCE_FUZZ_ENABLE_XNNPACK)
  TfLiteXNNPackDelegateOptions options = TfLiteXNNPackDelegateOptionsDefault();
  options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_QS8;
  options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_QU8;
  options.num_threads = 1;
  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> delegate(
      TfLiteXNNPackDelegateCreate(&options), TfLiteXNNPackDelegateDelete);
  if (delegate == nullptr) return kTfLiteError;
  return interpreter->ModifyGraphWithDelegate(std::move(delegate));
#else
  (void)interpreter;
  return kTfLiteError;
#endif
}

bool HasDelegateNode(const Interpreter& interpreter) {
  for (int node_index : interpreter.execution_plan()) {
    const auto* node_and_registration =
        interpreter.node_and_registration(node_index);
    if (node_and_registration != nullptr &&
        node_and_registration->second.builtin_code ==
            BuiltinOperator_DELEGATE) {
      return true;
    }
  }
  return false;
}

std::vector<int32_t> AxisShape(const ReduceCase& test_case,
                               size_t* axis_element_count) {
  switch (test_case.axis_shape_kind) {
    case AxisShapeKind::kVector:
      *axis_element_count = test_case.axis_values.size();
      return {static_cast<int32_t>(*axis_element_count)};
    case AxisShapeKind::kScalar:
      *axis_element_count = 1;
      return {};
    case AxisShapeKind::kMatrix: {
      const int32_t rows =
          1 + static_cast<int32_t>(test_case.axis_values.size() % 3);
      const int32_t columns =
          1 + static_cast<int32_t>(test_case.axis_values.size() % 4);
      *axis_element_count = static_cast<size_t>(rows * columns);
      return {rows, columns};
    }
    case AxisShapeKind::kEmpty:
      *axis_element_count = 0;
      return {0};
  }
}

std::vector<int64_t> MaterializeAxisValues(const std::vector<int32_t>& values,
                                           size_t count) {
  std::vector<int64_t> result(count, 0);
  for (size_t i = 0; i < count; ++i) {
    result[i] = values.empty() ? 0 : values[i % values.size()];
  }
  return result;
}

flatbuffers::Offset<QuantizationParameters> MaybeCreateQuantization(
    flatbuffers::FlatBufferBuilder* builder, ReduceKind reduce_kind,
    TensorType input_type, bool quantized) {
  const bool force_quantized_mean_or_sum =
      IsMeanOrSum(reduce_kind) && IsQuantizable(input_type);
  if ((!force_quantized_mean_or_sum && !quantized) ||
      !IsQuantizable(input_type)) {
    return 0;
  }
  return CreateQuantizationParameters(*builder, 0, 0,
                                      builder->CreateVector<float>({0.25f}),
                                      builder->CreateVector<int64_t>({0}));
}

RunResult RunReduceCase(const ReduceCase& test_case,
                        KernelVariant kernel_variant,
                        ExecutionMode execution_mode = ExecutionMode::kBuiltin) {
  const TensorType input_type = EffectiveInputType(test_case);
  size_t input_elements = 0;
  if (fuzzing::TypeSize(input_type) == 0 ||
      !fuzzing::CheckedShapeElementCount(test_case.input_shape,
                                         &input_elements) ||
      input_elements > kMaxGeneralInputElements) {
    return RunResult::kRejected;
  }

  std::vector<uint8_t> input_bytes =
      fuzzing::MakeValues(input_type, input_elements, 1);
  fuzzing::OverlayBytes(test_case.input_data, &input_bytes);
  fuzzing::ApplyCentralTensorInputInvariants(input_type, &input_bytes);

  size_t axis_element_count = 0;
  const std::vector<int32_t> axis_shape =
      AxisShape(test_case, &axis_element_count);
  std::vector<uint8_t> axis_bytes = fuzzing::MakeIntegerValues(
      TensorType_INT32,
      MaterializeAxisValues(test_case.axis_values, axis_element_count));
  fuzzing::OverlayBytes(test_case.axis_data, &axis_bytes);

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<Buffer>> buffers = {
      fuzzing::CreateAlignedBuffer(&builder, std::vector<uint8_t>{})};
  if (!test_case.dynamic_axis) {
    buffers.push_back(fuzzing::CreateAlignedBuffer(&builder, axis_bytes));
  }

  const auto input_shape = builder.CreateVector(test_case.input_shape);
  const auto axis_shape_offset = builder.CreateVector(axis_shape);
  const std::vector<int32_t> initial_output_shape = test_case.input_shape;
  const auto output_shape = builder.CreateVector(initial_output_shape);
  const auto quantization = MaybeCreateQuantization(
      &builder, test_case.reduce_kind, input_type, test_case.quantized);

  const auto input_tensor =
      CreateTensor(builder, input_shape, input_type, 0, 0, quantization);
  const auto axis_tensor =
      CreateTensor(builder, axis_shape_offset, TensorType_INT32,
                   test_case.dynamic_axis ? 0 : 1);
  const auto output_tensor =
      CreateTensor(builder, output_shape, input_type, 0, 0, quantization);

  std::vector<int32_t> model_inputs = {0};
  if (test_case.dynamic_axis) {
    model_inputs.push_back(1);
  }
  const std::vector<int32_t> op_inputs = {0, 1};
  const std::vector<int32_t> outputs = {2};

  fuzzing::OneOpModelSpec model_spec;
  model_spec.description = "reduce_fuzz";
  model_spec.builtin_operator =
      BuiltinOperatorForReduceKind(test_case.reduce_kind);
  model_spec.version = 1;
  model_spec.builtin_options_type = BuiltinOptions_ReducerOptions;
  model_spec.builtin_options =
      CreateReducerOptions(builder, test_case.keep_dims).Union();
  model_spec.tensors = {input_tensor, axis_tensor, output_tensor};
  model_spec.buffers = std::move(buffers);
  model_spec.model_inputs = std::move(model_inputs);
  model_spec.model_outputs = outputs;
  model_spec.op_inputs = op_inputs;
  model_spec.op_outputs = outputs;

  fuzzing::OneOpRunSpec run_spec;
  run_spec.registration =
      RegistrationForReduceKind(test_case.reduce_kind, kernel_variant);
  run_spec.min_version = 1;
  run_spec.max_version = 3;
  run_spec.max_live_allocation_bytes = kMaxFuzzerLiveAllocationBytes;
  run_spec.invoke = test_case.invoke;
  run_spec.runtime_tensors.push_back(
      {/*tensor_index=*/0, test_case.input_shape, std::move(input_bytes)});
  if (test_case.dynamic_axis) {
    run_spec.runtime_tensors.push_back(
        {/*tensor_index=*/1, axis_shape, std::move(axis_bytes)});
  }
  if (execution_mode == ExecutionMode::kXnnpack) {
    run_spec.post_allocate = [](Interpreter* interpreter) {
      return ApplyXnnpackDelegate(interpreter) == kTfLiteOk &&
                     HasDelegateNode(*interpreter)
                 ? RunResult::kSuccess
                 : RunResult::kRejected;
    };
  }

  return fuzzing::BuildAndRunOneOpModel(&builder, model_spec, run_spec);
}

ReduceCase MakeHighRankReduceCase(const HighRankReduceSpec& spec) {
  const int32_t rank = std::clamp(spec.rank, kMinHighRank, kMaxHighRank);
  ReduceCase test_case;
  test_case.input_shape.assign(rank, 1);
  test_case.input_shape.back() = 2;
  test_case.axis_values.resize(rank);
  std::iota(test_case.axis_values.begin(), test_case.axis_values.end(), 0);
  if (spec.negative_axes) {
    for (int32_t& axis : test_case.axis_values) {
      axis -= rank;
    }
  }
  if (spec.reverse_axes) {
    std::reverse(test_case.axis_values.begin(), test_case.axis_values.end());
  }
  if (spec.duplicate_axes && !test_case.axis_values.empty()) {
    test_case.axis_values.push_back(test_case.axis_values.front());
    test_case.axis_values.push_back(test_case.axis_values.back());
  }
  test_case.input_data = {};
  test_case.axis_data = {};
  test_case.reduce_kind = spec.reduce_kind;
  test_case.axis_shape_kind = AxisShapeKind::kVector;
  test_case.input_type = TensorType_FLOAT32;
  test_case.dynamic_axis = spec.dynamic_axis;
  test_case.keep_dims = spec.keep_dims;
  test_case.quantized = false;
  test_case.invoke = true;
  return test_case;
}

std::vector<int32_t> MakeXnnpackAxes(const std::vector<int32_t>& axis_values,
                                     int32_t rank) {
  std::vector<int32_t> axes;
  axes.reserve(rank);
  for (int32_t axis_value : axis_values) {
    int32_t axis = axis_value % rank;
    if (axis < 0) axis += rank;
    if (std::find(axes.begin(), axes.end(), axis) == axes.end()) {
      axes.push_back(axis);
      if (axes.size() == static_cast<size_t>(rank)) break;
    }
  }
  if (axes.empty()) {
    axes.push_back(0);
  }
  return axes;
}

ReduceCase MakeXnnpackReduceCase(const XnnpackReduceSpec& spec) {
  ReduceCase test_case;
  const int32_t rank = std::clamp(static_cast<int32_t>(spec.input_shape.size()),
                                  int32_t{1}, kMaxXnnpackReduceRank);
  test_case.input_shape.reserve(rank);
  for (int32_t i = 0; i < rank; ++i) {
    const int32_t dim =
        i < static_cast<int32_t>(spec.input_shape.size())
            ? spec.input_shape[static_cast<size_t>(i)]
            : 1;
    test_case.input_shape.push_back(std::clamp(dim, int32_t{1}, int32_t{4}));
  }
  test_case.axis_values = MakeXnnpackAxes(spec.axis_values, rank);
  test_case.input_data = spec.input_data;
  test_case.axis_data = {};
  test_case.reduce_kind = spec.reduce_kind;
  test_case.axis_shape_kind = AxisShapeKind::kVector;
  test_case.input_type = spec.input_type;
  test_case.dynamic_axis = false;
  test_case.keep_dims = spec.keep_dims;
  test_case.quantized = spec.input_type != TensorType_FLOAT32;
  test_case.invoke = true;
  return test_case;
}

auto AxisValueDomain() {
  return fuzztest::OneOf(
      fuzztest::InRange<int32_t>(-10, 11), fuzztest::Just<int32_t>(31),
      fuzztest::Just<int32_t>(32), fuzztest::Just<int32_t>(33),
      fuzztest::Just<int32_t>(-31), fuzztest::Just<int32_t>(-32),
      fuzztest::Just<int32_t>(-33),
      fuzztest::Just<int32_t>(std::numeric_limits<int32_t>::max()),
      fuzztest::Just<int32_t>(std::numeric_limits<int32_t>::min()));
}

auto ReduceCaseDomain() {
  return fuzztest::StructOf<ReduceCase>(
      fuzztest::VectorOf(fuzztest::InRange<int32_t>(0, 4))
          .WithMinSize(0)
          .WithMaxSize(8),
      fuzztest::VectorOf(AxisValueDomain()).WithMinSize(0).WithMaxSize(16),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(64),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(64),
      fuzztest::ElementOf<ReduceKind>({ReduceKind::kMean, ReduceKind::kSum,
                                       ReduceKind::kProd, ReduceKind::kMax,
                                       ReduceKind::kMin, ReduceKind::kAny,
                                       ReduceKind::kAll}),
      fuzztest::ElementOf<AxisShapeKind>(
          {AxisShapeKind::kVector, AxisShapeKind::kScalar,
           AxisShapeKind::kMatrix, AxisShapeKind::kEmpty}),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_UINT8,
                                       TensorType_INT8, TensorType_INT16,
                                       TensorType_INT32, TensorType_INT64,
                                       TensorType_BOOL}),
      fuzztest::Arbitrary<bool>(), fuzztest::Arbitrary<bool>(),
      fuzztest::Arbitrary<bool>(), fuzztest::Arbitrary<bool>());
}

auto HighRankReduceSpecDomain() {
  return fuzztest::StructOf<HighRankReduceSpec>(
      fuzztest::InRange<int32_t>(kMinHighRank, kMaxHighRank + 1),
      fuzztest::ElementOf<ReduceKind>({ReduceKind::kSum, ReduceKind::kProd,
                                       ReduceKind::kMax, ReduceKind::kMin}),
      fuzztest::Arbitrary<bool>(), fuzztest::Arbitrary<bool>(),
      fuzztest::Arbitrary<bool>(), fuzztest::Arbitrary<bool>(),
      fuzztest::Arbitrary<bool>());
}

auto XnnpackReduceSpecDomain() {
  return fuzztest::StructOf<XnnpackReduceSpec>(
      fuzztest::VectorOf(fuzztest::InRange<int32_t>(1, 5))
          .WithMinSize(1)
          .WithMaxSize(kMaxXnnpackReduceRank),
      fuzztest::VectorOf(fuzztest::InRange<int32_t>(-kMaxXnnpackReduceRank,
                                                    2 * kMaxXnnpackReduceRank))
          .WithMinSize(1)
          .WithMaxSize(kMaxXnnpackReduceRank),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(64),
      fuzztest::ElementOf<ReduceKind>({ReduceKind::kMean, ReduceKind::kSum,
                                       ReduceKind::kMax, ReduceKind::kMin}),
      fuzztest::ElementOf<TensorType>(
          {TensorType_FLOAT32, TensorType_INT8, TensorType_UINT8}),
      fuzztest::Arbitrary<bool>());
}

TEST(ReduceFuzzTest, ReferenceHighRankReduceAllSmoke) {
  const HighRankReduceSpec spec{
      /*rank=*/32,
      /*reduce_kind=*/ReduceKind::kSum,
      /*keep_dims=*/false,
      /*negative_axes=*/false,
      /*reverse_axes=*/false,
      /*duplicate_axes=*/false,
      /*dynamic_axis=*/true,
  };
  EXPECT_EQ(
      RunReduceCase(MakeHighRankReduceCase(spec), KernelVariant::kReference),
      RunResult::kSuccess);
}

void ReduceReferenceNeverCrashes(const ReduceCase& test_case) {
  EXPECT_NE(RunReduceCase(test_case, KernelVariant::kReference),
            RunResult::kHarnessFailure);
}

void ReduceOptimizedNeverCrashes(const ReduceCase& test_case) {
  EXPECT_NE(RunReduceCase(test_case, KernelVariant::kGenericOptimized),
            RunResult::kHarnessFailure);
}

void ReferenceHighRankReduceAllNeverCrashes(const HighRankReduceSpec& spec) {
  EXPECT_EQ(
      RunReduceCase(MakeHighRankReduceCase(spec), KernelVariant::kReference),
      RunResult::kSuccess);
}

FUZZ_TEST(ReduceFuzzTest, ReduceReferenceNeverCrashes)
    .WithDomains(ReduceCaseDomain());
FUZZ_TEST(ReduceFuzzTest, ReduceOptimizedNeverCrashes)
    .WithDomains(ReduceCaseDomain());
FUZZ_TEST(ReduceFuzzTest, ReferenceHighRankReduceAllNeverCrashes)
    .WithDomains(HighRankReduceSpecDomain());

#if defined(TFLITE_REDUCE_FUZZ_ENABLE_XNNPACK)
TEST(ReduceFuzzTest, ReduceXnnpackSmokeDelegates) {
  XnnpackReduceSpec spec;
  spec.input_shape = {2, 3, 4};
  spec.axis_values = {1};
  spec.input_data = {};
  spec.reduce_kind = ReduceKind::kMean;
  spec.input_type = TensorType_FLOAT32;
  spec.keep_dims = false;

  EXPECT_EQ(RunReduceCase(MakeXnnpackReduceCase(spec),
                          KernelVariant::kGenericOptimized,
                          ExecutionMode::kXnnpack),
            RunResult::kSuccess);
}

void ReduceXnnpackNeverCrashes(const XnnpackReduceSpec& spec) {
  EXPECT_NE(RunReduceCase(MakeXnnpackReduceCase(spec),
                          KernelVariant::kGenericOptimized,
                          ExecutionMode::kXnnpack),
            RunResult::kHarnessFailure);
}

FUZZ_TEST(ReduceFuzzTest, ReduceXnnpackNeverCrashes)
    .WithDomains(XnnpackReduceSpecDomain());
#endif

}  // namespace
}  // namespace tflite
