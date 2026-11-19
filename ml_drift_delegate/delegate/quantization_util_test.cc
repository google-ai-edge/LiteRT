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

#include "ml_drift_delegate/delegate/quantization_util.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tflite/array.h"
#include "tflite/core/c/common.h"

namespace ml_drift {
namespace delegate {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::Pointwise;

struct ScopedTestTensor {
  TfLiteTensor tensor{};
  tflite::IntArrayUniquePtr dims;

  ScopedTestTensor(TfLiteType type, const std::vector<int>& shape,
                   float scale = 1.0f, int32_t zero_point = 0,
                   void* data_ptr = nullptr, size_t byte_size = 0)
      : dims(tflite::BuildTfLiteArray(shape)) {
    tensor.type = type;
    tensor.dims = dims.get();
    tensor.params.scale = scale;
    tensor.params.zero_point = zero_point;
    tensor.data.data = data_ptr;
    tensor.bytes = byte_size;
    tensor.allocation_type = kTfLiteCustom;
  }
};

TEST(QuantizationUtilTest, Int8DequantizeAndQuantizeRoundTrip) {
  std::vector<int8_t> q_vals = {-20, -10, 0, 15, 30};
  ScopedTestTensor q_tensor(kTfLiteInt8, {1, 5}, /*scale=*/0.25f,
                            /*zero_point=*/-10, q_vals.data(), q_vals.size());

  EXPECT_THAT(ValidatedDequantizedBytes(q_tensor.tensor),
              IsOkAndHolds(5 * sizeof(float)));

  std::vector<float> floats(q_vals.size(), 0.0f);
  ASSERT_THAT(DequantizeTo(q_tensor.tensor, absl::MakeSpan(floats)), IsOk());
  EXPECT_THAT(floats,
              Pointwise(FloatNear(1e-6f),
                        std::vector<float>{-2.5f, 0.0f, 2.5f, 6.25f, 10.0f}));

  std::vector<int8_t> requant_vals(q_vals.size(), 0);
  ScopedTestTensor out_q_tensor(kTfLiteInt8, {1, 5}, /*scale=*/0.25f,
                                /*zero_point=*/-10, requant_vals.data(),
                                requant_vals.size());
  ASSERT_THAT(QuantizeFrom(absl::MakeConstSpan(floats), out_q_tensor.tensor),
              IsOk());
  EXPECT_THAT(requant_vals, ElementsAre(-20, -10, 0, 15, 30));
}

TEST(QuantizationUtilTest, UInt8DequantizeAndQuantizeRoundTrip) {
  std::vector<uint8_t> q_vals = {10, 20, 30, 50};
  ScopedTestTensor q_tensor(kTfLiteUInt8, {2, 2}, /*scale=*/0.1f,
                            /*zero_point=*/20, q_vals.data(), q_vals.size());

  EXPECT_THAT(ValidatedDequantizedBytes(q_tensor.tensor),
              IsOkAndHolds(4 * sizeof(float)));

  std::vector<float> floats(q_vals.size(), 0.0f);
  ASSERT_THAT(DequantizeTo(q_tensor.tensor, absl::MakeSpan(floats)), IsOk());
  EXPECT_THAT(floats, Pointwise(FloatNear(1e-6f),
                                std::vector<float>{-1.0f, 0.0f, 1.0f, 3.0f}));

  std::vector<uint8_t> requant_vals(q_vals.size(), 0);
  ScopedTestTensor out_q_tensor(kTfLiteUInt8, {2, 2}, /*scale=*/0.1f,
                                /*zero_point=*/20, requant_vals.data(),
                                requant_vals.size());
  ASSERT_THAT(QuantizeFrom(absl::MakeConstSpan(floats), out_q_tensor.tensor),
              IsOk());
  EXPECT_THAT(requant_vals, ElementsAre(10, 20, 30, 50));
}

TEST(QuantizationUtilTest, RejectsUnsupportedTensorTypes) {
  uint8_t dummy = 0;
  ScopedTestTensor int4_tensor(kTfLiteInt4, {2}, /*scale=*/1.0f,
                               /*zero_point=*/0, &dummy, sizeof(dummy));
  ScopedTestTensor fp32_tensor(kTfLiteFloat32, {1}, /*scale=*/1.0f,
                               /*zero_point=*/0, &dummy, sizeof(float));

  std::vector<float> floats(2, 0.0f);
  EXPECT_THAT(ValidatedDequantizedBytes(int4_tensor.tensor),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(DequantizeTo(int4_tensor.tensor, absl::MakeSpan(floats)),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(QuantizeFrom(absl::MakeConstSpan(floats), int4_tensor.tensor),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(DequantizeTo(fp32_tensor.tensor, absl::MakeSpan(floats)),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(QuantizationUtilTest, ZeroElementTensorSucceedsWithNullPointers) {
  ScopedTestTensor empty_tensor(kTfLiteInt8, {1, 0, 4}, /*scale=*/0.5f,
                                /*zero_point=*/0, /*data_ptr=*/nullptr,
                                /*byte_size=*/0);

  EXPECT_THAT(ValidatedDequantizedBytes(empty_tensor.tensor), IsOkAndHolds(0));
  EXPECT_THAT(DequantizeTo(empty_tensor.tensor, absl::Span<float>()), IsOk());
  EXPECT_THAT(QuantizeFrom(absl::Span<const float>(), empty_tensor.tensor),
              IsOk());
}

TEST(QuantizationUtilTest, RejectsZeroOrUndersizedTensorBytesForNonEmptyDims) {
  std::vector<uint8_t> storage(4, 1);
  std::vector<float> floats(4, 0.0f);

  ScopedTestTensor zero_bytes(kTfLiteUInt8, {4}, /*scale=*/0.1f,
                              /*zero_point=*/0, storage.data(),
                              /*byte_size=*/0);
  EXPECT_THAT(ValidatedDequantizedBytes(zero_bytes.tensor),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(DequantizeTo(zero_bytes.tensor, absl::MakeSpan(floats)),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(QuantizeFrom(absl::MakeConstSpan(floats), zero_bytes.tensor),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  ScopedTestTensor undersized_bytes(kTfLiteUInt8, {4}, /*scale=*/0.1f,
                                    /*zero_point=*/0, storage.data(),
                                    /*byte_size=*/3);
  EXPECT_THAT(ValidatedDequantizedBytes(undersized_bytes.tensor),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(DequantizeTo(undersized_bytes.tensor, absl::MakeSpan(floats)),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(
      QuantizeFrom(absl::MakeConstSpan(floats), undersized_bytes.tensor),
      StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(QuantizationUtilTest, RejectsNullDataAndUndersizedFloatSpans) {
  std::vector<int8_t> storage(4, 2);
  std::vector<float> short_floats(3, 1.0f);

  ScopedTestTensor null_data(kTfLiteInt8, {4}, /*scale=*/0.1f, /*zero_point=*/0,
                             /*data_ptr=*/nullptr, /*byte_size=*/4);
  std::vector<float> full_floats(4, 0.0f);
  EXPECT_THAT(ValidatedDequantizedBytes(null_data.tensor),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(DequantizeTo(null_data.tensor, absl::MakeSpan(full_floats)),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(QuantizeFrom(absl::MakeConstSpan(full_floats), null_data.tensor),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  ScopedTestTensor valid_q(kTfLiteInt8, {4}, /*scale=*/0.1f, /*zero_point=*/0,
                           storage.data(), storage.size());
  EXPECT_THAT(DequantizeTo(valid_q.tensor, absl::Span<float>()),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(DequantizeTo(valid_q.tensor, absl::MakeSpan(short_floats)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(QuantizeFrom(absl::Span<const float>(), valid_q.tensor),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(QuantizeFrom(absl::MakeConstSpan(short_floats), valid_q.tensor),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(QuantizationUtilTest, RejectsMalformedNegativeAndOverflowingDimensions) {
  uint8_t dummy = 1;
  std::vector<float> floats(16, 0.0f);

  TfLiteTensor null_dims_tensor{};
  null_dims_tensor.type = kTfLiteUInt8;
  null_dims_tensor.dims = nullptr;
  null_dims_tensor.data.uint8 = &dummy;
  null_dims_tensor.bytes = 1;
  EXPECT_THAT(ValidatedDequantizedBytes(null_dims_tensor),
              StatusIs(absl::StatusCode::kInvalidArgument));

  TfLiteIntArray neg_size_dims{};
  neg_size_dims.size = -1;
  TfLiteTensor neg_rank_tensor = null_dims_tensor;
  neg_rank_tensor.dims = &neg_size_dims;
  EXPECT_THAT(ValidatedDequantizedBytes(neg_rank_tensor),
              StatusIs(absl::StatusCode::kInvalidArgument));

  ScopedTestTensor neg_dim_tensor(kTfLiteInt8, {-4, -4}, /*scale=*/0.1f,
                                  /*zero_point=*/0, &dummy, 16);
  EXPECT_THAT(ValidatedDequantizedBytes(neg_dim_tensor.tensor),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(DequantizeTo(neg_dim_tensor.tensor, absl::MakeSpan(floats)),
              StatusIs(absl::StatusCode::kInvalidArgument));

  ScopedTestTensor int_overflow_tensor(kTfLiteInt8, {65536, 65536},
                                       /*scale=*/0.1f, /*zero_point=*/0, &dummy,
                                       std::numeric_limits<size_t>::max());
  EXPECT_THAT(ValidatedDequantizedBytes(int_overflow_tensor.tensor),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(DequantizeTo(int_overflow_tensor.tensor, absl::MakeSpan(floats)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(QuantizationUtilTest, DequantizeInputsAndQuantizeOutputsBoundsAndSync) {
  absl::flat_hash_map<int, int> quant_map = {{1, 0}, {0, 1}};
  EXPECT_THAT(DequantizeInputs(nullptr, {}, quant_map),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(QuantizeOutputs(nullptr, {}, quant_map),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(DequantizeInputs(nullptr, {1}, quant_map),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(QuantizeOutputs(nullptr, {1}, quant_map),
              StatusIs(absl::StatusCode::kInvalidArgument));

  TfLiteContext null_tensors_ctx{};
  null_tensors_ctx.tensors = nullptr;
  null_tensors_ctx.tensors_size = 2;
  EXPECT_THAT(DequantizeInputs(&null_tensors_ctx, {}, quant_map),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(QuantizeOutputs(&null_tensors_ctx, {}, quant_map),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(DequantizeInputs(&null_tensors_ctx, {1}, quant_map),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(QuantizeOutputs(&null_tensors_ctx, {1}, quant_map),
              StatusIs(absl::StatusCode::kInvalidArgument));

  std::vector<int8_t> q_buf = {-4, 0, 8, 12};
  std::vector<float> fp_buf(4, 0.0f);
  ScopedTestTensor q_tensor(kTfLiteInt8, {4}, /*scale=*/0.5f, /*zero_point=*/0,
                            q_buf.data(), q_buf.size());
  ScopedTestTensor fp_tensor(kTfLiteFloat32, {4}, /*scale=*/0.0f,
                             /*zero_point=*/0, fp_buf.data(),
                             fp_buf.size() * sizeof(float));

  std::vector<TfLiteTensor> tensors = {q_tensor.tensor, fp_tensor.tensor};
  TfLiteContext ctx{};
  ctx.tensors = tensors.data();
  ctx.tensors_size = tensors.size();

  // Out-of-range indices and int64 overflow indices are rejected.
  EXPECT_THAT(DequantizeInputs(&ctx, {-1}, quant_map),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(DequantizeInputs(&ctx, {2}, quant_map),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      QuantizeOutputs(
          &ctx, {static_cast<int64_t>(std::numeric_limits<int>::max()) + 1},
          quant_map),
      StatusIs(absl::StatusCode::kInvalidArgument));

  absl::flat_hash_map<int, int> bad_mapped_idx = {{1, 5}};
  EXPECT_THAT(DequantizeInputs(&ctx, {1}, bad_mapped_idx),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(QuantizeOutputs(&ctx, {1}, bad_mapped_idx),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Reverse-mapped quantized tensor index (0, type kTfLiteInt8) and unmapped
  // float tensor are skipped cleanly.
  EXPECT_THAT(DequantizeInputs(&ctx, {0}, quant_map), IsOk());
  EXPECT_THAT(QuantizeOutputs(&ctx, {0}, quant_map), IsOk());
  EXPECT_THAT(DequantizeInputs(&ctx, {1}, /*quant_conversion_map=*/{}), IsOk());

  // Valid forward dequantize and reverse quantize succeed.
  ASSERT_THAT(DequantizeInputs(&ctx, {1}, quant_map), IsOk());
  EXPECT_THAT(fp_buf, Pointwise(FloatNear(1e-6f),
                                std::vector<float>{-2.0f, 0.0f, 4.0f, 6.0f}));

  fp_buf = {1.0f, -1.5f, 3.0f, -5.0f};
  ASSERT_THAT(QuantizeOutputs(&ctx, {1}, quant_map), IsOk());
  EXPECT_THAT(q_buf, ElementsAre(2, -3, 6, -10));

  // Undersized fp32 temporary tensor byte count is rejected.
  tensors[1].bytes = 2 * sizeof(float);
  EXPECT_THAT(DequantizeInputs(&ctx, {1}, quant_map),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(QuantizeOutputs(&ctx, {1}, quant_map),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace delegate
}  // namespace ml_drift
