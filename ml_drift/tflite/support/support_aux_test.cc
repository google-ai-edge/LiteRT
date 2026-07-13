// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "third_party/odml/litert/ml_drift/tflite/support/support_aux.h"
#include <string>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {
namespace {

using ::testing::HasSubstr;
using ::testing::status::StatusIs;

TEST(SupportAuxTest, CheckFusedActivation) {
  // Check that the activation is rejected if the node has 2 outputs.
  TfLiteNode node;
  node.outputs = TfLiteIntArrayCreate(2);
  EXPECT_THAT(CheckFusedActivation(&node, kTfLiteActNone),
              StatusIs(absl::StatusCode::kInvalidArgument));
  TfLiteIntArrayFree(node.outputs);
  // Check that the activation is accepted if the node has 1 output.
  node.outputs = TfLiteIntArrayCreate(1);
  EXPECT_OK(CheckFusedActivation(&node, kTfLiteActNone));
  EXPECT_OK(CheckFusedActivation(&node, kTfLiteActRelu));
  EXPECT_OK(CheckFusedActivation(&node, kTfLiteActReluN1To1));
  EXPECT_OK(CheckFusedActivation(&node, kTfLiteActRelu6));
  EXPECT_OK(CheckFusedActivation(&node, kTfLiteActTanh));
  EXPECT_OK(CheckFusedActivation(&node, kTfLiteActSigmoid));
  EXPECT_OK(CheckFusedActivation(&node, kTfLiteActSignBit));
  TfLiteIntArrayFree(node.outputs);
}

// Test fixture for CheckPopulateTensor tests.
class CheckPopulateTensorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tensor_.quantization.type = kTfLiteAffineQuantization;
    quant_params_.quantized_dimension = 0;
    scale_arr_ = TfLiteFloatArrayCreate(1);
    zp_arr_ = TfLiteIntArrayCreate(1);
    quant_params_.scale = scale_arr_;
    quant_params_.zero_point = zp_arr_;
    tensor_.quantization.params = &quant_params_;
    dims_ = TfLiteIntArrayCreate(4);
    dims_->data[0] = 1;
    dims_->data[1] = 1;
    dims_->data[2] = 1;
    dims_->data[3] = 1;
    tensor_.dims = dims_;
    tensor_.bytes = 4 * sizeof(int);
    tensor_.name = "test_tensor";
  }

  void TearDown() override {
    TfLiteFloatArrayFree(scale_arr_);
    TfLiteIntArrayFree(zp_arr_);
    TfLiteIntArrayFree(dims_);
  }

  TfLiteTensor tensor_;
  TfLiteAffineQuantization quant_params_;
  TfLiteFloatArray* scale_arr_;
  TfLiteIntArray* zp_arr_;
  TfLiteIntArray* dims_;
};

TEST_F(CheckPopulateTensorTest, ValidTensors) {
  tensor_.bytes = 1;
  EXPECT_OK((CheckPopulateTensor<::ml_drift::OHWI, ::ml_drift::DataType::INT8>(
      &tensor_, /*enable_spanned_weights=*/false)));
  EXPECT_OK((CheckPopulateTensor<::ml_drift::OHWI, ::ml_drift::DataType::INT4>(
      &tensor_, /*enable_spanned_weights=*/false)));
  EXPECT_OK((CheckPopulateTensor<::ml_drift::OHWI, ::ml_drift::DataType::INT8>(
      &tensor_, /*enable_spanned_weights=*/true)));
  EXPECT_OK((CheckPopulateTensor<::ml_drift::OHWI, ::ml_drift::DataType::INT4>(
      &tensor_, /*enable_spanned_weights=*/true)));
  EXPECT_OK(
      (CheckPopulateTensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>(
          &tensor_, /*enable_spanned_weights=*/false)));
  EXPECT_OK(
      (CheckPopulateTensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>(
          &tensor_, /*enable_spanned_weights=*/true)));
  EXPECT_OK(
      (CheckPopulateTensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT16>(
          &tensor_, /*enable_spanned_weights=*/false)));
}


TEST_F(CheckPopulateTensorTest, InvalidQuantizationType) {
  tensor_.quantization.type = kTfLiteNoQuantization;
  EXPECT_THAT(
      (CheckPopulateTensor<::ml_drift::OHWI, ::ml_drift::DataType::INT8>(
          &tensor_, /*enable_spanned_weights=*/false)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("quantization.type must be "
                         "kTfLiteAffineQuantization")));
  EXPECT_THAT(
      (CheckPopulateTensor<::ml_drift::OHWI, ::ml_drift::DataType::INT4>(
          &tensor_, /*enable_spanned_weights=*/false)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("quantization.type must be "
                         "kTfLiteAffineQuantization")));
}

TEST_F(CheckPopulateTensorTest, InvalidQuantizedDimension) {
  quant_params_.quantized_dimension = 1;
  EXPECT_THAT(
      (CheckPopulateTensor<::ml_drift::OHWI, ::ml_drift::DataType::INT8>(
          &tensor_, /*enable_spanned_weights=*/false)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("quantized_dimension must be 0")));
  EXPECT_THAT(
      (CheckPopulateTensor<::ml_drift::OHWI, ::ml_drift::DataType::INT4>(
          &tensor_, /*enable_spanned_weights=*/false)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("quantized_dimension must be 0")));
}

TEST_F(CheckPopulateTensorTest, NullQuantParams) {
  quant_params_.scale = nullptr;
  EXPECT_THAT(
      (CheckPopulateTensor<::ml_drift::OHWI, ::ml_drift::DataType::INT8>(
          &tensor_, /*enable_spanned_weights=*/false)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("scale must not be null")));
  quant_params_.scale = scale_arr_;
  quant_params_.zero_point = nullptr;
  EXPECT_THAT(
      (CheckPopulateTensor<::ml_drift::OHWI, ::ml_drift::DataType::INT8>(
          &tensor_, /*enable_spanned_weights=*/false)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("zero_point must not be null")));
  quant_params_.zero_point = zp_arr_;
}

TEST_F(CheckPopulateTensorTest, UnsupportedTypeForZeroCopy) {
  EXPECT_THAT(
      (CheckPopulateTensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT16>(
          &tensor_, /*enable_spanned_weights=*/true)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Unsupported type for zero-copy")));
}

TEST_F(CheckPopulateTensorTest, DimSizeMismatch) {
  TfLiteIntArray* old_dims = tensor_.dims;
  tensor_.dims = TfLiteIntArrayCreate(1);
  tensor_.dims->data[0] = 1;
  EXPECT_THAT(
      (CheckPopulateTensor<::ml_drift::OHWI, ::ml_drift::DataType::INT8>(
          &tensor_, /*enable_spanned_weights=*/false)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Expected 2D, 4D, or 5D quantized tensor: test_tensor")));
  TfLiteIntArrayFree(tensor_.dims);
  tensor_.dims = old_dims;
}

TEST_F(CheckPopulateTensorTest, InvalidQuantizedDims) {
  TfLiteIntArray* old_dims = tensor_.dims;
  // Test 1D
  tensor_.dims = TfLiteIntArrayCreate(1);
  tensor_.dims->data[0] = 1;
  EXPECT_THAT(
      (CheckPopulateTensor<::ml_drift::OHWI, ::ml_drift::DataType::INT8>(
          &tensor_, /*enable_spanned_weights=*/false)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Expected 2D, 4D, or 5D quantized tensor: test_tensor")));
  TfLiteIntArrayFree(tensor_.dims);

  // Test 3D
  tensor_.dims = TfLiteIntArrayCreate(3);
  tensor_.dims->data[0] = 1;
  tensor_.dims->data[1] = 1;
  tensor_.dims->data[2] = 1;
  EXPECT_THAT(
      (CheckPopulateTensor<::ml_drift::OHWI, ::ml_drift::DataType::INT8>(
          &tensor_, /*enable_spanned_weights=*/false)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Expected 2D, 4D, or 5D quantized tensor: test_tensor")));
  TfLiteIntArrayFree(tensor_.dims);
  tensor_.dims = old_dims;
}

TEST_F(CheckPopulateTensorTest, CheckAllDimensionsForNonFloat) {
  TfLiteIntArray* old_dims = tensor_.dims;
  // Test FLOAT16 with mismatched dims (expects 4D for OHWI)
  tensor_.dims = TfLiteIntArrayCreate(2);
  tensor_.dims->data[0] = 1;
  tensor_.dims->data[1] = 1;
  EXPECT_THAT(
      (CheckPopulateTensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT16>(
          &tensor_, /*enable_spanned_weights=*/false)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Expected a 4D tensor of shape OxHxWxI")));
  TfLiteIntArrayFree(tensor_.dims);
  tensor_.dims = old_dims;
}

TEST(SupportAuxTest, ValidateTensorIds) {
  TfLiteContext context;
  context.tensors_size = 5;
  std::string error;

  // Valid ID.
  EXPECT_TRUE(ValidateTensorId(context, 0, "test_tensor", error));
  EXPECT_TRUE(ValidateTensorId(context, 4, "test_tensor", error));

  // Invalid ID: out of range.
  EXPECT_FALSE(ValidateTensorId(context, 5, "test_tensor", error));
  EXPECT_THAT(error, HasSubstr("Invalid tensor ID for test_tensor: 5"));

  // Invalid ID: negative (not optional).
  EXPECT_FALSE(ValidateTensorId(context, -2, "test_tensor", error));
  EXPECT_THAT(error, HasSubstr("Invalid tensor ID for test_tensor: -2"));

  // Optional tensor ID (not supported by default).
  EXPECT_FALSE(
      ValidateTensorId(context, kTfLiteOptionalTensor, "test_tensor", error));
  EXPECT_THAT(error, HasSubstr("Invalid tensor ID for test_tensor: -1"));
}

}  // namespace
}  // namespace litert::ml_drift::ir
