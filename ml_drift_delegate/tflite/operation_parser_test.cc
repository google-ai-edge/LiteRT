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

#include "ml_drift_delegate/tflite/operation_parser.h"

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <list>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/testing/matchers.h"

namespace litert::ml_drift {
namespace {

using ::testing::Not;
using ::testing::status::IsOk;
using ::testing::tflite::SimpleConstTensor;

// Convenience wrapper for TfLiteIntArray that frees it at destruction.
struct TfLiteIntArrayWrapper {
  TfLiteIntArrayWrapper() : ptr(TfLiteIntArrayCreate(0)) {}  // for `= {}`
  TfLiteIntArrayWrapper(std::initializer_list<int> il)
      : ptr(TfLiteIntArrayCreate(il.size())) {
    std::memcpy(ptr->data, std::data(il), il.size() * sizeof(int));
  }
  ~TfLiteIntArrayWrapper() { TfLiteIntArrayFree(ptr); }
  TfLiteIntArray* ptr;
};

TEST(OperationParserTest, CheckTensorShape) {
  {  // Success with 0D;
    TfLiteIntArrayWrapper dims = {};
    EXPECT_OK(CheckTensorShape(dims.ptr));
  }
  {  // Succeeds with 1D.
    TfLiteIntArrayWrapper dims = {42};
    EXPECT_OK(CheckTensorShape(dims.ptr));
  }
  {  // Succeeds with 2D.
    TfLiteIntArrayWrapper dims = {42, 69};
    EXPECT_OK(CheckTensorShape(dims.ptr));
  }
  {  // Succeeds with 3D.
    TfLiteIntArrayWrapper dims = {42, 69, 13};
    EXPECT_OK(CheckTensorShape(dims.ptr));
  }
  {  // Succeeds with 4D.
    TfLiteIntArrayWrapper dims = {42, 69, 13, 50};
    EXPECT_OK(CheckTensorShape(dims.ptr));
  }
  {  // Succeeds with 5D.
    TfLiteIntArrayWrapper dims = {42, 69, 13, 50, 87};
    EXPECT_OK(CheckTensorShape(dims.ptr));
  }
  {  // Fails with >=6D.
    TfLiteIntArrayWrapper dims = {42, 69, 13, 50, 87, 99};
    EXPECT_THAT(CheckTensorShape(dims.ptr), Not(IsOk()));
  }
}

TEST(OperationParserTest, CheckAllDimensionsScalar) {
  {  // Succeeds with 0D.
    TfLiteIntArrayWrapper dims = {};
    EXPECT_OK(CheckAllDimensions<::ml_drift::Scalar>(dims.ptr));
  }
  {  // Succeeds with special 1D [1].
    TfLiteIntArrayWrapper dims = {1};
    EXPECT_OK(CheckAllDimensions<::ml_drift::Scalar>(dims.ptr));
  }
  {  // Fails with general 1D.
    TfLiteIntArrayWrapper dims = {42};
    EXPECT_THAT(CheckAllDimensions<::ml_drift::Scalar>(dims.ptr), Not(IsOk()));
  }
  {  // Succeeds with special 2D [1, 1].
    TfLiteIntArrayWrapper dims = {1, 1};
    EXPECT_OK(CheckAllDimensions<::ml_drift::Scalar>(dims.ptr));
  }
  {  // Fails with general 2D.
    TfLiteIntArrayWrapper dims = {42, 69};
    EXPECT_THAT(CheckAllDimensions<::ml_drift::Scalar>(dims.ptr), Not(IsOk()));
  }
  {  // Succeeds with special 3D [1, 1, 1].
    TfLiteIntArrayWrapper dims = {1, 1, 1};
    EXPECT_OK(CheckAllDimensions<::ml_drift::Scalar>(dims.ptr));
  }
  {  // Fails with general 3D.
    TfLiteIntArrayWrapper dims = {42, 69, 13};
    EXPECT_THAT(CheckAllDimensions<::ml_drift::Scalar>(dims.ptr), Not(IsOk()));
  }
  {  // Succeeds with special 4D [1, 1, 1, 1].
    TfLiteIntArrayWrapper dims = {1, 1, 1, 1};
    EXPECT_OK(CheckAllDimensions<::ml_drift::Scalar>(dims.ptr));
  }
  {  // Fails with general 4D.
    TfLiteIntArrayWrapper dims = {42, 69, 13, 50};
    EXPECT_THAT(CheckAllDimensions<::ml_drift::Scalar>(dims.ptr), Not(IsOk()));
  }
}

TEST(OperationParserTest, CheckAllDimensionsLinear) {
  {  // Fails with 0D.
    TfLiteIntArrayWrapper dims = {};
    EXPECT_THAT(CheckAllDimensions<::ml_drift::Linear>(dims.ptr), Not(IsOk()));
  }
  {  // Succeeds with 1D.
    TfLiteIntArrayWrapper dims = {42};
    EXPECT_OK(CheckAllDimensions<::ml_drift::Linear>(dims.ptr));
  }
  {  // Succeeds with special 2D [1, x].
    TfLiteIntArrayWrapper dims = {1, 42};
    EXPECT_OK(CheckAllDimensions<::ml_drift::Linear>(dims.ptr));
  }
  {  // Fails with general 2D.
    TfLiteIntArrayWrapper dims = {42, 69};
    EXPECT_THAT(CheckAllDimensions<::ml_drift::Linear>(dims.ptr), Not(IsOk()));
  }
  {  // Succeeds with special 3D [1, 1, x].
    TfLiteIntArrayWrapper dims = {1, 1, 42};
    EXPECT_OK(CheckAllDimensions<::ml_drift::Linear>(dims.ptr));
  }
  {  // Fails with general 3D.
    TfLiteIntArrayWrapper dims = {42, 69, 13};
    EXPECT_THAT(CheckAllDimensions<::ml_drift::Linear>(dims.ptr), Not(IsOk()));
  }
  {  // Succeeds with special 4D [1, 1, 1, x].
    TfLiteIntArrayWrapper dims = {1, 1, 1, 42};
    EXPECT_OK(CheckAllDimensions<::ml_drift::Linear>(dims.ptr));
  }
  {  // Fails with general 4D.
    TfLiteIntArrayWrapper dims = {42, 69, 13, 50};
    EXPECT_THAT(CheckAllDimensions<::ml_drift::Linear>(dims.ptr), Not(IsOk()));
  }
}

TEST(OperationParserTest, CheckAllDimensionsHw) {
  {  // Fails with 0D.
    TfLiteIntArrayWrapper dims = {};
    EXPECT_THAT(CheckAllDimensions<::ml_drift::HW>(dims.ptr), Not(IsOk()));
  }
  {  // Fails with 1D.
    TfLiteIntArrayWrapper dims = {42};
    EXPECT_THAT(CheckAllDimensions<::ml_drift::HW>(dims.ptr), Not(IsOk()));
  }
  {  // Succeeds with 2D.
    TfLiteIntArrayWrapper dims = {42, 69};
    EXPECT_OK(CheckAllDimensions<::ml_drift::HW>(dims.ptr));
  }
  // TODO: who/impjdi - Should succeed with special 3D [1, x, y], but fails.
  // //third_party/ml_drift/common/operation_parser.cc;l=109-113;rcl=682602986
  // {
  //   TfLiteIntArrayWrapper dims = {1, 42, 69};
  //   EXPECT_OK(CheckAllDimensions<HW>(dims.ptr));
  // }
  {  // Fails with general 3D.
    TfLiteIntArrayWrapper dims = {42, 69, 13};
    EXPECT_THAT(CheckAllDimensions<::ml_drift::HW>(dims.ptr), Not(IsOk()));
  }
  // TODO: who/impjdi - Should succeed with special 4D [1, 1, x, y], but fails.
  // //third_party/ml_drift/common/operation_parser.cc;l=109-113;rcl=682602986
  // {
  //   TfLiteIntArrayWrapper dims = {1, 1, 42, 69};
  //   EXPECT_OK(CheckAllDimensions<HW>(dims.ptr));
  // }
  {  // Fails with general 4D.
    TfLiteIntArrayWrapper dims = {42, 69, 13, 50};
    EXPECT_THAT(CheckAllDimensions<::ml_drift::HW>(dims.ptr), Not(IsOk()));
  }
}

TEST(OperationParserTest, CheckAllDimensionsHWC) {
  {  // Fails with 0D.
    TfLiteIntArrayWrapper dims = {};
    EXPECT_THAT(CheckAllDimensions<::ml_drift::HWC>(dims.ptr), Not(IsOk()));
  }
  {  // Fails with 1D.
    TfLiteIntArrayWrapper dims = {42};
    EXPECT_THAT(CheckAllDimensions<::ml_drift::HWC>(dims.ptr), Not(IsOk()));
  }
  {  // Fails with 2D.
    TfLiteIntArrayWrapper dims = {42, 69};
    EXPECT_THAT(CheckAllDimensions<::ml_drift::HWC>(dims.ptr), Not(IsOk()));
  }
  {  // Succeeds with 3D.
    TfLiteIntArrayWrapper dims = {42, 69, 13};
    EXPECT_OK(CheckAllDimensions<::ml_drift::HWC>(dims.ptr));
  }
  {  // Succeeds with special 4D [1, x, y, z].
    TfLiteIntArrayWrapper dims = {1, 42, 69, 13};
    EXPECT_OK(CheckAllDimensions<::ml_drift::HWC>(dims.ptr));
  }
  {  // Fails with general 4D.
    TfLiteIntArrayWrapper dims = {42, 69, 13, 50};
    EXPECT_THAT(CheckAllDimensions<::ml_drift::HWC>(dims.ptr), Not(IsOk()));
  }
}

TEST(OperationParserTest, CheckAllDimensionsBHWC) {
  // In CheckAllDimensionsBWHC, it only check with CheckTensorShape.
  // Therefore 0 ~ 4D will be okay.
  ///
  // {  // Fails with 0D.
  //  TfLiteIntArrayWrapper dims = {};
  //  EXPECT_THAT(CheckAllDimensions(dims.ptr, &shape), Not(IsOk()));
  // }
  // TODO: who/impjdi - Should fail with 1D, but succeeds.
  // //third_party/ml_drift/common/model_builder_helper.cc;l=378;rcl=678476359
  // {
  //   TfLiteIntArrayWrapper dims = {42};
  //   EXPECT_THAT(CheckAllDimensions(dims.ptr, &shape), Not(IsOk()));
  // }
  // TODO: who/impjdi - Should fail with 2D, but succeeds.
  // //third_party/ml_drift/common/model_builder_helper.cc;l=378;rcl=678476359
  // {
  //   TfLiteIntArrayWrapper dims = {42, 69};
  //   EXPECT_THAT(CheckAllDimensions(dims.ptr, &shape), Not(IsOk()));
  // }
  // TODO: who/impjdi - Should fail with 3D, but succeeds.
  // //third_party/ml_drift/common/model_builder_helper.cc;l=378;rcl=678476359
  // {
  //   TfLiteIntArrayWrapper dims = {42, 69, 13};
  //   EXPECT_THAT(CheckAllDimensions(dims.ptr, &shape), Not(IsOk()));
  // }
  {  // Succeeds with 4D.
    TfLiteIntArrayWrapper dims = {42, 69, 13, 50};
    EXPECT_OK(CheckAllDimensions<::ml_drift::BHWC>(dims.ptr));
  }
}

TEST(OperationParserTest, CheckAllDimensionsOHWI) {
  {  // Fails with 0D.
    TfLiteIntArrayWrapper dims = {};
    EXPECT_THAT(CheckAllDimensions<::ml_drift::OHWI>(dims.ptr), Not(IsOk()));
  }
  {  // Fails with 1D.
    TfLiteIntArrayWrapper dims = {42};
    EXPECT_THAT(CheckAllDimensions<::ml_drift::OHWI>(dims.ptr), Not(IsOk()));
  }
  {  // Fails with 2D.
    TfLiteIntArrayWrapper dims = {42, 69};
    EXPECT_THAT(CheckAllDimensions<::ml_drift::OHWI>(dims.ptr), Not(IsOk()));
  }
  {  // Fails with 3D.
    TfLiteIntArrayWrapper dims = {42, 69, 13};
    EXPECT_THAT(CheckAllDimensions<::ml_drift::OHWI>(dims.ptr), Not(IsOk()));
  }
  {  // Succeeds with 4D.
    TfLiteIntArrayWrapper dims = {42, 69, 13, 50};
    EXPECT_OK(CheckAllDimensions<::ml_drift::OHWI>(dims.ptr));
  }
}

TEST(OperationParserTest, PreCheckCopyDataNonFloatSrc) {
  {  // Succeeds with float32 src.
    float data = 0.0f;
    SimpleConstTensor src(kTfLiteFloat32, {1}, absl::MakeSpan(&data, 1));
    uint32_t dst;
    EXPECT_OK(PreCheckCopyData(src, &dst));
  }
  {  // Fails with float16 src.
    ::ml_drift::half data(0.0f);
    SimpleConstTensor src(kTfLiteFloat16, {1}, absl::MakeSpan(&data, 1));
    uint16_t dst;
    EXPECT_THAT(PreCheckCopyData(src, &dst), Not(IsOk()));
  }
}

TEST(OperationParserTest, PreCheckCopyDataFloatSrc) {
  {  // Succeeds with float32 src.
    float data = 0.0f;
    SimpleConstTensor src(kTfLiteFloat32, {1}, absl::MakeSpan(&data, 1));
    float dst;
    EXPECT_OK(PreCheckCopyData(src, &dst));
  }
  {  // Succeeds with float16 src.
    ::ml_drift::half data(0.0f);
    SimpleConstTensor src(kTfLiteFloat16, {1}, absl::MakeSpan(&data, 1));
    float dst;
    EXPECT_OK(PreCheckCopyData(src, &dst));
  }
}

TEST(OperationParserTest, PreCheckTensorToTensorWrongShape) {
  float data = 0.0f;
  SimpleConstTensor src(kTfLiteFloat32, {1}, absl::MakeSpan(&data, 1));
  ::ml_drift::Tensor<::ml_drift::HW, ::ml_drift::DataType::FLOAT32> dst;
  EXPECT_THAT(PreCheckTensorToTensor(&src, &dst), Not(IsOk()));
}

TEST(OperationParserTest, PreCheckAxisFromIndex) {
  float data = 0.0f;
  {  // Succeeds with negative index.
    SimpleConstTensor src(kTfLiteFloat32, {1}, absl::MakeSpan(&data, 1));
    EXPECT_OK(PreCheckAxisFromIndex(src, -1));
  }
  {  // Succeeds with proper index.
    SimpleConstTensor src(kTfLiteFloat32, {1}, absl::MakeSpan(&data, 1));
    EXPECT_OK(PreCheckAxisFromIndex(src, 0));
  }
  {  // Fails with out-of-bound index.
    SimpleConstTensor src(kTfLiteFloat32, {1}, absl::MakeSpan(&data, 1));
    EXPECT_THAT(PreCheckAxisFromIndex(src, 1), Not(IsOk()));
  }
  {  // Fails with >4D tensor.
    SimpleConstTensor src(kTfLiteFloat32, {1, 1, 1, 1, 1},
                          absl::MakeSpan(&data, 1));
    EXPECT_THAT(PreCheckAxisFromIndex(src, 0), Not(IsOk()));
  }
}

TEST(OperationParserTest, PreCheckTfLiteShape) {
  float data = 0.0f;
  {
    // Fails with 0D tensor
    SimpleConstTensor src(kTfLiteFloat32, {}, absl::MakeSpan(&data, 1));
    src.name = nullptr;
    EXPECT_THAT(PreCheckTfLiteShape(src), Not(IsOk()));
  }
  {  // Succeeds with 1D tensor.
    SimpleConstTensor src(kTfLiteFloat32, {1}, absl::MakeSpan(&data, 1));
    EXPECT_OK(PreCheckTfLiteShape(src));
  }
  {  // Succeeds with 2D tensor.
    SimpleConstTensor src(kTfLiteFloat32, {1, 1}, absl::MakeSpan(&data, 1));
    EXPECT_OK(PreCheckTfLiteShape(src));
  }
  {  // Succeeds with 3D tensor.
    SimpleConstTensor src(kTfLiteFloat32, {1, 1, 1}, absl::MakeSpan(&data, 1));
    EXPECT_OK(PreCheckTfLiteShape(src));
  }
  {  // Succeeds with 4D tensor.
    SimpleConstTensor src(kTfLiteFloat32, {1, 1, 1, 1},
                          absl::MakeSpan(&data, 1));
    EXPECT_OK(PreCheckTfLiteShape(src));
  }
  {
    SimpleConstTensor src(kTfLiteFloat32, {1, 1, 1, 1, 1},
                          absl::MakeSpan(&data, 1));
    src.name = nullptr;
    EXPECT_THAT(PreCheckTfLiteShape(src), Not(IsOk()));
  }
}

TEST(OperationParserTest, PreCheckMaybeFuseActivation) {
  {  // Fails with more than one output.
    TfLiteNode node;
    node.outputs = TfLiteIntArrayCreate(2);
    EXPECT_THAT(PreCheckMaybeFuseActivation(&node, kTfLiteActNone),
                Not(IsOk()));
    TfLiteIntArrayFree(node.outputs);
  }
  {  // Succeeds with one output.
    TfLiteNode node;
    node.outputs = TfLiteIntArrayCreate(1);
    EXPECT_OK(PreCheckMaybeFuseActivation(&node, kTfLiteActNone));
    TfLiteIntArrayFree(node.outputs);
  }
}

TEST(OperationParserTest, PreCheckMaybeFuseActivationForElementwiseNode) {
  {  // Succeeds without the optional TfLiteAddParams.
    TfLiteNode node;
    node.builtin_data = nullptr;
    EXPECT_OK(PreCheckMaybeFuseActivationForElementwiseNode(
        ::ml_drift::OperationType::ADD, &node));
  }
  {  // Fails with more than one output.
    TfLiteNode node;
    TfLiteAddParams params;
    params.activation = kTfLiteActRelu;
    node.builtin_data = &params;
    node.outputs = TfLiteIntArrayCreate(2);
    EXPECT_THAT(PreCheckMaybeFuseActivationForElementwiseNode(
                    ::ml_drift::OperationType::ADD, &node),
                Not(IsOk()));
    TfLiteIntArrayFree(node.outputs);
  }
  {  // Succeeds with one output.
    TfLiteNode node;
    TfLiteAddParams params;
    params.activation = kTfLiteActRelu;
    node.builtin_data = &params;
    node.outputs = TfLiteIntArrayCreate(1);
    EXPECT_OK(PreCheckMaybeFuseActivationForElementwiseNode(
        ::ml_drift::OperationType::ADD, &node));
    TfLiteIntArrayFree(node.outputs);
  }
}

// TODO(b/371137307): To setup TfLiteContext is relatively complicated. Add
//                    test cases for helper functions with TfLiteContext as
//                    argument in the future.

}  // namespace
}  // namespace litert::ml_drift
