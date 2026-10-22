/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include <stdint.h>

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tflite/core/interpreter.h"
#include "tflite/kernels/test_util.h"
#include "tflite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

template <typename T>
class MatrixSetDiagOpModel : public SingleOpModel {
 public:
  explicit MatrixSetDiagOpModel(const TensorData& input, const TensorData& diag,
                                bool allocate_and_delegate = true) {
    input_ = AddInput(input);
    diag_ = AddInput(diag);
    output_ = AddOutput({input.type, {}});

    SetBuiltinOp(BuiltinOperator_MATRIX_SET_DIAG,
                 BuiltinOptions_MatrixSetDiagOptions,
                 CreateMatrixSetDiagOptions(builder_).Union());
    BuildInterpreter({GetShape(input_), GetShape(diag_)}, /*num_threads=*/-1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/true, allocate_and_delegate);
  }

  int input() { return input_; }
  int diag() { return diag_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }
  TfLiteType GetOutputType() {
    TfLiteTensor* t = interpreter_->tensor(output_);
    return t->type;
  }

 private:
  int input_;
  int diag_;
  int output_;
};

// Use the machinery of TYPED_TEST_SUITE to test all supported types.
// See
// https://github.com/google/googletest/blob/master/googletest/docs/advanced.md#typed-tests
// for details.
template <typename T>
class MatrixSetDiagOpTest : public ::testing::Test {};

using TypesUnderTest =
    ::testing::Types<TypeUnion<int32_t>, TypeUnion<float>, TypeUnion<int16_t>,
                     TypeUnion<int8_t>, TypeUnion<uint8_t>>;

TYPED_TEST_SUITE(MatrixSetDiagOpTest, TypesUnderTest);

TYPED_TEST(MatrixSetDiagOpTest, ThreeByThreeDiagScatter) {
  MatrixSetDiagOpModel<typename TypeParam::ScalarType> model(
      {TypeParam::tensor_type, {3, 3}}, {TypeParam::tensor_type, {3}});
  model.template PopulateTensor<typename TypeParam::ScalarType>(model.input(),
                                                                {7, 1, 2,  //
                                                                 3, 8, 4,  //
                                                                 5, 6, 9});
  model.template PopulateTensor<typename TypeParam::ScalarType>(model.diag(),
                                                                {0, 4, 2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({0, 1, 2,  //
                                                   3, 4, 4,  //
                                                   5, 6, 2}));
  EXPECT_THAT(model.GetOutputType(), TypeParam::tflite_type);
}

TEST(MatrixSetDiagTest, Int32TestMoreColumnsThanRows) {
  MatrixSetDiagOpModel<int32_t> model({TensorType_INT32, {2, 3}},
                                      {TensorType_INT32, {2}});
  model.PopulateTensor<int32_t>(model.input(), {0, 0, 0,  //
                                                9, 9, 9});
  model.PopulateTensor<int32_t>(model.diag(), {1, 1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 0, 0,  //
                                                   9, 1, 9}));
  EXPECT_THAT(model.GetOutputType(), TfLiteType::kTfLiteInt32);
}

TEST(MatrixSetDiagTest, Int32TestTwoDimDiag) {
  MatrixSetDiagOpModel<int32_t> model({TensorType_INT32, {2, 4, 4}},
                                      {TensorType_INT32, {2, 4}});
  model.PopulateTensor<int32_t>(model.input(), {5, 5, 5, 5,  //
                                                5, 5, 5, 5,  //
                                                5, 5, 5, 5,  //
                                                5, 5, 5, 5,  //
                                                1, 1, 1, 1,  //
                                                1, 1, 1, 1,  //
                                                1, 1, 1, 1,  //
                                                1, 1, 1, 1});
  model.PopulateTensor<int32_t>(model.diag(), {1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 4, 4));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 5, 5, 5,  //
                                                   5, 2, 5, 5,  //
                                                   5, 5, 3, 5,  //
                                                   5, 5, 5, 4,  //
                                                   5, 1, 1, 1,  //
                                                   1, 6, 1, 1,  //
                                                   1, 1, 7, 1,  //
                                                   1, 1, 1, 8}));
  EXPECT_THAT(model.GetOutputType(), TfLiteType::kTfLiteInt32);
}

TEST(MatrixSetDiagTest, RejectsMismatchedDiagonalType) {
  MatrixSetDiagOpModel<float> model({TensorType_FLOAT32, {2, 2}},
                                    {TensorType_INT32, {2}}, false);
  EXPECT_EQ(model.AllocateTensors(), kTfLiteError);
}

TEST(MatrixSetDiagTest, RejectsMismatchedDiagonalRank) {
  for (const std::vector<int>& shape :
       {std::vector<int>{}, std::vector<int>{1, 2}}) {
    MatrixSetDiagOpModel<int32_t> model({TensorType_INT32, {2, 2}},
                                        {TensorType_INT32, shape}, false);
    EXPECT_EQ(model.AllocateTensors(), kTfLiteError);
  }
}

TEST(MatrixSetDiagTest, RejectsMismatchedDiagonalBatchShape) {
  for (const std::vector<int>& shape :
       {std::vector<int>{1, 3, 4}, std::vector<int>{2, 2, 4}}) {
    MatrixSetDiagOpModel<int32_t> model({TensorType_INT32, {2, 3, 4, 5}},
                                        {TensorType_INT32, shape}, false);
    EXPECT_EQ(model.AllocateTensors(), kTfLiteError);
  }
}

TEST(MatrixSetDiagTest, RejectsMismatchedDiagonalLength) {
  for (int length : {1, 3}) {
    SCOPED_TRACE(length);
    MatrixSetDiagOpModel<int32_t> model({TensorType_INT32, {2, 3}},
                                        {TensorType_INT32, {length}}, false);
    EXPECT_EQ(model.AllocateTensors(), kTfLiteError);
  }
}

TEST(MatrixSetDiagTest, RejectsUnsupportedTypes) {
  for (TensorType type : {TensorType_BOOL, TensorType_FLOAT16}) {
    SCOPED_TRACE(type);
    MatrixSetDiagOpModel<uint8_t> model({type, {2, 2}}, {type, {2}});
    EXPECT_EQ(model.Invoke(), kTfLiteError);
  }
}

TEST(MatrixSetDiagTest, MoreRowsThanColumns) {
  MatrixSetDiagOpModel<int32_t> model({TensorType_INT32, {3, 2}},
                                      {TensorType_INT32, {2}});
  model.PopulateTensor<int32_t>(model.input(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int32_t>(model.diag(), {7, 8});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 2));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({7, 2, 3, 8, 5, 6}));
}

TEST(MatrixSetDiagTest, EmptyDiagonal) {
  MatrixSetDiagOpModel<float> model({TensorType_FLOAT32, {2, 0, 3}},
                                    {TensorType_FLOAT32, {2, 0}});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 0, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAre());
}

}  // namespace
}  // namespace tflite
