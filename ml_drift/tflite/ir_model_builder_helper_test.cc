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

#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"

#include <vector>

#include "testing/base/public/gunit.h"
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "tflite/c/builtin_op_data.h"

namespace litert::ml_drift::ir {
namespace {

TEST(IrModelBuilderHelperTest, ExtractTensorShape) {
  TfLiteIntArray* dims = TfLiteIntArrayCreate(0);
  EXPECT_EQ(ExtractTensorShape(dims), ::ml_drift::BHWDC(1, 1, 1, 1, 1));
  TfLiteIntArrayFree(dims);

  dims = TfLiteIntArrayCreate(1);
  dims->data[0] = 4;
  EXPECT_EQ(ExtractTensorShape(dims), ::ml_drift::BHWDC(4, 1, 1, 1, 1));
  TfLiteIntArrayFree(dims);

  dims = TfLiteIntArrayCreate(2);
  dims->data[0] = 2;
  dims->data[1] = 3;
  EXPECT_EQ(ExtractTensorShape(dims), ::ml_drift::BHWDC(2, 1, 1, 1, 3));
  TfLiteIntArrayFree(dims);

  dims = TfLiteIntArrayCreate(3);
  dims->data[0] = 2;
  dims->data[1] = 3;
  dims->data[2] = 4;
  EXPECT_EQ(ExtractTensorShape(dims), ::ml_drift::BHWDC(2, 1, 3, 1, 4));
  TfLiteIntArrayFree(dims);

  dims = TfLiteIntArrayCreate(4);
  dims->data[0] = 1;
  dims->data[1] = 2;
  dims->data[2] = 3;
  dims->data[3] = 4;
  EXPECT_EQ(ExtractTensorShape(dims), ::ml_drift::BHWDC(1, 2, 3, 1, 4));
  TfLiteIntArrayFree(dims);

  dims = TfLiteIntArrayCreate(5);
  dims->data[0] = 1;
  dims->data[1] = 2;
  dims->data[2] = 3;
  dims->data[3] = 4;
  dims->data[4] = 5;
  EXPECT_EQ(ExtractTensorShape(dims), ::ml_drift::BHWDC(1, 2, 3, 4, 5));
  TfLiteIntArrayFree(dims);
}

TEST(IrModelBuilderHelperTest, GetConcatAxis) {
  EXPECT_EQ(GetConcatAxis({::ml_drift::BHWDC(1, 2, 3, 1, 5)},
                          ::ml_drift::BHWDC(2, 2, 3, 1, 5)),
            ::ml_drift::Axis::BATCH);
  EXPECT_EQ(GetConcatAxis({::ml_drift::BHWDC(1, 2, 3, 1, 5)},
                          ::ml_drift::BHWDC(1, 4, 3, 1, 5)),
            ::ml_drift::Axis::HEIGHT);
  EXPECT_EQ(GetConcatAxis({::ml_drift::BHWDC(1, 2, 3, 1, 5)},
                          ::ml_drift::BHWDC(1, 2, 6, 1, 5)),
            ::ml_drift::Axis::WIDTH);
  EXPECT_EQ(GetConcatAxis({::ml_drift::BHWDC(1, 2, 3, 1, 5)},
                          ::ml_drift::BHWDC(1, 2, 3, 2, 5)),
            ::ml_drift::Axis::DEPTH);
  EXPECT_EQ(GetConcatAxis({::ml_drift::BHWDC(1, 2, 3, 1, 5)},
                          ::ml_drift::BHWDC(1, 2, 3, 1, 10)),
            ::ml_drift::Axis::CHANNELS);
}

TEST(IrModelBuilderHelperTest, UpdatePadding) {
  ::ml_drift::DepthwiseConvolution2DAttributes attr;
  attr.weights
      .emplace<
          ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>>()
      .shape = ::ml_drift::OHWI(1, 3, 3, 1);
  attr.strides = ::ml_drift::HW(2, 2);
  attr.dilations = ::ml_drift::HW(1, 1);
  UpdatePadding(kTfLitePaddingSame, ::ml_drift::BHWDC(1, 7, 7, 1, 1), &attr);
  EXPECT_EQ(attr.padding.prepended, ::ml_drift::HW(1, 1));
  EXPECT_EQ(attr.padding.appended, ::ml_drift::HW(1, 1));
  UpdatePadding(kTfLitePaddingValid, ::ml_drift::BHWDC(1, 7, 7, 1, 1), &attr);
  EXPECT_EQ(attr.padding.prepended, ::ml_drift::HW(0, 0));
  EXPECT_EQ(attr.padding.appended, ::ml_drift::HW(0, 0));
}


}  // namespace
}  // namespace litert::ml_drift::ir
