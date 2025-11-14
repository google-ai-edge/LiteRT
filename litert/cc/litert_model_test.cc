// Copyright 2024 Google LLC.
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

#include "litert/cc/litert_model.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_layout.h"
#include "litert/core/model/model.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"

// Tests for CC Wrapper classes around public C api.

namespace litert {

namespace {

constexpr int32_t kTensorDimensions[] = {1, 2, 3};

constexpr uint32_t kTensorStrides[] = {6, 3, 1};

constexpr LiteRtLayout kLayout = BuildLayout(kTensorDimensions);

constexpr LiteRtLayout kLayoutWithStrides =
    BuildLayout(kTensorDimensions, kTensorStrides);

constexpr LiteRtRankedTensorType kTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    /*.layout=*/kLayout,
};

//===----------------------------------------------------------------------===//
//                                CC Model                                    //
//===----------------------------------------------------------------------===//

TEST(CcModelTest, SimpleModel) {
  auto model = testing::LoadTestFileModel("one_mul.tflite");

  LiteRtParamIndex num_subgraphs;
  ASSERT_EQ(LiteRtGetNumModelSubgraphs(model.Get(), &num_subgraphs),
            kLiteRtStatusOk);
  EXPECT_EQ(num_subgraphs, 1);

  LiteRtParamIndex main_subgraph_index;
  ASSERT_EQ(LiteRtGetMainModelSubgraphIndex(model.Get(), &main_subgraph_index),
            kLiteRtStatusOk);
  EXPECT_EQ(main_subgraph_index, 0);

  LiteRtSubgraph litert_subgraph_0;
  ASSERT_EQ(LiteRtGetModelSubgraph(model.Get(), /*subgraph_index=*/0,
                                   &litert_subgraph_0),
            kLiteRtStatusOk);

  auto subgraph_0 = model.Subgraph(0);
  ASSERT_TRUE(subgraph_0);
  EXPECT_EQ(subgraph_0->Get(), litert_subgraph_0);

  auto main_subgraph = model.MainSubgraph();
  EXPECT_EQ(main_subgraph->Get(), subgraph_0->Get());
}

TEST(CcModelTest, SimpleModelSignature) {
  auto model = testing::LoadTestFileModel("reverse_signature_model.tflite");

  EXPECT_EQ(model.GetNumSignatures(), 1);
  auto signature_keys = model.GetSignatureKeys();
  ASSERT_TRUE(signature_keys);
  EXPECT_EQ(signature_keys->size(), 1);
  EXPECT_EQ(signature_keys->at(0), "serving_default");

  auto input_names = model.GetSignatureInputNames();
  ASSERT_TRUE(input_names);
  EXPECT_EQ(input_names->size(), 2);
  EXPECT_EQ(input_names->at(0), "y");
  EXPECT_EQ(input_names->at(1), "x");

  auto output_names = model.GetSignatureOutputNames();
  ASSERT_TRUE(output_names);
  EXPECT_EQ(output_names->size(), 2);
  EXPECT_EQ(output_names->at(0), "sum");
  EXPECT_EQ(output_names->at(1), "prod");
}

//===----------------------------------------------------------------------===//
//                                CC Signature                                //
//===----------------------------------------------------------------------===//

TEST(CcSignatureTest, Basic) {
  auto model = testing::LoadTestFileModel("one_mul.tflite");

  auto signatures = model.GetSignatures();
  ASSERT_TRUE(signatures);
  ASSERT_EQ(signatures->size(), 1);
  auto& signature = signatures->at(0);
  EXPECT_THAT(signature.Key(), Model::DefaultSignatureKey());
  auto input_names = signature.InputNames();
  EXPECT_THAT(input_names[0], "arg0");
  EXPECT_THAT(input_names[1], "arg1");
  auto output_names = signature.OutputNames();
  EXPECT_THAT(output_names[0], "tfl.mul");
}

TEST(CcSignatureTest, Lookup) {
  auto model = testing::LoadTestFileModel("one_mul.tflite");

  {
    auto signature = model.FindSignature("nonexistent");
    ASSERT_FALSE(signature);
  }
  auto signature = model.FindSignature(Model::DefaultSignatureKey());
  ASSERT_TRUE(signature);
  EXPECT_THAT(signature->Key(), Model::DefaultSignatureKey());
  auto input_names = signature->InputNames();
  EXPECT_THAT(input_names[0], "arg0");
  EXPECT_THAT(input_names[1], "arg1");
  auto output_names = signature->OutputNames();
  EXPECT_THAT(output_names[0], "tfl.mul");
}

//===----------------------------------------------------------------------===//
//                                CC Layout                                   //
//===----------------------------------------------------------------------===//

TEST(CcLayoutTest, NoStrides) {
  Layout layout(kLayout);

  ASSERT_EQ(layout.Rank(), kLayout.rank);
  for (auto i = 0; i < layout.Rank(); ++i) {
    ASSERT_EQ(layout.Dimensions()[i], kLayout.dimensions[i]);
  }
  ASSERT_FALSE(layout.HasStrides());
}

TEST(CcLayoutTest, WithStrides) {
  Layout layout(kLayoutWithStrides);

  ASSERT_EQ(layout.Rank(), kLayoutWithStrides.rank);
  for (auto i = 0; i < layout.Rank(); ++i) {
    ASSERT_EQ(layout.Dimensions()[i], kLayoutWithStrides.dimensions[i]);
  }
  ASSERT_TRUE(layout.HasStrides());
  for (auto i = 0; i < layout.Rank(); ++i) {
    ASSERT_EQ(layout.Strides()[i], kLayoutWithStrides.strides[i]);
  }
}

TEST(CcLayoutTest, Equal) {
  Layout layout1(Dimensions({2, 2}));
  Layout layout2(Dimensions({2, 2}));
  ASSERT_TRUE(layout1 == layout2);
}

TEST(CcLayoutTest, NotEqual) {
  Layout layout1(Dimensions({2, 2}));
  Layout layout2(Dimensions({2, 2}), litert::Strides({6, 3, 1}));
  ASSERT_FALSE(layout1 == layout2);
}

TEST(CcLayoutTest, NumElements) {
  Layout layout(Dimensions({2, 2, 3}));
  auto num_elements = layout.NumElements();
  ASSERT_TRUE(num_elements);
  EXPECT_EQ(*num_elements, 12);
}

//===----------------------------------------------------------------------===//
//                           CC RankedTensorType                              //
//===----------------------------------------------------------------------===//

TEST(CcRankedTensorTypeTest, Accessors) {
  Layout layout(kLayout);
  RankedTensorType tensor_type(kTensorType);
  ASSERT_EQ(tensor_type.ElementType(),
            static_cast<ElementType>(kTensorType.element_type));
  ASSERT_TRUE(tensor_type.Layout() == layout);
}

TEST(CcTensorTest, Name) {
  constexpr absl::string_view kName = "foo";
  LiteRtTensorT tensor;
  tensor.SetName(std::string(kName));

  Tensor cc_tensor(&tensor);
  EXPECT_EQ(cc_tensor.Name(), kName);
}

TEST(CcTensorTest, Index) {
  constexpr std::uint32_t kIndex = 1;
  LiteRtTensorT tensor;
  tensor.SetTensorIndex(kIndex);

  Tensor cc_tensor(&tensor);
  EXPECT_EQ(cc_tensor.TensorIndex(), kIndex);
}

//===----------------------------------------------------------------------===//
//                               CC ElementType                               //
//===----------------------------------------------------------------------===//

TEST(CcElementTypeTest, GetByteWidth) {
  const auto width = GetByteWidth<ElementType::Bool>().NumBytes();
  EXPECT_EQ(width, 1);
}

TEST(CcElementTypeTest, GetElementType) {
  ElementType ty = GetElementType<float>();
  EXPECT_EQ(ty, ElementType::Float32);
}

}  // namespace
}  // namespace litert
