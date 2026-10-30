// Copyright 2026 The LiteRT Authors. All Rights Reserved.
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

#include "litert/vendors/arm_vulkan_ml/compiler/op_registry.h"

#include <set>
#include <utility>

#include "gtest/gtest.h"

namespace litert::arm_vulkan_ml {
namespace {

TEST(OpRegistry, UniqueSortedCodesAndTypePairs) {
  const auto& registrations = GetOperationRegistrations();
  EXPECT_EQ(registrations.size(), 109);
  int previous = -1;
  for (const auto& registration : registrations) {
    EXPECT_GT(registration.code, previous);
    previous = registration.code;
    EXPECT_GE(registration.data_input, 0);
    EXPECT_FALSE(registration.variants.empty());
    EXPECT_EQ(FindOperationRegistration(registration.code), &registration);
    std::set<std::pair<LiteRtElementType, LiteRtElementType>> types;
    for (const auto& variant : registration.variants) {
      EXPECT_TRUE(types.emplace(variant.input_type, variant.output_type).second);
      EXPECT_NE(variant.constraints[0], '\0');
      EXPECT_EQ(FindOperationVariant(registration, variant.input_type,
                                    variant.output_type), &variant);
    }
  }
}

TEST(OpRegistry, RoutingDependsOnTypesAndUsesTheCorrectDataInput) {
  auto route = [](LiteRtOpCode code, LiteRtElementType type) {
    const auto* registration = FindOperationRegistration(code);
    EXPECT_NE(registration, nullptr);
    const auto* variant = FindOperationVariant(*registration, type, type);
    EXPECT_NE(variant, nullptr);
    return variant->route;
  };
  EXPECT_EQ(route(kLiteRtOpCodeTflSum, kLiteRtElementTypeInt8), LoweringRoute::kGeneral);
  EXPECT_EQ(route(kLiteRtOpCodeTflSum, kLiteRtElementTypeFloat32), LoweringRoute::kExtended);
  EXPECT_EQ(route(kLiteRtOpCodeTflSub, kLiteRtElementTypeFloat32), LoweringRoute::kPortable);
  EXPECT_EQ(route(kLiteRtOpCodeTflAdd, kLiteRtElementTypeFloat32), LoweringRoute::kGeneral);
  EXPECT_EQ(FindOperationRegistration(kLiteRtOpCodeTflSplit)->data_input, 1);
  EXPECT_EQ(FindOperationRegistration(kLiteRtOpCodeTflScatterNd)->data_input, 1);
  EXPECT_EQ(FindOperationRegistration(kLiteRtOpCodeTflTransposeConv)->data_input, 2);
}

TEST(OpRegistry, RejectsUnregisteredTypesAndControlFlow) {
  EXPECT_EQ(FindOperationRegistration(kLiteRtOpCodeTflWhile), nullptr);
  const auto* softmax = FindOperationRegistration(kLiteRtOpCodeTflSoftmax);
  ASSERT_NE(softmax, nullptr);
  EXPECT_EQ(FindOperationVariant(*softmax, kLiteRtElementTypeInt16,
                                kLiteRtElementTypeInt16), nullptr);
  const auto* mean = FindOperationRegistration(kLiteRtOpCodeTflMean);
  ASSERT_NE(mean, nullptr);
  EXPECT_NE(FindOperationVariant(*mean, kLiteRtElementTypeInt8,
                                kLiteRtElementTypeInt8), nullptr);
  const auto* quantize = FindOperationRegistration(kLiteRtOpCodeTflQuantize);
  ASSERT_NE(quantize, nullptr);
  EXPECT_NE(FindOperationVariant(*quantize, kLiteRtElementTypeFloat32,
                                kLiteRtElementTypeInt8), nullptr);
  EXPECT_EQ(FindOperationVariant(*quantize, kLiteRtElementTypeInt8,
                                kLiteRtElementTypeFloat32), nullptr);
}

}  // namespace
}  // namespace litert::arm_vulkan_ml
