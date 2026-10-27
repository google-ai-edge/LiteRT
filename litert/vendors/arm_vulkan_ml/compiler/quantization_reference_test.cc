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

#include "litert/vendors/arm_vulkan_ml/compiler/quantization.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include "tflite/kernels/internal/common.h"
#include "tflite/kernels/internal/quantization_util.h"
#include "tflite/kernels/internal/reference/hard_swish.h"
#include "tflite/kernels/internal/types.h"
namespace litert::arm_vulkan_ml {
namespace {
TEST(HardSwishTable, ExhaustiveSignedCodesMatchBuiltinFixedPoint) {
  const std::array<std::array<float,4>,5> cases={{{.10759072005748749f,-18,.06274118274450302f,-122},
    {.08598509430885315f,17,.03870214894413948f,-118},
    {.024599973112344742f,54,.007206030189990997f,-76},
    {.2786346673965454f,15,.1233467161655426f,-125},
    {.017f,-128,.006f,-100}}};
  for(const auto& q:cases) {
    auto table=HardSwishTable(q[0],static_cast<int>(q[1]),q[2],static_cast<int>(q[3]));ASSERT_TRUE(table);
    tflite::HardSwishParams p{};p.input_zero_point=q[1];p.output_zero_point=q[3];
    const float hires=q[0]/128.0f;int32_t multiplier;
    tflite::QuantizeMultiplier(static_cast<float>(hires/q[2]),&multiplier,&p.output_multiplier_exponent);
    tflite::DownScaleInt32ToInt16Multiplier(multiplier,&p.output_multiplier_fixedpoint_int16);
    tflite::QuantizeMultiplier(static_cast<float>(hires/(3.0f/32768.0f)),&multiplier,&p.reluish_multiplier_exponent);
    tflite::DownScaleInt32ToInt16Multiplier(multiplier,&p.reluish_multiplier_fixedpoint_int16);
    std::array<int8_t,256> input,output;
    for(int i=0;i<256;++i)input[i]=i-128;
    tflite::RuntimeShape shape({256});
    tflite::reference_ops::HardSwish(p,shape,input.data(),shape,output.data());
    EXPECT_TRUE(std::equal(table->begin(),table->end(),output.begin()));
  }
  EXPECT_FALSE(HardSwishTable(0,0,1,0));
  EXPECT_FALSE(HardSwishTable(1,128,1,0));
  EXPECT_FALSE(HardSwishTable(1,0,0,0));
}
TEST(RequantizeTable, ExhaustiveSignedCodesMatchReferenceRounding) {
  for (float scale : {0.000001f, 0.125f, 0.4785989f, 0.75f, 1.0f, 1.22785f, 2.0f, 16.0f}) {
    for (int zero_point : {-128, -11, 0, 127}) {
      auto table = RequantizeTable(scale, zero_point, 1.0f, -6);
      ASSERT_TRUE(table);
      int32_t multiplier;
      int shift;
      tflite::QuantizeMultiplier(static_cast<double>(scale), &multiplier, &shift);
      for (int code = -128; code <= 127; ++code) {
        int value = tflite::MultiplyByQuantizedMultiplier(code - zero_point, multiplier, shift) - 6;
        EXPECT_EQ((*table)[code + 128], std::clamp(value, -128, 127))
            << "scale=" << scale << " zero_point=" << zero_point << " code=" << code;
      }
    }
  }
  EXPECT_FALSE(RequantizeTable(0, 0, 1, 0));
  EXPECT_FALSE(RequantizeTable(1, 128, 1, 0));
  EXPECT_FALSE(RequantizeTable(1e30f, 0, 1, 0));
  auto tiny = RequantizeTable(1e-30f, 0, 1, -7);
  ASSERT_TRUE(tiny);
  EXPECT_TRUE(std::all_of(tiny->begin(), tiny->end(), [](int8_t x) { return x == -7; }));
}

}  // namespace
}  // namespace litert::arm_vulkan_ml
