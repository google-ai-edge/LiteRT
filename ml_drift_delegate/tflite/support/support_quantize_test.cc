// Copyright 2026 Google LLC.
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

#include "ml_drift_delegate/tflite/support/support_quantize.h"

#include <string>
#include <tuple>

#include "testing/base/public/gunit.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {
namespace {

class SupportQuantizeTest
    : public ::testing::TestWithParam<std::tuple<TfLiteType, TfLiteType>> {};

TEST_P(SupportQuantizeTest, SupportedTypes) {
  TfLiteContext context = {};
  context.tensors_size = 2;
  TfLiteNode node = {};
  TfLiteRegistration registration = {};
  registration.version = 1;

  int inputs[] = {1, 0};
  int outputs[] = {1, 1};
  node.inputs = reinterpret_cast<TfLiteIntArray*>(inputs);
  node.outputs = reinterpret_cast<TfLiteIntArray*>(outputs);

  TfLiteTensor tensors[2] = {};
  tensors[0].type = std::get<0>(GetParam());
  tensors[1].type = std::get<1>(GetParam());
  context.tensors = tensors;

  TfLiteAffineQuantization quant_params = {};
  tensors[1].quantization.params = &quant_params;
  tensors[1].quantization.type = kTfLiteAffineQuantization;

  std::string error;
  EXPECT_TRUE(IsQuantizeSupported(&context, &node, &registration, &error))
      << error;
}

INSTANTIATE_TEST_SUITE_P(
    SupportQuantizeTest, SupportQuantizeTest,
    ::testing::Combine(::testing::Values(kTfLiteFloat32, kTfLiteFloat16),
                       ::testing::Values(kTfLiteInt8, kTfLiteUInt8, kTfLiteInt4,
                                         kTfLiteUInt4, kTfLiteInt2)));

TEST(SupportQuantizeTest, UnsupportedVersion) {
  TfLiteContext context = {};
  context.tensors_size = 2;
  TfLiteNode node = {};
  TfLiteRegistration registration = {};
  registration.version = 3;

  int inputs[] = {1, 0};
  int outputs[] = {1, 1};
  node.inputs = reinterpret_cast<TfLiteIntArray*>(inputs);
  node.outputs = reinterpret_cast<TfLiteIntArray*>(outputs);

  TfLiteTensor tensors[2] = {};
  tensors[0].type = kTfLiteFloat32;
  tensors[1].type = kTfLiteInt8;
  context.tensors = tensors;

  TfLiteAffineQuantization quant_params = {};
  tensors[1].quantization.params = &quant_params;
  tensors[1].quantization.type = kTfLiteAffineQuantization;

  std::string error;
  EXPECT_FALSE(IsQuantizeSupported(&context, &node, &registration, &error));
  EXPECT_EQ(error, "Unsupported version.");
}

TEST(SupportQuantizeTest, UnsupportedInputType) {
  TfLiteContext context = {};
  context.tensors_size = 2;
  TfLiteNode node = {};
  TfLiteRegistration registration = {};
  registration.version = 1;

  int inputs[] = {1, 0};
  int outputs[] = {1, 1};
  node.inputs = reinterpret_cast<TfLiteIntArray*>(inputs);
  node.outputs = reinterpret_cast<TfLiteIntArray*>(outputs);

  TfLiteTensor tensors[2] = {};
  tensors[0].type = kTfLiteInt32;
  tensors[1].type = kTfLiteInt8;
  context.tensors = tensors;

  TfLiteAffineQuantization quant_params = {};
  tensors[1].quantization.params = &quant_params;
  tensors[1].quantization.type = kTfLiteAffineQuantization;

  std::string error;
  EXPECT_FALSE(IsQuantizeSupported(&context, &node, &registration, &error));
  EXPECT_EQ(error, "Unsupported dtype for input: 2");
}

TEST(SupportQuantizeTest, UnsupportedOutputType) {
  TfLiteContext context = {};
  context.tensors_size = 2;
  TfLiteNode node = {};
  TfLiteRegistration registration = {};
  registration.version = 1;

  int inputs[] = {1, 0};
  int outputs[] = {1, 1};
  node.inputs = reinterpret_cast<TfLiteIntArray*>(inputs);
  node.outputs = reinterpret_cast<TfLiteIntArray*>(outputs);

  TfLiteTensor tensors[2] = {};
  tensors[0].type = kTfLiteFloat32;
  tensors[1].type = kTfLiteFloat32;
  context.tensors = tensors;

  TfLiteAffineQuantization quant_params = {};
  tensors[1].quantization.params = &quant_params;
  tensors[1].quantization.type = kTfLiteAffineQuantization;

  std::string error;
  EXPECT_FALSE(IsQuantizeSupported(&context, &node, &registration, &error));
  EXPECT_EQ(error, "Unsupported dtype for output: 1");
}

TEST(SupportQuantizeTest, MissingQuantParams) {
  TfLiteContext context = {};
  context.tensors_size = 2;
  TfLiteNode node = {};
  TfLiteRegistration registration = {};
  registration.version = 1;

  int inputs[] = {1, 0};
  int outputs[] = {1, 1};
  node.inputs = reinterpret_cast<TfLiteIntArray*>(inputs);
  node.outputs = reinterpret_cast<TfLiteIntArray*>(outputs);

  TfLiteTensor tensors[2] = {};
  tensors[0].type = kTfLiteFloat32;
  tensors[1].type = kTfLiteInt8;
  context.tensors = tensors;

  std::string error;
  EXPECT_FALSE(IsQuantizeSupported(&context, &node, &registration, &error));
  EXPECT_EQ(error, "Encountered Quantize output with no quant params");
}

TEST(SupportQuantizeTest, UnsupportedPerAxisScale) {
  TfLiteContext context = {};
  context.tensors_size = 2;
  TfLiteNode node = {};
  TfLiteRegistration registration = {};
  registration.version = 1;

  int inputs[] = {1, 0};
  int outputs[] = {1, 1};
  node.inputs = reinterpret_cast<TfLiteIntArray*>(inputs);
  node.outputs = reinterpret_cast<TfLiteIntArray*>(outputs);

  TfLiteTensor tensors[2] = {};
  tensors[0].type = kTfLiteFloat32;
  tensors[1].type = kTfLiteInt8;
  context.tensors = tensors;

  // Per-axis (per-channel) scale with size > 1.
  struct {
    int size;
    float data[2];
  } scale = {2, {0.1f, 0.2f}};
  TfLiteAffineQuantization quant_params = {};
  quant_params.scale = reinterpret_cast<TfLiteFloatArray*>(&scale);
  tensors[1].quantization.params = &quant_params;
  tensors[1].quantization.type = kTfLiteAffineQuantization;

  std::string error;
  EXPECT_FALSE(IsQuantizeSupported(&context, &node, &registration, &error));
  EXPECT_EQ(error, "Unsupported quantization scale size");
}

}  // namespace
}  // namespace litert::ml_drift::ir
