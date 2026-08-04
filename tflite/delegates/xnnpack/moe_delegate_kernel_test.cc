/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tflite/delegates/xnnpack/moe_delegate_kernel.h"

#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace tflite {
namespace xnnpack {
namespace {

void DummyReportError(TfLiteContext*, const char*, ...) {}

TEST(MoeExpertsDelegateKernelTest, SupportsInt8WeightType) {
  TfLiteContext context = {};
  context.ReportError = DummyReportError;
  context.tensors_size = 11;
  std::vector<TfLiteTensor> tensors(11);
  context.tensors = tensors.data();

  // Define 10 input tensors for int8 mode + 1 output tensor.
  // indices 0: src (fp32), 1: top_weights (fp32), 2: top_indices (int32)
  tensors[0].type = kTfLiteFloat32;
  tensors[1].type = kTfLiteFloat32;
  tensors[2].type = kTfLiteInt32;
  // indices 3, 5, 7: gate/ff1/linear weights (int8 constant)
  tensors[3].type = kTfLiteInt8;
  tensors[3].allocation_type = kTfLiteMmapRo;
  tensors[5].type = kTfLiteInt8;
  tensors[5].allocation_type = kTfLiteMmapRo;
  tensors[7].type = kTfLiteInt8;
  tensors[7].allocation_type = kTfLiteMmapRo;
  // indices 4, 6, 8: gate/ff1/linear scales (fp32 constant)
  tensors[4].type = kTfLiteFloat32;
  tensors[4].allocation_type = kTfLiteMmapRo;
  tensors[6].type = kTfLiteFloat32;
  tensors[6].allocation_type = kTfLiteMmapRo;
  tensors[8].type = kTfLiteFloat32;
  tensors[8].allocation_type = kTfLiteMmapRo;
  // index 9: per_expert_scale (fp32 constant)
  tensors[9].type = kTfLiteFloat32;
  tensors[9].allocation_type = kTfLiteMmapRo;
  // index 10: output (fp32)
  tensors[10].type = kTfLiteFloat32;

  std::vector<int> inputs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> outputs = {10};
  TfLiteIntArray* input_array = TfLiteIntArrayCreate(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) input_array->data[i] = inputs[i];
  TfLiteIntArray* output_array = TfLiteIntArrayCreate(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    output_array->data[i] = outputs[i];
  }

  TfLiteNode node = {};
  node.inputs = input_array;
  node.outputs = output_array;

  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("num_experts", 2);
    fbb.Int("num_active_experts", 1);
    fbb.Int("model_dim", 4);
    fbb.Int("hidden_dim", 8);
    fbb.String("weight_type", "int8");
  });
  fbb.Finish();
  node.custom_initial_data = fbb.GetBuffer().data();
  node.custom_initial_data_size = fbb.GetBuffer().size();

  TfLiteRegistration registration = {};
  registration.builtin_code = kTfLiteBuiltinCustom;
  registration.custom_name = "moe";

  EXPECT_EQ(
      MoeExpertsDelegateKernel::IsSupported(&context, &node, &registration, 0),
      kTfLiteOk);

  TfLiteIntArrayFree(input_array);
  TfLiteIntArrayFree(output_array);
}

}  // namespace
}  // namespace xnnpack
}  // namespace tflite
