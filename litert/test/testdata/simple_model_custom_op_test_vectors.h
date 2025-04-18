// Copyright 2025 Google LLC.
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

#ifndef ODML_LITERT_LITERT_TEST_TESTDATA_SIMPLE_MODEL_CUSTOM_OP_TEST_VECTORS_H_
#define ODML_LITERT_LITERT_TEST_TESTDATA_SIMPLE_MODEL_CUSTOM_OP_TEST_VECTORS_H_

#include <cstddef>
#include <cstdint>

#include "litert/c/litert_model.h"
#include "litert/cc/litert_layout.h"

constexpr const char* kModelFileName = "simple_model_custom_op.tflite";

constexpr const int32_t kTestInput0Dimensions[] = {2};
constexpr const int32_t kTestInput1Dimensions[] = {2};
constexpr const int32_t kTestOutputDimensions[] = {2};

constexpr const int32_t kNumTestInput0Dimensions = 1;
constexpr const int32_t kNumTestInput1Dimensions = 1;
constexpr const int32_t kNumTestOutputDimensions = 1;

constexpr const float kTestInput0Tensor[] = {1, 2};
constexpr const float kTestInput1Tensor[] = {10, 20};
constexpr const float kTestOutputTensor[] = {11, 22};

constexpr const size_t kTestInput0Size = 2;
constexpr const size_t kTestInput1Size = 2;
constexpr const size_t kTestOutputSize = 2;

constexpr const LiteRtRankedTensorType kInput0TensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    ::litert::BuildLayout(kTestInput0Dimensions)};

constexpr const LiteRtRankedTensorType kInput1TensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    ::litert::BuildLayout(kTestInput1Dimensions)};

constexpr const LiteRtRankedTensorType kOutputTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    ::litert::BuildLayout(kTestOutputDimensions)};

#endif  // ODML_LITERT_LITERT_TEST_TESTDATA_SIMPLE_MODEL_CUSTOM_OP_TEST_VECTORS_H_
