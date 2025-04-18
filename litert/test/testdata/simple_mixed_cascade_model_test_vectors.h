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

#ifndef ODML_LITERT_LITERT_TEST_TESTDATA_SIMPLE_MIXED_CASCADE_MODEL_TEST_VECTORS_H_
#define ODML_LITERT_LITERT_TEST_TESTDATA_SIMPLE_MIXED_CASCADE_MODEL_TEST_VECTORS_H_

#include <cstddef>
#include <cstdint>

#include "litert/c/litert_model.h"
#include "litert/cc/litert_layout.h"

constexpr const char* kModelFileName = "simple_mixed_cascade_model_npu.tflite";
constexpr const char* kQualcommNpuBytecodeFileName =
    "simple_model_qualcomm.bin";
constexpr const char* kGoogleTensorNpuBytecodeFileName =
    "simple_model_google_tensor.bin";
constexpr const char* kMediaTekNpuBytecodeFileName = "simple_model_mtk.bin";

constexpr const int32_t kTestInput0Dimensions[] = {2};
constexpr const int32_t kTestInput1Dimensions[] = {2};
constexpr const int32_t kTestInput2Dimensions[] = {2};
constexpr const int32_t kTestInput3Dimensions[] = {2};
constexpr const int32_t kTestInput4Dimensions[] = {2};
constexpr const int32_t kTestOutputDimensions[] = {2};

constexpr const int32_t kNumTestInput0Dimensions = 1;
constexpr const int32_t kNumTestInput1Dimensions = 1;
constexpr const int32_t kNumTestInput2Dimensions = 1;
constexpr const int32_t kNumTestInput3Dimensions = 1;
constexpr const int32_t kNumTestInput4Dimensions = 1;
constexpr const int32_t kNumTestOutputDimensions = 1;

constexpr const float kTestInput0Tensor_1[] = {1, 2};
constexpr const float kTestInput1Tensor_1[] = {10, 20};
constexpr const float kTestInput2Tensor_1[] = {100, 200};
constexpr const float kTestInput3Tensor_1[] = {10, 20};
constexpr const float kTestInput4Tensor_1[] = {100, 200};
constexpr const float kTestOutputTensor_1[] = {221, 442};

constexpr const float kTestInput0Tensor_2[] = {2, 1};
constexpr const float kTestInput1Tensor_2[] = {20, 10};
constexpr const float kTestInput2Tensor_2[] = {200, 100};
constexpr const float kTestInput3Tensor_2[] = {20, 10};
constexpr const float kTestInput4Tensor_2[] = {200, 100};
constexpr const float kTestOutputTensor_2[] = {442, 221};

constexpr const float kTestInput0Tensor_3[] = {1, 4};
constexpr const float kTestInput1Tensor_3[] = {10, 40};
constexpr const float kTestInput2Tensor_3[] = {100, 400};
constexpr const float kTestInput3Tensor_3[] = {1, 40};
constexpr const float kTestInput4Tensor_3[] = {10, 400};
constexpr const float kTestOutputTensor_3[] = {122, 884};

constexpr const size_t kTestInput0Size = 2;
constexpr const size_t kTestInput1Size = 2;
constexpr const size_t kTestInput2Size = 2;
constexpr const size_t kTestInput3Size = 2;
constexpr const size_t kTestInput4Size = 2;
constexpr const size_t kTestOutputSize = 2;

constexpr const LiteRtRankedTensorType kInput0TensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    ::litert::BuildLayout(kTestInput0Dimensions)};

constexpr const LiteRtRankedTensorType kInput1TensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    ::litert::BuildLayout(kTestInput1Dimensions)};

constexpr const LiteRtRankedTensorType kInput2TensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    ::litert::BuildLayout(kTestInput2Dimensions)};

constexpr const LiteRtRankedTensorType kOutputTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    ::litert::BuildLayout(kTestOutputDimensions)};

#endif  // ODML_LITERT_LITERT_TEST_TESTDATA_SIMPLE_MIXED_CASCADE_MODEL_TEST_VECTORS_H_
