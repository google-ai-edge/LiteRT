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
//
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_TEST_TESTDATA_SIMPLE_MODEL_TEST_VECTORS_H_
#define ODML_LITERT_LITERT_TEST_TESTDATA_SIMPLE_MODEL_TEST_VECTORS_H_

#include <cstddef>
#include <cstdint>

#include "litert/c/litert_model.h"
#include "litert/cc/litert_layout.h"

constexpr const char* kModelFileName = "simple_model.tflite";
constexpr const char* kNpuModelFileName = "simple_model_npu.tflite";
constexpr const char* kQualcommModelFileName = "simple_model_qualcomm.bin";
constexpr const char* kQualcommUInt16ModelFileName =
    "simple_add_uint16_qualcomm.bin";
constexpr const char* kGoogleTensorModelFileName =
    "simple_model_google_tensor.bin";
constexpr const char* kMediaTekModelFileName = "simple_model_mtk.bin";
constexpr const char* kOpenvinoModelBlobFileName = "ov_model_add1d_FP32.blob";

constexpr const int32_t kTestInput0Dimensions[] = {2};
constexpr const int32_t kNumTestInput0Dimensions =
    sizeof(kTestInput0Dimensions) / sizeof(kTestInput0Dimensions[0]);
constexpr const int32_t kTestInput1Dimensions[] = {2};
constexpr const int32_t kNumTestInput1Dimensions =
    sizeof(kTestInput1Dimensions) / sizeof(kTestInput1Dimensions[0]);
constexpr const int32_t kTestOutputDimensions[] = {2};
constexpr const int32_t kNumTestOutputDimensions =
    sizeof(kTestOutputDimensions) / sizeof(kTestOutputDimensions[0]);

constexpr const float kTestInput0Tensor[] = {1, 2};
constexpr const float kTestInput1Tensor[] = {10, 20};
constexpr const float kTestOutputTensor[] = {11, 22};

constexpr const float kTestInput0Tensor_2[] = {10, 20};
constexpr const float kTestInput1Tensor_2[] = {100, 200};
constexpr const float kTestOutputTensor_2[] = {110, 220};

constexpr const float kTestInput0Tensor_3[] = {0.5, 0.5};
constexpr const float kTestInput1Tensor_3[] = {0.5, 0.5};
constexpr const float kTestOutputTensor_3[] = {1.0, 1.0};

constexpr const size_t kTestInput0Size =
    sizeof(kTestInput0Tensor) / sizeof(kTestInput0Tensor[0]);
constexpr const size_t kTestInput1Size =
    sizeof(kTestInput1Tensor) / sizeof(kTestInput1Tensor[0]);
constexpr const size_t kTestOutputSize =
    sizeof(kTestOutputTensor) / sizeof(kTestOutputTensor[0]);

constexpr const LiteRtRankedTensorType kInput0TensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    ::litert::BuildLayout(kTestInput0Dimensions)};

constexpr const LiteRtRankedTensorType kInput1TensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    ::litert::BuildLayout(kTestInput1Dimensions)};

constexpr const LiteRtRankedTensorType kOutputTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    ::litert::BuildLayout(kTestOutputDimensions)};

constexpr const LiteRtRankedTensorType kInput0TensorType_3 = {
    /*.element_type=*/kLiteRtElementTypeInt16,
    ::litert::BuildLayout(kTestInput0Dimensions)};

constexpr const LiteRtRankedTensorType kInput1TensorType_3 = {
    /*.element_type=*/kLiteRtElementTypeInt16,
    ::litert::BuildLayout(kTestInput1Dimensions)};

constexpr const LiteRtRankedTensorType kOutputTensorType_3 = {
    /*.element_type=*/kLiteRtElementTypeInt16,
    ::litert::BuildLayout(kTestOutputDimensions)};

// Mock symmetric quant params for qint16 model.
constexpr float kInputScale = 3.05159e-5;
constexpr std::int16_t kInputZeroPoint = 0;
constexpr float kOutputScale = 5.99496e-5;
constexpr std::int16_t kOutputZeroPoint = 0;

#endif  // ODML_LITERT_LITERT_TEST_TESTDATA_SIMPLE_MODEL_TEST_VECTORS_H_
