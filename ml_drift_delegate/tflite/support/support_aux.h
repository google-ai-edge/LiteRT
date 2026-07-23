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

// This file provides auxiliary functions for the support functions between
// TFLite and MLDrift IR.

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_SUPPORT_SUPPORT_AUX_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_SUPPORT_SUPPORT_AUX_H_

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

// Checks if the number of inputs and outputs matches the expected counts.
bool CheckInputOutputCounts(const TfLiteNode& node, int expected_inputs,
                            int expected_outputs, std::string& error);

// Checks if a tensor is constant.
bool CheckIsConstant(const TfLiteTensor& tensor, const std::string& tensor_name,
                     std::string& error);

// Checks if a tensor is not constant.
bool CheckNotConstant(const TfLiteTensor& tensor,
                      const std::string& tensor_name, std::string& error);

// Checks if the number of dimensions is within the allowed range.
bool CheckTensorDims(const TfLiteTensor& tensor, int min_dims, int max_dims,
                     const std::string& tensor_name, std::string& error);

// Checks if the dtype of a tensor is in the supported set.
bool CheckTensorDtype(const TfLiteTensor& tensor,
                      const absl::flat_hash_set<TfLiteType>& supported_dtypes,
                      const std::string& tensor_name, std::string& error);

// Checks if a TfLiteFusedActivation is supported.
absl::Status CheckFusedActivation(const TfLiteNode* node,
                                  TfLiteFusedActivation activation);

// Checks if a TfLiteFusedActivation is supported, without checking the output
// count.
absl::Status CheckFusedActivationSkipSize(TfLiteFusedActivation activation);


// Validates a single tensor ID.
bool ValidateTensorId(const TfLiteContext& context, int tensor_id,
                      const std::string& tensor_name, std::string& error);

// Validates all tensor IDs in an array.
bool ValidateTensorIds(const TfLiteContext& context,
                       const TfLiteIntArray& tensor_ids,
                       const std::string& array_name, std::string& error);

// Checks for any possible fatal paths in PopulateTensor.
template <typename ShapeT, ::ml_drift::DataType Type>
inline absl::Status CheckPopulateTensor(
    const TfLiteTensor* absl_nonnull tflite_tensor,
    bool enable_spanned_weights = false) {
  if constexpr (Type == ::ml_drift::DataType::INT2 ||
                Type == ::ml_drift::DataType::INT4 ||
                Type == ::ml_drift::DataType::INT8) {
    if (tflite_tensor->dims->size != 2 && tflite_tensor->dims->size != 4 &&
        tflite_tensor->dims->size != 5) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Expected 2D, 4D, or 5D quantized tensor: ", tflite_tensor->name));
    }
    if (!enable_spanned_weights && tflite_tensor->bytes % SizeOf(Type) != 0) {
      return absl::InvalidArgumentError(
          "tflite_tensor->bytes must be divisible by SizeOf(Type).");
    }
    if (tflite_tensor->quantization.type != kTfLiteAffineQuantization) {
      return absl::InvalidArgumentError(
          "tflite_tensor->quantization.type must be kTfLiteAffineQuantization "
          "for quantized tensors.");
    }
    const auto* quant_params = static_cast<TfLiteAffineQuantization*>(
        tflite_tensor->quantization.params);
    if (!quant_params) {
      return absl::InvalidArgumentError("Missing quantization.params.");
    }
    if (quant_params->quantized_dimension != 0) {
      return absl::InvalidArgumentError(
          "quant_params->quantized_dimension must be 0 for quantized "
          "tensors.");
    }
    if (!quant_params->scale) {
      return absl::InvalidArgumentError(
          "quant_params->scale must not be null for quantized tensors.");
    }
    if (!quant_params->zero_point) {
      return absl::InvalidArgumentError(
          "quant_params->zero_point must not be null for quantized tensors.");
    }
  } else if constexpr (Type != ::ml_drift::DataType::FLOAT32) {
    ABSL_RETURN_IF_ERROR(CheckAllDimensions<ShapeT>(tflite_tensor->dims));
    if (enable_spanned_weights) {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported type for zero-copy: ", ToString(Type)));
    }
  }
  return absl::OkStatus();
}

}  // namespace litert::ml_drift::ir

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_SUPPORT_SUPPORT_AUX_H_
