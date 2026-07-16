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

#include "ml_drift_delegate/tflite/support/support_aux.h"

#include <string>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/status.h"  // from @ml_drift
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

bool CheckInputOutputCounts(const TfLiteNode& node, int expected_inputs,
                            int expected_outputs, std::string& error) {
  if (node.inputs->size != expected_inputs) {
    error = absl::StrCat("Invalid number of inputs: ", node.inputs->size,
                         ", while expected ", expected_inputs, ".");
    return false;
  }
  if (node.outputs->size != expected_outputs) {
    error = absl::StrCat("Invalid number of outputs: ", node.outputs->size,
                         ", while expected ", expected_outputs, ".");
    return false;
  }
  return true;
}

bool CheckIsConstant(const TfLiteTensor& tensor, const std::string& tensor_name,
                     std::string& error) {
  if (!::tflite::IsConstantTensor(&tensor)) {
    error = absl::StrCat(tensor_name, " should be a constant tensor.");
    return false;
  }
  return true;
}

bool CheckNotConstant(const TfLiteTensor& tensor,
                      const std::string& tensor_name, std::string& error) {
  if (::tflite::IsConstantTensor(&tensor)) {
    error = absl::StrCat(tensor_name, " should not be a constant tensor.");
    return false;
  }
  return true;
}

bool CheckTensorDims(const TfLiteTensor& tensor, int min_dims, int max_dims,
                     const std::string& tensor_name, std::string& error) {
  if (!tensor.dims) {
    error = absl::StrCat(tensor_name, " has null dims.");
    return false;
  }
  if (tensor.dims->size < min_dims || tensor.dims->size > max_dims) {
    error = absl::StrCat("Invalid number of dimensions for ", tensor_name, ": ",
                         tensor.dims->size, ", while expected to be in [",
                         min_dims, ", ", max_dims, "].");
    return false;
  }
  return true;
}

bool CheckTensorDtype(const TfLiteTensor& tensor,
                      const absl::flat_hash_set<TfLiteType>& supported_dtypes,
                      const std::string& tensor_name, std::string& error) {
  if (supported_dtypes.find(tensor.type) == supported_dtypes.end()) {
    error =
        absl::StrCat("Unsupported dtype for ", tensor_name, ": ", tensor.type);
    return false;
  }
  return true;
}

absl::Status CheckFusedActivation(const TfLiteNode* node,
                                  TfLiteFusedActivation activation) {
  if (node->outputs->size != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Node ", node->outputs->size, " has ", node->outputs->size,
                     " outputs, but should have 1."));
  }
  return CheckFusedActivationSkipSize(activation);
}

absl::Status CheckFusedActivationSkipSize(TfLiteFusedActivation activation) {
  switch (activation) {
    case kTfLiteActNone:
    case kTfLiteActRelu:
    case kTfLiteActReluN1To1:
    case kTfLiteActRelu6:
    case kTfLiteActTanh:
    case kTfLiteActSigmoid:
    case kTfLiteActSignBit:
      return absl::OkStatus();
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unsupported fused activation: ", activation));
  }
}

bool ValidateTensorId(const TfLiteContext& context, int tensor_id,
                      const std::string& tensor_name, std::string& error) {
  if (tensor_id < 0 || tensor_id >= context.tensors_size) {
    error =
        absl::StrCat("Invalid tensor ID for ", tensor_name, ": ", tensor_id,
                     ", which should be in [0, ", context.tensors_size, ")");
    return false;
  }
  return true;
}

bool ValidateTensorIds(const TfLiteContext& context,
                       const TfLiteIntArray& tensor_ids,
                       const std::string& array_name, std::string& error) {
  for (int i = 0; i < tensor_ids.size; ++i) {
    if (!ValidateTensorId(context, tensor_ids.data[i],
                          absl::StrCat(array_name, "[", i, "]"), error)) {
      return false;
    }
  }
  return true;
}

}  // namespace litert::ml_drift::ir
