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

#include "third_party/odml/litert/ml_drift/tflite/support/support_elementwise.h"

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
#include "third_party/odml/litert/ml_drift/tflite/support/support_aux.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

using ::tflite::GetShapeDebugString;
using ::tflite::IsConstantTensor;

namespace litert::ml_drift::ir {
namespace {

bool IsBinaryOpSupported(
    const TfLiteContext* absl_nonnull context,
    const TfLiteNode* absl_nonnull node,
    const TfLiteRegistration* absl_nonnull registration,
    const int supported_max_version,
    const absl::flat_hash_set<TfLiteType>& supported_dtypes,
    std::string* absl_nonnull error) {
  // Check version.
  if (registration->version < 1 ||
      registration->version > supported_max_version) {
    *error = absl::StrCat("Unsupported version: ", registration->version);
    return false;
  }
  // Check number of inputs and outputs.
  if (!CheckInputOutputCounts(*node, /*expected_inputs=*/2,
                              /*expected_outputs=*/1, *error)) {
    return false;
  }

  const TfLiteIntArray* inputs = node->inputs;
  const TfLiteIntArray* outputs = node->outputs;

  // Validate tensor IDs.
  if (!ValidateTensorIds(*context, *inputs, "inputs", *error)) return false;
  if (!ValidateTensorIds(*context, *outputs, "outputs", *error)) return false;

  const int input0_id = inputs->data[0];
  const int input1_id = inputs->data[1];
  const int output_id = outputs->data[0];

  // Check dtype.
  const TfLiteTensor& input0 = context->tensors[input0_id];
  if (!CheckTensorDtype(input0, supported_dtypes, "inputs[0]", *error)) {
    return false;
  }
  const TfLiteTensor& input1 = context->tensors[input1_id];
  if (!CheckTensorDtype(input1, supported_dtypes, "inputs[1]", *error)) {
    return false;
  }
  const TfLiteTensor& output = context->tensors[output_id];
  if (!CheckTensorDtype(output, supported_dtypes, "outputs[0]", *error)) {
    return false;
  }
  // Check dims.
  if (!TfLiteIntArrayEqual(input0.dims, input1.dims) &&
      !IsBroadcastable(input0.dims, input1.dims)) {
    *error = absl::StrCat("input0 and input1 dimension mismatch: ",
                          GetShapeDebugString(input0.dims), ", ",
                          GetShapeDebugString(input1.dims));
    return false;
  }
  if (!TfLiteIntArrayEqual(input0.dims, output.dims) &&
      !TfLiteIntArrayEqual(input1.dims, output.dims) &&
      !IsBroadcastable(input0.dims, output.dims) &&
      !IsBroadcastable(input1.dims, output.dims)) {
    absl::StrAppend(error, "Input and output dimension mismatch: ",
                    GetShapeDebugString(input0.dims), ", ",
                    GetShapeDebugString(input1.dims), ", ",
                    GetShapeDebugString(output.dims));
    return false;
  }
  // Check 2 const inputs.
  if (IsConstantTensor(&input0) && IsConstantTensor(&input1)) {
    absl::StrAppend(error, "Invalid 2 constant inputs: ", input0_id, " & ",
                    input1_id);
    return false;
  }
  // Check fused activation.
  TfLiteFusedActivation activation;
  bool needs_activation_check = true;
  if (registration->builtin_code == kTfLiteBuiltinAdd) {
    const auto* params =
        static_cast<const TfLiteAddParams*>(node->builtin_data);
    if (!params) {
      *error = "Invalid params";
      return false;
    }
    activation = params->activation;
  } else if (registration->builtin_code == kTfLiteBuiltinSub) {
    const auto* params =
        static_cast<const TfLiteSubParams*>(node->builtin_data);
    if (!params) {
      *error = "Invalid params";
      return false;
    }
    activation = params->activation;
  } else if (registration->builtin_code == kTfLiteBuiltinMul) {
    const auto* params =
        static_cast<const TfLiteMulParams*>(node->builtin_data);
    if (!params) {
      *error = "Invalid params";
      return false;
    }
    activation = params->activation;
  } else if (registration->builtin_code == kTfLiteBuiltinDiv) {
    const auto* params =
        static_cast<const TfLiteDivParams*>(node->builtin_data);
    if (!params) {
      *error = "Invalid params";
      return false;
    }
    activation = params->activation;
  } else {
    needs_activation_check = false;
  }

  if (needs_activation_check &&
      !CheckFusedActivation(node, activation).ok()) {
    *error = absl::StrCat("Unsupported fused activation: ", activation);
    return false;
  }
  return true;
}

bool IsUnaryOpSupported(const TfLiteContext* absl_nonnull context,
                        const TfLiteNode* absl_nonnull node,
                        const TfLiteRegistration* absl_nonnull registration,
                        const int supported_max_version,
                        const absl::flat_hash_set<TfLiteType>& supported_dtypes,
                        std::string* absl_nonnull error) {
  // Check version.
  if (registration->version < 1 ||
      registration->version > supported_max_version) {
    *error = absl::StrCat("Unsupported version: ", registration->version);
    return false;
  }
  // Check number of inputs and outputs.
  if (!CheckInputOutputCounts(*node, /*expected_inputs=*/1,
                              /*expected_outputs=*/1, *error)) {
    return false;
  }

  const TfLiteIntArray* inputs = node->inputs;
  const TfLiteIntArray* outputs = node->outputs;

  // Validate tensor IDs.
  if (!ValidateTensorIds(*context, *inputs, "inputs", *error)) return false;
  if (!ValidateTensorIds(*context, *outputs, "outputs", *error)) return false;

  const int input_id = inputs->data[0];
  const int output_id = outputs->data[0];

  // Some unary ops support all types.
  auto op_type = registration->builtin_code;
  bool supports_all_types = op_type == kTfLiteBuiltinCast;
  absl::flat_hash_set<TfLiteType> dtypes_supported = supported_dtypes;
  if (supports_all_types) {
    dtypes_supported.insert(kTfLiteBool);
  }
  // Check dtype.
  const TfLiteTensor& input = context->tensors[input_id];
  if (!CheckTensorDtype(input, dtypes_supported, "inputs[0]", *error)) {
    return false;
  }
  const TfLiteTensor& output = context->tensors[output_id];
  if (!CheckTensorDtype(output, dtypes_supported, "outputs[0]", *error)) {
    return false;
  }
  // Check dims.
  if (!TfLiteIntArrayEqual(input.dims, output.dims)) {
    absl::StrAppend(error, "Input and output dimension mismatch: ",
                    GetShapeDebugString(input.dims), ", ",
                    GetShapeDebugString(output.dims));
    return false;
  }
  // Check const input.
  if (!CheckNotConstant(input, "inputs[0]", *error)) {
    return false;
  }
  return true;
}

}  // namespace

bool IsBinaryArithmeticOpSupported(
    const TfLiteContext* absl_nonnull context,
    const TfLiteNode* absl_nonnull node,
    const TfLiteRegistration* absl_nonnull registration,
    const int supported_max_version, std::string* absl_nonnull error) {
  const absl::flat_hash_set<TfLiteType> supported_dtypes = {
      // clang-format off
      // go/keep-sorted start numeric=yes
      kTfLiteBFloat16,
      kTfLiteFloat16,
      kTfLiteFloat32,
      kTfLiteInt4,
      kTfLiteInt8,
      kTfLiteInt16,
      kTfLiteInt32,
      kTfLiteUInt8,
      kTfLiteUInt16,
      kTfLiteUInt32,
      // go/keep-sorted end
      // clang-format on
  };
  return IsBinaryOpSupported(context, node, registration, supported_max_version,
                             supported_dtypes, error);
}

bool IsBinaryLogicalOpSupported(const TfLiteContext* context,
                                const TfLiteNode* node,
                                const TfLiteRegistration* registration,
                                const int supported_max_version,
                                std::string* absl_nonnull error) {
  const absl::flat_hash_set<TfLiteType> supported_dtypes = {
      // clang-format off
      // go/keep-sorted start numeric=yes
      kTfLiteBFloat16,
      kTfLiteBool,
      kTfLiteFloat16,
      kTfLiteFloat32,
      kTfLiteInt4,
      kTfLiteInt8,
      kTfLiteInt16,
      kTfLiteInt32,
      kTfLiteUInt8,
      kTfLiteUInt16,
      kTfLiteUInt32,
      // go/keep-sorted end
      // clang-format on
  };
  return IsBinaryOpSupported(context, node, registration, supported_max_version,
                             supported_dtypes, error);
}

bool IsUnaryArithmeticOpSupported(
    const TfLiteContext* absl_nonnull context,
    const TfLiteNode* absl_nonnull node,
    const TfLiteRegistration* absl_nonnull registration,
    const int supported_max_version, std::string* absl_nonnull error) {
  const absl::flat_hash_set<TfLiteType> supported_dtypes = {
      // clang-format off
      // go/keep-sorted start numeric=yes
      kTfLiteBFloat16,
      kTfLiteFloat16,
      kTfLiteFloat32,
      kTfLiteInt4,
      kTfLiteInt8,
      kTfLiteInt16,
      kTfLiteInt32,
      kTfLiteUInt8,
      kTfLiteUInt16,
      kTfLiteUInt32,
      // go/keep-sorted end
      // clang-format on
  };
  return IsUnaryOpSupported(context, node, registration, supported_max_version,
                            supported_dtypes, error);
}

bool IsUnaryLogicalOpSupported(
    const TfLiteContext* absl_nonnull context,
    const TfLiteNode* absl_nonnull node,
    const TfLiteRegistration* absl_nonnull registration,
    const int supported_max_version, std::string* absl_nonnull error) {
  const absl::flat_hash_set<TfLiteType> supported_dtypes = {
      // clang-format off
      // go/keep-sorted start numeric=yes
      kTfLiteBFloat16,
      kTfLiteBool,
      kTfLiteFloat16,
      kTfLiteFloat32,
      kTfLiteInt4,
      kTfLiteInt8,
      kTfLiteInt16,
      kTfLiteInt32,
      kTfLiteUInt8,
      kTfLiteUInt16,
      kTfLiteUInt32,
      // go/keep-sorted end
      // clang-format on
  };
  return IsUnaryOpSupported(context, node, registration, supported_max_version,
                            supported_dtypes, error);
}

}  // namespace litert::ml_drift::ir
