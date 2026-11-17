// Copyright 2026 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_QSA_PARSER_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_QSA_PARSER_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift {

inline bool IsQsaFloatType(TfLiteType type) {
  return type == kTfLiteFloat32 || type == kTfLiteFloat16;
}

// Computes the positive element count of `tensor` (`1` for rank-0 scalars),
// returning -1 if `dims` is null, has negative rank, contains non-positive
// dimensions, or overflows `int32_t`.
inline int64_t NumQsaTensorElements(const TfLiteTensor& tensor) {
  if (tensor.dims == nullptr || tensor.dims->size < 0) {
    return -1;
  }
  int64_t count = 1;
  const int64_t max_elems = std::numeric_limits<int32_t>::max();
  for (int i = 0; i < tensor.dims->size; ++i) {
    const int dim = tensor.dims->data[i];
    if (dim <= 0 || count > max_elems / dim) {
      return -1;
    }
    count *= dim;
  }
  return count;
}

inline absl::Status CheckQsaTensorRank(const TfLiteTensor& tensor,
                                       int expected_rank,
                                       absl::string_view op_name,
                                       absl::string_view tensor_name) {
  if (tensor.dims == nullptr || tensor.dims->size != expected_rank) {
    return absl::UnimplementedError(absl::StrCat(
        op_name, " expects rank-", expected_rank, " ", tensor_name, "."));
  }
  if (NumQsaTensorElements(tensor) <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        op_name, " has invalid dimensions in ", tensor_name, "."));
  }
  return absl::OkStatus();
}

// Validates a scalar or unit-shaped 1D..4D tensor (`{}`, `{1}`, `{1, 1}`, ...).
inline absl::Status CheckQsaScalarElementTensor(const TfLiteTensor& tensor,
                                                absl::string_view op_name,
                                                absl::string_view tensor_name) {
  if (tensor.dims == nullptr || tensor.dims->size < 0 ||
      tensor.dims->size > 4) {
    return absl::UnimplementedError(absl::StrCat(
        op_name, " expects rank 0..4 for ", tensor_name, "."));
  }
  for (int i = 0; i < tensor.dims->size; ++i) {
    if (tensor.dims->data[i] != 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          op_name, " expects unit dimensions in ", tensor_name, "."));
    }
  }
  return absl::OkStatus();
}

// Verifies and extracts the FlexBuffers attribute map from a StableHLO
// composite or custom op node, returning `std::nullopt` when attributes are
// absent and `InvalidArgumentError` when an attribute buffer is malformed or
// does not have a map root.
inline absl::StatusOr<std::optional<flexbuffers::Map>>
GetQsaCompositeAttrMap(const TfLiteNode& tflite_node) {
  const uint8_t* buffer_t = nullptr;
  size_t length = 0;
  if (tflite_node.custom_initial_data != nullptr &&
      tflite_node.custom_initial_data_size != 0) {
    if (tflite_node.custom_initial_data_size < 0) {
      return absl::InvalidArgumentError(
          "Negative custom_initial_data_size in QSA composite.");
    }
    buffer_t =
        reinterpret_cast<const uint8_t*>(tflite_node.custom_initial_data);
    length = static_cast<size_t>(tflite_node.custom_initial_data_size);
  } else if (tflite_node.builtin_data != nullptr) {
    const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
        tflite_node.builtin_data);
    if (params != nullptr && params->attributes != nullptr &&
        params->attributes_size > 0) {
      buffer_t = reinterpret_cast<const uint8_t*>(params->attributes);
      length = params->attributes_size;
    }
  }
  if (buffer_t == nullptr || length == 0) {
    return std::nullopt;
  }
  constexpr size_t kMaxAttrBufferBytes = 1 << 20;
  if (length > kMaxAttrBufferBytes ||
      !flexbuffers::VerifyBuffer(buffer_t, length)) {
    return absl::InvalidArgumentError(
        "Malformed FlexBuffers attribute buffer in QSA composite.");
  }
  const flexbuffers::Reference root = flexbuffers::GetRoot(buffer_t, length);
  if (!root.IsMap()) {
    return absl::InvalidArgumentError(
        "Expected FlexBuffers map root for QSA composite attributes.");
  }
  return root.AsMap();
}

// Validates `context`, `tflite_node`, `input_idx` bounds, `tensor_idx` bounds,
// and non-null `dims` before `PreCheckReadValue` or constant checks.
inline absl::StatusOr<const TfLiteTensor*> GetValidatedQsaInputTensor(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    int input_idx) {
  if (context == nullptr || context->tensors == nullptr ||
      tflite_node == nullptr || tflite_node->inputs == nullptr) {
    return absl::InvalidArgumentError("Null TFLite context or node.");
  }
  if (input_idx < 0 || input_idx >= tflite_node->inputs->size) {
    return absl::InvalidArgumentError("Input index out of range.");
  }
  const int tensor_idx = tflite_node->inputs->data[input_idx];
  if (tensor_idx < 0) {
    return absl::UnavailableError("Optional input tensor is not supported.");
  }
  if (tensor_idx >= context->tensors_size) {
    return absl::InvalidArgumentError("Input tensor index out of bounds.");
  }
  const TfLiteTensor& t = context->tensors[tensor_idx];
  if (t.dims == nullptr) {
    return absl::InvalidArgumentError("Input tensor has null dims.");
  }
  return &t;
}

inline absl::StatusOr<size_t> GetQsaConstantElementSize(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return sizeof(float);
    case kTfLiteFloat16:
      return sizeof(uint16_t);
    case kTfLiteInt32:
      return sizeof(int32_t);
    case kTfLiteBool:
      return sizeof(bool);
    default:
      return absl::UnimplementedError(
          "Unsupported constant tensor type in QSA composite.");
  }
}

// Checks an input that must be a non-constant runtime tensor.
inline absl::Status PreCheckQsaRuntimeOnlyInput(const TfLiteContext* context,
                                                const TfLiteNode* tflite_node,
                                                int input_idx) {
  ABSL_ASSIGN_OR_RETURN(
      const TfLiteTensor* t,
      GetValidatedQsaInputTensor(context, tflite_node, input_idx));
  if (tflite::IsConstantTensor(t)) {
    return absl::UnimplementedError(
        "Constant tensor is not supported on runtime-only QSA input.");
  }
  return PreCheckReadValue(context, tflite_node, input_idx);
}

// Checks an input that may be either a runtime tensor or a constant tensor.
inline absl::Status PreCheckQsaRuntimeOrConstantInput(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    int input_idx) {
  ABSL_ASSIGN_OR_RETURN(
      const TfLiteTensor* t,
      GetValidatedQsaInputTensor(context, tflite_node, input_idx));
  if (tflite::IsConstantTensor(t)) {
    if (t->data.raw_const == nullptr) {
      return absl::InvalidArgumentError("Constant tensor has null data.");
    }
    if (t->sparsity != nullptr) {
      return absl::UnimplementedError(
          "Sparse constant tensor is not supported in QSA composite.");
    }
    if (t->dims->size < 0 || t->dims->size > 4) {
      return absl::InvalidArgumentError(
          "Invalid constant tensor rank in QSA composite.");
    }
    ABSL_RETURN_IF_ERROR(CheckTensorShape(t->dims, t->name));
    const int64_t num_elems = NumQsaTensorElements(*t);
    if (num_elems <= 0) {
      return absl::InvalidArgumentError(
          "Invalid constant tensor dimensions in QSA composite.");
    }
    ABSL_ASSIGN_OR_RETURN(const size_t elem_size,
                          GetQsaConstantElementSize(t->type));
    // Compare in `uint64_t` so the product cannot wrap on 32-bit `size_t`.
    if (static_cast<uint64_t>(t->bytes) !=
        static_cast<uint64_t>(num_elems) * elem_size) {
      return absl::InvalidArgumentError(
          "Constant tensor byte size does not match shape and type.");
    }
    return absl::OkStatus();
  }
  return PreCheckReadValue(context, tflite_node, input_idx);
}

// Validates output tensor indices and non-null `dims` before
// `PreCheckOutputs`, which dereferences `dims` unconditionally.
inline absl::Status PreCheckQsaOutputs(const TfLiteContext* context,
                                       const TfLiteNode* tflite_node) {
  if (context == nullptr || context->tensors == nullptr ||
      tflite_node == nullptr || tflite_node->outputs == nullptr) {
    return absl::InvalidArgumentError("Null TFLite context or node.");
  }
  for (int i = 0; i < tflite_node->outputs->size; ++i) {
    const int tensor_idx = tflite_node->outputs->data[i];
    if (tensor_idx < 0 || tensor_idx >= context->tensors_size) {
      return absl::InvalidArgumentError("Output tensor index out of bounds.");
    }
    if (context->tensors[tensor_idx].dims == nullptr) {
      return absl::InvalidArgumentError("Output tensor has null dims.");
    }
  }
  return PreCheckOutputs(context, tflite_node);
}

inline void AddQsaRuntimeOrConstInputs(const TfLiteNode& tflite_node,
                                       ::ml_drift::Node* node,
                                       ::ml_drift::GraphFloat32* graph,
                                       ObjectReader* reader,
                                       absl::Span<const int> skip_inputs = {}) {
  for (int i = 0; i < tflite_node.inputs->size; ++i) {
    if (absl::c_linear_search(skip_inputs, i)) {
      continue;
    }
    if (reader->IsConstantTensor(i)) {
      const ::ml_drift::Value* input = reader->AddConstInput(i, /*layout=*/{});
      if (input != nullptr) {
        graph->AddConsumer(node->id, input->id);
      }
    } else {
      reader->AddInput(node, i);
    }
  }
}

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_QSA_PARSER_UTILS_H_
