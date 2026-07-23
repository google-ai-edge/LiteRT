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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_OPERATION_PARSER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_OPERATION_PARSER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/object_reader.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/kernels/kernel_util.h"
#include "tflite/tools/versioning/gpu_compatibility.h"
#include "tflite/tools/versioning/op_signature.h"

namespace litert::ml_drift {

// Parses TFLite operation and updates provided GraphFloat32.
class TFLiteOperationParser {
 public:
  virtual ~TFLiteOperationParser() = default;

  // Parses TFLite operation. This method allows expanding fused operations
  // into more than one node.
  virtual void Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     ::ml_drift::GraphFloat32* graph, ObjectReader* reader) = 0;

  // Verifies whether passed tflite node may be built by GPU delegate or not.
  virtual absl::Status IsSupported(const TfLiteContext* context,
                                   const TfLiteNode* tflite_node,
                                   const TfLiteRegistration* registration) = 0;

  // Returns the value IDs in the graph that correspond to the updated values of
  // the variable input tensor.
  virtual absl::flat_hash_map<int, ::ml_drift::ValueId>
  GetNewValueIdsForVariableInputNodes() {
    return {};
  }

 protected:
  struct ParserValidationOptions {
    int max_version = -1;
    int min_inputs = -1;
    int max_inputs = -1;
    int num_outputs = -1;
    int required_runtime_inputs = -1;
    int required_const_inputs = -1;
    bool check_gpu_compatibility = true;
    tflite::GpuCompatibilityFlags gpu_flags =
        tflite::GpuCompatibilityFlags::kStandard;
  };

  absl::Status ValidateSupport(const TfLiteContext* context,
                               const TfLiteNode* tflite_node,
                               const TfLiteRegistration* registration,
                               const ParserValidationOptions& options);
};

// Factory to create StableHLO composite operation parsers.
class TFLiteStablehloCompositeParserFactory {
 public:
  virtual ~TFLiteStablehloCompositeParserFactory() = default;

  // Creates a parser for the given StableHLO composite operation name.
  virtual std::unique_ptr<TFLiteOperationParser> Create(
      std::string_view op_name) = 0;

  // Whether the given StableHLO composite operation supports Int32 type.
  virtual bool SupportsIntegerTypes(std::string_view op_name) = 0;

  // Whether the given StableHLO composite operation supports BOOL type.
  virtual bool SupportsBoolTypes(std::string_view op_name) = 0;
};

absl::Status CheckMaxSupportedOpVersion(const TfLiteRegistration* registration,
                                        int max_version);

template <typename AttrT>
void UpdatePadding(const TfLitePadding& padding,
                   const ::ml_drift::BHWC& input_shape, AttrT* attr) {
  if (padding == kTfLitePaddingSame) {
    attr->padding = CalculateSamePadding(input_shape, *attr);
  } else {
    attr->padding.prepended = ::ml_drift::HW(0, 0);
    attr->padding.appended = ::ml_drift::HW(0, 0);
  }
}

// Helper functions for IsSupported() checkings, these functions are the flow
// of checking errors in functions used in Parse(). To fix the handshake issue
// the checkings are extracted from the functions and keep as helper functions
// here.
//
// The "PreCheckXXX" functions just do checkings the conditions without any
// actual actions.
// The "PreXXX" functions will do actual actions.
// For example, PreCheckReadTensor() will just check the error conditions but
// PreReadTensor() will actually read the data out.
//
// The functions may fail in Parse(), corresponding checkings should be added in
// IsSupported(), here are some generic rules:
// 1. reader->GetInputTensor() reader->GetOutputTensor:
//    Use PreGetInputTensor(), PreGetOutputTensor to check.
// 2. reader->AddInput():
//    Use PreCheckReadValue() to check
// 3. reader->AddOutput(), reader->AddOutputs:
//    Use PreCheckOutput(), PreCheckOutputs()
// 4. reader->ReadTensor():
//    * Use PreCheckReadTensor to check
//    * Use PreReadTensor to read the tensor data
// 5. ExtractTensorShape():
//    * Use PreCheckTensorShape() to check
//    * Use ExtractTensorShape() to get tensor shape
// 6. ExtractAxisFromIndex:
//    Use PreCheckAxisFromIndex to check.
// 7. MaybeFuseActivation():
//    Use PreCheckMaybeFuseActivation() to check.
// 8. graph->FindInput(Node *node)
//    For such scenario, it usually uses the information of type and shape.
//    * For tensor type, just use PreGetOutputTensor()
//    * For tensor shape, use PreGetOutputTensor() and ExtractTensorShape
//      to get tensor shape and type.
//    * For reading output tensor data, use PreCheckReadValueByTensorIdx().
// 9. For dedicated flow, the error checkings must be extracted manually.

absl::Status PreGetInputTensor(const TfLiteContext* context,
                               const TfLiteNode* node, int32_t index,
                               const TfLiteTensor** input);

absl::Status PreGetOutputTensor(const TfLiteContext* context,
                                const TfLiteNode* node, int32_t index,
                                TfLiteTensor** output);

// Checking extracted from ExtractTensorShape
absl::Status CheckTensorShape(const TfLiteIntArray* dims,
                              const char* tensor_name = nullptr);

// CheckAllDimemsions - check the dimension constrains of different shape type
template <typename ShapeT>
absl::Status CheckAllDimensions(const TfLiteIntArray* dimensions);

template <>
absl::Status CheckAllDimensions<::ml_drift::Scalar>(
    const TfLiteIntArray* dimensions);
template <>
absl::Status CheckAllDimensions<::ml_drift::Linear>(
    const TfLiteIntArray* dimensions);
template <>
absl::Status CheckAllDimensions<::ml_drift::HWC>(
    const TfLiteIntArray* dimensions);
template <>
absl::Status CheckAllDimensions<::ml_drift::HW>(
    const TfLiteIntArray* dimensions);
template <>
absl::Status CheckAllDimensions<::ml_drift::OHWI>(
    const TfLiteIntArray* dimensions);
template <>
absl::Status CheckAllDimensions<::ml_drift::BHWC>(
    const TfLiteIntArray* dimensions);

// Checkings extracted from CreateVectorCopyData
template <typename T>
absl::Status PreCheckCopyData(const TfLiteTensor& src, T* dst) {
  if (src.data.raw_const == nullptr) {
    return absl::InvalidArgumentError("src has no data.");
  }
  if (src.bytes % sizeof(T) != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Input data size ", src.bytes,
                     " is not aligned to expected type: ", sizeof(T)));
  }
  if (const int n = ::tflite::NumElements(&src); n * sizeof(T) != src.bytes) {
    return absl::InvalidArgumentError("src has wrong size.");
  }
  switch (src.type) {
    case kTfLiteFloat32:
    case kTfLiteInt32:
    case kTfLiteUInt8:
    case kTfLiteInt64:
    case kTfLiteBool:
    case kTfLiteInt16:
    case kTfLiteInt8:
    case kTfLiteFloat64:
    case kTfLiteUInt64:
    case kTfLiteUInt32:
    case kTfLiteUInt16:
    case kTfLiteInt4:
      return absl::OkStatus();
    case kTfLiteNoType:
      return absl::InvalidArgumentError("src has no type.");
    case kTfLiteString:
      return absl::UnimplementedError("src can't be string.");
    case kTfLiteComplex64:
      return absl::UnimplementedError("src can't be complex64.");
    case kTfLiteFloat16:
      return absl::UnimplementedError("src can't be float16.");
    case kTfLiteComplex128:
      return absl::UnimplementedError("src can't be complex128.");
    case kTfLiteResource:
      return absl::UnimplementedError("src can't be resource.");
    case kTfLiteVariant:
      return absl::UnimplementedError("src can't be variant.");
    default:
      return absl::UnimplementedError(
          absl::StrCat("src type not allowed: ", src.type));
  }
}

// Checkings extracted from CreateVectorCopyData
template <>
absl::Status PreCheckCopyData(const TfLiteTensor& src, float* dst);

// Checkings extracted from TfLiteTensorToTensor
template <typename TensorT>
absl::Status PreCheckTensorToTensor(const TfLiteTensor* const tflite_tensor,
                                    TensorT* tensor) {
  // Since size flag is dropped, larger size is picked for checking.
  tensor->data.resize(::tflite::NumElements(tflite_tensor) +
                      XNN_EXTRA_BYTES / sizeof(float));

  if (tflite_tensor->sparsity) {
    return absl::UnimplementedError("ML Drift doesn't support sparsity.");
  }
  ABSL_RETURN_IF_ERROR(PreCheckCopyData(*tflite_tensor, &tensor->data[0]));
  ABSL_RETURN_IF_ERROR(
      CheckAllDimensions<typename TensorT::ShapeType>(tflite_tensor->dims));
  return absl::OkStatus();
}

absl::Status GetTensorId(const TfLiteContext* context, const TfLiteNode* node,
                         int input_id, int* tensor_id);

// Checkings extracted from ReadTensor
template <typename TensorT>
absl::Status PreCheckReadTensor(const TfLiteContext* context,
                                const TfLiteNode* node, uint32_t index,
                                TensorT* tensor) {
  int32_t tensor_id = 0;
  ABSL_RETURN_IF_ERROR(GetTensorId(context, node, index, &(tensor_id)));
  tensor->id = tensor_id;
  const TfLiteTensor* tflite_tensor = context->tensors + tensor_id;
  return PreCheckTensorToTensor(tflite_tensor, tensor);
}

// Checkings extracted from ReadTensor
template <typename TensorT>
absl::Status PreReadTensor(const TfLiteContext* context, const TfLiteNode* node,
                           uint32_t index, TensorT* tensor,
                           ReadTensorFlags flags) {
  int32_t tensor_id = 0;
  ABSL_RETURN_IF_ERROR(GetTensorId(context, node, index, &(tensor_id)));
  tensor->id = tensor_id;
  const TfLiteTensor* tflite_tensor = context->tensors + tensor_id;
  ABSL_RETURN_IF_ERROR(PreCheckTensorToTensor(tflite_tensor, tensor));
  TfLiteTensorToTensorCopyData(tflite_tensor, tensor, flags);
  return absl::OkStatus();
}

// Tries to read the input tensor at the given index. Sets 'optional_tensor'
// if the index is valid and the tensor is accessible; otherwise, propagates
// errors or leaves 'optional_tensor' unset if out of range.
template <typename TensorT>
absl::Status PreReadTensor(const TfLiteContext* context, const TfLiteNode* node,
                           uint32_t index,
                           std::optional<TensorT>& optional_tensor,
                           ReadTensorFlags flags) {
  if (index < 0 || index >= node->inputs->size) {
    return absl::OkStatus();
  } else {
    TensorT tensor;
    absl::Status status = PreReadTensor(context, node, index, &tensor, flags);
    if (status.ok()) {
      optional_tensor = std::move(tensor);
    }
    return status;
  }
}

// Checkings extracted from ExtractAxisFromIndex
absl::Status PreCheckAxisFromIndex(const TfLiteTensor& tflite_tensor,
                                   int index);

// Checkings extracted from ExtractTensorShape
absl::Status PreCheckTensorShape(const TfLiteTensor& tflite_tensor);

// Checkings extracted from ExtractTfLiteShape
absl::Status PreCheckTfLiteShape(const TfLiteTensor& tflite_tensor);

// Checkings extracted from ReadValue
absl::Status PreCheckReadValue(const TfLiteContext* context,
                               const TfLiteNode* node, uint32_t tensor_idx);

// Checkings extracted from ReadValueByTensorIdx
absl::Status PreCheckReadValueByTensorIdx(const TfLiteContext* context,
                                          uint32_t tensor_idx);

// Checkings extracted from ReadQuantizedValueByTensorIdx
absl::Status PreCheckReadQuantizedValueByTensorIdx(const TfLiteContext* context,
                                                   uint32_t tensor_idx);

absl::Status PreCheckRuntimeOrConstantInput(const TfLiteContext* context,
                                            const TfLiteNode* node,
                                            uint32_t tensor_idx);

// Checkings extracted from AddOutput / AddOutputs
absl::Status PreCheckOutput(const TfLiteContext* context,
                            const TfLiteNode* node, int id);

absl::Status PreCheckOutputs(const TfLiteContext* context,
                             const TfLiteNode* node);

// Checkings extracted from MaybeFuseActivation*
absl::Status PreCheckMaybeFuseActivation(
    const TfLiteNode* node, TfLiteFusedActivation fused_activation);

absl::Status PreCheckMaybeFuseActivationSkipSize(
    const TfLiteNode* node, TfLiteFusedActivation fused_activation);

template <typename ParamsT>
absl::Status PreCheckBuiltinData(const TfLiteNode* tflite_node,
                                 const ParamsT** tf_options) {
  *tf_options = static_cast<const ParamsT*>(tflite_node->builtin_data);
  if (!*tf_options) {
    return absl::InternalError("Unable to retrieve builtin_data.");
  }
  return absl::OkStatus();
}

absl::Status PreCheckMaybeFuseActivationForElementwiseNode(
    ::ml_drift::OperationType operation_type, const TfLiteNode* tflite_node);

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_OPERATION_PARSER_H_
