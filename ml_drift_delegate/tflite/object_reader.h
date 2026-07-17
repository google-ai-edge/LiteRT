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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_OBJECT_READER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_OBJECT_READER_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <unordered_map>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/model_builder_helper.h"
#include "ml_drift_delegate/tflite/shared_const_tensor_map.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift {

enum class ReadTensorFlags {
  kNoExtraBytes,
  kExtraBytes,
};

struct SizedLayout {
  ::ml_drift::Layout layout_1d = ::ml_drift::Layout::BHWC;  // Bx1x1x1
  ::ml_drift::Layout layout_2d = ::ml_drift::Layout::BHWC;  // Bx1x1xC
  ::ml_drift::Layout layout_3d = ::ml_drift::Layout::BHWC;  // Bx1xWxC
  ::ml_drift::Layout layout_4d = ::ml_drift::Layout::BHWC;  // BxHxWxC
};

template <typename ShapeT, ::ml_drift::DataType Type>
void TfLiteTensorToTensorCopyData(
    const TfLiteTensor* const tflite_tensor,
    ::ml_drift::Tensor<ShapeT, Type>* tensor, ReadTensorFlags flags,
    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>* scale =
        nullptr,
    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT32>*
        zero_point = nullptr) {
  const int extra_elements = flags == ReadTensorFlags::kExtraBytes
                                 ? XNN_EXTRA_BYTES / sizeof(float)
                                 : 0;
  if (scale != nullptr && (Type == ::ml_drift::DataType::INT8 ||
                           Type == ::ml_drift::DataType::INT4)) {
    ABSL_QCHECK_EQ(tflite_tensor->bytes % SizeOf(Type), 0);
    tensor->data.resize(tflite_tensor->bytes / SizeOf(Type) + extra_elements);
    std::memcpy(tensor->data.data(), tflite_tensor->data.raw_const,
                tflite_tensor->bytes);

    TfLiteAffineQuantization* quant_params =
        static_cast<TfLiteAffineQuantization*>(
            tflite_tensor->quantization.params);
    // TODO: b/378522761 - Support blockwise quantized tensors.
    ABSL_QCHECK_EQ(tflite_tensor->quantization.type, kTfLiteAffineQuantization);
    ABSL_QCHECK_EQ(quant_params->quantized_dimension, 0);
    ABSL_CHECK(quant_params->scale);
    ABSL_CHECK(quant_params->zero_point);
    scale->shape = ::ml_drift::OHWI(quant_params->scale->size, 1, 1, 1);
    zero_point->shape =
        ::ml_drift::OHWI(quant_params->zero_point->size, 1, 1, 1);
    if (quant_params->scale->size > 1) {
      scale->data.resize(quant_params->scale->size + extra_elements);
      std::memcpy(scale->data.data(), quant_params->scale->data,
                  quant_params->scale->size * sizeof(float));
      zero_point->data.resize(quant_params->zero_point->size + extra_elements);
      std::memcpy(zero_point->data.data(), quant_params->zero_point->data,
                  quant_params->zero_point->size * sizeof(int));
    } else {
      scale->data = {tflite_tensor->params.scale};
      zero_point->data = {tflite_tensor->params.zero_point};
    }
  } else {
    tensor->data.resize(::tflite::NumElements(tflite_tensor) + extra_elements);
    CopyData(*tflite_tensor, &tensor->data[0]);
  }

  // Axis and data layout depend on operation this tensor is used in. So,
  // postpone resolutions until operations are parsed.
  SetAllDimensions(tflite_tensor->dims, &tensor->shape);
}

template <typename ShapeT, ::ml_drift::DataType Type>
void TfLiteTensorToTensorZeroCopy(
    const TfLiteTensor* const tflite_tensor,
    ::ml_drift::Tensor<ShapeT, Type>* tensor,
    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>* scale =
        nullptr,
    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT32>*
        zero_point = nullptr) {
  // TODO: b/378522761 - Support other types.
  if constexpr (Type == ::ml_drift::DataType::FLOAT32) {
    tensor->spanned_data = absl::MakeSpan(tflite_tensor->data.f,
                                          ::tflite::NumElements(tflite_tensor));
  } else if constexpr (Type == ::ml_drift::DataType::INT4 ||
                       Type == ::ml_drift::DataType::INT8) {
    ABSL_CHECK(scale);
    ABSL_CHECK(zero_point);
    tensor->spanned_data =
        absl::MakeSpan(tflite_tensor->data.int8, tflite_tensor->bytes);

    TfLiteAffineQuantization* quant_params =
        static_cast<TfLiteAffineQuantization*>(
            tflite_tensor->quantization.params);
    // TODO: b/378522761 - Support blockwise quantized tensors.
    ABSL_CHECK(tflite_tensor->quantization.type == kTfLiteAffineQuantization);
    ABSL_QCHECK_EQ(quant_params->quantized_dimension, 0);
    ABSL_CHECK(quant_params->scale);
    ABSL_CHECK(quant_params->zero_point);
    scale->shape = ::ml_drift::OHWI(quant_params->scale->size, 1, 1, 1);
    zero_point->shape =
        ::ml_drift::OHWI(quant_params->zero_point->size, 1, 1, 1);
    if (quant_params->scale->size > 1) {
      scale->spanned_data =
          absl::MakeSpan(quant_params->scale->data, quant_params->scale->size);
      zero_point->spanned_data = absl::MakeSpan(quant_params->zero_point->data,
                                                quant_params->zero_point->size);
    } else {
      scale->data = {tflite_tensor->params.scale};
      zero_point->data = {tflite_tensor->params.zero_point};
    }
  } else {
    ABSL_LOG(FATAL) << "Unsupported type: " << ToString(Type);
  }
  SetAllDimensions(tflite_tensor->dims, &tensor->shape);
}

// TFLite constant tensors may share the same buffer. This map links a
// tflite::Subgraph constant tensor index to its global buffer id.
// NOLINTNEXTLINE: unordered_map is used in the tflite::Subgraph API.
using TensorIndexToBufferIdMap = std::unordered_map<size_t, size_t>;
// Maps a TFLite tensor index to an external buffer id.
using TensorIndexToExternalBufferIdMap = std::unordered_map<size_t, size_t>;

// If quantized tensors exist in the graph & quant_conversion_map is non-null,
// the mapping between the original tensors (fixed-point) & GPU values (fp) is
// stored in quant_conversion_map.
class ObjectReader {
 public:
  static bool CanReadNonConstantTensor(
      TfLiteContext* context,
      absl::flat_hash_map<int, ::ml_drift::Value*>* tensor_to_value,
      int tensor_idx, bool shared_tensor);

  // MUST check ObjectReader::CanReadNonConstantTensor a priori.
  static ::ml_drift::Value* ReadNonConstantTensor(
      TfLiteContext* context,
      absl::flat_hash_map<int, ::ml_drift::Value*>* tensor_to_value,
      absl::flat_hash_map<int, int>* quant_conversion_map,
      ::ml_drift::GraphFloat32* graph, int tensor_idx,
      bool shared_tensor = false);

  static absl::Status ReadSharedTensor(
      TfLiteTensor* tflite_tensor,
      absl::flat_hash_map<int, ::ml_drift::Value*>* tensor_to_value,
      uint32_t tensor_idx, ::ml_drift::GraphFloat32* graph,
      ::ml_drift::Value** input);

  ObjectReader(
      ::ml_drift::GraphFloat32* graph, TfLiteContext* context,
      const TfLiteNode* node,
      absl::flat_hash_map<int, ::ml_drift::Value*>* tensor_to_value,
      absl::flat_hash_map<int, int>* quant_conversion_map = nullptr,
      const TensorIndexToBufferIdMap* tensor_to_buffer_id_map = nullptr,
      const TensorIndexToExternalBufferIdMap* tensor_to_external_buffer_id_map =
          nullptr,
      SharedConstTensorsMap* shared_tensor_map = nullptr)
      : graph_(graph),
        context_(context),
        node_(node),
        tensor_to_value_(tensor_to_value),
        quant_conversion_map_(quant_conversion_map),
        tensor_to_buffer_id_map_(tensor_to_buffer_id_map),
        tensor_to_external_buffer_id_map_(tensor_to_external_buffer_id_map),
        shared_tensor_map_(shared_tensor_map) {}

  bool CanReadValue(int input_idx) const;

  // MUST check ObjectReader::CanReadValue a priori.
  ::ml_drift::Value* ReadValue(int input_idx);

  bool CanReadValueByTensorIdx(int tensor_idx) const;

  // MUST check ObjectReader::CanReadValueByTensorIdx a priori.
  ::ml_drift::Value* ReadValueByTensorIdx(int tensor_idx);

  void ReadQuantizedValueByTensorIdx(uint32_t tensor_idx,
                                     ::ml_drift::Value** input_int8);

  int GetNumberOfRuntimeInputs() const;

  int GetTensorId(int node_input_index) const;

  absl::Status GetTensorDims(uint32_t idx, TfLiteIntArray* dimensions) const;

  // Checks whether TFLite constant input sharing is possible.
  bool SharingEnabled() const;

  // Allowlists the inputs that can be shared.
  //
  // Not all constant tensor inputs of an op can be shared.
  //
  // WARNING: this MUST be called at the beginning of the op parsing.
  void AllowSharingInput(int input_index);

  struct ConstantInputSharingInfo {
    static constexpr size_t kInvalidId = std::numeric_limits<size_t>::max();
    size_t buffer_id = kInvalidId;
    size_t external_buffer_id = kInvalidId;

    // Returns true if the tensor has a valid buffer id.
    bool HasBufferId() const { return buffer_id != kInvalidId; }
    // Returns true if the tensor has a valid external buffer id.
    bool HasExternalBufferId() const {
      return external_buffer_id != kInvalidId;
    }
    // Returns true if the tensor is shared.
    bool IsShared() const { return HasExternalBufferId() || HasBufferId(); }
    // Returns the preferred buffer id for the tensor.
    size_t PreferredId() const {
      return HasExternalBufferId() ? external_buffer_id : buffer_id;
    }
    // Returns a ConstantInputSharingInfo object that is not shared.
    static constexpr ConstantInputSharingInfo BuildNotShared() {
      return {kInvalidId, kInvalidId};
    }
  };

  // Checks if the TFLite tensor id references a shared constant tensor.
  //
  // Note: this will work even if `SharingEnabled()` returns false.
  ConstantInputSharingInfo GetSharingInfoByTensorIndex(int tensor_id) const;

  // Checks if the TFLite input id references a shared constant tensor.
  //
  // Note: this will work even if `SharingEnabled()` returns false.
  ConstantInputSharingInfo GetSharingInfoByNodeInputIndex(
      int node_input_index) const;

  // Sets the mapping between a graph value id and a shared TFLite tensor.
  //
  // The `tensor` parameter should consist of the `global_id` returned by
  // `GetSharingInfoFor` and the associated tensor. The `dequant_forced`
  // parameter affects the way the tensor is shared: if true, the tensor is
  // first dequantized before sharing. The `layout` parameter is used to
  // enforce the specific layout when creating the spatial tensor.
  //
  // WARNING: this should not be called is `SharingEnabled()` returns false.
  void SetSharedTensor(::ml_drift::ValueId graph_value_id, int global_id,
                       int tflite_tensor_id, bool dequant_forced,
                       std::optional<::ml_drift::Layout> layout);

  inline bool IsNodeInputTensorPresent(int node_input_index) const {
    return node_input_index < node_->inputs->size &&
           node_->inputs->data[node_input_index] >= 0;
  }

  inline bool IsNodeOutputTensorPresent(int node_output_index) const {
    return node_output_index < node_->outputs->size &&
           node_->outputs->data[node_output_index] >= 0;
  }

  bool IsLinearTensor(int index) const {
    const TfLiteTensor* t = GetInputTensor(index);
    return t && IsLinearConvertible(t->dims);
  }

  bool IsConstantTensor(int index) const {
    const TfLiteTensor* t = GetInputTensor(index);
    return t && ::tflite::IsConstantTensor(t);
  }

  // Any failure here will result in a hard crash.
  // If you need a precondition check, use PreReadTensor.
  template <typename TensorT>
  void ReadTensor(
      int node_input_index, TensorT* tensor, ReadTensorFlags flags,
      bool enable_spanned_weights = false,
      ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>*
          scale = nullptr,
      ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT32>*
          zero_point = nullptr) const {
    tensor->id = node_->inputs->data[node_input_index];
    const TfLiteTensor* const tflite_tensor = context_->tensors + tensor->id;
    enable_spanned_weights
        ? TfLiteTensorToTensorZeroCopy(tflite_tensor, tensor, scale, zero_point)
        : TfLiteTensorToTensorCopyData(tflite_tensor, tensor, flags, scale,
                                       zero_point);
  }

  void AddInput(const ::ml_drift::Node* node, int node_input_index);
  void AddUpdate(const ::ml_drift::Node* node, int node_input_index);
  void AddOutput(const ::ml_drift::Node* node, int node_output_index);
  void AddOutputs(const ::ml_drift::Node* node);

  inline TfLiteTensor* GetInputTensor(int index) const {
    return index >= 0 && index < node_->inputs->size
               ? context_->tensors + node_->inputs->data[index]
               : nullptr;
  }

  inline TfLiteTensor* GetOutputTensor(int index) const {
    return index >= 0 && index < node_->outputs->size
               ? context_->tensors + node_->outputs->data[index]
               : nullptr;
  }

  inline absl::Status VerifyInputsConstsOutputs(const TfLiteNode* node,
                                                int runtime_inputs,
                                                int const_inputs, int outputs) {
    return CheckInputsConstsOutputs(context_, node, runtime_inputs,
                                    const_inputs, outputs);
  }

  // Returns a tensor using its absolute `tensor_id` from TfLiteContext's
  // main tensor array.
  //
  // This differs from GetInput/OutputTensor(index), where `index` is a local
  // index into the node's list of inputs/outputs. This function is necessary
  // for TfLiteBlockwiseQuantization that operate on global tensor indices.
  //
  // For better encapsulation, prefer GetInput/OutputTensor().
  inline const TfLiteTensor* GetTensor(int tensor_id) const {
    return context_->tensors + tensor_id;
  }

  ::ml_drift::Value* AddConstInput(int index, const SizedLayout& layout);

 private:
  ::ml_drift::GraphFloat32* graph_;
  TfLiteContext* context_;
  const TfLiteNode* node_;
  absl::flat_hash_map<int, ::ml_drift::Value*>* tensor_to_value_;
  absl::flat_hash_map<int, int>* quant_conversion_map_;
  const TensorIndexToBufferIdMap* tensor_to_buffer_id_map_;
  const TensorIndexToExternalBufferIdMap* tensor_to_external_buffer_id_map_;
  SharedConstTensorsMap* shared_tensor_map_;
  // Stores the tensor index of the tensors that are allowed to be shared.
  absl::flat_hash_set<int> allowed_shared_tensors_;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_OBJECT_READER_H_
