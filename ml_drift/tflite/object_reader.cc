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

#include "third_party/odml/litert/ml_drift/tflite/object_reader.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/model_builder_helper.h"
#include "third_party/odml/litert/ml_drift/tflite/shared_const_tensor_map.h"
#include "tflite/c/common.h"
#include "tflite/delegates/utils.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift {
namespace {

::ml_drift::BHWC GetShape(const ::ml_drift::BHWC& shape,
                          const SizedLayout& layout, int num_dims) {
  if ((num_dims == 0 || num_dims == 1) &&
      layout.layout_1d == ::ml_drift::Layout::SCALAR) {
    return ::ml_drift::BHWC(1, 1, 1, shape.b);
  } else if (num_dims == 2 && layout.layout_2d == ::ml_drift::Layout::HW) {
    return ::ml_drift::BHWC(1, 1, shape.b, shape.c);
  } else if (num_dims == 3 && layout.layout_3d == ::ml_drift::Layout::HWC) {
    return ::ml_drift::BHWC(1, shape.b, shape.w, shape.c);
  } else {
    return shape;
  }
}

// Helper to copy tensor data and set value/attribute fields.
template <typename TensorType>
void SetValueAndAttrFromTfLiteTensor(const TfLiteTensor* tfl_tensor,
                                     const SizedLayout& layout,
                                     ::ml_drift::Value* value,
                                     ::ml_drift::ConstTensorAttributes& attr) {
  TensorType t;
  TfLiteTensorToTensorCopyData(tfl_tensor, &t, ReadTensorFlags::kNoExtraBytes);
  value->tensor.type = t.kType;
  value->tensor.shape = GetShape(t.shape, layout, tfl_tensor->dims->size);
  attr.tensor = std::move(t);
}

}  // namespace

::ml_drift::Value* ObjectReader::AddConstInput(int index,
                                               const SizedLayout& layout) {
  const TfLiteTensor* tfl_tensor = GetInputTensor(index);
  ABSL_CHECK(
      tfl_tensor &&
      (tfl_tensor->type == kTfLiteFloat32 ||
       tfl_tensor->type == kTfLiteFloat16 || tfl_tensor->type == kTfLiteInt8 ||
       tfl_tensor->type == kTfLiteUInt8 || tfl_tensor->type == kTfLiteInt4 ||
       tfl_tensor->type == kTfLiteInt2 || tfl_tensor->type == kTfLiteBool ||
       tfl_tensor->type == kTfLiteInt32));
  ::ml_drift::Node* node = graph_->NewNode();
  node->operation.type = ToString(::ml_drift::OperationType::CONSTANT);
  ::ml_drift::Value* value = graph_->NewValue();
  graph_->SetProducer(node->id, value->id);
  ::ml_drift::ConstTensorAttributes attr;
  value->tensor.ref = -1;
  if (tfl_tensor->type == kTfLiteFloat16) {
    SetValueAndAttrFromTfLiteTensor<::ml_drift::TensorFloat16>(
        tfl_tensor, layout, value, attr);
  } else if (tfl_tensor->type == kTfLiteFloat32 ||
             tfl_tensor->type == kTfLiteInt8 ||
             tfl_tensor->type == kTfLiteUInt8 ||
             tfl_tensor->type == kTfLiteInt4 ||
             tfl_tensor->type == kTfLiteInt2) {
    // Note: kTfLiteInt8, kTfLiteUInt8, kTfLiteInt4, and kTfLiteInt2 are read as
    // TensorFloat32.
    SetValueAndAttrFromTfLiteTensor<::ml_drift::TensorFloat32>(
        tfl_tensor, layout, value, attr);
  } else if (tfl_tensor->type == kTfLiteBool) {
    SetValueAndAttrFromTfLiteTensor<::ml_drift::TensorBool>(tfl_tensor, layout,
                                                            value, attr);
  } else if (tfl_tensor->type == kTfLiteInt32) {
    SetValueAndAttrFromTfLiteTensor<::ml_drift::TensorInt32>(tfl_tensor, layout,
                                                             value, attr);
  }
  node->operation.attributes = std::move(attr);
  return value;
}

// static
bool ObjectReader::CanReadNonConstantTensor(
    TfLiteContext* context,
    absl::flat_hash_map<int, ::ml_drift::Value*>* tensor_to_value,
    int tensor_id, bool shared_tensor) {
  if (tensor_id < 0 || tensor_id >= context->tensors_size) return false;
  if (tensor_to_value->find(tensor_id) == tensor_to_value->end()) {
    TfLiteTensor* tflite_tensor = context->tensors + tensor_id;
    if (!shared_tensor && tflite::IsConstantTensor(tflite_tensor)) {
      return false;
    }
  }
  return true;
}

// static
::ml_drift::Value* ObjectReader::ReadNonConstantTensor(
    TfLiteContext* context,
    absl::flat_hash_map<int, ::ml_drift::Value*>* tensor_to_value,
    absl::flat_hash_map<int, int>* quant_conversion_map,
    ::ml_drift::GraphFloat32* graph, int tensor_idx, bool shared_tensor) {
  if (tensor_to_value->find(tensor_idx) == tensor_to_value->end()) {
    TfLiteTensor* tflite_tensor = &context->tensors[tensor_idx];
    if (!shared_tensor &&
        (tflite_tensor->type == kTfLiteInt8 ||
         tflite_tensor->type == kTfLiteUInt8) &&
        quant_conversion_map &&
        tflite_tensor->quantization.type ==
            TfLiteQuantizationType::kTfLiteAffineQuantization) {
      // Quantized case
      if (quant_conversion_map->find(tensor_idx) ==
          quant_conversion_map->end()) {
        // Since the original tensor is fixed-point, add a new float tensor to
        // the TFLite graph to represent the dequantized data.
        int fp_tensor_index = 0;
        TfLiteTensor* fp_tflite_tensor;
        ABSL_CHECK_EQ(tflite::delegates::CreateNewTensorWithDifferentType(
                          context, tensor_idx, kTfLiteFloat32,
                          &fp_tflite_tensor, &fp_tensor_index),
                      kTfLiteOk);
        // `tflite_tensor` value could be invalid when the `context->tensors`
        // is reallocated. Thus reassigning `tflite_tensor` with a fresh value.
        tflite_tensor = &context->tensors[tensor_idx];

        // Remember this tensor for later.
        (*quant_conversion_map)[fp_tensor_index] = tensor_idx;
        (*quant_conversion_map)[tensor_idx] = fp_tensor_index;
        // Add a new GPU Value for the new dequantized floating-point tensor.
        ::ml_drift::Value* value = graph->NewValue();
        value->tensor = ::ml_drift::TensorRef<::ml_drift::BHWC>{
            /*type=*/ToDataType(fp_tflite_tensor->type),
            /*shape=*/ExtractTensorShape(fp_tflite_tensor),
        };
        value->tensor.ref = fp_tensor_index;
        value->tensor.is_variable_input = tflite_tensor->is_variable;
        value->quant_params.emplace();
        PopulateQuantParams(*tflite_tensor, &value->quant_params.value());
        (*tensor_to_value)[fp_tensor_index] = value;
      }
      // We do not use the original tensor index as reference for the GPU
      // Value, instead pointing at the corresponding float version.
      tensor_idx = quant_conversion_map->at(tensor_idx);
    } else {
      // Floating-point case.
      ::ml_drift::Value* value = graph->NewValue();
      value->tensor = ::ml_drift::TensorRef<::ml_drift::BHWC>{
          /*type=*/ToDataType(tflite_tensor->type),
          /*shape=*/ExtractTensorShape(tflite_tensor),
      };
      value->tensor.ref = tensor_idx;
      value->tensor.is_variable_input = tflite_tensor->is_variable;
      (*tensor_to_value)[tensor_idx] = value;
    }
  }
  return (*tensor_to_value)[tensor_idx];
}

absl::Status ObjectReader::ReadSharedTensor(
    TfLiteTensor* tflite_tensor,
    absl::flat_hash_map<int, ::ml_drift::Value*>* tensor_to_value,
    uint32_t tensor_idx, ::ml_drift::GraphFloat32* graph,
    ::ml_drift::Value** input) {
  int batch = 0;
  int channel = 0;
  if (tflite_tensor->dims->size == 2) {
    batch = tflite_tensor->dims->data[0];
    channel = tflite_tensor->dims->data[1];
  } else if (tflite_tensor->dims->size == 4) {
    ABSL_CHECK(tflite_tensor->dims->data[1] == 1 &&
               tflite_tensor->dims->data[2] == 1)
        << "Expected 4D quantized weights with height of 1 and width of 1.";
    batch = tflite_tensor->dims->data[0];
    channel = tflite_tensor->dims->data[3];
  } else {
    return absl::InvalidArgumentError("Expected 2D or 4D quantized weights.");
  }
  ABSL_CHECK(tflite_tensor->type == kTfLiteInt8 ||
             tflite_tensor->type == kTfLiteInt4 ||
             tflite_tensor->type == kTfLiteInt2)
      << "Expected Int8, Int4, or Int2 quantized weights.";
  *input = graph->NewValue();

  if (tflite_tensor->type == kTfLiteInt8) {
    (*input)->tensor.type = ::ml_drift::DataType::INT8;
  } else if (tflite_tensor->type == kTfLiteInt4) {
    (*input)->tensor.type = ::ml_drift::DataType::INT4;
  } else if (tflite_tensor->type == kTfLiteInt2) {
    (*input)->tensor.type = ::ml_drift::DataType::INT2;
  }
  (*input)->tensor.shape = ::ml_drift::BHWC(batch, 1, 1, channel);
  (*input)->tensor.ref = tensor_idx;
  (*input)->tensor.is_variable_input = tflite_tensor->is_variable;
  (*tensor_to_value)[tensor_idx] = *input;
  return absl::OkStatus();
}

bool ObjectReader::CanReadValue(int input_idx) const {
  return input_idx < node_->inputs->size &&
         CanReadValueByTensorIdx(node_->inputs->data[input_idx]);
}

::ml_drift::Value* ObjectReader::ReadValue(int input_idx) {
  return ReadValueByTensorIdx(node_->inputs->data[input_idx]);
}

void ObjectReader::ReadQuantizedValueByTensorIdx(
    uint32_t tensor_idx, ::ml_drift::Value** input_int8) {
  TfLiteTensor* tflite_tensor = &context_->tensors[tensor_idx];
  ABSL_CHECK(tflite_tensor->quantization.type == kTfLiteAffineQuantization ||
             tflite_tensor->quantization.type ==
                 TfLiteQuantizationType::kTfLiteBlockwiseQuantization)
      << absl::StrCat(
             "ReadQuantizedValueByTensorIdx: value is not quantized tensor",
             tensor_idx);
  int batch = 0;
  int channel = 0;
  if (tflite_tensor->dims->size == 2) {
    batch = tflite_tensor->dims->data[0];
    channel = tflite_tensor->dims->data[1];
  } else if (tflite_tensor->dims->size == 4) {
    ABSL_CHECK(tflite_tensor->dims->data[1] == 1 &&
               tflite_tensor->dims->data[2] == 1)
        << "Expected 4D quantized weights with height of 1 and width of 1.";
    batch = tflite_tensor->dims->data[0];
    channel = tflite_tensor->dims->data[3];
  }
  ABSL_CHECK(tflite_tensor->type == kTfLiteInt8 ||
             tflite_tensor->type == kTfLiteInt4 ||
             tflite_tensor->type == kTfLiteInt2)
      << "Expected Int8, Int4, or Int2 quantized weights.";
  *input_int8 = graph_->NewValue();

  if (tflite_tensor->type == kTfLiteInt8) {
    (*input_int8)->tensor.type = ::ml_drift::DataType::INT8;
  } else if (tflite_tensor->type == kTfLiteInt4) {
    (*input_int8)->tensor.type = ::ml_drift::DataType::INT4;
  } else if (tflite_tensor->type == kTfLiteInt2) {
    (*input_int8)->tensor.type = ::ml_drift::DataType::INT2;
  }
  (*input_int8)->tensor.shape = ::ml_drift::BHWC(batch, 1, 1, channel);
  (*input_int8)->tensor.ref = tensor_idx;
  (*input_int8)->tensor.is_variable_input = tflite_tensor->is_variable;
  (*tensor_to_value_)[tensor_idx] = *input_int8;
}

bool ObjectReader::CanReadValueByTensorIdx(int tensor_idx) const {
  const bool shared_tensor = GetSharingInfoByTensorIndex(tensor_idx).IsShared();
  return CanReadNonConstantTensor(context_, tensor_to_value_, tensor_idx,
                                  shared_tensor);
}

::ml_drift::Value* ObjectReader::ReadValueByTensorIdx(int tensor_idx) {
  const bool shared_tensor = GetSharingInfoByTensorIndex(tensor_idx).IsShared();
  return ReadNonConstantTensor(context_, tensor_to_value_,
                               quant_conversion_map_, graph_, tensor_idx,
                               shared_tensor);
}

int ObjectReader::GetNumberOfRuntimeInputs() const {
  return GetNumberOfRuntimeInputsForNode(context_, node_);
}

int ObjectReader::GetTensorId(int node_input_index) const {
  return node_->inputs->data[node_input_index];
}

bool ObjectReader::SharingEnabled() const {
  return (tensor_to_buffer_id_map_ || tensor_to_external_buffer_id_map_) &&
         shared_tensor_map_;
}

void ObjectReader::AllowSharingInput(int input_index) {
  if (input_index < node_->inputs->size) {
    const int tensor_id = node_->inputs->data[input_index];
    allowed_shared_tensors_.insert(tensor_id);
  }
}

ObjectReader::ConstantInputSharingInfo
ObjectReader::GetSharingInfoByTensorIndex(int tensor_id) const {
  if (!SharingEnabled() || !allowed_shared_tensors_.count(tensor_id)) {
    return ConstantInputSharingInfo::BuildNotShared();
  }
  ConstantInputSharingInfo info;
  if (tensor_to_buffer_id_map_) {
    auto it = tensor_to_buffer_id_map_->find(tensor_id);
    if (it != tensor_to_buffer_id_map_->end()) {
      info.buffer_id = it->second;
    }
  }
  if (tensor_to_external_buffer_id_map_) {
    auto it = tensor_to_external_buffer_id_map_->find(tensor_id);
    if (it != tensor_to_external_buffer_id_map_->end()) {
      info.external_buffer_id = it->second;
    }
  }
  return info.IsShared() ? info : ConstantInputSharingInfo::BuildNotShared();
}

ObjectReader::ConstantInputSharingInfo
ObjectReader::GetSharingInfoByNodeInputIndex(int node_input_index) const {
  if (node_input_index < 0 || node_input_index >= node_->inputs->size) {
    return ConstantInputSharingInfo::BuildNotShared();
  }
  const int tensor_id = node_->inputs->data[node_input_index];
  if (tensor_id < 0 || tensor_id >= context_->tensors_size) {
    return ConstantInputSharingInfo::BuildNotShared();
  }
  return GetSharingInfoByTensorIndex(tensor_id);
}

void ObjectReader::SetSharedTensor(const ::ml_drift::ValueId graph_value_id,
                                   const int global_id,
                                   const int tflite_tensor_id,
                                   bool dequant_forced,
                                   std::optional<::ml_drift::Layout> layout) {
  SharedTfliteTensor shared_tensor;
  shared_tensor.tflite_tensor_id = tflite_tensor_id;
  shared_tensor.global_id = global_id;
  shared_tensor.dequant_forced = dequant_forced;
  shared_tensor.layout = layout;
  shared_tensor_map_->try_emplace(graph_value_id, shared_tensor);
}

absl::Status ObjectReader::GetTensorDims(uint32_t idx,
                                         TfLiteIntArray* dimensions) const {
  if (idx >= node_->inputs->size) {
    return absl::OutOfRangeError(absl::StrCat("Input tensor index: ", idx));
  }
  const int tensor_idx = node_->inputs->data[idx];
  if (tensor_idx < 0 || tensor_idx > context_->tensors_size) {
    return absl::OutOfRangeError(absl::StrCat("Tensor index: ", tensor_idx));
  }
  const TfLiteTensor& tflite_tensor = context_->tensors[tensor_idx];
  *dimensions = *tflite_tensor.dims;
  return absl::OkStatus();
}

void ObjectReader::AddInput(const ::ml_drift::Node* node,
                            int node_input_index) {
  ABSL_CHECK_LT(node_input_index, node_->inputs->size);
  ABSL_CHECK(CanReadValue(node_input_index));
  const ::ml_drift::Value* input = ReadValue(node_input_index);
  graph_->AddConsumer(node->id, input->id);
}

void ObjectReader::AddUpdate(const ::ml_drift::Node* node,
                             int node_input_index) {
  ABSL_CHECK_LT(node_input_index, node_->inputs->size);
  const int tensor_id = node_->inputs->data[node_input_index];
  TfLiteTensor* update_tensor = context_->tensors + tensor_id;
  ABSL_CHECK(update_tensor->is_variable) << "tensor_id: " << tensor_id;
  ABSL_CHECK(CanReadValueByTensorIdx(tensor_id)) << "tensor_id: " << tensor_id;
  ::ml_drift::Value* value = ReadValueByTensorIdx(tensor_id);
  ABSL_CHECK(value->tensor.is_variable_input);

  // We cannot create a cycle in the graph. The way around this when a node
  // updates a tensor in place would be to add a new value to the graph that
  // points to the same tensor.
  ::ml_drift::Value* updated_value = graph_->NewValue();
  updated_value->tensor = value->tensor;
  updated_value->quant_params = value->quant_params;
  graph_->SetProducer(node->id, updated_value->id);

  // We also need to update the tensor_to_value arrays so that the nodes added
  // after the current node will access the tensor with the updated value rather
  // than the initial value.
  if (quant_conversion_map_ != nullptr &&
      quant_conversion_map_->find(tensor_id) != quant_conversion_map_->end()) {
    // If quantization conversion map exists, then the index provided is not the
    // actual tensor idx. We need to find the float version of the tensor from
    // the map.
    tensor_to_value_->at(quant_conversion_map_->at(tensor_id)) = updated_value;
  } else {
    tensor_to_value_->at(tensor_id) = updated_value;
  }
}

void ObjectReader::AddOutput(const ::ml_drift::Node* node,
                             int node_output_index) {
  ABSL_CHECK_LT(node_output_index, node_->outputs->size);
  const int tensor_id = node_->outputs->data[node_output_index];
  ABSL_CHECK(CanReadValueByTensorIdx(tensor_id)) << "tensor_id: " << tensor_id;
  const ::ml_drift::Value* value = ReadValueByTensorIdx(tensor_id);
  graph_->SetProducer(node->id, value->id);
}

void ObjectReader::AddOutputs(const ::ml_drift::Node* node) {
  for (int i = 0; i < node_->outputs->size; ++i) AddOutput(node, i);
}

}  // namespace litert::ml_drift
