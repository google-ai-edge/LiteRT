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

#include "ml_drift_delegate/delegate/shared_memory_manager/ir_graph_adapter.h"

#include <cstdint>
#include <string>
#include <vector>

#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "tflite/core/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace ml_drift {

BHWC IrModelAdapter::GetValueShape(uint32_t value_id) const {
  return graph_.tensor(value_id)->desc.GetBHWCShape();
}

void IrModelAdapter::SetValueType(uint32_t value_id, DataType type) {
  graph_.GetMutableTensor(value_id)->desc.SetDataType(type);
}

void IrModelAdapter::SetValueShapeAndType(uint32_t value_id, const BHWC& shape,
                                          DataType type) {
  // Mutate the descriptor in place, preserving storage type / layout and
  // keeping the tensor id stable (matching GraphFloat32's in-place mutation).
  ir::IrTensor* tensor = graph_.GetMutableTensor(value_id);
  tensor->desc.SetBHWCShape(shape);
  tensor->desc.SetDataType(type);
}

DataType IrModelAdapter::ResolveSharedTensorType(
    uint32_t shared_tensor_id, DataType default_data_type) const {
  const DataType graph_value_type =
      graph_.tensor(shared_tensor_id)->desc.GetDataType();
  if (IsFloatType(graph_value_type)) {
    if (default_data_type == DataType::FLOAT32) {
      std::vector<uint32_t> consumers = FindConsumerOps(shared_tensor_id);
      if (consumers.size() == 1 && OpHasInputs(consumers[0])) {
        DataType input_type = GetOpFirstInputType(consumers[0]);
        if (IsFloatType(input_type)) {
          return input_type;
        }
      }
    }
    return default_data_type;
  }
  return graph_value_type;
}

void IrModelAdapter::UploadTensorData(const TfLiteTensor& tensor,
                                      const float* weights_data_ptr,
                                      TensorDescriptor& tensor_desc) const {
  // Support uploading raw data for integer tensors, float16 data for float16
  // tensors, otherwise upload float data. This is used for models with fp16
  // weights from MediaPipe (e.g. inpainting models) or non-float constants
  // (e.g. shape/pack/lookup tensors).
  switch (tensor_desc.GetDataType()) {
    case DataType::INT8:
      tensor_desc.UploadData<int8_t>(tensor.data.int8);
      break;
    case DataType::UINT8:
      tensor_desc.UploadData<uint8_t>(tensor.data.uint8);
      break;
    case DataType::INT32:
      tensor_desc.UploadData<int32_t>(tensor.data.i32);
      break;
    case DataType::INT64:
      tensor_desc.UploadData<int64_t>(tensor.data.i64);
      break;
    case DataType::BOOL:
      tensor_desc.UploadData<bool>(tensor.data.b);
      break;
    case DataType::FLOAT16:
      if (tensor.type == TfLiteType::kTfLiteFloat16) {
        tensor_desc.UploadData<half>(
            reinterpret_cast<const half*>(tensor.data.f16));
      } else {
        int num_elements = tflite::NumElements(&tensor);
        std::vector<half> half_data(num_elements);
        for (int i = 0; i < num_elements; ++i) {
          half_data[i] = half(weights_data_ptr[i]);
        }
        tensor_desc.UploadData<half>(half_data.data());
      }
      break;
    default:  // FLOAT32
      if (tensor.type == TfLiteType::kTfLiteFloat16) {
        int num_elements = tflite::NumElements(&tensor);
        std::vector<float> float_data(num_elements);
        const half* f16_ptr = reinterpret_cast<const half*>(tensor.data.f16);
        for (int i = 0; i < num_elements; ++i) {
          float_data[i] = static_cast<float>(f16_ptr[i]);
        }
        tensor_desc.UploadData<float>(float_data.data());
      } else {
        tensor_desc.UploadData<float>(weights_data_ptr);
      }
      break;
  }
}

std::vector<uint32_t> IrModelAdapter::FindConsumerOps(uint32_t value_id) const {
  std::vector<ir::IrOp*> consumers = graph_.FindConsumers(value_id);
  std::vector<uint32_t> op_ids;
  op_ids.reserve(consumers.size());
  for (const ir::IrOp* op : consumers) {
    op_ids.push_back(static_cast<uint32_t>(op->id));
  }
  return op_ids;
}

std::string IrModelAdapter::GetOpTypeName(uint32_t op_id) const {
  return graph_.op(op_id)->name;
}

bool IrModelAdapter::OpHasInputs(uint32_t op_id) const {
  return !graph_.op(op_id)->inputs.empty();
}

BHWC IrModelAdapter::GetOpFirstInputShape(uint32_t op_id) const {
  const ir::IrTensorId input_id = graph_.op(op_id)->inputs[0];
  return graph_.tensor(input_id)->desc.GetBHWCShape();
}

DataType IrModelAdapter::GetOpFirstInputType(uint32_t op_id) const {
  const ir::IrTensorId input_id = graph_.op(op_id)->inputs[0];
  return graph_.tensor(input_id)->desc.GetDataType();
}

uint32_t IrModelAdapter::AddConstantInput(uint32_t global_tensor_id,
                                          const BHWC& shape, DataType type,
                                          uint32_t consumer_op_id) {
  ir::IrTensor* value = graph_.add_tensor(type, shape);
  value->buffer_source = ir::BufferSource{
      .is_shared = true,
      .global_id = global_tensor_id,
  };
  graph_.AddConsumer(value->id, consumer_op_id);
  return static_cast<uint32_t>(value->id);
}

}  // namespace ml_drift
