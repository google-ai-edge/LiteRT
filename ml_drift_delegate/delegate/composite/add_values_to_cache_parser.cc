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

#include "ml_drift_delegate/delegate/composite/add_values_to_cache_parser.h"

#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/model_builder_helper.h"
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

absl::Status AddValuesToCacheOperationParser::IsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration*) {
  if (GetNumberOfRuntimeInputsForNode(context, tflite_node) != 3 &&
      GetNumberOfRuntimeInputsForNode(context, tflite_node) != 7) {
    return absl::UnavailableError("odml.cache_update expects 3 or 7 inputs.");
  }
  RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
  RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 1));
  RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 2));
  RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));

  if (tflite_node->outputs->size != 2) {
    return absl::InvalidArgumentError("odml.cache_update expects 2 outputs.");
  }

  const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
      tflite_node->builtin_data);
  if (params) {
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(params->attributes, params->attributes_size)
            .AsMap();
    if (flexbuffer_map["kv_cache_batch_size"].IsNull()) {
      return absl::InvalidArgumentError(
          "odml.cache_update is missing kv_cache_batch_size.");
    }
    if (flexbuffer_map["cache_size"].IsNull()) {
      return absl::InvalidArgumentError(
          "odml.cache_update is missing cache_size.");
    }
    if (flexbuffer_map["head_size"].IsNull()) {
      return absl::InvalidArgumentError(
          "odml.cache_update is missing head_size.");
    }
  }

  return absl::OkStatus();
}

void AddValuesToCacheOperationParser::Parse(const TfLiteNode* tflite_node,
                                            const TfLiteRegistration*,
                                            ::ml_drift::GraphFloat32* graph,
                                            ObjectReader* reader) {
  auto* node = graph->NewNode();
  node->operation.type = kAddValuesToCacheType;
  // The operator might have 7 inputs. We only read the first 3. It is a
  // composite op, and cpu decomposition needs additional tensors to do the same
  // thing (in-place dynamic update slice). It needs to a part of the op
  // signature, even if for the GPU we do not need those tensors.
  reader->AddInput(node, 0);  // src_k
  reader->AddInput(node, 1);  // src_v
  reader->AddInput(node, 2);  // runtime_param_tensor
  reader->AddOutputs(node);

  // TODO(b/482104479): allow native int8 tensors in MLDrift. Currently, MLDrift
  // will automatically dequantize int8 tensors to float32 if the tflite tensor
  // is quantized (AffineQuantization present).
  if (reader->GetOutputTensor(0)->type == kTfLiteInt8) {
    auto output_1 = graph->FindOutputs(node->id)[0];
    auto output_2 = graph->FindOutputs(node->id)[1];
    // reader->AddOutputs() created new FP32 tflite tensors, and uses those as
    // the tflite tensor reference (tensor index). We change it back to the
    // original tensor index to keep the mapping of the value to the original
    // int8 tflite tensor, as we expect to use that int8 tensor to map to the
    // GPU value used in the kernel, without any dequantize in the middle.
    output_1->tensor.ref = tflite_node->outputs->data[0];
    output_2->tensor.ref = tflite_node->outputs->data[1];
    if (output_1->quant_params.has_value()) {
      output_1->quant_params.reset();
    }
    if (output_2->quant_params.has_value()) {
      output_2->quant_params.reset();
    }
    // we expect uint8 instead of int8.
    output_1->tensor.type = ::ml_drift::DataType::UINT8;
    output_2->tensor.type = ::ml_drift::DataType::UINT8;
  }

  const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
      tflite_node->builtin_data);
  const flexbuffers::Map flexbuffer_map =
      flexbuffers::GetRoot(params->attributes, params->attributes_size).AsMap();
  ml_drift::AddValuesToCacheAttributes attr;
  attr.kv_cache_batch_size = flexbuffer_map["kv_cache_batch_size"].AsInt32();
  attr.cache_size = flexbuffer_map["cache_size"].AsInt32();
  attr.head_size = flexbuffer_map["head_size"].AsInt32();
  if (!flexbuffer_map["scale_k"].IsNull()) {
    attr.scale_k = flexbuffer_map["scale_k"].AsFloat();
  }
  if (!flexbuffer_map["scale_v"].IsNull()) {
    attr.scale_v = flexbuffer_map["scale_v"].AsFloat();
  }
  node->operation.attributes = std::move(attr);
}

}  // namespace litert::ml_drift
