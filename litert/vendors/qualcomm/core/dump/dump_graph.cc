// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @com_github_nlohmann_json
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnTypes.h"  // from @qairt

namespace qnn {

namespace {
template <typename T>
nlohmann::json ReshapeDataRecursive(uint32_t& cur_index, T* data,
                                    uint32_t num_elements,
                                    absl::Span<uint32_t> dims) {
  if (dims.empty()) {
    return nlohmann::json();
  }

  uint32_t size = dims[0];
  nlohmann::json nested_array = nlohmann::json::array();
  if (dims.size() == 1) {
    for (uint32_t i = 0; i < size; ++i) {
      if (cur_index < num_elements) {
        nested_array.emplace_back(data[cur_index++]);
      } else {
        QNN_LOG_ERROR("The data size for tensor_params does not match.");
        // Fill with 0 if array is smaller than the specified dimensions.
        nested_array.emplace_back(0);
      }
    }
  } else {
    absl::Span<uint32_t> sub_dims = dims.subspan(1);
    for (uint32_t i = 0; i < size; ++i) {
      nested_array.emplace_back(
          ReshapeDataRecursive<T>(cur_index, data, num_elements, sub_dims));
    }
  }
  return nested_array;
}

template <typename T>
nlohmann::json ReshapeData(void* buffer_data, uint32_t buffer_size,
                           absl::Span<uint32_t> dims) {
  T* data = static_cast<T*>(buffer_data);
  uint32_t ind = 0;
  return ReshapeDataRecursive<T>(ind, data, buffer_size / sizeof(T), dims);
}
}  // namespace

nlohmann::json SerializeQuantParamToJson(
    const Qnn_QuantizeParams_t& quant_params) {
  nlohmann::json qnn_quant_params = {
      {"definition", quant_params.encodingDefinition},
      {"encoding", quant_params.quantizationEncoding}};
  // TODO (jiunkaiy): Support more quant encoding.
  if (quant_params.quantizationEncoding ==
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    qnn_quant_params["scale_offset"] = {
        {"scale", quant_params.scaleOffsetEncoding.scale},
        {"offset", quant_params.scaleOffsetEncoding.offset}};
  } else if (quant_params.quantizationEncoding ==
             QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET) {
    qnn_quant_params["scale_offset"] = {
        {"bitwidth", quant_params.bwScaleOffsetEncoding.bitwidth},
        {"scale", quant_params.bwScaleOffsetEncoding.scale},
        {"offset", quant_params.bwScaleOffsetEncoding.offset}};
  } else if (quant_params.quantizationEncoding ==
             QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    std::vector<nlohmann::json> scale_offsets;
    uint32_t num_scale_offsets =
        quant_params.axisScaleOffsetEncoding.numScaleOffsets;
    scale_offsets.reserve(num_scale_offsets);
    for (int i = 0; i < num_scale_offsets; ++i) {
      scale_offsets.emplace_back(nlohmann::json{
          {"scale", quant_params.axisScaleOffsetEncoding.scaleOffset[i].scale},
          {"offset",
           quant_params.axisScaleOffsetEncoding.scaleOffset[i].offset}});
    }
    qnn_quant_params["axis_scale_offset"] = {
        {"axis", quant_params.axisScaleOffsetEncoding.axis},
        {"num_scale_offsets", num_scale_offsets},
        {"scale_offsets", scale_offsets},
    };
  } else if (quant_params.quantizationEncoding ==
             QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
    std::vector<nlohmann::json> scale_offsets;
    uint32_t num_elements = quant_params.bwAxisScaleOffsetEncoding.numElements;
    scale_offsets.reserve(num_elements);
    for (int i = 0; i < num_elements; ++i) {
      scale_offsets.emplace_back(nlohmann::json{
          {"scale", quant_params.bwAxisScaleOffsetEncoding.scales[i]},
          {"offset", quant_params.bwAxisScaleOffsetEncoding.offsets[i]}});
    }
    qnn_quant_params["axis_scale_offset"] = {
        {"bitwidth", quant_params.bwAxisScaleOffsetEncoding.bitwidth},
        {"axis", quant_params.bwAxisScaleOffsetEncoding.axis},
        {"num_scale_offsets", num_elements},
        {"scale_offsets", scale_offsets},
    };
  } else {
    QNN_LOG_WARNING(
        "Quantization encoding: %u is not supported in Qnn "
        "Json dump",
        quant_params.quantizationEncoding);
  }
  return qnn_quant_params;
}

nlohmann::json SerializeTensorToJson(const Qnn_TensorV1_t& qnn_tensor) {
  nlohmann::json qnn_tensor_json;
  qnn_tensor_json["id"] = qnn_tensor.id;
  qnn_tensor_json["type"] = qnn_tensor.type;
  qnn_tensor_json["dataFormat"] = qnn_tensor.dataFormat;
  qnn_tensor_json["data_type"] = qnn_tensor.dataType;
  qnn_tensor_json["dims"] =
      absl::Span<uint32_t>(qnn_tensor.dimensions, qnn_tensor.rank);

  const Qnn_QuantizeParams_t& quant_params = qnn_tensor.quantizeParams;
  if (quant_params.encodingDefinition != QNN_DEFINITION_DEFINED) {
    return qnn_tensor_json;
  }
  // Add basic key-value pairs for quant_params.
  qnn_tensor_json["quant_params"] = SerializeQuantParamToJson(quant_params);
  return qnn_tensor_json;
}

nlohmann::json SerializeScalarParamToJson(const Qnn_Scalar_t& scalar) {
  Qnn_DataType_t datatype = scalar.dataType;
  switch (datatype) {
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
      return nlohmann::json{{std::to_string(datatype), scalar.int8Value}};
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
      return nlohmann::json{{std::to_string(datatype), scalar.int16Value}};
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
      return nlohmann::json{{std::to_string(datatype), scalar.int32Value}};
    case QNN_DATATYPE_INT_64:
      return nlohmann::json{{std::to_string(datatype), scalar.int64Value}};
    case QNN_DATATYPE_BOOL_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
      return nlohmann::json{{std::to_string(datatype), scalar.uint8Value}};
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      return nlohmann::json{{std::to_string(datatype), scalar.uint16Value}};
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32:
      return nlohmann::json{{std::to_string(datatype), scalar.uint32Value}};
    case QNN_DATATYPE_UINT_64:
      return nlohmann::json{{std::to_string(datatype), scalar.uint64Value}};
    case QNN_DATATYPE_FLOAT_32:
      return nlohmann::json{{std::to_string(datatype), scalar.floatValue}};
    case QNN_DATATYPE_FLOAT_64:
      return nlohmann::json{{std::to_string(datatype), scalar.doubleValue}};
    case QNN_DATATYPE_STRING:
      return nlohmann::json{{std::to_string(datatype), scalar.stringValue}};
    default:
      QNN_LOG_WARNING(
          "Datatype: %u is not supported for scalar_params in Qnn Json dump",
          datatype)
      break;
  }
  return nlohmann::json::object();
}

nlohmann::json SerializeTensorParamToJson(const Qnn_TensorV1_t& qnn_tensor) {
  void* data = qnn_tensor.clientBuf.data;
  uint32_t size = qnn_tensor.clientBuf.dataSize;
  absl::Span<uint32_t> dims(qnn_tensor.dimensions, qnn_tensor.rank);
  switch (qnn_tensor.dataType) {
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
      return ReshapeData<int8_t>(data, size, dims);
    case QNN_DATATYPE_BOOL_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
      return ReshapeData<uint8_t>(data, size, dims);
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
      return ReshapeData<int16_t>(data, size, dims);
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      return ReshapeData<uint16_t>(data, size, dims);
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
      return ReshapeData<int32_t>(data, size, dims);
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32:
      return ReshapeData<uint32_t>(data, size, dims);
    case QNN_DATATYPE_FLOAT_32:
      return ReshapeData<float>(data, size, dims);
    case QNN_DATATYPE_FLOAT_64:
      return ReshapeData<double>(data, size, dims);
    case QNN_DATATYPE_INT_64:
      return ReshapeData<int64_t>(data, size, dims);
    case QNN_DATATYPE_UINT_64:
      return ReshapeData<uint64_t>(data, size, dims);
    default:
      QNN_LOG_ERROR(
          "Datatype: %u is not supported for tensor_params in Qnn Json dump",
          qnn_tensor.dataType);
      break;
  }
  return nlohmann::json();
}

nlohmann::json SerializeOpToJson(const Qnn_OpConfig_t& op_config) {
  nlohmann::json qnn_node_json;
  // Create input_names and output_names.
  qnn_node_json["input_names"] = nlohmann::json::array();
  for (uint32_t i = 0; i < op_config.v1.numOfInputs; ++i) {
    qnn_node_json["input_names"].emplace_back(
        op_config.v1.inputTensors[i].v1.name);
  }
  qnn_node_json["output_names"] = nlohmann::json::array();
  for (uint32_t i = 0; i < op_config.v1.numOfOutputs; ++i) {
    qnn_node_json["output_names"].emplace_back(
        op_config.v1.outputTensors[i].v1.name);
  }
  // Create macs_per_inferences and op type.
  qnn_node_json["macs_per_inference"] = "";
  qnn_node_json["type"] = op_config.v1.typeName;
  // Create scalar_params and tensor_params.
  qnn_node_json["scalar_params"] = nlohmann::json::object();
  qnn_node_json["tensor_params"] = nlohmann::json::object();
  for (uint32_t i = 0; i < op_config.v1.numOfParams; ++i) {
    if (op_config.v1.params[i].paramType == QNN_PARAMTYPE_SCALAR) {
      qnn_node_json["scalar_params"][op_config.v1.params[i].name] =
          SerializeScalarParamToJson(op_config.v1.params[i].scalarParam);
    } else if (op_config.v1.params[i].paramType == QNN_PARAMTYPE_TENSOR) {
      const Qnn_TensorV1_t& qnn_tensor = op_config.v1.params[i].tensorParam.v1;
      nlohmann::json qnn_tensor_json = SerializeTensorToJson(qnn_tensor);
      qnn_tensor_json["data"] = SerializeTensorParamToJson(qnn_tensor);
      qnn_node_json["tensor_params"][op_config.v1.params[i].name]
                   [qnn_tensor.name] = qnn_tensor_json;
    }
  }
  return qnn_node_json;
}

void DumpIrJson(
    const absl::flat_hash_set<const TensorWrapper*>& tensor_wrappers,
    std::vector<OpWrapper>& graph_op_wrappers, std::string_view json_dir,
    std::string_view graph_name) {
  CreateDirectoryRecursive(json_dir);

  std::filesystem::path ir_json_path = json_dir;
  ir_json_path /= std::string(graph_name) + ".json";
  QNN_LOG_INFO("Qnn Json Path: %s", ir_json_path.c_str());

  nlohmann::json ir_json = {
      {"model.cpp", "N/A"},
      {"model.bin", "N/A"},
      {"converter_command", ""},
      {"copyright_str",
       "Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved."},
      {"op_types", nlohmann::json::array()},
      {"Total parameters", ""},
      {"Total MACs per inference", ""},
      {"graph",
       {{"tensors", nlohmann::json::object()},
        {"nodes", nlohmann::json::object()}}}};

  // Dump Qnn Ops.
  // Note that only the static data of tensorParam is stored in QNN Json.
  absl::flat_hash_set<uint32_t> param_tensor_ids;
  absl::flat_hash_set<std::string> op_types;
  for (auto& op : graph_op_wrappers) {
    const Qnn_OpConfig_t op_config = op.GetOpConfig();
    ir_json["graph"]["nodes"][op_config.v1.name] = SerializeOpToJson(op_config);
    // Record seen op type in a set.
    op_types.emplace(op_config.v1.typeName);
    // Record tensor param IDs to avoid adding them to graph tensors.
    for (uint32_t i = 0; i < op_config.v1.numOfParams; ++i) {
      if (op_config.v1.params[i].paramType == QNN_PARAMTYPE_TENSOR) {
        param_tensor_ids.emplace(op_config.v1.params[i].tensorParam.v1.id);
      }
    }
  }
  // Dump Qnn Tensors.
  for (const TensorWrapper* tensor : tensor_wrappers) {
    // Skip tensor params.
    if (param_tensor_ids.count(tensor->GetId()) > 0) {
      continue;
    }
    // Create tensors.
    nlohmann::json qnn_tensor_json =
        SerializeTensorToJson(tensor->GetQnnTensor().v1);
    ir_json["graph"]["tensors"][tensor->GetName()] = qnn_tensor_json;
  }
  // Dumpe Qnn op types.
  ir_json["op_types"] = op_types;
  // Write the JSON string to a file.
  std::ofstream outFile(ir_json_path);
  if (outFile.is_open()) {
    outFile << ir_json.dump(/*indent=*/4);
    outFile.close();
  } else {
    QNN_LOG_ERROR("Unable to open qnn_litert.json for writing.");
  }
}

}  // namespace qnn
