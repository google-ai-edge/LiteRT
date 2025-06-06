// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <string>
#include <unordered_set>

#include "QnnTypes.h"                      // from @qairt
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "nlohmann/json.hpp"
namespace qnn {

namespace {
void AddQuantParams(const Qnn_QuantizeParams_t& quant_params,
                    nlohmann::json& qnn_tensor_json) {
  if (quant_params.encodingDefinition != QNN_DEFINITION_DEFINED) {
    return;
  }
  // Add basic key-value pairs for quant_params.
  qnn_tensor_json["quant_params"] = {
      {"definition", quant_params.encodingDefinition},
      {"encoding", quant_params.quantizationEncoding}};
  // TODO (jiunkaiy): Support more quant encoding.
  if (quant_params.quantizationEncoding ==
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    qnn_tensor_json["quant_params"]["scale_offset"] = {
        {"scale", quant_params.scaleOffsetEncoding.scale},
        {"offset", quant_params.scaleOffsetEncoding.offset}};
  } else if (quant_params.quantizationEncoding ==
             QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET) {
    qnn_tensor_json["quant_params"]["scale_offset"] = {
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
    qnn_tensor_json["quant_params"]["axis_scale_offset"] = {
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
    qnn_tensor_json["quant_params"]["axis_scale_offset"] = {
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
}

void AddScalarParams(const Qnn_Param_t& params, nlohmann::json& qnn_node_json) {
  const char* name = params.name;
  Qnn_DataType_t datatype = params.scalarParam.dataType;
  const Qnn_Scalar_t& scalar = params.scalarParam;
  switch (datatype) {
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
      qnn_node_json["scalar_params"][name] =
          nlohmann::json{{std::to_string(datatype), scalar.int8Value}};
      break;
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
      qnn_node_json["scalar_params"][name] =
          nlohmann::json{{std::to_string(datatype), scalar.int16Value}};
      break;
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
      qnn_node_json["scalar_params"][name] =
          nlohmann::json{{std::to_string(datatype), scalar.int32Value}};
      break;
    case QNN_DATATYPE_INT_64:
      qnn_node_json["scalar_params"][name] =
          nlohmann::json{{std::to_string(datatype), scalar.int64Value}};
      break;
    case QNN_DATATYPE_BOOL_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
      qnn_node_json["scalar_params"][name] =
          nlohmann::json{{std::to_string(datatype), scalar.uint8Value}};
      break;
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      qnn_node_json["scalar_params"][name] =
          nlohmann::json{{std::to_string(datatype), scalar.uint16Value}};
      break;
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32:
      qnn_node_json["scalar_params"][name] =
          nlohmann::json{{std::to_string(datatype), scalar.uint32Value}};
      break;
    case QNN_DATATYPE_UINT_64:
      qnn_node_json["scalar_params"][name] =
          nlohmann::json{{std::to_string(datatype), scalar.uint64Value}};
      break;
    case QNN_DATATYPE_FLOAT_32:
      qnn_node_json["scalar_params"][name] =
          nlohmann::json{{std::to_string(datatype), scalar.floatValue}};
      break;
    case QNN_DATATYPE_FLOAT_64:
      qnn_node_json["scalar_params"][name] =
          nlohmann::json{{std::to_string(datatype), scalar.doubleValue}};
      break;
    case QNN_DATATYPE_STRING:
      qnn_node_json["scalar_params"][name] =
          nlohmann::json{{std::to_string(datatype), scalar.stringValue}};
      break;
    default:
      QNN_LOG_WARNING("Datatype: %u is not supported in Qnn Json dump",
                      datatype)
      break;
  }
}

}  // namespace

void DumpQnnJson(
    const absl::flat_hash_set<const TensorWrapper*>& tensor_wrappers,
    std::vector<OpWrapper>& graph_op_wrappers) {
  nlohmann::json qnn_json = {
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
  std::unordered_set<uint32_t> param_tensor_ids;
  std::unordered_set<std::string> op_types;
  for (auto& op : graph_op_wrappers) {
    const Qnn_OpConfig_t op_config = op.GetOpConfig();
    nlohmann::json qnn_node_json = nlohmann::json::object();
    qnn_node_json["input_names"] = nlohmann::json::array();
    for (uint32_t i = 0; i < op_config.v1.numOfInputs; ++i) {
      qnn_node_json["input_names"].emplace_back(
          op_config.v1.inputTensors[i].v2.name);
    }
    qnn_node_json["macs_per_inference"] = "";
    // Record seen op type in a set.
    qnn_node_json["type"] = op_config.v1.typeName;
    op_types.emplace(op_config.v1.typeName);
    // Create scalar_params and tensor_params.
    qnn_node_json["scalar_params"] = nlohmann::json::object();
    qnn_node_json["tensor_params"] = nlohmann::json::object();
    for (uint32_t i = 0; i < op_config.v1.numOfParams; ++i) {
      if (op_config.v1.params[i].paramType == QNN_PARAMTYPE_SCALAR) {
        AddScalarParams(op_config.v1.params[i], qnn_node_json);
      } else if (op_config.v1.params[i].paramType == QNN_PARAMTYPE_TENSOR) {
        nlohmann::json qnn_tensor_json = nlohmann::json::object();
        const Qnn_TensorV1_t& tensor = op_config.v1.params[i].tensorParam.v1;
        qnn_tensor_json["id"] = tensor.id;
        qnn_tensor_json["type"] = tensor.type;
        qnn_tensor_json["dataFormat"] = tensor.dataFormat;
        qnn_tensor_json["data_type"] = tensor.dataType;
        qnn_tensor_json["dims"] = nlohmann::json::array();
        for (size_t j = 0; j < tensor.rank; ++j) {
          qnn_tensor_json["dims"].emplace_back(tensor.dimensions[j]);
        }
        AddQuantParams(tensor.quantizeParams, qnn_tensor_json);
        qnn_node_json["tensor_params"][op_config.v1.params[i].name]
                     [tensor.name] = qnn_tensor_json;
        // Record tensor param IDs to avoid adding them to graph tensors.
        param_tensor_ids.emplace(tensor.id);
      }
    }

    qnn_json["graph"]["nodes"][op_config.v1.name] = qnn_node_json;
  }
  // Dump Qnn Tensors.
  for (const TensorWrapper* tensor : tensor_wrappers) {
    // Skip tensor params.
    if (param_tensor_ids.count(tensor->GetQnnTensorID()) > 0) {
      continue;
    }
    // Create tensors.
    nlohmann::json qnn_tensor_json = nlohmann::json::object();
    qnn_tensor_json["id"] = tensor->GetQnnTensorID();
    qnn_tensor_json["type"] = tensor->GetQnnTensorType();
    qnn_tensor_json["dataFormat"] = tensor->GetQnnTensorDataType();
    qnn_tensor_json["data_type"] = tensor->GetQnnTensorDataType();
    qnn_tensor_json["dims"] = tensor->GetDims();
    AddQuantParams(tensor->GetQnnTensorQuantParams(), qnn_tensor_json);
    qnn_json["graph"]["tensors"][tensor->GetName()] = qnn_tensor_json;
  }
  // Dumpe Qnn op types.
  qnn_json["op_types"] = op_types;
  // Convert the JSON object to a string.
  std::string jsonString = qnn_json.dump(4);

  // Print the JSON string using printf.
  QNN_LOG_DEBUG("%s\n", jsonString.c_str());

  // Write the JSON string to a file.
  std::ofstream outFile("qnn_litert.json");
  if (outFile.is_open()) {
    outFile << jsonString;
    outFile.close();
  } else {
    QNN_LOG_ERROR("Unable to open qnn_litert.json for writing.");
  }
}

}  // namespace qnn
