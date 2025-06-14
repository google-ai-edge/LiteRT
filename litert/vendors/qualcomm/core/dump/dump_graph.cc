// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <string>
#include <unordered_set>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "nlohmann/json.hpp"
#include "QnnTypes.h"  // from @qairt

namespace qnn {

namespace {
void AddQnnTensor(const Qnn_TensorV1_t& qnn_tensor,
                  nlohmann::json& qnn_tensor_json) {
  qnn_tensor_json["id"] = qnn_tensor.id;
  qnn_tensor_json["type"] = qnn_tensor.type;
  qnn_tensor_json["dataFormat"] = qnn_tensor.dataFormat;
  qnn_tensor_json["data_type"] = qnn_tensor.dataType;
  qnn_tensor_json["dims"] =
      absl::Span<uint32_t>(qnn_tensor.dimensions, qnn_tensor.rank);

  const Qnn_QuantizeParams_t& quant_params = qnn_tensor.quantizeParams;
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
      QNN_LOG_WARNING(
          "Datatype: %u is not supported for scalar_params in Qnn Json dump",
          datatype)
      break;
  }
}

template <typename T>
nlohmann::json ReshapeData(absl::Span<T> data, absl::Span<uint32_t> dims,
                           uint32_t& index) {
  if (dims.empty()) {
    return nlohmann::json();
  }

  uint32_t size = dims[0];
  nlohmann::json nested_array = nlohmann::json::array();
  if (dims.size() == 1) {
    for (uint32_t i = 0; i < size; ++i) {
      if (index < data.size()) {
        nested_array.emplace_back(data[index++]);
      } else {
        // Fill with 0 if array is smaller than the specified dimensions.
        nested_array.emplace_back(0);
      }
    }
  } else {
    absl::Span<uint32_t> sub_dims = dims.subspan(1);
    for (uint32_t i = 0; i < size; ++i) {
      nested_array.emplace_back(ReshapeData(data, sub_dims, index));
    }
  }
  return nested_array;
}

nlohmann::json GetData(Qnn_DataType_t datatype,
                       const Qnn_ClientBuffer_t& buffer,
                       absl::Span<uint32_t> dims) {
  uint32_t index = 0;
  if (datatype == QNN_DATATYPE_INT_8 ||
      datatype == QNN_DATATYPE_SFIXED_POINT_8) {
    absl::Span<int8_t> data(static_cast<int8_t*>(buffer.data),
                            buffer.dataSize / sizeof(int8_t));
    return ReshapeData<int8_t>(data, dims, index);
  } else if (datatype == QNN_DATATYPE_BOOL_8 ||
             datatype == QNN_DATATYPE_UINT_8 ||
             datatype == QNN_DATATYPE_UFIXED_POINT_8) {
    absl::Span<uint8_t> data(static_cast<uint8_t*>(buffer.data),
                             buffer.dataSize / sizeof(uint8_t));
    return ReshapeData<uint8_t>(data, dims, index);
  } else if (datatype == QNN_DATATYPE_INT_16 ||
             datatype == QNN_DATATYPE_SFIXED_POINT_16) {
    absl::Span<int16_t> data(static_cast<int16_t*>(buffer.data),
                             buffer.dataSize / sizeof(int16_t));
    return ReshapeData<int16_t>(data, dims, index);
  } else if (datatype == QNN_DATATYPE_UINT_16 ||
             datatype == QNN_DATATYPE_UFIXED_POINT_16) {
    absl::Span<uint16_t> data(static_cast<uint16_t*>(buffer.data),
                              buffer.dataSize / sizeof(uint16_t));
    return ReshapeData<uint16_t>(data, dims, index);
  } else if (datatype == QNN_DATATYPE_INT_32 ||
             datatype == QNN_DATATYPE_SFIXED_POINT_32) {
    absl::Span<int32_t> data(static_cast<int32_t*>(buffer.data),
                             buffer.dataSize / sizeof(int32_t));
    return ReshapeData<int32_t>(data, dims, index);
  } else if (datatype == QNN_DATATYPE_UINT_32 ||
             datatype == QNN_DATATYPE_UFIXED_POINT_32) {
    absl::Span<uint32_t> data(static_cast<uint32_t*>(buffer.data),
                              buffer.dataSize / sizeof(uint32_t));
    return ReshapeData<uint32_t>(data, dims, index);
  } else if (datatype == QNN_DATATYPE_FLOAT_32) {
    absl::Span<float> data(static_cast<float*>(buffer.data),
                           buffer.dataSize / sizeof(float));
    return ReshapeData<float>(data, dims, index);
  } else if (datatype == QNN_DATATYPE_FLOAT_64) {
    absl::Span<double> data(static_cast<double*>(buffer.data),
                            buffer.dataSize / sizeof(double));
    return ReshapeData<double>(data, dims, index);
  } else if (datatype == QNN_DATATYPE_INT_64) {
    absl::Span<int64_t> data(static_cast<int64_t*>(buffer.data),
                             buffer.dataSize / sizeof(int64_t));
    return ReshapeData<int64_t>(data, dims, index);
  } else if (datatype == QNN_DATATYPE_UINT_64) {
    absl::Span<uint64_t> data(static_cast<uint64_t*>(buffer.data),
                              buffer.dataSize / sizeof(uint64_t));
    return ReshapeData<uint64_t>(data, dims, index);
  } else {
    QNN_LOG_WARNING(
        "Datatype: %u is not supported for tensor_params in Qnn Json dump",
        datatype)
  }
}

}  // namespace

void DumpQnnJson(
    const absl::flat_hash_set<const TensorWrapper*>& tensor_wrappers,
    std::vector<OpWrapper>& graph_op_wrappers, const char* json_path) {
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
    // Record seen op type in a set.
    op_types.emplace(op_config.v1.typeName);
    // Create scalar_params and tensor_params.
    qnn_node_json["scalar_params"] = nlohmann::json::object();
    qnn_node_json["tensor_params"] = nlohmann::json::object();
    for (uint32_t i = 0; i < op_config.v1.numOfParams; ++i) {
      if (op_config.v1.params[i].paramType == QNN_PARAMTYPE_SCALAR) {
        AddScalarParams(op_config.v1.params[i], qnn_node_json);
      } else if (op_config.v1.params[i].paramType == QNN_PARAMTYPE_TENSOR) {
        nlohmann::json qnn_tensor_json = nlohmann::json::object();
        const Qnn_TensorV1_t& qnn_tensor =
            op_config.v1.params[i].tensorParam.v1;
        AddQnnTensor(qnn_tensor, qnn_tensor_json);
        qnn_tensor_json["data"] = GetData(
            qnn_tensor.dataType, qnn_tensor.clientBuf,
            absl::Span<uint32_t>(qnn_tensor.dimensions, qnn_tensor.rank));
        qnn_node_json["tensor_params"][op_config.v1.params[i].name]
                     [qnn_tensor.name] = qnn_tensor_json;
        // Record tensor param IDs to avoid adding them to graph tensors.
        param_tensor_ids.emplace(qnn_tensor.id);
      }
    }

    qnn_json["graph"]["nodes"][op_config.v1.name] = qnn_node_json;
  }
  // Dump Qnn Tensors.
  for (const TensorWrapper* tensor : tensor_wrappers) {
    // Skip tensor params.
    if (param_tensor_ids.count(tensor->GetQnnTensorId()) > 0) {
      continue;
    }
    // Create tensors.
    nlohmann::json qnn_tensor_json = nlohmann::json::object();
    AddQnnTensor(tensor->GetQnnTensor().v1, qnn_tensor_json);
    qnn_json["graph"]["tensors"][tensor->GetName()] = qnn_tensor_json;
  }
  // Dumpe Qnn op types.
  qnn_json["op_types"] = op_types;

  // Write the JSON string to a file.
  std::ofstream outFile(json_path);
  if (outFile.is_open()) {
    outFile << qnn_json.dump(4);
    outFile.close();
  } else {
    QNN_LOG_ERROR("Unable to open qnn_litert.json for writing.");
  }
}

}  // namespace qnn
