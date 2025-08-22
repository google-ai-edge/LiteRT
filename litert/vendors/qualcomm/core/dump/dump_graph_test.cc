// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/dump/dump_graph.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @com_github_nlohmann_json
#include "litert/vendors/qualcomm/core/builders/matmul_op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnTypes.h"  // from @qairt

namespace qnn {
namespace {

TEST(IrJsonDump, SerializeOpToJson) {
  TensorPool tensor_pool;
  std::vector<OpWrapper> graph_op_wrappers;
  QuantizeParamsWrapperVariant quant_param;
  quant_param.emplace<ScaleOffsetQuantizeParamsWrapper>(0.001, 0);

  auto& input0 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 1, 512, 256});
  auto& input1 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 1, 1280, 256});
  auto& output0 = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 512, 1280});
  auto matmul0 =
      BuildMatmulOp(tensor_pool, {input0, input1}, {output0}, false, true);
  nlohmann::json qnn_op = SerializeOpToJson(matmul0[0].GetOpConfig());

  ASSERT_TRUE(qnn_op.contains("input_names"));
  EXPECT_EQ(qnn_op["input_names"][0], "0_qnn");
  EXPECT_EQ(qnn_op["input_names"][1], "1_qnn");
  ASSERT_TRUE(qnn_op.contains("output_names"));
  EXPECT_EQ(qnn_op["output_names"][0], "2_qnn");
  ASSERT_TRUE(qnn_op.contains("scalar_params"));
  ASSERT_TRUE(qnn_op["scalar_params"].contains("transpose_in0"));
  ASSERT_TRUE(qnn_op["scalar_params"].contains("transpose_in1"));
  ASSERT_TRUE(qnn_op.contains("tensor_params"));
  ASSERT_TRUE(qnn_op.contains("type"));
  EXPECT_EQ(qnn_op["type"], "MatMul");
}

TEST(IrJsonDump, SerializeQuantParamToJson) {
  const Qnn_QuantizeParams_t quant_params = {
      QNN_DEFINITION_DEFINED,                 /*encodingDefinition*/
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET, /*quantizationEncoding*/
      {{
          0.003f, /*scale*/
          0       /*offset*/
      }}};
  nlohmann::json quant_info = SerializeQuantParamToJson(quant_params);
  ASSERT_TRUE(quant_info.contains("definition"));
  ASSERT_TRUE(quant_info.contains("encoding"));
  ASSERT_TRUE(quant_info.contains("scale_offset"));
  ASSERT_TRUE(quant_info["scale_offset"].contains("scale"));
  ASSERT_TRUE(quant_info["scale_offset"].contains("offset"));
  EXPECT_EQ(quant_info["scale_offset"]["scale"], 0.003f);
  EXPECT_EQ(quant_info["scale_offset"]["offset"], 0);
}

TEST(IrJsonDump, SerializeScalarParamToJson) {
  const Qnn_Scalar_t qnn_scalar = {QNN_DATATYPE_FLOAT_32, /*dataType*/
                                   {
                                       1e-6f /*floatValue*/
                                   }};
  nlohmann::json tensor_info = SerializeScalarParamToJson(qnn_scalar);
  ASSERT_TRUE(tensor_info.contains(std::to_string(qnn_scalar.dataType)));
  EXPECT_EQ(tensor_info[std::to_string(qnn_scalar.dataType)], 1e-6f);
}

TEST(IrJsonDump, SerializeTensorAndParamToJson) {
  std::array<uint32_t, 1> axes = {3};
  std::array<uint32_t, 1> dims = {1};
  const Qnn_TensorV1_t qnn_tensor = {
      79u,                                /*id*/
      "83_qnn",                           /*name*/
      QNN_TENSOR_TYPE_STATIC,             /*type*/
      QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, /*dataFormat*/
      QNN_DATATYPE_UINT_32,               /*dataType*/
      QNN_QUANTIZE_PARAMS_INIT,           /*quantizeParams*/
      1u,                                 /*rank*/
      dims.data(),                        /*dimensions*/
      QNN_TENSORMEMTYPE_RAW,              /*memType*/
      {{
          axes.data(),                  /*data*/
          axes.size() * sizeof(axes[0]) /*dataSize*/
      }}};
  nlohmann::json tensor_info = SerializeTensorToJson(qnn_tensor);
  EXPECT_EQ(tensor_info["dataFormat"], qnn_tensor.dataFormat);
  EXPECT_EQ(tensor_info["data_type"], qnn_tensor.dataType);
  EXPECT_EQ(tensor_info["id"], qnn_tensor.id);
  EXPECT_EQ(tensor_info["type"], qnn_tensor.type);
  ASSERT_EQ(tensor_info["dims"].size(), dims.size());
  EXPECT_EQ(tensor_info["dims"][0], dims[0]);

  nlohmann::json data = SerializeTensorParamToJson(qnn_tensor);
  ASSERT_EQ(data.size(), axes.size());
  EXPECT_EQ(data[0], axes[0]);
}

TEST(IrJsonDump, MatMul) {
  TensorPool tensor_pool;
  std::vector<OpWrapper> graph_op_wrappers;
  QuantizeParamsWrapperVariant quant_param;
  quant_param.emplace<ScaleOffsetQuantizeParamsWrapper>(0.001, 0);

  auto& input0 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 1, 512, 256});
  auto& input1 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 1, 1280, 256});
  auto& output0 = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 512, 1280});
  auto matmul0 =
      BuildMatmulOp(tensor_pool, {input0, input1}, {output0}, false, true);
  std::move(matmul0.begin(), matmul0.end(),
            std::back_inserter(graph_op_wrappers));
  absl::flat_hash_set<const ::qnn::TensorWrapper*> created_tensors;
  for (auto& op_wrapper : graph_op_wrappers) {
    for (const auto& tensor_wrapper_ref : op_wrapper.GetAllTensors()) {
      created_tensors.emplace(&tensor_wrapper_ref.get());
    }
  }
  const char* filename = "/tmp/qnn_graph.json";
  DumpIrJson(created_tensors, graph_op_wrappers, "/tmp/", "qnn_graph");

  // Retrieve Qnn JSON file.
  std::ifstream input_file(filename);

  // Parse the JSON data.
  nlohmann::json qnn_ir;
  input_file >> qnn_ir;
  input_file.close();
  // Check op_types.
  ASSERT_TRUE(qnn_ir.contains("op_types"));
  ASSERT_EQ(qnn_ir["op_types"].size(), 1);
  EXPECT_EQ(qnn_ir["op_types"][0], "MatMul");
  // Check tensors.
  ASSERT_TRUE(qnn_ir.contains("graph"));
  ASSERT_TRUE(qnn_ir["graph"].contains("tensors"));
  ASSERT_EQ(qnn_ir["graph"]["tensors"].size(), 3);
  const auto& tensor = qnn_ir["graph"]["tensors"];
  for (const auto& op_name : {"0_qnn", "1_qnn", "2_qnn"}) {
    ASSERT_TRUE(tensor.contains(op_name));
    // Check dataFormat.
    ASSERT_TRUE(tensor[op_name].contains("dataFormat"));
    EXPECT_EQ(tensor[op_name]["dataFormat"], 0);
    // Check data_type.
    ASSERT_TRUE(tensor[op_name].contains("data_type"));
    EXPECT_EQ(tensor[op_name]["data_type"], 790);
    // Check dims.
    ASSERT_TRUE(tensor[op_name].contains("dims"));
    ASSERT_EQ(tensor[op_name]["dims"].size(), 4);
    EXPECT_EQ(tensor[op_name]["dims"][0], 1);
    EXPECT_EQ(tensor[op_name]["dims"][1], 1);
    if (strcmp(op_name, "0_qnn") == 0) {
      EXPECT_EQ(tensor[op_name]["dims"][2], 512);
      EXPECT_EQ(tensor[op_name]["dims"][3], 256);
    } else if (strcmp(op_name, "1_qnn") == 0) {
      EXPECT_EQ(tensor[op_name]["dims"][2], 1280);
      EXPECT_EQ(tensor[op_name]["dims"][3], 256);
    } else {
      EXPECT_EQ(tensor[op_name]["dims"][2], 512);
      EXPECT_EQ(tensor[op_name]["dims"][3], 1280);
    }
    // Check quant_params.
    ASSERT_TRUE(tensor[op_name].contains("quant_params"));
    const auto& quant_params = tensor[op_name]["quant_params"];
    ASSERT_TRUE(quant_params.contains("definition"));
    EXPECT_EQ(quant_params["definition"], 1);
    ASSERT_TRUE(quant_params.contains("encoding"));
    EXPECT_EQ(quant_params["encoding"], 0);
    ASSERT_TRUE(quant_params.contains("scale_offset"));
    double scale = quant_params["scale_offset"]["scale"].get<double>();
    EXPECT_EQ(std::abs(scale - 1e-3) < 1e-4, true);
    EXPECT_EQ(quant_params["scale_offset"]["offset"], 0);
    // Check type.
    ASSERT_TRUE(tensor[op_name].contains("type"));
    EXPECT_EQ(tensor[op_name]["type"], 3);
  }
  // Check nodes.
  ASSERT_TRUE(qnn_ir["graph"].contains("nodes"));
  ASSERT_EQ(qnn_ir["graph"]["nodes"].size(), 1);
  auto it = qnn_ir["graph"]["nodes"].begin();
  const auto& node = it.value();
  // Check input_names.
  ASSERT_TRUE(node.contains("input_names"));
  EXPECT_EQ(node["input_names"][0], "0_qnn");
  EXPECT_EQ(node["input_names"][1], "1_qnn");
  // Check output_names.
  ASSERT_TRUE(node.contains("output_names"));
  EXPECT_EQ(node["output_names"][0], "2_qnn");
  // Check macs_per_inference.
  ASSERT_TRUE(node.contains("macs_per_inference"));
  // Check scalar_params.
  ASSERT_TRUE(node.contains("scalar_params"));
  ASSERT_TRUE(node["scalar_params"].contains("transpose_in0"));
  ASSERT_TRUE(node["scalar_params"]["transpose_in0"].contains("1288"));
  EXPECT_EQ(node["scalar_params"]["transpose_in0"]["1288"], 0);
  ASSERT_TRUE(node["scalar_params"].contains("transpose_in1"));
  ASSERT_TRUE(node["scalar_params"]["transpose_in1"].contains("1288"));
  EXPECT_EQ(node["scalar_params"]["transpose_in1"]["1288"], 1);
  // Check tensor_params.
  ASSERT_TRUE(node.contains("tensor_params"));
  // Check type.
  ASSERT_TRUE(node.contains("type"));
  EXPECT_EQ(node["type"], "MatMul");

  ASSERT_EQ(std::remove(filename), 0);
}
}  // namespace
}  // namespace qnn
