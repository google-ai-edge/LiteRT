// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/dump/dump_graph.h"

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <iostream>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/builders/matmul_op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "nlohmann/json.hpp"

namespace qnn {
namespace {
TEST(QnnJsonDump, MatMul) {
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
  const char* filename = "tmp.json";
  DumpQnnJson(created_tensors, graph_op_wrappers, filename);

  // Retrieve Qnn JSON file.
  std::ifstream input_file(filename);

  // Parse the JSON data.
  nlohmann::json qnn_ir;
  input_file >> qnn_ir;
  input_file.close();
  // Check op_types.
  ASSERT_EQ(qnn_ir.contains("op_types"), true);
  ASSERT_EQ(qnn_ir["op_types"].size(), 1);
  ASSERT_EQ(qnn_ir["op_types"][0], "MatMul");
  // Check tensors.
  ASSERT_EQ(qnn_ir.contains("graph"), true);
  ASSERT_EQ(qnn_ir["graph"].contains("tensors"), true);
  ASSERT_EQ(qnn_ir["graph"]["tensors"].size(), 3);
  const auto& tensor = qnn_ir["graph"]["tensors"];
  for (const auto& op_name : {"0_qnn", "1_qnn", "2_qnn"}) {
    ASSERT_EQ(tensor.contains(op_name), true);
    // Check dataFormat.
    ASSERT_EQ(tensor[op_name].contains("dataFormat"), true);
    ASSERT_EQ(tensor[op_name]["dataFormat"], 0);
    // Check data_type.
    ASSERT_EQ(tensor[op_name].contains("data_type"), true);
    ASSERT_EQ(tensor[op_name]["data_type"], 790);
    // Check dims.
    ASSERT_EQ(tensor[op_name].contains("dims"), true);
    ASSERT_EQ(tensor[op_name]["dims"].size(), 4);
    ASSERT_EQ(tensor[op_name]["dims"][0], 1);
    ASSERT_EQ(tensor[op_name]["dims"][1], 1);
    if (op_name == "0_qnn") {
      ASSERT_EQ(tensor[op_name]["dims"][2], 512);
      ASSERT_EQ(tensor[op_name]["dims"][3], 256);
    } else if (op_name == "1_qnn") {
      ASSERT_EQ(tensor[op_name]["dims"][2], 1280);
      ASSERT_EQ(tensor[op_name]["dims"][3], 256);
    } else {
      ASSERT_EQ(tensor[op_name]["dims"][2], 512);
      ASSERT_EQ(tensor[op_name]["dims"][3], 1280);
    }
    // Check quant_params.
    ASSERT_EQ(tensor[op_name].contains("quant_params"), true);
    const auto& quant_params = tensor[op_name]["quant_params"];
    ASSERT_EQ(quant_params.contains("definition"), true);
    ASSERT_EQ(quant_params["definition"], 1);
    ASSERT_EQ(quant_params.contains("encoding"), true);
    ASSERT_EQ(quant_params["encoding"], 0);
    ASSERT_EQ(quant_params.contains("scale_offset"), true);
    double scale = quant_params["scale_offset"]["scale"].get<double>();
    ASSERT_EQ(std::abs(scale - 1e-3) < 1e-4, true);
    ASSERT_EQ(quant_params["scale_offset"]["offset"], 0);
    // Check type.
    ASSERT_EQ(tensor[op_name].contains("type"), true);
    ASSERT_EQ(tensor[op_name]["type"], 3);
  }
  // Check nodes.
  ASSERT_EQ(qnn_ir["graph"].contains("nodes"), true);
  ASSERT_EQ(qnn_ir["graph"]["nodes"].size(), 1);
  ASSERT_EQ(qnn_ir["graph"]["nodes"].contains("MatMul_0"), true);
  const auto& node = qnn_ir["graph"]["nodes"]["MatMul_0"];
  // Check input_names.
  ASSERT_EQ(node.contains("input_names"), true);
  ASSERT_EQ(node["input_names"][0], "0_qnn");
  ASSERT_EQ(node["input_names"][1], "1_qnn");
  // Check output_names.
  ASSERT_EQ(node.contains("output_names"), true);
  ASSERT_EQ(node["output_names"][0], "2_qnn");
  // Check macs_per_inference.
  ASSERT_EQ(node.contains("macs_per_inference"), true);
  // Check scalar_params.
  ASSERT_EQ(node.contains("scalar_params"), true);
  ASSERT_EQ(node["scalar_params"].contains("transpose_in0"), true);
  ASSERT_EQ(node["scalar_params"]["transpose_in0"].contains("1288"), true);
  ASSERT_EQ(node["scalar_params"]["transpose_in0"]["1288"], 0);
  ASSERT_EQ(node["scalar_params"].contains("transpose_in1"), true);
  ASSERT_EQ(node["scalar_params"]["transpose_in1"].contains("1288"), true);
  ASSERT_EQ(node["scalar_params"]["transpose_in1"]["1288"], 1);
  // Check tensor_params.
  ASSERT_EQ(node.contains("tensor_params"), true);
  // Check type.
  ASSERT_EQ(node.contains("type"), true);
  ASSERT_EQ(node["type"], "MatMul");

  // ASSERT_EQ(std::remove(filename), 0);
}
}  // namespace
}  // namespace qnn
