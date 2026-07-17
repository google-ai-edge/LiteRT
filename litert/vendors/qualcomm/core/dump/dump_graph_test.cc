// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/dump/dump_graph.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
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
  auto matmul0 = CreateMatmulOp(input0, input1, output0, false, true);
  nlohmann::json qnn_op = SerializeOpToJson(matmul0.GetOpConfig());

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
  ASSERT_TRUE(qnn_op.contains("package"));
  EXPECT_EQ(qnn_op["package"], QNN_OP_PACKAGE_NAME_QTI_AISW);
  // param_map: scalar params map to 2, tensor params map to 1.
  ASSERT_TRUE(qnn_op.contains("param_map"));
  ASSERT_TRUE(qnn_op["param_map"].contains("transpose_in0"));
  EXPECT_EQ(qnn_op["param_map"]["transpose_in0"], 2);
  ASSERT_TRUE(qnn_op["param_map"].contains("transpose_in1"));
  EXPECT_EQ(qnn_op["param_map"]["transpose_in1"], 2);
  // macs_per_inference: hardcoded "0" placeholder.
  ASSERT_TRUE(qnn_op.contains("macs_per_inference"));
  EXPECT_EQ(qnn_op["macs_per_inference"], "0");
}

TEST(IrJsonDump, SerializeQuantParamToJson) {
  const Qnn_QuantizeParams_t quant_params = {
      QNN_DEFINITION_DEFINED,                 /*encodingDefinition*/
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET, /*quantizationEncoding*/
      {{
          0.003f, /*scale*/
          0       /*offset*/
      }}};
  // UFIXED_POINT_8: unsigned 8-bit, min=0, max=255*scale, is_symmetric=false.
  nlohmann::json quant_info =
      SerializeQuantParamToJson(quant_params, QNN_DATATYPE_UFIXED_POINT_8);
  ASSERT_TRUE(quant_info.contains("definition"));
  ASSERT_TRUE(quant_info.contains("encoding"));
  ASSERT_TRUE(quant_info.contains("scale_offset"));
  const auto& so = quant_info["scale_offset"];
  ASSERT_TRUE(so.contains("scale"));
  EXPECT_EQ(so["scale"], 0.003f);
  ASSERT_TRUE(so.contains("offset"));
  EXPECT_EQ(so["offset"], 0);
  ASSERT_TRUE(so.contains("bitwidth"));
  EXPECT_EQ(so["bitwidth"], 8u);
  ASSERT_TRUE(so.contains("minimum"));
  EXPECT_FLOAT_EQ(so["minimum"].get<float>(), 0.0f);
  ASSERT_TRUE(so.contains("maximum"));
  EXPECT_FLOAT_EQ(so["maximum"].get<float>(), 0.003f * 255);
  ASSERT_TRUE(so.contains("is_symmetric"));
  EXPECT_FALSE(so["is_symmetric"].get<bool>());
  ASSERT_TRUE(so.contains("is_fixed_point"));
  EXPECT_TRUE(so["is_fixed_point"].get<bool>());
  // is_overridden: always true since quant params come from TFLite.
  ASSERT_TRUE(quant_info.contains("is_overridden"));
  EXPECT_TRUE(quant_info["is_overridden"].get<bool>());

  // SFIXED_POINT_8 with offset=0: signed symmetric, min=-128*scale,
  // max=127*scale.
  const Qnn_QuantizeParams_t signed_quant_params = {
      QNN_DEFINITION_DEFINED,
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
      {{0.003f, 0}}};
  nlohmann::json signed_info = SerializeQuantParamToJson(
      signed_quant_params, QNN_DATATYPE_SFIXED_POINT_8);
  EXPECT_TRUE(signed_info["scale_offset"]["is_symmetric"].get<bool>());
  EXPECT_FLOAT_EQ(signed_info["scale_offset"]["minimum"].get<float>(),
                  0.003f * (-128));
  EXPECT_FLOAT_EQ(signed_info["scale_offset"]["maximum"].get<float>(),
                  0.003f * 127);

  // UFIXED_POINT_8 with a non-zero offset: dequantization uses
  // real = scale * (quantized + offset), so min/max shift by offset and the
  // encoding is asymmetric.
  const Qnn_QuantizeParams_t offset_quant_params = {
      QNN_DEFINITION_DEFINED,
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
      {{0.003f, -128}}};
  nlohmann::json offset_info = SerializeQuantParamToJson(
      offset_quant_params, QNN_DATATYPE_UFIXED_POINT_8);
  EXPECT_EQ(offset_info["scale_offset"]["offset"], -128);
  EXPECT_FALSE(offset_info["scale_offset"]["is_symmetric"].get<bool>());
  EXPECT_FLOAT_EQ(offset_info["scale_offset"]["minimum"].get<float>(),
                  0.003f * (0 + -128));
  EXPECT_FLOAT_EQ(offset_info["scale_offset"]["maximum"].get<float>(),
                  0.003f * (255 + -128));

  // Check unsupported (non-fixed-point) data type.
  nlohmann::json unsupported_info =
      SerializeQuantParamToJson(quant_params, QNN_DATATYPE_FLOAT_32);
  ASSERT_TRUE(unsupported_info.contains("definition"));
  ASSERT_TRUE(unsupported_info.contains("encoding"));
  EXPECT_FALSE(unsupported_info.contains("scale_offset"));
  EXPECT_FALSE(unsupported_info.contains("is_overridden"));
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

  ASSERT_TRUE(tensor_info.contains("unquantized_data_type"));
  EXPECT_EQ(tensor_info["unquantized_data_type"],
            static_cast<uint32_t>(QNN_DATATYPE_UINT_32));

  ASSERT_TRUE(tensor_info.contains("permute_order_to_src"));
  ASSERT_EQ(tensor_info["permute_order_to_src"].size(), 1u);
  EXPECT_EQ(tensor_info["permute_order_to_src"][0], 0u);

  ASSERT_TRUE(tensor_info.contains("is_dynamic_dims"));
  EXPECT_TRUE(tensor_info["is_dynamic_dims"].empty());

  ASSERT_TRUE(tensor_info.contains("is_quantizable"));
  EXPECT_FALSE(tensor_info["is_quantizable"].get<bool>());

  ASSERT_TRUE(tensor_info.contains("is_updateable"));
  EXPECT_FALSE(tensor_info["is_updateable"].get<bool>());

  nlohmann::json data = SerializeTensorParamToJson(qnn_tensor);
  ASSERT_EQ(data.size(), axes.size());
  EXPECT_EQ(data[0], axes[0]);
}

TEST(IrJsonDump, SerializeTensorToJsonQuantizableFloat32) {
  std::array<uint32_t, 1> dims{1};
  const Qnn_TensorV1_t qnn_tensor = {
      1u,                                 /*id*/
      "float_qnn",                        /*name*/
      QNN_TENSOR_TYPE_NATIVE,             /*type*/
      QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, /*dataFormat*/
      QNN_DATATYPE_FLOAT_32,              /*dataType*/
      QNN_QUANTIZE_PARAMS_INIT,           /*quantizeParams*/
      1u,                                 /*rank*/
      dims.data(),                        /*dimensions*/
      QNN_TENSORMEMTYPE_RAW,              /*memType*/
      {{nullptr, 0}}};
  nlohmann::json tensor_info = SerializeTensorToJson(qnn_tensor);
  // FLOAT_32 tensors are the only ones marked quantizable, and a
  // non-fixed-point data type maps unquantized_data_type back to itself.
  ASSERT_TRUE(tensor_info.contains("is_quantizable"));
  EXPECT_TRUE(tensor_info["is_quantizable"].get<bool>());
  EXPECT_EQ(tensor_info["unquantized_data_type"],
            static_cast<uint32_t>(QNN_DATATYPE_FLOAT_32));
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
  graph_op_wrappers.emplace_back(
      CreateMatmulOp(input0, input1, output0, false, true));
  absl::flat_hash_set<const ::qnn::TensorWrapper*> created_tensors;
  for (auto& op_wrapper : graph_op_wrappers) {
    for (const auto& tensor_wrapper_ref : op_wrapper.GetAllTensors()) {
      created_tensors.emplace(&tensor_wrapper_ref.get());
    }
  }
#ifdef __ANDROID__
  constexpr const char* kGraphDir = "/data/local/tmp/";
#else
  constexpr const char* kGraphDir = "/tmp/";
#endif
  const auto graph_file = std::filesystem::path(kGraphDir) / "qnn_graph.json";
  DumpIrJson(created_tensors, graph_op_wrappers, kGraphDir, "qnn_graph");

  // Retrieve Qnn JSON file.
  std::ifstream input_file(graph_file);
  ASSERT_TRUE(input_file.is_open());

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
    ASSERT_TRUE(quant_params["scale_offset"].contains("bitwidth"));
    EXPECT_EQ(quant_params["scale_offset"]["bitwidth"], 16u);
    ASSERT_TRUE(quant_params["scale_offset"].contains("is_symmetric"));
    EXPECT_TRUE(quant_params["scale_offset"]["is_symmetric"].get<bool>());
    ASSERT_TRUE(quant_params["scale_offset"].contains("is_fixed_point"));
    EXPECT_TRUE(quant_params["scale_offset"]["is_fixed_point"].get<bool>());
    ASSERT_TRUE(quant_params.contains("is_overridden"));
    EXPECT_TRUE(quant_params["is_overridden"].get<bool>());
    // Check tensor.
    ASSERT_TRUE(tensor[op_name].contains("unquantized_data_type"));
    EXPECT_EQ(tensor[op_name]["unquantized_data_type"],
              static_cast<uint32_t>(QNN_DATATYPE_FLOAT_32));
    ASSERT_TRUE(tensor[op_name].contains("permute_order_to_src"));
    ASSERT_EQ(tensor[op_name]["permute_order_to_src"].size(), 4u);
    for (uint32_t i = 0; i < 4; ++i) {
      EXPECT_EQ(tensor[op_name]["permute_order_to_src"][i], i);
    }
    ASSERT_TRUE(tensor[op_name].contains("is_dynamic_dims"));
    EXPECT_TRUE(tensor[op_name]["is_dynamic_dims"].empty());
    ASSERT_TRUE(tensor[op_name].contains("is_quantizable"));
    // is_quantizable is true only for FLOAT_32 tensors; these are
    // SFIXED_POINT_16.
    EXPECT_FALSE(tensor[op_name]["is_quantizable"].get<bool>());
    ASSERT_TRUE(tensor[op_name].contains("is_updateable"));
    EXPECT_FALSE(tensor[op_name]["is_updateable"].get<bool>());
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
  EXPECT_EQ(node["macs_per_inference"], "0");
  // Check package.
  ASSERT_TRUE(node.contains("package"));
  EXPECT_EQ(node["package"], QNN_OP_PACKAGE_NAME_QTI_AISW);
  // Check param_map. Scalar params map to 2, tensor params map to 1.
  ASSERT_TRUE(node.contains("param_map"));
  EXPECT_EQ(node["param_map"]["transpose_in0"], 2);
  EXPECT_EQ(node["param_map"]["transpose_in1"], 2);
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

  ASSERT_TRUE(std::filesystem::remove(graph_file));
}

TEST(IrJsonDump, ParamsCount) {
  TensorPool tensor_pool;
  const QuantizeParamsWrapperVariant quant_param{
      ScaleOffsetQuantizeParamsWrapper{0.001, 0}};

  // MatMul with two static weight tensors so params_count is emitted. Element
  // counts: 3072 and 1024, total 4096, i.e. 75% and 25% of the parameters.
  static constexpr std::array<int16_t, 3072> weight0{};
  static constexpr std::array<int16_t, 1024> weight1{};
  auto& input0 = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 12, 256},
      weight0.size() * sizeof(int16_t), weight0.data());
  auto& input1 = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 4, 256},
      weight1.size() * sizeof(int16_t), weight1.data());
  auto& output0 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                 quant_param, {1, 1, 12, 4});
  std::vector<OpWrapper> graph_op_wrappers =
      MakeVector(CreateMatmulOp(input0, input1, output0, false, true));
  absl::flat_hash_set<const ::qnn::TensorWrapper*> created_tensors;
  for (auto& op_wrapper : graph_op_wrappers) {
    for (const auto& tensor_wrapper_ref : op_wrapper.GetAllTensors()) {
      created_tensors.emplace(&tensor_wrapper_ref.get());
    }
  }
#ifdef __ANDROID__
  constexpr const char* kGraphDir = "/data/local/tmp/";
#else
  constexpr const char* kGraphDir = "/tmp/";
#endif
  const auto graph_file =
      std::filesystem::path(kGraphDir) / "qnn_params_count.json";
  DumpIrJson(created_tensors, graph_op_wrappers, kGraphDir, "qnn_params_count");

  std::ifstream input_file(graph_file);
  ASSERT_TRUE(input_file.is_open());
  nlohmann::json qnn_ir;
  input_file >> qnn_ir;
  input_file.close();

  const auto& tensors = qnn_ir["graph"]["tensors"];
  // Static tensors report "count (percentage%)"; the native output tensor has
  // no params_count field.
  ASSERT_TRUE(tensors["0_qnn"].contains("params_count"));
  EXPECT_EQ(tensors["0_qnn"]["params_count"], "3072 (75%)");
  ASSERT_TRUE(tensors["1_qnn"].contains("params_count"));
  EXPECT_EQ(tensors["1_qnn"]["params_count"], "1024 (25%)");
  EXPECT_FALSE(tensors["2_qnn"].contains("params_count"));

  ASSERT_TRUE(std::filesystem::remove(graph_file));
}
}  // namespace
}  // namespace qnn
