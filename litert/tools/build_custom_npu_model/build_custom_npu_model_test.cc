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

#include "litert/tools/build_custom_npu_model/build_custom_npu_model.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/dispatch_op_schema.h"

namespace litert::tools {
namespace {

// =============================================================================
// Success Test Cases
// =============================================================================

TEST(BuildCustomNpuModelTest, ParseElementTypeSuccessTest) {
  auto res1 = ParseElementType("f32");
  ASSERT_TRUE(res1.HasValue());
  EXPECT_EQ(res1.Value(), kLiteRtElementTypeFloat32);

  auto res2 = ParseElementType("i32");
  ASSERT_TRUE(res2.HasValue());
  EXPECT_EQ(res2.Value(), kLiteRtElementTypeInt32);

  auto res3 = ParseElementType("u8");
  ASSERT_TRUE(res3.HasValue());
  EXPECT_EQ(res3.Value(), kLiteRtElementTypeUInt8);

  auto res4 = ParseElementType("i8");
  ASSERT_TRUE(res4.HasValue());
  EXPECT_EQ(res4.Value(), kLiteRtElementTypeInt8);

  auto res5 = ParseElementType("i16");
  ASSERT_TRUE(res5.HasValue());
  EXPECT_EQ(res5.Value(), kLiteRtElementTypeInt16);

  auto res6 = ParseElementType("f16");
  ASSERT_TRUE(res6.HasValue());
  EXPECT_EQ(res6.Value(), kLiteRtElementTypeFloat16);

  auto res7 = ParseElementType("bool");
  ASSERT_TRUE(res7.HasValue());
  EXPECT_EQ(res7.Value(), kLiteRtElementTypeBool);
}

TEST(BuildCustomNpuModelTest, ParseDimensionsAxBxCxDFilesTest) {
  // Dimension format '1x224x224x3'
  auto dims_4d = ParseDimensions("1x224x224x3");
  ASSERT_TRUE(dims_4d.HasValue());
  EXPECT_EQ(*dims_4d, (std::vector<int32_t>{1, 224, 224, 3}));

  // Dimension format '1x10'
  auto dims_2d = ParseDimensions("1x10");
  ASSERT_TRUE(dims_2d.HasValue());
  EXPECT_EQ(*dims_2d, (std::vector<int32_t>{1, 10}));

  // Single dimension '512'
  auto dims_1d = ParseDimensions("512");
  ASSERT_TRUE(dims_1d.HasValue());
  EXPECT_EQ(*dims_1d, (std::vector<int32_t>{512}));
}

TEST(BuildCustomNpuModelTest, ParseTensorInfoListCustomNamesTest) {
  // Test multiple input parsing with custom names
  auto list = ParseTensorInfoList("1x224x224x3,1x10,1x512", "f32,i32,u8",
                                  "image,mask,query", "input");
  ASSERT_TRUE(list.HasValue());
  ASSERT_EQ(list->size(), 3);

  EXPECT_EQ((*list)[0].name, "image");
  EXPECT_EQ((*list)[0].element_type, kLiteRtElementTypeFloat32);
  EXPECT_EQ((*list)[0].dimensions, (std::vector<int32_t>{1, 224, 224, 3}));

  EXPECT_EQ((*list)[1].name, "mask");
  EXPECT_EQ((*list)[1].element_type, kLiteRtElementTypeInt32);
  EXPECT_EQ((*list)[1].dimensions, (std::vector<int32_t>{1, 10}));

  EXPECT_EQ((*list)[2].name, "query");
  EXPECT_EQ((*list)[2].element_type, kLiteRtElementTypeUInt8);
  EXPECT_EQ((*list)[2].dimensions, (std::vector<int32_t>{1, 512}));

  // Test default naming fallback when names flag is empty
  auto default_names =
      ParseTensorInfoList("1x1000,1x100", "f32,f16", "", "output");
  ASSERT_TRUE(default_names.HasValue());
  ASSERT_EQ(default_names->size(), 2);
  EXPECT_EQ((*default_names)[0].name, "output_0");
  EXPECT_EQ((*default_names)[1].name, "output_1");
}

TEST(BuildCustomNpuModelTest, MultiInputMultiOutputModelSignatureTest) {
  const std::string dummy_bytecode =
      "MULTI_INPUT_MULTI_OUTPUT_NPU_BYTECODE_PAYLOAD_123456789";
  BufferRef<uint8_t> bytecode_ref(
      reinterpret_cast<const uint8_t*>(dummy_bytecode.data()),
      dummy_bytecode.size());

  BuildCustomNpuModelOptions options;
  options.soc_manufacturer = "Qualcomm";
  options.soc_model = "SM8750";
  options.entry_point_name = "multi_io_graph";
  options.signature_key = "serving_default";

  TensorInfo in0;
  in0.name = "image";
  in0.element_type = kLiteRtElementTypeFloat32;
  in0.dimensions = {1, 224, 224, 3};
  options.input_tensors.push_back(in0);

  TensorInfo in1;
  in1.name = "mask";
  in1.element_type = kLiteRtElementTypeInt32;
  in1.dimensions = {1, 128};
  options.input_tensors.push_back(in1);

  TensorInfo out0;
  out0.name = "logits";
  out0.element_type = kLiteRtElementTypeFloat32;
  out0.dimensions = {1, 1000};
  options.output_tensors.push_back(out0);

  TensorInfo out1;
  out1.name = "embeddings";
  out1.element_type = kLiteRtElementTypeFloat16;
  out1.dimensions = {1, 128};
  options.output_tensors.push_back(out1);

  auto serialized_res = BuildCustomNpuModelMemory(options, bytecode_ref);
  ASSERT_TRUE(serialized_res.HasValue()) << serialized_res.Error().Message();
  auto& serialized = *serialized_res;

  BufferRef<uint8_t> serialized_buf_ref(serialized.Data(), serialized.Size());
  auto model_res = ExtendedModel::CreateFromBuffer(serialized_buf_ref);
  ASSERT_TRUE(model_res.HasValue()) << model_res.Error().Message();
  auto& model = *model_res;

  EXPECT_EQ(model.NumSubgraphs(), 1);
  auto subgraph_res = model.Subgraph(0);
  ASSERT_TRUE(subgraph_res.HasValue());
  auto& subgraph = *subgraph_res;

  EXPECT_EQ(subgraph.Inputs().size(), 2);
  EXPECT_EQ(subgraph.Outputs().size(), 2);

  // Verify Tensor Names
  EXPECT_EQ(subgraph.Inputs()[0].Name(), "image");
  EXPECT_EQ(subgraph.Inputs()[1].Name(), "mask");
  EXPECT_EQ(subgraph.Outputs()[0].Name(), "logits");
  EXPECT_EQ(subgraph.Outputs()[1].Name(), "embeddings");

  // Verify Signature Key
  auto sig_res = model.FindSignature("serving_default");
  ASSERT_TRUE(sig_res.HasValue())
      << "Signature 'serving_default' not found in model";

  auto ops = subgraph.Ops();
  EXPECT_EQ(ops.size(), 1);
  auto op = ops.front();
  EXPECT_EQ(op.Code(), kLiteRtOpCodeTflCustom);
  EXPECT_EQ(op.Inputs().size(), 2);
  EXPECT_EQ(op.Outputs().size(), 2);

  // Inspect DispatchOp custom options
  auto custom_opts = op.CustomOptions();
  ASSERT_TRUE(custom_opts.HasValue());
  BufferRef<uint8_t> custom_opts_buf(custom_opts->data(), custom_opts->size());
  auto dispatch_opts = internal::GetDispatchOpOptions(custom_opts_buf);
  EXPECT_EQ(dispatch_opts.name, "multi_io_graph");
  EXPECT_EQ(dispatch_opts.bytecode_size, dummy_bytecode.size());

  // Verify raw payload matches
  const uint8_t* payload_ptr =
      serialized.Data() + dispatch_opts.bytecode_offset;
  EXPECT_EQ(
      std::memcmp(payload_ptr, dummy_bytecode.data(), dummy_bytecode.size()),
      0);
}

// =============================================================================
// Failure / Edge Case Test Cases
// =============================================================================

TEST(BuildCustomNpuModelTest, ParseInvalidElementTypeFailsTest) {
  auto invalid_type = ParseElementType("float32");
  EXPECT_FALSE(invalid_type.HasValue());
  EXPECT_NE(invalid_type.Error().Message().find("Unsupported data type"),
            std::string::npos);
}

TEST(BuildCustomNpuModelTest, ParseInvalidDimensionsFailsTest) {
  // Zero dimension
  auto zero_dim = ParseDimensions("1x0x224");
  EXPECT_FALSE(zero_dim.HasValue());

  // Negative dimension
  auto neg_dim = ParseDimensions("1x-5x224");
  EXPECT_FALSE(neg_dim.HasValue());

  // Non-numeric text
  auto text_dim = ParseDimensions("1xABCx3");
  EXPECT_FALSE(text_dim.HasValue());
}

TEST(BuildCustomNpuModelTest, NonExistentInputFileFailsTest) {
  BuildCustomNpuModelOptions options;
  options.npu_bytecode_path = "/tmp/non_existent_file_12345.bin";
  options.output_model_path = "/tmp/dummy_out.tflite";
  options.input_tensors.push_back(
      {"input_0", kLiteRtElementTypeFloat32, {1, 10}});
  options.output_tensors.push_back(
      {"output_0", kLiteRtElementTypeFloat32, {1, 10}});

  auto res = BuildCustomNpuModel(options);
  EXPECT_FALSE(res.HasValue());
  EXPECT_NE(res.Error().Message().find("Cannot open binary file"),
            std::string::npos)
      << "Actual error message: " << res.Error().Message();
}

TEST(BuildCustomNpuModelTest, EmptyBytecodeBufferFailsTest) {
  BufferRef<uint8_t> empty_bytecode(static_cast<const uint8_t*>(nullptr), 0);

  BuildCustomNpuModelOptions options;
  options.input_tensors.push_back(
      {"input_0", kLiteRtElementTypeFloat32, {1, 10}});
  options.output_tensors.push_back(
      {"output_0", kLiteRtElementTypeFloat32, {1, 10}});

  auto res = BuildCustomNpuModelMemory(options, empty_bytecode);
  EXPECT_FALSE(res.HasValue());
  EXPECT_NE(res.Error().Message().find("bytecode buffer is empty"),
            std::string::npos);
}

TEST(BuildCustomNpuModelTest, MissingTensorSpecsFailsTest) {
  const std::string dummy_bytecode_file = "/tmp/dummy_bytecode_test.bin";
  {
    std::ofstream ofs(dummy_bytecode_file, std::ios::binary);
    ofs << "DUMMY_BYTECODE_CONTENT";
  }

  BuildCustomNpuModelOptions options;
  options.npu_bytecode_path = dummy_bytecode_file;
  options.output_model_path = "/tmp/dummy_out.tflite";

  auto res = BuildCustomNpuModel(options);
  EXPECT_FALSE(res.HasValue());
  EXPECT_NE(res.Error().Message().find(
                "Input and output tensor specifications must be provided"),
            std::string::npos)
      << "Actual error message: " << res.Error().Message();
}

TEST(BuildCustomNpuModelTest, InvalidOutputFileLocationFailsTest) {
  const std::string dummy_bytecode_file = "/tmp/dummy_bytecode_test2.bin";
  {
    std::ofstream ofs(dummy_bytecode_file, std::ios::binary);
    ofs << "DUMMY_BYTECODE_CONTENT";
  }

  BuildCustomNpuModelOptions options;
  options.npu_bytecode_path = dummy_bytecode_file;
  options.output_model_path = "/non_existent_directory_12345/out.tflite";
  options.input_tensors.push_back(
      {"input_0", kLiteRtElementTypeFloat32, {1, 10}});
  options.output_tensors.push_back(
      {"output_0", kLiteRtElementTypeFloat32, {1, 10}});

  auto res = BuildCustomNpuModel(options);
  EXPECT_FALSE(res.HasValue());
  EXPECT_NE(res.Error().Message().find("Cannot open output file"),
            std::string::npos);
}

}  // namespace
}  // namespace litert::tools
