// Copyright 2024 Google LLC.
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

#include "litert/tools/apply_plugin.h"

#include <array>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_model.h"
#include "litert/core/build_stamp.h"
#include "litert/core/dispatch_op_schema.h"
#include "litert/core/model/model.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/tools/outstream.h"
#include "tflite/schema/schema_generated.h"

namespace litert::tools {
namespace {

using ::litert::internal::kLiteRtBuildStampKey;
using ::litert::internal::ParseBuildStamp;
using ::litert::testing::GetLiteRtPath;
using ::testing::HasSubstr;
using ::testing::litert::IsError;

static constexpr absl::string_view kSocManufacturer = "ExampleSocManufacturer";

static constexpr absl::string_view kSocModel = "ExampleSocModel";

absl::string_view TestModelPath(absl::string_view filename) {
  static char kModelPath[512] = {};
  const auto model_path = ::litert::testing::GetTestFilePath(filename);
  ABSL_CHECK(model_path.size() < 512);
  model_path.copy(kModelPath, model_path.size(), 0);
  return kModelPath;
}

ApplyPluginRun::Ptr MakeBaseRun(
    ApplyPluginRun::Cmd cmd, absl::string_view model_path = "one_mul.tflite") {
  auto run = std::make_unique<ApplyPluginRun>();
  run->cmd = cmd;
  run->lib_search_paths.push_back(GetLiteRtPath("vendors/examples/"));
  run->model.emplace(TestModelPath(model_path));
  run->soc_manufacturer.emplace(std::string(kSocManufacturer));
  run->soc_models.push_back(std::string(kSocModel));
  run->outs.clear();
  run->dump_out = UserStream(std::cerr);
  return run;
}

TEST(TestApplyPluginTool, TestInfoBadConfig) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::INFO);
  run->lib_search_paths.clear();
  EXPECT_THAT(ApplyPlugin(std::move(run)),
              IsError(kLiteRtStatusErrorInvalidToolConfig));
}

TEST(TestApplyPluginTool, TestInfo) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::INFO);
  std::stringstream out;
  run->outs.push_back(out);
  LITERT_ASSERT_OK(ApplyPlugin(std::move(run)));
  EXPECT_THAT(out.str(),
              ::testing::HasSubstr(
                  "< LiteRtCompilerPlugin > \"ExampleSocManufacturer\" | "
                  "\"ExampleSocModel\""));
}

TEST(TestApplyPluginTool, TestNoopBadConfig) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::NOOP);
  run->model.reset();
  EXPECT_THAT(ApplyPlugin(std::move(run)),
              IsError(kLiteRtStatusErrorInvalidToolConfig));
}

TEST(TestApplyPluginTool, TestNoop) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::NOOP);
  std::stringstream out;
  run->outs.push_back(out);
  LITERT_ASSERT_OK(ApplyPlugin(std::move(run)));

  std::string out_str = out.str();
  BufferRef<uint8_t> serialized(out_str.data(), out_str.size());
  LITERT_ASSERT_OK_AND_ASSIGN(auto model, Model::CreateFromBuffer(serialized));

  EXPECT_EQ(model.Get()->NumSubgraphs(), 1);
}

TEST(TestApplyPluginTool, TestPartitionBadConfig) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::PARTITION);
  run->model.reset();
  EXPECT_THAT(ApplyPlugin(std::move(run)),
              IsError(kLiteRtStatusErrorInvalidToolConfig));
}

TEST(TestApplyPluginTool, TestPartition) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::PARTITION);
  std::stringstream out;
  run->outs.push_back(out);
  LITERT_ASSERT_OK(ApplyPlugin(std::move(run)));
  EXPECT_FALSE(out.str().empty());
}

TEST(TestApplyPluginTool, TestCompileBadConfig) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::COMPILE);
  run->model.reset();
  EXPECT_THAT(ApplyPlugin(std::move(run)),
              IsError(kLiteRtStatusErrorInvalidToolConfig));
}

TEST(TestApplyPluginTool, TestCompile) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::COMPILE);
  std::stringstream out;
  run->outs.push_back(out);
  LITERT_ASSERT_OK(ApplyPlugin(std::move(run)));
  EXPECT_FALSE(out.str().empty());
  EXPECT_THAT(out.str(), HasSubstr("inputs:0,1\noutputs:2\nconst_map:\ntensors:"
                                   "[2x2],[2x2],[2x2]\nops:mul(0,1)(2)"));
}

TEST(TestApplyPluginTool, TestApplyBadConfig) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::APPLY);
  run->model.reset();
  EXPECT_THAT(ApplyPlugin(std::move(run)),
              IsError(kLiteRtStatusErrorInvalidToolConfig));
}

TEST(TestApplyPluginTool, TestApply) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::APPLY);
  std::stringstream out;
  run->outs.push_back(out);
  LITERT_ASSERT_OK(ApplyPlugin(std::move(run)));

  const auto out_str = out.str();
  BufferRef<uint8_t> serialized(out_str.data(), out_str.size());

  auto model = Model::CreateFromBuffer(serialized);
  EXPECT_EQ(model->Get()->NumSubgraphs(), 1);

  {
    auto stamp_buffer = model->Get()->FindMetadata(kLiteRtBuildStampKey);
    auto stamp = ParseBuildStamp(*stamp_buffer);
    auto [man, soc_model] = *stamp;
    EXPECT_EQ(man, kSocManufacturer);
    EXPECT_EQ(soc_model, kSocModel);
  }

  auto* op = model->Get()->MainSubgraph()->Ops().front();
  ASSERT_EQ(op->OpCode(), kLiteRtOpCodeTflCustom);

  const auto options = internal::GetDispatchOpOptions(op->CustomOptions());
  const auto& [size, offset, name] = options;
  EXPECT_EQ(name, "partition_0");
  ASSERT_LE(offset + size, serialized.Size());

  EXPECT_THAT(serialized.StrView().substr(offset, size),
              HasSubstr("inputs:0,1\noutputs:2\nconst_map:\ntensors:[2x2],[2x2]"
                        ",[2x2]\nops:mul(0,1)(2)"));
}

TEST(TestApplyPluginTool, TestCompileToMultiByteCode) {
  auto run =
      MakeBaseRun(ApplyPluginRun::Cmd::COMPILE, "multi_subgraph_mul.tflite");
  std::stringstream out_0;
  std::stringstream out_1;
  run->outs.push_back(out_0);
  run->outs.push_back(out_1);

  LITERT_ASSERT_OK(ApplyPlugin(std::move(run)));
  EXPECT_FALSE(out_0.str().empty());
  EXPECT_FALSE(out_1.str().empty());
  EXPECT_THAT(out_0.str(),
              HasSubstr("inputs:0,1\noutputs:2\nconst_map:\ntensors:[2x2],[2x2]"
                        ",[2x2]\nops:mul(0,1)(2)"));
  EXPECT_THAT(out_1.str(),
              HasSubstr("inputs:0,1\noutputs:2\nconst_map:\ntensors:[4x4],[4x4]"
                        ",[4x4]\nops:mul(0,1)(2)"));
}

TEST(TestApplyPluginTool, TestApplyWithExternalWeights) {
  auto flatbuffer = internal::FlatbufferWrapper::CreateFromTflFile(
      ::litert::testing::GetTestFilePath("add_cst.tflite"));
  ASSERT_TRUE(flatbuffer);
  auto tfl_model = flatbuffer->get()->Unpack();

  tfl_model->operator_codes[0]->builtin_code = ::tflite::BuiltinOperator_MUL;
  const int32_t cst_tensor_idx =
      tfl_model->subgraphs[0]->operators[0]->inputs[1];
  auto& tfl_cst_tensor = *tfl_model->subgraphs[0]->tensors[cst_tensor_idx];
  tfl_model->buffers[tfl_cst_tensor.buffer]->data.clear();
  tfl_cst_tensor.buffer = 0;
  tfl_cst_tensor.external_buffer = 1;

  auto group0 = std::make_unique<tflite::ExternalBufferGroupT>();
  group0->name = "";
  auto group1 = std::make_unique<tflite::ExternalBufferGroupT>();
  group1->name = "ext_weights.bin";
  tfl_model->external_buffer_groups.push_back(std::move(group0));
  tfl_model->external_buffer_groups.push_back(std::move(group1));

  constexpr std::array<float, 4> kExternalWeights = {7.0f, 8.0f, 9.0f, 10.0f};
  auto ext_buf = std::make_unique<tflite::ExternalBufferT>();
  ext_buf->id = 1;
  ext_buf->group = 1;
  ext_buf->offset = 0;
  ext_buf->length = sizeof(kExternalWeights);
  tfl_model->external_buffers.push_back(std::move(ext_buf));

  auto initial_serialized = internal::SerializeFlatbuffer(*tfl_model);
  auto temp_dir = ::litert::testing::UniqueTestDirectory::Create();
  ASSERT_TRUE(temp_dir);
  std::filesystem::path temp_path(std::string(temp_dir->Str()));

  std::filesystem::path model_path = temp_path / "ext_mul_cst.tflite";
  {
    std::ofstream model_file(model_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(model_file.is_open());
    model_file.write(initial_serialized.StrData(), initial_serialized.Size());
  }
  std::filesystem::path weights_path = temp_path / "ext_weights.bin";
  {
    std::ofstream weights_file(weights_path,
                               std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(weights_file.is_open());
    weights_file.write(reinterpret_cast<const char*>(kExternalWeights.data()),
                       sizeof(kExternalWeights));
  }

  std::string model_path_str = model_path.string();
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::APPLY);
  run->model.emplace(model_path_str);
  std::stringstream out;
  run->outs.push_back(out);
  LITERT_ASSERT_OK(ApplyPlugin(std::move(run)));

  const auto out_str = out.str();
  BufferRef<uint8_t> serialized(out_str.data(), out_str.size());
  auto model = Model::CreateFromBuffer(serialized);
  ASSERT_TRUE(model);

  auto* op = model->Get()->MainSubgraph()->Ops().front();
  ASSERT_EQ(op->OpCode(), kLiteRtOpCodeTflCustom);
  EXPECT_EQ(op->Inputs().size(), 1u);

  const auto options = internal::GetDispatchOpOptions(op->CustomOptions());
  const auto& [size, offset, name] = options;
  EXPECT_EQ(name, "partition_0");
  ASSERT_LE(offset + size, serialized.Size());
  EXPECT_THAT(serialized.StrView().substr(offset, size),
              HasSubstr("DATA:7,8,9,10"));
}

TEST(TestApplyPluginTool, TestCompileAndPartitionWithExternalWeights) {
  auto flatbuffer = internal::FlatbufferWrapper::CreateFromTflFile(
      ::litert::testing::GetTestFilePath("add_cst.tflite"));
  ASSERT_TRUE(flatbuffer);
  auto tfl_model = flatbuffer->get()->Unpack();

  tfl_model->operator_codes[0]->builtin_code = ::tflite::BuiltinOperator_MUL;
  const int32_t cst_tensor_idx =
      tfl_model->subgraphs[0]->operators[0]->inputs[1];
  auto& tfl_cst_tensor = *tfl_model->subgraphs[0]->tensors[cst_tensor_idx];
  tfl_model->buffers[tfl_cst_tensor.buffer]->data.clear();
  tfl_cst_tensor.buffer = 0;
  tfl_cst_tensor.external_buffer = 1;

  auto group0 = std::make_unique<tflite::ExternalBufferGroupT>();
  group0->name = "";
  auto group1 = std::make_unique<tflite::ExternalBufferGroupT>();
  group1->name = "ext_weights.bin";
  tfl_model->external_buffer_groups.push_back(std::move(group0));
  tfl_model->external_buffer_groups.push_back(std::move(group1));

  constexpr std::array<float, 4> kExternalWeights = {7.0f, 8.0f, 9.0f, 10.0f};
  auto ext_buf = std::make_unique<tflite::ExternalBufferT>();
  ext_buf->id = 1;
  ext_buf->group = 1;
  ext_buf->offset = 0;
  ext_buf->length = sizeof(kExternalWeights);
  tfl_model->external_buffers.push_back(std::move(ext_buf));

  auto initial_serialized = internal::SerializeFlatbuffer(*tfl_model);
  auto temp_dir = ::litert::testing::UniqueTestDirectory::Create();
  ASSERT_TRUE(temp_dir);
  std::filesystem::path temp_path(std::string(temp_dir->Str()));

  std::filesystem::path model_path = temp_path / "ext_mul_cst.tflite";
  {
    std::ofstream model_file(model_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(model_file.is_open());
    model_file.write(initial_serialized.StrData(), initial_serialized.Size());
  }
  std::filesystem::path weights_path = temp_path / "ext_weights.bin";
  {
    std::ofstream weights_file(weights_path,
                               std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(weights_file.is_open());
    weights_file.write(reinterpret_cast<const char*>(kExternalWeights.data()),
                       sizeof(kExternalWeights));
  }

  std::string model_path_str = model_path.string();

  // 1. Test Cmd::COMPILE directly on the external weight model.
  {
    auto compile_run = MakeBaseRun(ApplyPluginRun::Cmd::COMPILE);
    compile_run->model.emplace(model_path_str);
    std::stringstream compile_out;
    compile_run->outs.push_back(compile_out);
    LITERT_ASSERT_OK(ApplyPlugin(std::move(compile_run)));
    EXPECT_THAT(compile_out.str(), HasSubstr("DATA:7,8,9,10"));
  }

  // 2. Test Cmd::PARTITION on the external weight model: the partitioned
  // subgraph is transferred back into the serialized model, so external_buffer
  // 1 must still be preserved in the partitioned output flatbuffer.
  {
    auto partition_run = MakeBaseRun(ApplyPluginRun::Cmd::PARTITION);
    partition_run->model.emplace(model_path_str);
    std::stringstream partition_out;
    partition_run->outs.push_back(partition_out);
    LITERT_ASSERT_OK(ApplyPlugin(std::move(partition_run)));

    const auto part_str = partition_out.str();
    BufferRef<uint8_t> part_buf(part_str.data(), part_str.size());
    auto part_fb = internal::FlatbufferWrapper::CreateFromBuffer(part_buf);
    ASSERT_TRUE(part_fb);
    auto unpacked = part_fb->get()->Unpack();
    ASSERT_EQ(unpacked->subgraphs.size(), 2u);
    ASSERT_EQ(unpacked->external_buffers.size(), 1u);
    EXPECT_EQ(unpacked->external_buffers[0]->id, 1u);
  }
}

TEST(TestApplyPluginTool, TestApplyPartialCompilationWithExternalWeights) {
  auto flatbuffer = internal::FlatbufferWrapper::CreateFromTflFile(
      ::litert::testing::GetTestFilePath("add_cst.tflite"));
  ASSERT_TRUE(flatbuffer);
  auto tfl_model = flatbuffer->get()->Unpack();

  auto mul_opcode = std::make_unique<internal::TflOpCode>();
  mul_opcode->builtin_code = ::tflite::BuiltinOperator_MUL;
  tfl_model->operator_codes.push_back(std::move(mul_opcode));
  const uint32_t mul_opcode_idx = tfl_model->operator_codes.size() - 1;

  auto& sg = *tfl_model->subgraphs[0];
  const int32_t orig_input_idx = sg.operators[0]->inputs[0];
  const int32_t cst1_idx = sg.operators[0]->inputs[1];
  const int32_t orig_output_idx = sg.operators[0]->outputs[0];

  tfl_model->buffers[sg.tensors[cst1_idx]->buffer]->data.clear();
  sg.tensors[cst1_idx]->buffer = 0;
  sg.tensors[cst1_idx]->external_buffer = 1;

  auto tmp_tensor =
      std::make_unique<internal::TflTensor>(*sg.tensors[orig_output_idx]);
  tmp_tensor->name = "mul_out";
  tmp_tensor->buffer = 0;
  tmp_tensor->external_buffer = 0;
  sg.tensors.push_back(std::move(tmp_tensor));
  const int32_t tmp_tensor_idx = sg.tensors.size() - 1;

  auto cst2_tensor =
      std::make_unique<internal::TflTensor>(*sg.tensors[cst1_idx]);
  cst2_tensor->name = "ext_cst_2";
  cst2_tensor->buffer = 0;
  cst2_tensor->external_buffer = 2;
  sg.tensors.push_back(std::move(cst2_tensor));
  const int32_t cst2_idx = sg.tensors.size() - 1;

  sg.operators[0]->opcode_index = mul_opcode_idx;
  sg.operators[0]->inputs = {orig_input_idx, cst1_idx};
  sg.operators[0]->outputs = {tmp_tensor_idx};

  auto add_op = std::make_unique<internal::TflOp>();
  add_op->opcode_index = 0;  // ADD
  add_op->inputs = {tmp_tensor_idx, cst2_idx};
  add_op->outputs = {orig_output_idx};
  sg.operators.push_back(std::move(add_op));

  auto group0 = std::make_unique<tflite::ExternalBufferGroupT>();
  group0->name = "";
  auto group1 = std::make_unique<tflite::ExternalBufferGroupT>();
  group1->name = "ext_weights.bin";
  tfl_model->external_buffer_groups.push_back(std::move(group0));
  tfl_model->external_buffer_groups.push_back(std::move(group1));

  constexpr std::array<float, 8> kExternalWeights = {
      2.0f,  3.0f,  4.0f,  5.0f,   // ext_buf 1 (MUL -> NPU)
      10.0f, 20.0f, 30.0f, 40.0f,  // ext_buf 2 (ADD -> CPU)
  };
  auto ext_buf1 = std::make_unique<tflite::ExternalBufferT>();
  ext_buf1->id = 1;
  ext_buf1->group = 1;
  ext_buf1->offset = 0;
  ext_buf1->length = 4 * sizeof(float);
  auto ext_buf2 = std::make_unique<tflite::ExternalBufferT>();
  ext_buf2->id = 2;
  ext_buf2->group = 1;
  ext_buf2->offset = 4 * sizeof(float);
  ext_buf2->length = 4 * sizeof(float);
  tfl_model->external_buffers.push_back(std::move(ext_buf1));
  tfl_model->external_buffers.push_back(std::move(ext_buf2));

  auto initial_serialized = internal::SerializeFlatbuffer(*tfl_model);
  auto temp_dir = ::litert::testing::UniqueTestDirectory::Create();
  ASSERT_TRUE(temp_dir);
  std::filesystem::path temp_path(std::string(temp_dir->Str()));

  std::filesystem::path model_path = temp_path / "ext_mul_add.tflite";
  {
    std::ofstream model_file(model_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(model_file.is_open());
    model_file.write(initial_serialized.StrData(), initial_serialized.Size());
  }
  {
    std::ofstream weights_file(temp_path / "ext_weights.bin",
                               std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(weights_file.is_open());
    weights_file.write(reinterpret_cast<const char*>(kExternalWeights.data()),
                       sizeof(kExternalWeights));
  }

  std::string model_path_str = model_path.string();
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::APPLY);
  run->model.emplace(model_path_str);
  std::stringstream out;
  run->outs.push_back(out);
  LITERT_ASSERT_OK(ApplyPlugin(std::move(run)));

  const auto out_str = out.str();
  BufferRef<uint8_t> serialized(out_str.data(), out_str.size());
  auto fb = internal::FlatbufferWrapper::CreateFromBuffer(serialized);
  ASSERT_TRUE(fb);
  auto unpacked = fb->get()->Unpack();

  ASSERT_EQ(unpacked->external_buffers.size(), 1u);
  EXPECT_EQ(unpacked->external_buffers[0]->id, 2u);
}

}  // namespace
}  // namespace litert::tools
