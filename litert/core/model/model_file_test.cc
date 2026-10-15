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

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT
#include <fstream>
#include <functional>
#include <ios>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// schema/mutable/schema_generated.h and schema/schema_generated.h (included
// through flatbuffer_tools.h via model.h) have the same #ifdef, thus this line
// need to be put at the top to ensure we get the "mutable" version.
#if 1
#include "tflite/converter/schema/mutable/schema_generated.h"
#endif

#include <unordered_map>

#include <gmock/gmock.h>  // IWYU pragma: keep
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_consts.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/internal/litert_model_predicates.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/dispatch_op_schema.h"
#include "litert/core/model/buffer_manager.h"
#include "litert/core/model/graph_validation.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_file_test_util.h"
#include "litert/core/model/model_load.h"
#include "litert/core/model/model_serialize.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/common.h"
#include "litert/test/load_test_model.h"
#include "litert/test/matchers.h"
#include "litert/test/test_models.h"
#include "tflite/schema/mutable/schema_generated.h"

namespace litert::internal {
namespace {

using ::litert::testing::GetTestFilePath;
using ::testing::Each;
using ::testing::ElementsAreArray;
using ::testing::FloatEq;
using ::testing::Values;
using ::testing::litert::IsError;
using ::testing::litert::IsOkAndHolds;

using ModelFactory = std::function<Expected<ExtendedModel>()>;

static constexpr absl::string_view kAddSimple = "add_simple.tflite";
static constexpr absl::string_view kAddCst = "add_cst.tflite";
static constexpr absl::string_view kDynamicShapeModel =
    "dynamic_shape_tensor.tflite";
static constexpr absl::string_view kSimpleMultiOp = "simple_multi_op.tflite";
static constexpr absl::string_view kOneMul = "one_mul.tflite";
static constexpr absl::string_view kSimpleMultiSubgraph =
    "multi_subgraph.tflite";
static constexpr absl::string_view kCstMultiSubgraph =
    "cst_multi_subgraph.tflite";
static constexpr absl::string_view kPreCompiledModel =
    "simple_add_op.sm8750.tflite";

// Store buffers to ensure they outlive the models in tests
static std::unordered_map<std::string, OwningBufferRef<uint8_t>>&
GetTestBufferCache() {
  static std::unordered_map<std::string, OwningBufferRef<uint8_t>>* cache =
      new std::unordered_map<std::string, OwningBufferRef<uint8_t>>();
  return *cache;
}

// Load a model, then serialize and re-load. Used to test serialization.
Expected<ExtendedModel> LoadModelThroughRoundTrip(absl::string_view filename) {
  LITERT_ASSIGN_OR_RETURN(auto env, Environment::Create({}));
  auto model = ExtendedModel::CreateFromFile(env, GetTestFilePath(filename));
  if (!model) {
    return model.Error();
  }

  OwningBufferRef buf;
  auto [data, size, offset] = buf.GetWeak();

  const auto opts = litert::SerializationOptions::Defaults();
  LITERT_RETURN_IF_ERROR(LiteRtSerializeModel(model->Release(), &data, &size,
                                              &offset, true, opts));

  // Store buffer in cache to keep it alive
  std::string key(filename);
  GetTestBufferCache()[key] = std::move(buf);

  // Reload model using the cached buffer
  LiteRtModel result = nullptr;
  auto& cached_buf = GetTestBufferCache()[key];
  LITERT_RETURN_IF_ERROR(LiteRtCreateModelFromBuffer(
      env.Get(), cached_buf.Data(), cached_buf.Size(), &result));

  return ExtendedModel::CreateFromOwnedHandle(result);
}

ModelFactory MakeRoundTripFactory(absl::string_view filename) {
  return [=]() { return LoadModelThroughRoundTrip(filename); };
}

ModelFactory MakeLoadFactory(absl::string_view filename) {
  return [=]() -> Expected<ExtendedModel> {
    return ExtendedModel::CreateFromFile(GetTestFilePath(filename));
  };
}

// Test fixture parameterized by a file path to test model.
class TestWithModelPath : public ::testing::TestWithParam<absl::string_view> {
 protected:
  std::string GetTestModelPath() const {
    return testing::GetTestFilePath(GetParam());
  }
};

// Test fixture pareterized by a function that loads a model.
class TestWithModelFactory : public ::testing::TestWithParam<ModelFactory> {
 protected:
  Expected<ExtendedModel> LoadModel() { return GetParam()(); }
};

// Simple tests
//===---------------------------------------------------------------------------

TEST(ModelLoadTest, BadFilepath) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  LiteRtModel model = nullptr;
  EXPECT_THAT(LiteRtCreateModelFromFile(env.Get(), "bad_path", &model),
              IsError(kLiteRtStatusErrorFileIO));
}

TEST(ModelLoadTest, BadFileData) {
  // NOLINTBEGIN
#ifndef NDEBUG
  // In debug mode, flatbuffers will `assert` while verifying. This will
  // cause this test to crash (as expected).
  GTEST_SKIP();
#endif
  std::filesystem::path test_file_path(::testing::TempDir());
  test_file_path.append("bad_file.txt");

  std::ofstream bad_file;
  bad_file.open(test_file_path.c_str());
  bad_file << "not_tflite";
  bad_file.close();

  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  LiteRtModel model = nullptr;
#ifdef _WIN32
  const std::string test_file_path_str = test_file_path.string();
  const char* bad_file_path = test_file_path_str.c_str();
#else
  const char* bad_file_path = test_file_path.c_str();
#endif
  EXPECT_THAT(LiteRtCreateModelFromFile(env.Get(), bad_file_path, &model),
              IsError(kLiteRtStatusErrorFileIO));
  // NOLINTEND
}

TEST(ModelLoadTest, GetCustomOpCode) {
  auto model = litert::testing::LoadTestFileModel("simple_model_npu.tflite");
  ASSERT_TRUE(model);
  const auto& litert_model = *model.Get();
  const auto& op = *litert_model.MainSubgraph()->Ops().front();
  auto custom_op_code = GetCustomOpCode(litert_model, op);
  ASSERT_TRUE(custom_op_code.has_value());
  EXPECT_EQ(*custom_op_code, "DISPATCH_OP");
}

TEST(ModelLoadTest, WithMetadata) {
  constexpr static absl::string_view kMetadataName = "an_soc_manufacturer";
  constexpr static absl::string_view kMetadataData = "My_Meta_Data";

  auto flatbuffer =
      FlatbufferWrapper::CreateFromTflFile(GetTestFilePath(kAddSimple));
  auto tfl_model = flatbuffer->get()->Unpack();
  PushMetadata(kMetadataName, *tfl_model,
               BufferRef<uint8_t>(kMetadataData.data(), kMetadataData.size()));
  auto serialialized = SerializeFlatbuffer(*tfl_model);

  auto litert_model = LoadModelFromBuffer(serialialized);
  ASSERT_TRUE(litert_model);

  auto metadata = litert_model->get()->FindMetadata(kMetadataName);
  ASSERT_TRUE(metadata);
  EXPECT_EQ(metadata->StrView(), kMetadataData);
}

TEST(ModelSerializeTest, WithMetadata) {
  auto model = litert::testing::LoadTestFileModel(kAddSimple);

  constexpr static absl::string_view kMetadataName = "an_soc_manufacturer";
  constexpr static absl::string_view kMetadataData = "My_Meta_Data";

  LITERT_ASSERT_OK(model.Get()->PushMetadata(
      kMetadataName, OwningBufferRef<uint8_t>(kMetadataData)));

  auto serialized = SerializeModel(std::move(*model.Get()));
  EXPECT_TRUE(VerifyFlatbuffer(serialized->Span()));

  auto re_loaded = LoadModelFromBuffer(*serialized);
  auto metadata = re_loaded->get()->FindMetadata(kMetadataName);
  EXPECT_EQ(metadata->StrView(), kMetadataData);
}

TEST(ModelLoadTest, WithSignature) {
  auto model = litert::testing::LoadTestFileModel(kAddSimple);
  auto& litert_model = *model.Get();

  auto signature = litert_model.FindSignature(litert::kDefaultSignatureKey);
  ASSERT_TRUE(signature);

  EXPECT_EQ(signature->get().InputNames().size(), 1);
  EXPECT_EQ(signature->get().OutputNames().size(), 1);
  EXPECT_EQ(&signature->get().GetSubgraph(), litert_model.MainSubgraph());
}

TEST(ModelLoadTest, NoSignature) {
  auto model = litert::testing::LoadTestFileModel("add_simple.tflite");
  auto& litert_model = *model.Get();
  auto signature = litert_model.FindSignature(litert::kDefaultSignatureKey);
  ASSERT_TRUE(signature);
  EXPECT_EQ(signature->get().InputNames().size(), 1);
  EXPECT_EQ(signature->get().OutputNames().size(), 1);
  EXPECT_EQ(&signature->get().GetSubgraph(), litert_model.MainSubgraph());
}

TEST(ModelSerializeTest, WithSignature) {
  auto model = litert::testing::LoadTestFileModel(kAddSimple);
  auto& litert_model = *model.Get();

  static constexpr char kInput[] = "foo";
  static constexpr char kOutput[] = "bar";
  static constexpr char kKey[] = "newKey";

  LiteRtSubgraph subgraph = litert_model.MainSubgraph();
  std::vector<std::string> input_names = {kInput};
  std::vector<LiteRtTensor> input_tensors = {subgraph->Inputs()[0]};
  std::vector<std::string> output_names = {kOutput};
  std::vector<LiteRtTensor> output_tensors = {subgraph->Outputs()[0]};
  LiteRtSignatureT signature(subgraph, std::move(input_names),
                             std::move(input_tensors), std::move(output_names),
                             std::move(output_tensors), kKey);
  litert_model.EmplaceSignature(std::move(signature));

  auto serialized = SerializeModel(std::move(*model.Get()));
  EXPECT_TRUE(VerifyFlatbuffer(serialized->Span()));

  auto re_loaded = LoadModelFromBuffer(*serialized);
  auto re_loaded_signature = re_loaded->get()->FindSignature(kKey);
  ASSERT_TRUE(re_loaded_signature);
  const auto& sig = re_loaded_signature->get();

  const auto& inputs = sig.InputNames();
  const auto& outputs = sig.OutputNames();
  EXPECT_THAT(inputs, ElementsAreArray({kInput}));
  EXPECT_THAT(outputs, ElementsAreArray({kOutput}));
  EXPECT_EQ(&sig.GetSubgraph(), re_loaded->get()->MainSubgraph());
}

TEST(ModelLoadTest, ReverseSignature) {
  auto model =
      litert::testing::LoadTestFileModel("reverse_signature_model.tflite");
  ASSERT_TRUE(model);
  auto& litert_model = *model.Get();

  auto signature = litert_model.FindSignature("serving_default");
  ASSERT_TRUE(signature);

  // Check if the input and output names are in the order of the subgraph
  // inputs and outputs instead of the signature appearance order.
  const auto& sig = signature->get();
  ASSERT_EQ(sig.InputNames().size(), 2);
  EXPECT_STREQ(sig.InputNames()[0].c_str(), "y");
  EXPECT_STREQ(sig.InputNames()[1].c_str(), "x");
  ASSERT_EQ(sig.OutputNames().size(), 2);
  EXPECT_STREQ(sig.OutputNames()[0].c_str(), "sum");
  EXPECT_STREQ(sig.OutputNames()[1].c_str(), "prod");

  auto serialized = SerializeModel(std::move(*model.Get()));
  EXPECT_TRUE(VerifyFlatbuffer(serialized->Span()));

  auto re_loaded = LoadModelFromBuffer(*serialized);
  auto re_loaded_signature = re_loaded->get()->FindSignature("serving_default");
  ASSERT_TRUE(re_loaded_signature);

  // Check again with the serialized model.
  const auto& re_sig = re_loaded_signature->get();
  ASSERT_EQ(re_sig.InputNames().size(), 2);
  EXPECT_STREQ(re_sig.InputNames()[0].c_str(), "y");
  EXPECT_STREQ(re_sig.InputNames()[1].c_str(), "x");
  ASSERT_EQ(re_sig.OutputNames().size(), 2);
  EXPECT_STREQ(re_sig.OutputNames()[0].c_str(), "sum");
  EXPECT_STREQ(re_sig.OutputNames()[1].c_str(), "prod");
}

TEST(ModelLoadTest, WithOffsetTensorBuffer) {
  static constexpr absl::string_view kTensorData = "SOME_TENSOR_DATA";

  auto flatbuffer =
      FlatbufferWrapper::CreateFromTflFile(GetTestFilePath(kAddSimple));
  auto tfl_model = flatbuffer->get()->Unpack();
  const auto buf_ind = tfl_model->subgraphs[0]->tensors[0]->buffer;
  auto& tfl_buffer = tfl_model->buffers[buf_ind];
  tfl_buffer->offset = 1;
  tfl_buffer->size = 1;
  auto model_buf = SerializeFlatbuffer(*tfl_model);
  auto* packed_tfl = tflite::GetMutableModel(model_buf.Data());
  auto* buf = packed_tfl->mutable_buffers()->GetMutableObject(buf_ind);
  ASSERT_TRUE(buf->mutate_offset(model_buf.Size()));
  ASSERT_TRUE(buf->mutate_size(kTensorData.size()));
  OwningBufferRef<uint8_t> final_serializd(kTensorData.size() +
                                           model_buf.Size());
  std::memcpy(final_serializd.Data(), model_buf.Data(), model_buf.Size());
  std::memcpy(final_serializd.Data() + model_buf.Size(), kTensorData.data(),
              kTensorData.size());

  auto litert_model = LoadModelFromBuffer(final_serializd);
  ASSERT_TRUE(litert_model);

  const auto& weights_buffer =
      litert_model->get()->Subgraph(0).Tensor(0).Weights();
  EXPECT_EQ(weights_buffer.Buffer().StrView(), kTensorData);

  // The loaded buffer should indicate that it should be also serialized as
  // external.
  LITERT_ASSERT_OK_AND_ASSIGN(auto weights_buffer_context,
                              weights_buffer.GetBufferManager()->GetContext(
                                  weights_buffer.GetBufferId()));
  const auto will_append = weights_buffer_context.get().should_append;
  EXPECT_TRUE(will_append);

  // All tensors in the first subgraph should have the same buffer manager as
  // the model.
  for (auto* tensor : litert_model->get()->Subgraph(0).Tensors()) {
    EXPECT_EQ(tensor->Weights().GetBufferManager(),
              litert_model->get()->Buffers());
  }
}

TEST(ModelLoadTest, RejectsExternalBufferPastAllocation) {
  auto flatbuffer =
      FlatbufferWrapper::CreateFromTflFile(GetTestFilePath(kAddSimple));
  auto tfl_model = flatbuffer->get()->Unpack();
  const auto buf_ind = tfl_model->subgraphs[0]->tensors[0]->buffer;
  auto& tfl_buffer = tfl_model->buffers[buf_ind];
  tfl_buffer->offset = 1;
  tfl_buffer->size = 1;

  auto model_buf = SerializeFlatbuffer(*tfl_model);
  auto* packed_tfl = tflite::GetMutableModel(model_buf.Data());
  auto* buf = packed_tfl->mutable_buffers()->GetMutableObject(buf_ind);
  ASSERT_TRUE(buf->mutate_offset(model_buf.Size() + 1));
  ASSERT_TRUE(buf->mutate_size(1));

  auto litert_model = LoadModelFromBuffer(model_buf);
  ASSERT_FALSE(litert_model);
  EXPECT_EQ(litert_model.Error().Status(), kLiteRtStatusErrorInvalidFlatbuffer);
}

TEST(ModelLoadTest, RejectsInvalidOperatorTensorIndex) {
  auto flatbuffer =
      FlatbufferWrapper::CreateFromTflFile(GetTestFilePath(kAddSimple));
  auto tfl_model = flatbuffer->get()->Unpack();
  auto& subgraph = *tfl_model->subgraphs[0];
  subgraph.operators[0]->inputs[0] = subgraph.tensors.size();

  auto serialized = SerializeFlatbuffer(*tfl_model);
  auto litert_model = LoadModelFromBuffer(serialized);
  ASSERT_FALSE(litert_model);
  EXPECT_EQ(litert_model.Error().Status(), kLiteRtStatusErrorInvalidFlatbuffer);
}

TEST(ModelLoadTest, RejectsInvalidSignatureTensorIndex) {
  auto flatbuffer = FlatbufferWrapper::CreateFromTflFile(
      GetTestFilePath("reverse_signature_model.tflite"));
  auto tfl_model = flatbuffer->get()->Unpack();
  ASSERT_FALSE(tfl_model->signature_defs.empty());
  ASSERT_FALSE(tfl_model->signature_defs[0]->inputs.empty());

  tfl_model->signature_defs[0]->inputs[0]->tensor_index =
      tfl_model->subgraphs[0]->tensors.size();

  auto serialized = SerializeFlatbuffer(*tfl_model);
  auto litert_model = LoadModelFromBuffer(serialized);
  ASSERT_FALSE(litert_model);
  EXPECT_EQ(litert_model.Error().Status(), kLiteRtStatusErrorInvalidFlatbuffer);
}

TEST(ModelLoadTest, RejectsDispatchBytecodePastAllocation) {
  static constexpr absl::string_view kByteCode = "SOME_BYTE_CODE";

  LiteRtModelT root;
  auto& sg = root.EmplaceSubgraph();
  auto& op = sg.EmplaceOp();
  OwningBufferRef<uint8_t> buffer(kByteCode);
  const auto buf_id = root.Buffers()->RegisterOwnedBuffer(std::move(buffer));
  root.AttachAssetToOp(&op, buf_id, "foo");

  auto serialized = SerializeModel(std::move(root));
  ASSERT_TRUE(serialized);

  auto flatbuffer = FlatbufferWrapper::CreateFromBuffer(*serialized);
  ASSERT_TRUE(flatbuffer);
  auto tfl_model = flatbuffer->get()->Unpack();
  auto& opts = tfl_model->subgraphs[0]->operators[0]->custom_options;
  MutableBufferRef<uint8_t> opts_buffer(opts.data(), opts.size());
  auto dispatch_opts = GetDispatchOpOptions(opts_buffer);
  dispatch_opts.bytecode_offset = std::numeric_limits<size_t>::max();
  ASSERT_TRUE(UpdateDispatchOpOptionsInPlace(dispatch_opts, opts_buffer));

  auto mutated = SerializeFlatbuffer(*tfl_model);
  const std::filesystem::path test_file_path =
      std::filesystem::path(::testing::TempDir()) / "bad_dispatch.tflite";
  std::ofstream output(test_file_path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(mutated.Data()), mutated.Size());
  output.close();

  auto litert_model = LoadModelFromFile(test_file_path.string());
  ASSERT_FALSE(litert_model);
  EXPECT_EQ(litert_model.Error().Status(), kLiteRtStatusErrorInvalidFlatbuffer);
}

TEST(ModelTestCApi, Int2) {
  auto model = LoadModelThroughRoundTrip("FFW-2-bit.tflite");
  ASSERT_TRUE(model);

  const auto& litert_model = *model->Get();
  const auto& int2_tensor = litert_model.Subgraph(0).Tensor(2);

  const auto& int2_type = int2_tensor.Type().second.ranked_tensor_type;
  const auto rank = int2_type.layout.rank;
  EXPECT_EQ(int2_type.element_type, kLiteRtElementTypeInt2);

  size_t num_elements = 1;
  for (int i = 0; i < rank; ++i) {
    num_elements *= int2_type.layout.dimensions[i];
  }

  const auto& weights = int2_tensor.Weights();
  const auto num_bytes = weights.Buffer().Size();
  EXPECT_EQ(num_bytes, std::ceil(num_elements / 4.0f));

  const auto& [int2_quant_id, int2_quant_detail] = int2_tensor.Qparams();
  ASSERT_EQ(int2_quant_id, kLiteRtQuantizationPerChannel);
  const auto q_dim = int2_quant_detail.per_channel.quantized_dimension;
  const auto num_channels = int2_quant_detail.per_channel.num_channels;
  ASSERT_EQ(num_channels, int2_type.layout.dimensions[q_dim]);

  EXPECT_THAT(absl::MakeConstSpan(int2_quant_detail.per_channel.zero_points,
                                  num_channels),
              Each(0));
}

TEST(ModelTestCCApi, Int2) {
  auto model = LoadModelThroughRoundTrip("FFW-2-bit.tflite");
  ASSERT_TRUE(model);

  const auto& litert_model = *model;
  LITERT_ASSERT_OK_AND_ASSIGN(const auto sg, litert_model.MainSubgraph());
  const auto ops = sg.Ops();

  const auto& fc = ops[1];
  ASSERT_EQ(fc.Code(), kLiteRtOpCodeTflFullyConnected);

  const auto inputs = fc.Inputs();
  const auto& int2_tensor = inputs[1];

  LITERT_ASSERT_OK_AND_ASSIGN(const auto rtt, int2_tensor.RankedTensorType());
  ASSERT_EQ(rtt.ElementType(), ElementType::Int2);
  const auto& layout = rtt.Layout();
  LITERT_ASSERT_OK_AND_ASSIGN(const auto num_elements, layout.NumElements());
  LITERT_ASSERT_OK_AND_ASSIGN(const auto num_bytes, rtt.Bytes());
  EXPECT_EQ(num_bytes, std::ceil(num_elements / 4.0f));
  ASSERT_TRUE(int2_tensor.IsConstant());

  const auto q_params = int2_tensor.PerChannelQuantization();
  const auto q_dim = q_params.quantized_dimension;
  const auto num_channels = q_params.num_channels;
  ASSERT_EQ(num_channels, layout.Dimensions()[q_dim]);
  EXPECT_THAT(absl::MakeConstSpan(q_params.zero_points, num_channels), Each(0));
}

TEST(ModelTestCCApi, BlockWiseQuantization) {
  auto model = LoadModelThroughRoundTrip("blockwise_quantized.tflite");
  ASSERT_TRUE(model);

  const auto& litert_model = *model;
  LITERT_ASSERT_OK_AND_ASSIGN(const auto sg, litert_model.MainSubgraph());
  const auto ops = sg.Ops();

  ASSERT_EQ(ops.size(), 1);
  const auto& add = ops[0];
  ASSERT_EQ(add.Code(), kLiteRtOpCodeTflAdd);

  const auto inputs = add.Inputs();
  ASSERT_EQ(inputs.size(), 2);

  Tensor quantized_tensor(nullptr);
  for (const auto& input : inputs) {
    if (input.Name() == "quantized_tensor") {
      quantized_tensor = input;
      break;
    }
  }
  ASSERT_NE(quantized_tensor.Get(), nullptr);

  LITERT_ASSERT_OK_AND_ASSIGN(const auto rtt,
                              quantized_tensor.RankedTensorType());
  ASSERT_EQ(rtt.ElementType(), ElementType::Int8);

  ASSERT_TRUE(quantized_tensor.HasQuantization());
  ASSERT_EQ(quantized_tensor.QTypeId(), kLiteRtQuantizationBlockWise);

  const auto q_params = quantized_tensor.BlockWiseQuantization();
  EXPECT_EQ(q_params.block_size, 32);

  ASSERT_NE(q_params.scales, nullptr);
  const char* scales_name;
  LITERT_ASSERT_OK(LiteRtGetTensorName(q_params.scales, &scales_name));
  EXPECT_STREQ(scales_name, "scales");

  ASSERT_NE(q_params.zero_points, nullptr);
  const char* zps_name;
  LITERT_ASSERT_OK(LiteRtGetTensorName(q_params.zero_points, &zps_name));
  EXPECT_STREQ(zps_name, "zps");
}

TEST(ModelTestCCApi, QwenBlockWiseQuantization) {
  auto model = LoadModelThroughRoundTrip("qwen_fc_blockwise_quantized.tflite");
  ASSERT_TRUE(model);

  const auto& litert_model = *model;
  LITERT_ASSERT_OK_AND_ASSIGN(const auto sg, litert_model.MainSubgraph());
  const auto ops = sg.Ops();

  ASSERT_EQ(ops.size(), 1);
  const auto& fc = ops[0];
  ASSERT_EQ(fc.Code(), kLiteRtOpCodeTflFullyConnected);

  const auto inputs = fc.Inputs();
  ASSERT_GE(inputs.size(), 2);

  const auto& weights_tensor = inputs[1];

  LITERT_ASSERT_OK_AND_ASSIGN(const auto rtt,
                              weights_tensor.RankedTensorType());
  ASSERT_EQ(rtt.ElementType(), ElementType::Int4);

  ASSERT_TRUE(weights_tensor.HasQuantization());
  ASSERT_EQ(weights_tensor.QTypeId(), kLiteRtQuantizationBlockWise);

  const auto q_params = weights_tensor.BlockWiseQuantization();
  EXPECT_EQ(q_params.block_size, 32);

  ASSERT_NE(q_params.scales, nullptr);
}

TEST(ModelSerializeTest, WithOffsetTensorBuffer) {
  static constexpr absl::string_view kTensorData = "SOME_TENSOR_DATA";

  LiteRtModelT root;
  auto& sg = root.EmplaceSubgraph();
  auto& tensor = sg.EmplaceTensor();
  sg.EmplaceOp();
  tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));
  auto& weights = tensor.Weights();
  weights.SetBufferManager(root.Buffers());

  OwningBufferRef<uint8_t> buffer(kTensorData);
  BufferContext context;
  context.should_append = true;
  SetWeightsFromOwnedBuffer(weights, std::move(buffer), context);

  auto serialized = SerializeModel(std::move(root));
  ASSERT_TRUE(serialized);

  // Verify the op contains an offset and size to the byte code and the correct
  // name.
  auto fb = FlatbufferWrapper::CreateFromBuffer(*serialized);
  ASSERT_TRUE(fb);

  auto tfl = fb->get()->Unpack();
  const auto& tfl_tensor = tfl->subgraphs[0]->tensors[0];
  const auto tfl_buffer_ind = tfl_tensor->buffer;
  const auto& tfl_buffer = tfl->buffers[tfl_buffer_ind];

  auto data =
      serialized->StrView().substr(tfl_buffer->offset, tfl_buffer->size);
  EXPECT_EQ(data, kTensorData);
}

TEST(ModelSerializeTest, WithMultipleOffsetTensorBuffer) {
  static constexpr absl::string_view kTensorData = "SOME_TENSOR_DATA";
  static constexpr absl::string_view kTensorData2 = "SOME_TENSOR_DATA2";

  LiteRtModelT root;
  auto& sg = root.EmplaceSubgraph();
  sg.EmplaceOp();

  {
    auto& tensor = sg.EmplaceTensor();
    tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));
    auto& weights = tensor.Weights();
    weights.SetBufferManager(root.Buffers());

    OwningBufferRef<uint8_t> buffer(kTensorData);
    BufferContext context;
    context.should_append = true;
    SetWeightsFromOwnedBuffer(weights, std::move(buffer), context);
  }

  {
    auto& tensor = sg.EmplaceTensor();
    tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));
    auto& weights = tensor.Weights();
    weights.SetBufferManager(root.Buffers());

    OwningBufferRef<uint8_t> buffer(kTensorData2);
    BufferContext context;
    context.should_append = true;
    SetWeightsFromOwnedBuffer(weights, std::move(buffer), context);
  }

  auto serialized = SerializeModel(std::move(root));
  ASSERT_TRUE(serialized);

  // Verify the op contains an offset and size to the byte code and the correct
  // name.
  auto fb = FlatbufferWrapper::CreateFromBuffer(*serialized);
  ASSERT_TRUE(fb);

  auto tfl = fb->get()->Unpack();

  {
    const auto& tfl_tensor = tfl->subgraphs[0]->tensors[0];
    const auto tfl_buffer_ind = tfl_tensor->buffer;
    const auto& tfl_buffer = tfl->buffers[tfl_buffer_ind];

    auto data =
        serialized->StrView().substr(tfl_buffer->offset, tfl_buffer->size);
    EXPECT_EQ(data, kTensorData);
  }

  {
    const auto& tfl_tensor = tfl->subgraphs[0]->tensors[1];
    const auto tfl_buffer_ind = tfl_tensor->buffer;
    const auto& tfl_buffer = tfl->buffers[tfl_buffer_ind];

    auto data =
        serialized->StrView().substr(tfl_buffer->offset, tfl_buffer->size);
    EXPECT_EQ(data, kTensorData2);
  }
}

TEST(ModelSerializeTest, WithSingleExternalBuffer) {
  static constexpr absl::string_view kByteCode = "SOME_BYTE_CODE";
  static constexpr absl::string_view kName = "foo";

  LiteRtModelT root;
  auto& sg = root.EmplaceSubgraph();
  auto& op = sg.EmplaceOp();

  OwningBufferRef<uint8_t> buffer(kByteCode);
  const auto buf_id = root.Buffers()->RegisterOwnedBuffer(std::move(buffer));
  root.AttachAssetToOp(&op, buf_id, std::string(kName));

  auto serialized = SerializeModel(std::move(root));
  ASSERT_TRUE(serialized);

  // Verify the op contains an offset and size to the byte code and the correct
  // name.
  auto fb = FlatbufferWrapper::CreateFromBuffer(*serialized);
  ASSERT_TRUE(fb);

  auto tfl = fb->get()->Unpack();
  const auto& opts = tfl->subgraphs[0]->operators[0]->custom_options;
  BufferRef<uint8_t> opts_buffer(opts.data(), opts.size());

  auto dispatch_opts = GetDispatchOpOptions(opts_buffer);
  EXPECT_EQ(dispatch_opts.name, kName);
  EXPECT_EQ(serialized->StrView().substr(dispatch_opts.bytecode_offset,
                                         dispatch_opts.bytecode_size),
            kByteCode);
}

TEST(ModelSerializeTest, WithMultipleUniqueExternalBuffer) {
  static constexpr absl::string_view kByteCode = "SOME_BYTE_CODE";
  static constexpr absl::string_view kName = "foo";
  static constexpr absl::string_view kByteCode2 = "SOME_BYTE_CODE2";
  static constexpr absl::string_view kName2 = "bar";

  LiteRtModelT root;
  auto& sg = root.EmplaceSubgraph();
  auto& op = sg.EmplaceOp();
  auto& op2 = sg.EmplaceOp();

  OwningBufferRef<uint8_t> buffer(kByteCode);
  const auto buf_id = root.Buffers()->RegisterOwnedBuffer(std::move(buffer));
  root.AttachAssetToOp(&op, buf_id, std::string(kName));

  OwningBufferRef<uint8_t> buffer2(kByteCode2);
  const auto buf_id2 = root.Buffers()->RegisterOwnedBuffer(std::move(buffer2));
  root.AttachAssetToOp(&op2, buf_id2, std::string(kName2));

  auto serialized = SerializeModel(std::move(root));
  ASSERT_TRUE(serialized);

  // Verify both ops contains an offset and size to the byte code and the
  // correct name.
  auto fb = FlatbufferWrapper::CreateFromBuffer(*serialized);
  ASSERT_TRUE(fb);

  auto tfl = fb->get()->Unpack();

  {
    const auto& opts = tfl->subgraphs[0]->operators[0]->custom_options;
    BufferRef<uint8_t> opts_buffer(opts.data(), opts.size());

    auto dispatch_opts = GetDispatchOpOptions(opts_buffer);
    EXPECT_EQ(dispatch_opts.name, kName);
    EXPECT_EQ(serialized->StrView().substr(dispatch_opts.bytecode_offset,
                                           dispatch_opts.bytecode_size),
              kByteCode);
  }

  {
    const auto& opts = tfl->subgraphs[0]->operators[1]->custom_options;
    BufferRef<uint8_t> opts_buffer(opts.data(), opts.size());

    auto dispatch_opts = GetDispatchOpOptions(opts_buffer);
    EXPECT_EQ(dispatch_opts.name, kName2);
    EXPECT_EQ(serialized->StrView().substr(dispatch_opts.bytecode_offset,
                                           dispatch_opts.bytecode_size),
              kByteCode2);
  }
}

TEST(ModelSerializeTest, WithSharedExternalBuffer) {
  static constexpr absl::string_view kByteCode = "SOME_BYTE_CODE";
  static constexpr absl::string_view kName = "foo";
  static constexpr absl::string_view kName2 = "bar";

  LiteRtModelT root;
  auto& sg = root.EmplaceSubgraph();
  auto& op = sg.EmplaceOp();
  auto& op2 = sg.EmplaceOp();

  OwningBufferRef<uint8_t> buffer(kByteCode);
  const auto buf_id = root.Buffers()->RegisterOwnedBuffer(std::move(buffer));

  root.AttachAssetToOp(&op, buf_id, std::string(kName));
  root.AttachAssetToOp(&op2, buf_id, std::string(kName2));

  auto serialized = SerializeModel(std::move(root));
  ASSERT_TRUE(serialized);

  // Verify both ops point to the same appended buffer.
  auto fb = FlatbufferWrapper::CreateFromBuffer(*serialized);
  ASSERT_TRUE(fb);

  auto tfl = fb->get()->Unpack();

  {
    const auto& opts = tfl->subgraphs[0]->operators[0]->custom_options;
    BufferRef<uint8_t> opts_buffer(opts.data(), opts.size());

    auto dispatch_opts = GetDispatchOpOptions(opts_buffer);
    EXPECT_EQ(dispatch_opts.name, kName);
    EXPECT_EQ(serialized->StrView().substr(dispatch_opts.bytecode_offset,
                                           dispatch_opts.bytecode_size),
              kByteCode);
  }

  {
    const auto& opts = tfl->subgraphs[0]->operators[1]->custom_options;
    BufferRef<uint8_t> opts_buffer(opts.data(), opts.size());

    auto dispatch_opts = GetDispatchOpOptions(opts_buffer);
    EXPECT_EQ(dispatch_opts.name, kName2);
    EXPECT_EQ(serialized->StrView().substr(dispatch_opts.bytecode_offset,
                                           dispatch_opts.bytecode_size),
              kByteCode);
  }
}

TEST(ModelSerializeTest, WithOffsetTensorBufferAndOpAsset) {
  static constexpr absl::string_view kTensorData = "SOME_TENSOR_DATA";
  static constexpr absl::string_view kByteCode = "SOME_BYTE_CODE";
  static constexpr absl::string_view kName = "name";

  LiteRtModelT root;
  auto& sg = root.EmplaceSubgraph();
  auto& op = sg.EmplaceOp();
  auto& tensor = sg.EmplaceTensor();
  tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));
  auto& weights = tensor.Weights();
  weights.SetBufferManager(root.Buffers());

  {
    OwningBufferRef<uint8_t> buffer(kTensorData);
    BufferContext context;
    context.should_append = true;
    SetWeightsFromOwnedBuffer(weights, std::move(buffer), context);
  }

  {
    OwningBufferRef<uint8_t> buffer(kByteCode);
    const auto buf_id = root.Buffers()->RegisterOwnedBuffer(std::move(buffer));
    root.AttachAssetToOp(&op, buf_id, std::string(kName));
  }

  auto serialized = SerializeModel(std::move(root));
  ASSERT_TRUE(serialized);

  auto fb = FlatbufferWrapper::CreateFromBuffer(*serialized);
  ASSERT_TRUE(fb);
  auto tfl = fb->get()->Unpack();

  {
    const auto& tfl_tensor = tfl->subgraphs[0]->tensors[0];
    const auto tfl_buffer_ind = tfl_tensor->buffer;
    const auto& tfl_buffer = tfl->buffers[tfl_buffer_ind];

    auto data =
        serialized->StrView().substr(tfl_buffer->offset, tfl_buffer->size);
    EXPECT_EQ(data, kTensorData);
  }

  {
    const auto& opts = tfl->subgraphs[0]->operators[0]->custom_options;
    BufferRef<uint8_t> opts_buffer(opts.data(), opts.size());

    auto dispatch_opts = GetDispatchOpOptions(opts_buffer);
    EXPECT_EQ(dispatch_opts.name, kName);
    EXPECT_EQ(serialized->StrView().substr(dispatch_opts.bytecode_offset,
                                           dispatch_opts.bytecode_size),
              kByteCode);
  }
}

TEST(ModelSerializeTest, WithOffsetTensorBufferAndOpAssetHasAlignment) {
  static constexpr absl::string_view kTensorData = "SOME_TENSOR_DATA";
  static constexpr absl::string_view kByteCode = "SOME_BYTE_CODE";
  static constexpr absl::string_view kName = "name";
  static constexpr size_t kAlignment = 32;

  LiteRtModelT root;
  auto& sg = root.EmplaceSubgraph();
  auto& op = sg.EmplaceOp();
  auto& tensor = sg.EmplaceTensor();
  tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));
  auto& weights = tensor.Weights();
  weights.SetBufferManager(root.Buffers());

  {
    OwningBufferRef<uint8_t> buffer(kTensorData);
    BufferContext context;
    context.should_append = true;
    SetWeightsFromOwnedBuffer(weights, std::move(buffer), context);
  }

  {
    OwningBufferRef<uint8_t> buffer(kByteCode);
    const auto buf_id = root.Buffers()->RegisterOwnedBuffer(std::move(buffer));
    root.AttachAssetToOp(&op, buf_id, std::string(kName));
  }

  auto serialized = SerializeModel(std::move(root), kAlignment);
  ASSERT_TRUE(serialized);

  auto fb = FlatbufferWrapper::CreateFromBuffer(*serialized);
  ASSERT_TRUE(fb);
  auto tfl = fb->get()->Unpack();

  {
    const auto& tfl_tensor = tfl->subgraphs[0]->tensors[0];
    const auto tfl_buffer_ind = tfl_tensor->buffer;
    const auto& tfl_buffer = tfl->buffers[tfl_buffer_ind];

    auto data =
        serialized->StrView().substr(tfl_buffer->offset, tfl_buffer->size);
    EXPECT_EQ(data, kTensorData);
  }

  {
    const auto& opts = tfl->subgraphs[0]->operators[0]->custom_options;
    BufferRef<uint8_t> opts_buffer(opts.data(), opts.size());

    auto dispatch_opts = GetDispatchOpOptions(opts_buffer);
    EXPECT_EQ(dispatch_opts.name, kName);
    ASSERT_EQ(dispatch_opts.bytecode_offset % kAlignment, 0);
    EXPECT_EQ(serialized->StrView().substr(dispatch_opts.bytecode_offset,
                                           dispatch_opts.bytecode_size),
              kByteCode);
  }
}

TEST(ModelSerializeTest, TransferAndSerializeNoConstants) {
  LiteRtModelT root;
  auto& sg = root.EmplaceSubgraph();
  sg.EmplaceOp();
  auto& tensor = sg.EmplaceTensor();
  tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));
  auto& weights = tensor.Weights();
  weights.SetBufferManager(root.Buffers());

  auto other = root.Yank({0});
  ASSERT_EQ(other.Subgraphs().size(), 1);

  auto serialized = SerializeModel(std::move(other));
  ASSERT_TRUE(serialized);
}

TEST(ModelSerializeTest, TransferAndSerializeWithConstants) {
  static constexpr absl::string_view kTensorData = "SOME_TENSOR_DATA";

  LiteRtModelT root;
  auto& sg = root.EmplaceSubgraph();
  sg.EmplaceOp();
  auto& tensor = sg.EmplaceTensor();
  tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));
  auto& weights = tensor.Weights();
  weights.SetBufferManager(root.Buffers());

  {
    OwningBufferRef<uint8_t> buffer(kTensorData);
    BufferContext context;
    context.should_append = false;
    SetWeightsFromOwnedBuffer(weights, std::move(buffer), context);
  }

  auto other = root.Yank({0});
  ASSERT_EQ(other.Subgraphs().size(), 1);

  auto serialized = SerializeModel(std::move(other));
  ASSERT_TRUE(serialized);
}

TEST(ModelSerializeTest, TransferAndSerializeOffsetBuffers) {
  static constexpr absl::string_view kTensorData = "SOME_TENSOR_DATA";

  LiteRtModelT root;
  auto& sg = root.EmplaceSubgraph();
  sg.EmplaceOp();
  auto& tensor = sg.EmplaceTensor();
  tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));
  auto& weights = tensor.Weights();
  weights.SetBufferManager(root.Buffers());

  {
    OwningBufferRef<uint8_t> buffer(kTensorData);
    BufferContext context;
    context.should_append = true;
    SetWeightsFromOwnedBuffer(weights, std::move(buffer), context);
  }

  auto other = root.Yank({0});
  ASSERT_EQ(other.Subgraphs().size(), 1);

  auto serialized = SerializeModel(std::move(other));
  ASSERT_TRUE(serialized);
}

TEST(ModelSerializeTest,
     LoadModelWithBytecodeThenAddMetadataSerializationSuccess) {
  static constexpr absl::string_view kMetadataKey = "metadata_key";
  static constexpr absl::string_view kMetadataValue = "metadata_value";
  static constexpr absl::string_view kNpuModelPath = "simple_model_npu.tflite";

  auto model = litert::testing::LoadTestFileModel(kNpuModelPath);
  ASSERT_TRUE(model);
  LITERT_ASSERT_OK(
      model.AddMetadata(kMetadataKey.data(), kMetadataValue.data()));

  auto serialized = SerializeModel(std::move(*model.Get()));
  EXPECT_TRUE(VerifyFlatbuffer(serialized->Span()));

  // Reload model using the cached buffer.
  LiteRtModel result_model = nullptr;
  EXPECT_EQ(
      LiteRtCreateModelFromBuffer(/*environment=*/nullptr, serialized->Data(),
                                  serialized->Size(), &result_model),
      kLiteRtStatusOk);

  // Check dispatch op is present.
  ASSERT_TRUE(result_model);
  const auto& op = result_model->MainSubgraph()->Ops().front();
  auto custom_op_code = GetCustomOpCode(*result_model, *op);
  ASSERT_TRUE(custom_op_code.has_value());
  EXPECT_EQ(*custom_op_code, "DISPATCH_OP");

  // Check metadata is present.
  const void* metadata;
  size_t metadata_size;
  LITERT_ASSERT_OK(LiteRtGetModelMetadata(result_model, kMetadataKey.data(),
                                          &metadata, &metadata_size));
  EXPECT_EQ(BufferRef(metadata, metadata_size).StrView(), kMetadataValue);
  LiteRtDestroyModel(result_model);
}

TEST(ModelLoadSerializeTest, LoadAndSerializeWithBytecode) {
  Expected<std::unique_ptr<FlatbufferWrapper>> flatbuffer =
      FlatbufferWrapper::CreateFromTflFile(GetTestFilePath(kPreCompiledModel));
  ASSERT_TRUE(flatbuffer);

  ExtendedModel model = litert::testing::LoadTestFileModel(kPreCompiledModel);

  Expected<OwningBufferRef<unsigned char>> serialized =
      SerializeModel(std::move(*model.Get()));
  EXPECT_TRUE(VerifyFlatbuffer(serialized->Span()));

  EXPECT_EQ(serialized->Size(), flatbuffer->get()->Buf().Size());
}

// Tests that explicitly check litert graph structure.
//===---------------------------------------------------------------------------

using AddSimpleTest = TestWithModelFactory;

TEST_P(AddSimpleTest, CheckGraph) {
  auto model = LoadModel();
  ASSERT_TRUE(model);

  // func(arg0)
  //  output = tfl.add(arg0, arg0)
  //  return(output)
  //

  auto subgraph = model->MainSubgraph();
  const auto subgraph_inputs = subgraph->Inputs();
  const auto subgraph_outputs = subgraph->Outputs();
  const auto ops = subgraph->Ops();

  ASSERT_EQ(subgraph_inputs.size(), 1);
  ASSERT_EQ(subgraph_outputs.size(), 1);

  const auto& internal_ops = subgraph->Get()->Ops();
  ASSERT_TRUE(
      ValidateLocalTopology(internal_ops.cbegin(), internal_ops.cend()));
  ASSERT_TRUE(ValidateSubgraphIO(*subgraph->Get()));

  ASSERT_EQ(ops.size(), 1);
  const auto& op = ops.front();

  const TensorTypeInfo float_2by2_type(ElementType::Float32, {2, 2});
  ASSERT_TRUE(
      MatchOpType(op, {float_2by2_type, float_2by2_type}, {float_2by2_type}));
  EXPECT_EQ(op.Code(), kLiteRtOpCodeTflAdd);

  const auto op_inputs = op.Inputs();
  ASSERT_EQ(op_inputs.size(), 2);
  ASSERT_EQ(op_inputs.front().Get(), subgraph_inputs.front().Get());
  ASSERT_EQ(op_inputs.front().Get(), op_inputs.back().Get());

  const auto op_outputs = op.Outputs();
  ASSERT_EQ(op_outputs.size(), 1);
  ASSERT_EQ(op_outputs.front().Get(), subgraph_outputs.front().Get());

  ASSERT_FALSE(subgraph_outputs.front().IsConstant());
  ASSERT_FALSE(subgraph_inputs.front().IsConstant());
}

INSTANTIATE_TEST_SUITE_P(ModelLoadTests, AddSimpleTest,
                         Values(MakeLoadFactory(kAddSimple)));

INSTANTIATE_TEST_SUITE_P(ModelSerializeTests, AddSimpleTest,
                         Values(MakeRoundTripFactory(kAddSimple)));

using AddCstTest = TestWithModelFactory;

TEST_P(AddCstTest, CheckGraph) {
  auto model = LoadModel();
  ASSERT_TRUE(model);

  // func(arg0)
  //  cst = ConstantTensor([1, 2, 3, 4])
  //  output = tfl.add(arg0, cst)
  //  return(output)
  //

  auto subgraph = model->MainSubgraph();
  const auto subgraph_inputs = subgraph->Inputs();
  const auto subgraph_outputs = subgraph->Outputs();
  const auto ops = subgraph->Ops();

  ASSERT_EQ(subgraph_inputs.size(), 1);
  ASSERT_EQ(subgraph_outputs.size(), 1);

  const auto& internal_ops = subgraph->Get()->Ops();
  ASSERT_TRUE(
      ValidateLocalTopology(internal_ops.cbegin(), internal_ops.cend()));
  ASSERT_TRUE(ValidateSubgraphIO(*subgraph->Get()));

  ASSERT_EQ(ops.size(), 1);
  const auto& op = ops.front();

  const TensorTypeInfo float_by4_type(ElementType::Float32, {4});
  ASSERT_TRUE(
      MatchOpType(op, {float_by4_type, float_by4_type}, {float_by4_type}));
  EXPECT_EQ(op.Code(), kLiteRtOpCodeTflAdd);

  const auto op_inputs = op.Inputs();
  ASSERT_EQ(op_inputs.size(), 2);
  ASSERT_EQ(op_inputs.front().Get(), subgraph_inputs.front().Get());
  ASSERT_TRUE(MatchWeights(op_inputs.back(),
                           absl::Span<const float>({1.0, 2.0, 3.0, 4.0})));

  const auto op_outputs = op.Outputs();
  ASSERT_EQ(op_outputs.size(), 1);
  ASSERT_EQ(op_outputs.front().Get(), subgraph_outputs.front().Get());

  ASSERT_FALSE(subgraph_outputs.front().IsConstant());
  ASSERT_FALSE(subgraph_inputs.front().IsConstant());
}

INSTANTIATE_TEST_SUITE_P(ModelLoadTests, AddCstTest,
                         Values(MakeLoadFactory(kAddCst)));

INSTANTIATE_TEST_SUITE_P(ModelSerializeTests, AddCstTest,
                         Values(MakeRoundTripFactory(kAddCst)));

using SimpleMultiOpTest = TestWithModelFactory;

TEST_P(SimpleMultiOpTest, CheckGraph) {
  auto model = LoadModel();
  ASSERT_TRUE(model);

  // func.func @main(arg0)
  //   0 = tfl.add arg0, arg0
  //   1 = tfl.mul 0, 0
  //   2 = tfl.mul 1, 1
  //   3 = tfl.add 2, 2
  //   return 3

  auto subgraph = model->MainSubgraph();
  const auto subgraph_inputs = subgraph->Inputs();
  const auto subgraph_outputs = subgraph->Outputs();
  const auto ops = subgraph->Ops();

  ASSERT_EQ(subgraph_inputs.size(), 1);
  ASSERT_EQ(subgraph_outputs.size(), 1);

  const auto& internal_ops = subgraph->Get()->Ops();
  ASSERT_TRUE(
      ValidateLocalTopology(internal_ops.cbegin(), internal_ops.cend()));
  ASSERT_TRUE(ValidateSubgraphIO(*subgraph->Get()));

  ASSERT_EQ(ops.size(), 4);

  for (const auto& op : ops) {
    const auto inputs = op.Inputs();
    ASSERT_EQ(inputs.size(), 2);
    ASSERT_EQ(inputs.front().Get(), inputs.back().Get());
  }

  const TensorTypeInfo float_2by2_type(ElementType::Float32, {2, 2});

  ASSERT_TRUE(MatchOpType(ops.at(2), {float_2by2_type, float_2by2_type},
                          {float_2by2_type}));
  EXPECT_EQ(ops.at(2).Code(), kLiteRtOpCodeTflMul);
}

INSTANTIATE_TEST_SUITE_P(ModelLoadTests, SimpleMultiOpTest,
                         Values(MakeLoadFactory(kSimpleMultiOp)));

INSTANTIATE_TEST_SUITE_P(ModelSerializeTests, SimpleMultiOpTest,
                         Values(MakeRoundTripFactory(kSimpleMultiOp)));

using SimpleMultiSubgraphTest = TestWithModelFactory;

TEST_P(SimpleMultiSubgraphTest, CheckGraph) {
  auto model_wrap = LoadModel();
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap->Get();

  ASSERT_EQ(model.NumSubgraphs(), 3);

  {
    auto& main = *model.MainSubgraph();
    EXPECT_EQ(main.NumInputs(), 1);
    EXPECT_EQ(main.NumOutputs(), 1);
    EXPECT_EQ(main.Ops().size(), 1);
    EXPECT_EQ(main.Tensors().size(), 3);
    auto& op = main.Op(0);
    auto* cst = op.Inputs().back();
    auto data = Tensor(cst).WeightsData<float>();
    ASSERT_TRUE(data);
    EXPECT_THAT(*data, Each(FloatEq(-1.0)));
    EXPECT_TRUE(ValidateLocalTopology(main.Ops().cbegin(), main.Ops().cend()));
    EXPECT_TRUE(ValidateSubgraphIO(main));
  }

  {
    auto& func1 = model.Subgraph(1);
    EXPECT_EQ(func1.NumInputs(), 1);
    EXPECT_EQ(func1.NumOutputs(), 1);
    EXPECT_EQ(func1.Ops().size(), 1);
    EXPECT_EQ(func1.Tensors().size(), 3);
    auto& op = func1.Op(0);
    auto* cst = op.Inputs().back();
    auto data = Tensor(cst).WeightsData<float>();
    ASSERT_TRUE(data);
    EXPECT_THAT(*data, Each(FloatEq(1.0)));
    EXPECT_TRUE(
        ValidateLocalTopology(func1.Ops().cbegin(), func1.Ops().cend()));
    EXPECT_TRUE(ValidateSubgraphIO(func1));
  }

  {
    auto& func2 = model.Subgraph(2);
    EXPECT_EQ(func2.NumInputs(), 1);
    EXPECT_EQ(func2.NumOutputs(), 1);
    EXPECT_EQ(func2.Ops().size(), 1);
    EXPECT_EQ(func2.Tensors().size(), 3);
    auto& op = func2.Op(0);
    auto* cst = op.Inputs().back();
    auto data = Tensor(cst).WeightsData<float>();
    ASSERT_TRUE(data);
    EXPECT_THAT(*data, Each(FloatEq(2.0)));
    EXPECT_TRUE(
        ValidateLocalTopology(func2.Ops().cbegin(), func2.Ops().cend()));
    EXPECT_TRUE(ValidateSubgraphIO(func2));
  }
}

INSTANTIATE_TEST_SUITE_P(ModelLoadTests, SimpleMultiSubgraphTest,
                         Values(MakeLoadFactory(kSimpleMultiSubgraph)));

INSTANTIATE_TEST_SUITE_P(ModelSerializeTests, SimpleMultiSubgraphTest,
                         Values(MakeRoundTripFactory(kSimpleMultiSubgraph)));

// Test when flatbuffer export has optimized multiple tensors to share the
// same buffer.
using MultiSubgraphDupeConstTest = TestWithModelFactory;

TEST_P(MultiSubgraphDupeConstTest, CheckGraph) {
  static constexpr std::array kWeights = {1.0, 2.0, 3.0, 4.0};

  auto model_wrap = LoadModel();
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap->Get();

  ASSERT_EQ(model.NumSubgraphs(), 2);

  {
    ASSERT_EQ(model.Subgraph(0).Ops().size(), 1);
    ASSERT_EQ(model.Subgraph(0).Tensors().size(), 3);
    auto& cst = model.Subgraph(0).Op(0).Input(1);
    Tensor t(&cst);
    EXPECT_THAT(t.WeightsData<float>(),
                IsOkAndHolds(ElementsAreArray(kWeights)));
  }

  {
    ASSERT_EQ(model.Subgraph(1).Ops().size(), 1);
    ASSERT_EQ(model.Subgraph(1).Tensors().size(), 3);
    auto& cst = model.Subgraph(1).Op(0).Input(1);
    Tensor t(&cst);
    EXPECT_THAT(t.WeightsData<float>(),
                IsOkAndHolds(ElementsAreArray(kWeights)));
  }
  auto buf_id_0 = model.Subgraph(0).Op(0).Input(1).Weights().GetBufferId();
  auto buf_id_1 = model.Subgraph(1).Op(0).Input(1).Weights().GetBufferId();
  ASSERT_EQ(buf_id_0, buf_id_1);
}

INSTANTIATE_TEST_SUITE_P(ModelLoadTests, MultiSubgraphDupeConstTest,
                         Values(MakeLoadFactory(kCstMultiSubgraph)));

INSTANTIATE_TEST_SUITE_P(ModelSerializeTests, MultiSubgraphDupeConstTest,
                         Values(MakeRoundTripFactory(kCstMultiSubgraph)));

// Tests that programmatically check litert against tflite models.
//===---------------------------------------------------------------------------

using ModelLoadOpCheckTest = TestWithModelPath;

TEST_P(ModelLoadOpCheckTest, CheckOps) {
  const auto model_path = GetTestModelPath();

  auto flatbuffer = FlatbufferWrapper::CreateFromTflFile(model_path);
  ASSERT_TRUE(flatbuffer);
  auto expected_fb = flatbuffer->get()->Unpack();

  auto model = LoadModelFromFile(model_path);
  ASSERT_TRUE(model);

  const auto* subgraph = model->get()->MainSubgraph();
  const auto& ops = subgraph->Ops();

  const auto& fb_subgraph = *expected_fb->subgraphs.front();
  const auto& fb_ops = fb_subgraph.operators;
  const auto& fb_tensors = fb_subgraph.tensors;

  ASSERT_EQ(ops.size(), fb_ops.size());

  auto get_tfl_tensor = [&](uint32_t ind) -> const TflTensor& {
    return *fb_tensors.at(ind);
  };

  for (auto i = 0; i < ops.size(); ++i) {
    ASSERT_TRUE(EqualsFbOp(*ops.at(i), *fb_ops.at(i), get_tfl_tensor));
  }
}

INSTANTIATE_TEST_SUITE_P(ModelLoadQuantizedOpCheckTest, ModelLoadOpCheckTest,
                         ::testing::ValuesIn(kAllQModels));

INSTANTIATE_TEST_SUITE_P(ModelLoadDynamicOpCheckTest, ModelLoadOpCheckTest,
                         ::testing::ValuesIn({kDynamicShapeModel}));

using ModelSerializeOpCheckTest = TestWithModelPath;

TEST_P(ModelSerializeOpCheckTest, CheckOps) {
  const auto model_path = GetTestModelPath();

  // Save the initial fb for comparison.
  auto expected_fb_data = FlatbufferWrapper::CreateFromTflFile(model_path);
  ASSERT_TRUE(expected_fb_data);
  auto expected_fb = expected_fb_data->get()->Unpack();

  // Round trip the model.
  auto model = LoadModelFromFile(model_path);
  ASSERT_TRUE(model);
  auto serialized = SerializeModel(std::move(**model));

  auto actual_fb_data = FlatbufferWrapper::CreateFromBuffer(*serialized);
  ASSERT_TRUE(actual_fb_data);
  auto actual_fb = actual_fb_data->get()->Unpack();

  const auto& expected_fb_subgraph = *expected_fb->subgraphs.front();
  const auto& expected_fb_ops = expected_fb_subgraph.operators;
  const auto& expected_fb_tensors = expected_fb_subgraph.tensors;

  const auto& actual_fb_subgraph = *actual_fb->subgraphs.front();
  const auto& actual_fb_ops = actual_fb_subgraph.operators;
  const auto& actual_fb_tensors = actual_fb_subgraph.tensors;

  ASSERT_EQ(expected_fb_ops.size(), actual_fb_ops.size());
  for (auto i = 0; i < actual_fb_ops.size(); ++i) {
    const auto& expected = *expected_fb_ops.at(i);
    const auto& actual = *actual_fb_ops.at(i);
    EXPECT_EQ(expected.inputs.size(), actual.inputs.size());
    EXPECT_EQ(expected.outputs.size(), actual.outputs.size());
  }

  ASSERT_EQ(expected_fb_tensors.size(), actual_fb_tensors.size());
  for (auto i = 0; i < actual_fb_tensors.size(); ++i) {
    const auto& expected = *expected_fb_tensors.at(i);
    const auto& actual = *actual_fb_tensors.at(i);

    EXPECT_EQ(actual.type, expected.type);
    EXPECT_EQ(actual.shape, expected.shape);
    EXPECT_EQ(actual.shape_signature, expected.shape_signature);

    const auto expected_q_params = expected.quantization.get();
    const auto actual_q_params = actual.quantization.get();

    const auto neither_quantized =
        !IsQuantized(expected_q_params) && !IsQuantized(actual_q_params);
    const auto both_per_tensor = IsPerTensorQuantized(expected_q_params) &&
                                 IsPerTensorQuantized(actual_q_params);
    ASSERT_TRUE(neither_quantized || both_per_tensor);

    if (both_per_tensor) {
      const auto expected_per_tensor = AsPerTensorQparams(expected_q_params);
      const auto actual_per_tensor = AsPerTensorQparams(actual_q_params);
      EXPECT_EQ(*expected_per_tensor, *actual_per_tensor);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(ModelSerializeOpCheckTest, ModelSerializeOpCheckTest,
                         ::testing::ValuesIn({kOneMul, kDynamicShapeModel}));

INSTANTIATE_TEST_SUITE_P(ModelSerializeQuantizedOpCheckTest,
                         ModelSerializeOpCheckTest,
                         ::testing::ValuesIn(kAllQModels));

TEST(ModelLoadTest, IgnoreNonDispatchCustomOp) {
  Expected<FlatbufferWrapper::Ptr> flatbuffer =
      FlatbufferWrapper::CreateFromTflFile(GetTestFilePath(kAddSimple));
  auto tfl_model = flatbuffer->get()->Unpack();

  // Add a non-dispatch custom op with some dummy custom options
  auto op_code = std::make_unique<tflite::OperatorCodeT>();
  op_code->builtin_code = tflite::BuiltinOperator_CUSTOM;
  op_code->custom_code = "NOT_DISPATCH_OP";
  tfl_model->operator_codes.push_back(std::move(op_code));

  auto op = std::make_unique<tflite::OperatorT>(
      *tfl_model->subgraphs[0]->operators[0]);
  op->opcode_index = tfl_model->operator_codes.size() - 1;
  // A dummy custom_options that isn't a valid DispatchOpOptions
  op->custom_options = {1, 2, 3, 4};

  auto tensor = std::make_unique<tflite::TensorT>();
  tensor->name = "dummy_tensor";
  tensor->type = tflite::TensorType_FLOAT32;
  tfl_model->subgraphs[0]->tensors.push_back(std::move(tensor));
  // We just `push_back` a tensor into `tensor`, so the size of `tensors` should
  // be non-zero.
  int new_tensor_index = tfl_model->subgraphs[0]->tensors.size() - 1;

  op->outputs.clear();
  op->outputs.push_back(new_tensor_index);

  tfl_model->subgraphs[0]->operators.push_back(std::move(op));

  auto serialized = SerializeFlatbuffer(*tfl_model);
  auto litert_model = LoadModelFromBuffer(serialized);
  ASSERT_TRUE(litert_model);

  // The model should load correctly but the custom op shouldn't have an asset
  // attached
  const auto litert_op = litert_model->get()->MainSubgraph()->Ops().back();
  EXPECT_EQ(litert_op->OpCode(), kLiteRtOpCodeTflCustom);
  EXPECT_FALSE(litert_model->get()->FindOpAsset(litert_op));
}

TEST(ModelSerializeTest, SerializeWithExistingDispatchOpCodeNotAtEnd) {
  auto model = litert::testing::LoadTestFileModel(kAddSimple);
  ASSERT_TRUE(model);
  auto litert_model_val = std::move(*model.Get());
  auto op = litert_model_val.MainSubgraph()->Ops().front();

  OwningBufferRef<uint8_t> buffer("SOME_BYTE_CODE");
  const auto buf_id =
      litert_model_val.Buffers()->RegisterOwnedBuffer(std::move(buffer));
  litert_model_val.AttachAssetToOp(op, buf_id, "foo");

  op->SetOpCode(kLiteRtOpCodeTflCustom);
  litert::internal::SetTflOpCodeInd(*op,
                                    litert::internal::kDispatchOpCodeTflInd);

  auto serialized1 = SerializeModel(std::move(litert_model_val));
  ASSERT_TRUE(serialized1);

  auto flatbuffer = FlatbufferWrapper::CreateFromBuffer(*serialized1);
  auto tfl_model = flatbuffer->get()->Unpack();

  // Add another custom op code AFTER the dispatch op code
  auto op_code = std::make_unique<tflite::OperatorCodeT>();
  op_code->builtin_code = tflite::BuiltinOperator_CUSTOM;
  op_code->custom_code = "OTHER_CUSTOM_OP";
  tfl_model->operator_codes.push_back(std::move(op_code));

  auto serialized2 = SerializeFlatbuffer(*tfl_model);
  auto litert_model2 = LoadModelFromBuffer(serialized2);
  ASSERT_TRUE(litert_model2);

  // Now serialize it again
  auto serialized3 = SerializeModel(std::move(*litert_model2->get()));
  ASSERT_TRUE(serialized3);

  // Verify the new flatbuffer uses the correct custom code
  auto fb3 = FlatbufferWrapper::CreateFromBuffer(*serialized3);
  auto tfl3 = fb3->get()->Unpack();

  const auto& tfl_op = tfl3->subgraphs[0]->operators[0];
  const auto& code = tfl3->operator_codes[tfl_op->opcode_index];
  EXPECT_EQ(code->custom_code, "DISPATCH_OP");
}

TEST(ModelLoadTest, LoadModelFromFileWithExternalBuffer) {
  auto flatbuffer =
      FlatbufferWrapper::CreateFromTflFile(GetTestFilePath(kAddCst));
  ASSERT_TRUE(flatbuffer);
  auto tfl_model = flatbuffer->get()->Unpack();

  // Convert the inline constant tensor (input 1 of the add op) into an
  // externalized weight tensor referencing external_buffer 1 in group 1.
  const int32_t cst_tensor_idx =
      tfl_model->subgraphs[0]->operators[0]->inputs[1];
  auto& tfl_cst_tensor = *tfl_model->subgraphs[0]->tensors[cst_tensor_idx];
  const uint32_t old_buf_idx = tfl_cst_tensor.buffer;
  tfl_model->buffers[old_buf_idx]->data.clear();
  tfl_cst_tensor.buffer = 0;
  tfl_cst_tensor.external_buffer = 1;

  auto group0 = std::make_unique<tflite::ExternalBufferGroupT>();
  group0->name = "";
  auto group1 = std::make_unique<tflite::ExternalBufferGroupT>();
  group1->name = "ext_weights.bin";
  tfl_model->external_buffer_groups.push_back(std::move(group0));
  tfl_model->external_buffer_groups.push_back(std::move(group1));

  constexpr std::array<float, 4> kExternalWeights = {10.0f, 20.0f, 30.0f,
                                                     40.0f};
  auto ext_buf = std::make_unique<tflite::ExternalBufferT>();
  ext_buf->id = 1;
  ext_buf->group = 1;
  ext_buf->offset = 0;
  ext_buf->length = sizeof(kExternalWeights);
  tfl_model->external_buffers.push_back(std::move(ext_buf));

  auto serialized = SerializeFlatbuffer(*tfl_model);
  auto temp_dir = ::litert::testing::UniqueTestDirectory::Create();
  ASSERT_TRUE(temp_dir);

  std::filesystem::path temp_path(std::string(temp_dir->Str()));
  std::filesystem::path model_path = temp_path / "ext_add_cst.tflite";
  {
    std::ofstream model_file(model_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(model_file.is_open());
    model_file.write(serialized.StrData(), serialized.Size());
  }

  std::filesystem::path weights_path = temp_path / "ext_weights.bin";
  {
    std::ofstream weights_file(weights_path,
                               std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(weights_file.is_open());
    weights_file.write(reinterpret_cast<const char*>(kExternalWeights.data()),
                       sizeof(kExternalWeights));
  }

  auto loaded_model = LoadModelFromFile(model_path.string());
  ASSERT_TRUE(loaded_model);

  auto* main_subgraph = loaded_model->get()->MainSubgraph();
  ASSERT_NE(main_subgraph, nullptr);
  ASSERT_EQ(main_subgraph->Ops().size(), 1);
  auto& cst_tensor = main_subgraph->Op(0).Input(1);
  EXPECT_TRUE(IsConstant(cst_tensor));

  Tensor cc_cst(&cst_tensor);
  EXPECT_THAT(cc_cst.WeightsData<float>(),
              IsOkAndHolds(ElementsAreArray(kExternalWeights)));

  const auto* ext_buf_meta =
      GetTflExternalBuffer(**loaded_model, cst_tensor.Weights().GetBufferId());
  ASSERT_NE(ext_buf_meta, nullptr);
  EXPECT_EQ(ext_buf_meta->id(), 1u);
}

TEST(ModelLoadTest, LoadModelFromFileWithZeroIndexedExternalBufferGroup) {
  auto flatbuffer =
      FlatbufferWrapper::CreateFromTflFile(GetTestFilePath(kAddCst));
  ASSERT_TRUE(flatbuffer);
  auto tfl_model = flatbuffer->get()->Unpack();

  // Match MLIR flatbuffer_export.cc convention: external_buffer_id has MSB set
  // ((1u << 31) | 0u) and external_buffer_groups is 0-indexed (group = 0).
  constexpr uint32_t kConverterExtBufId = (1u << 31) | 0u;
  const int32_t cst_tensor_idx =
      tfl_model->subgraphs[0]->operators[0]->inputs[1];
  auto& tfl_cst_tensor = *tfl_model->subgraphs[0]->tensors[cst_tensor_idx];
  tfl_model->buffers[tfl_cst_tensor.buffer]->data.clear();
  tfl_cst_tensor.buffer = 0;
  tfl_cst_tensor.external_buffer = kConverterExtBufId;

  auto group0 = std::make_unique<tflite::ExternalBufferGroupT>();
  group0->name = "ext_weights.bin";
  tfl_model->external_buffer_groups.push_back(std::move(group0));

  constexpr std::array<float, 4> kExternalWeights = {10.0f, 20.0f, 30.0f,
                                                     40.0f};
  auto ext_buf = std::make_unique<tflite::ExternalBufferT>();
  ext_buf->id = kConverterExtBufId;
  ext_buf->group = 0;
  ext_buf->offset = 0;
  ext_buf->length = sizeof(kExternalWeights);
  tfl_model->external_buffers.push_back(std::move(ext_buf));

  auto serialized = SerializeFlatbuffer(*tfl_model);
  auto temp_dir = ::litert::testing::UniqueTestDirectory::Create();
  ASSERT_TRUE(temp_dir);

  std::filesystem::path temp_path(std::string(temp_dir->Str()));
  std::filesystem::path model_path = temp_path / "ext_group0.tflite";
  {
    std::ofstream model_file(model_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(model_file.is_open());
    model_file.write(serialized.StrData(), serialized.Size());
  }
  {
    std::ofstream weights_file(temp_path / "ext_weights.bin",
                               std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(weights_file.is_open());
    weights_file.write(reinterpret_cast<const char*>(kExternalWeights.data()),
                       sizeof(kExternalWeights));
  }

  auto loaded_model = LoadModelFromFile(model_path.string());
  ASSERT_TRUE(loaded_model);

  auto& cst_tensor = loaded_model->get()->MainSubgraph()->Op(0).Input(1);
  EXPECT_TRUE(IsConstant(cst_tensor));
  EXPECT_THAT(Tensor(&cst_tensor).WeightsData<float>(),
              IsOkAndHolds(ElementsAreArray(kExternalWeights)));
}

TEST(ModelLoadTest, LoadModelFromFileWithOutOfBoundsExternalBufferFails) {
  auto flatbuffer =
      FlatbufferWrapper::CreateFromTflFile(GetTestFilePath(kAddCst));
  ASSERT_TRUE(flatbuffer);
  auto tfl_model = flatbuffer->get()->Unpack();

  const int32_t cst_tensor_idx =
      tfl_model->subgraphs[0]->operators[0]->inputs[1];
  auto& tfl_cst_tensor = *tfl_model->subgraphs[0]->tensors[cst_tensor_idx];
  tfl_cst_tensor.buffer = 0;
  tfl_cst_tensor.external_buffer = 1;

  auto group0 = std::make_unique<tflite::ExternalBufferGroupT>();
  group0->name = "";
  auto group1 = std::make_unique<tflite::ExternalBufferGroupT>();
  group1->name = "ext_weights.bin";
  tfl_model->external_buffer_groups.push_back(std::move(group0));
  tfl_model->external_buffer_groups.push_back(std::move(group1));

  constexpr std::array<float, 4> kExternalWeights = {1.0f, 2.0f, 3.0f, 4.0f};
  auto ext_buf = std::make_unique<tflite::ExternalBufferT>();
  ext_buf->id = 1;
  ext_buf->group = 1;
  ext_buf->offset = 8;
  ext_buf->length = sizeof(kExternalWeights);  // 8 + 16 > 16 (out of bounds)
  tfl_model->external_buffers.push_back(std::move(ext_buf));

  auto serialized = SerializeFlatbuffer(*tfl_model);
  auto temp_dir = ::litert::testing::UniqueTestDirectory::Create();
  ASSERT_TRUE(temp_dir);

  std::filesystem::path temp_path(std::string(temp_dir->Str()));
  std::filesystem::path model_path = temp_path / "oob_ext.tflite";
  {
    std::ofstream model_file(model_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(model_file.is_open());
    model_file.write(serialized.StrData(), serialized.Size());
  }

  std::filesystem::path weights_path = temp_path / "ext_weights.bin";
  {
    std::ofstream weights_file(weights_path,
                               std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(weights_file.is_open());
    weights_file.write(reinterpret_cast<const char*>(kExternalWeights.data()),
                       sizeof(kExternalWeights));
  }

  EXPECT_THAT(LoadModelFromFile(model_path.string()),
              IsError(kLiteRtStatusErrorInvalidFlatbuffer));
}

TEST(ModelSerializeTest, PreservesExternalBuffersWithoutInlining) {
  auto flatbuffer =
      FlatbufferWrapper::CreateFromTflFile(GetTestFilePath(kAddCst));
  ASSERT_TRUE(flatbuffer);
  auto tfl_model = flatbuffer->get()->Unpack();

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

  constexpr std::array<float, 4> kExternalWeights = {5.0f, 6.0f, 7.0f, 8.0f};
  auto ext_buf = std::make_unique<tflite::ExternalBufferT>();
  ext_buf->id = 1;
  ext_buf->group = 1;
  ext_buf->offset = 0;
  ext_buf->length = sizeof(kExternalWeights);
  tfl_model->external_buffers.push_back(std::move(ext_buf));

  auto initial_serialized = SerializeFlatbuffer(*tfl_model);
  auto temp_dir = ::litert::testing::UniqueTestDirectory::Create();
  ASSERT_TRUE(temp_dir);

  std::filesystem::path temp_path(std::string(temp_dir->Str()));
  std::filesystem::path model_path = temp_path / "ext_add_cst.tflite";
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

  auto loaded_model = LoadModelFromFile(model_path.string());
  ASSERT_TRUE(loaded_model);

  auto reserialized = SerializeModel(std::move(**loaded_model));
  ASSERT_TRUE(reserialized);

  auto fb = FlatbufferWrapper::CreateFromBuffer(*reserialized);
  ASSERT_TRUE(fb);
  auto unpacked = fb->get()->Unpack();

  const int32_t new_cst_idx = unpacked->subgraphs[0]->operators[0]->inputs[1];
  const auto& new_cst_tensor = *unpacked->subgraphs[0]->tensors[new_cst_idx];
  EXPECT_EQ(new_cst_tensor.external_buffer, 1u);
  EXPECT_EQ(new_cst_tensor.buffer, 0u);

  // Ensure external weights were not duplicated into inline tensor buffers.
  for (const auto& tfl_tensor : unpacked->subgraphs[0]->tensors) {
    EXPECT_TRUE(unpacked->buffers[tfl_tensor->buffer]->data.empty());
  }

  ASSERT_EQ(unpacked->external_buffers.size(), 1u);
  EXPECT_EQ(unpacked->external_buffers[0]->id, 1u);
  EXPECT_EQ(unpacked->external_buffers[0]->group, 1u);
  EXPECT_EQ(unpacked->external_buffers[0]->offset, 0u);
  EXPECT_EQ(unpacked->external_buffers[0]->length, sizeof(kExternalWeights));

  ASSERT_EQ(unpacked->external_buffer_groups.size(), 2u);
  EXPECT_EQ(unpacked->external_buffer_groups[1]->name, "ext_weights.bin");
}

TEST(ModelLoadTest, LoadModelFromFileWithMultipleExternalBufferGroups) {
  auto flatbuffer =
      FlatbufferWrapper::CreateFromTflFile(GetTestFilePath(kAddCst));
  ASSERT_TRUE(flatbuffer);
  auto tfl_model = flatbuffer->get()->Unpack();

  auto& sg = *tfl_model->subgraphs[0];
  const int32_t orig_input_idx = sg.operators[0]->inputs[0];
  const int32_t cst1_idx = sg.operators[0]->inputs[1];
  const int32_t orig_output_idx = sg.operators[0]->outputs[0];

  tfl_model->buffers[sg.tensors[cst1_idx]->buffer]->data.clear();
  sg.tensors[cst1_idx]->buffer = 0;
  sg.tensors[cst1_idx]->external_buffer = 1;

  auto tmp_tensor = std::make_unique<TflTensor>(*sg.tensors[orig_output_idx]);
  tmp_tensor->name = "tmp_out";
  tmp_tensor->buffer = 0;
  tmp_tensor->external_buffer = 0;
  sg.tensors.push_back(std::move(tmp_tensor));
  const int32_t tmp_idx = sg.tensors.size() - 1;

  auto cst2_tensor = std::make_unique<TflTensor>(*sg.tensors[cst1_idx]);
  cst2_tensor->name = "ext_cst_2";
  cst2_tensor->buffer = 0;
  cst2_tensor->external_buffer = 2;
  sg.tensors.push_back(std::move(cst2_tensor));
  const int32_t cst2_idx = sg.tensors.size() - 1;

  sg.operators[0]->inputs = {orig_input_idx, cst1_idx};
  sg.operators[0]->outputs = {tmp_idx};

  auto op2 = std::make_unique<TflOp>(*sg.operators[0]);
  op2->inputs = {tmp_idx, cst2_idx};
  op2->outputs = {orig_output_idx};
  sg.operators.push_back(std::move(op2));

  auto group0 = std::make_unique<tflite::ExternalBufferGroupT>();
  group0->name = "";
  auto group1 = std::make_unique<tflite::ExternalBufferGroupT>();
  group1->name = "group1.bin";
  auto group2 = std::make_unique<tflite::ExternalBufferGroupT>();
  group2->name = "group2.bin";
  tfl_model->external_buffer_groups.push_back(std::move(group0));
  tfl_model->external_buffer_groups.push_back(std::move(group1));
  tfl_model->external_buffer_groups.push_back(std::move(group2));

  constexpr std::array<float, 4> kGroup1Weights = {1.5f, 2.5f, 3.5f, 4.5f};
  constexpr std::array<float, 4> kGroup2Weights = {11.0f, 22.0f, 33.0f, 44.0f};
  constexpr uint64_t kGroup2Offset = 16;

  auto ext_buf1 = std::make_unique<tflite::ExternalBufferT>();
  ext_buf1->id = 1;
  ext_buf1->group = 1;
  ext_buf1->offset = 0;
  ext_buf1->length = sizeof(kGroup1Weights);
  auto ext_buf2 = std::make_unique<tflite::ExternalBufferT>();
  ext_buf2->id = 2;
  ext_buf2->group = 2;
  ext_buf2->offset = kGroup2Offset;
  ext_buf2->length = sizeof(kGroup2Weights);
  tfl_model->external_buffers.push_back(std::move(ext_buf1));
  tfl_model->external_buffers.push_back(std::move(ext_buf2));

  auto serialized = SerializeFlatbuffer(*tfl_model);
  auto temp_dir = ::litert::testing::UniqueTestDirectory::Create();
  ASSERT_TRUE(temp_dir);
  std::filesystem::path temp_path(std::string(temp_dir->Str()));

  std::filesystem::path model_path = temp_path / "multi_group.tflite";
  {
    std::ofstream model_file(model_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(model_file.is_open());
    model_file.write(serialized.StrData(), serialized.Size());
  }
  {
    std::ofstream g1_file(temp_path / "group1.bin",
                          std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(g1_file.is_open());
    g1_file.write(reinterpret_cast<const char*>(kGroup1Weights.data()),
                  sizeof(kGroup1Weights));
  }
  {
    std::vector<uint8_t> g2_bytes(kGroup2Offset + sizeof(kGroup2Weights), 0);
    std::memcpy(g2_bytes.data() + kGroup2Offset, kGroup2Weights.data(),
                sizeof(kGroup2Weights));
    std::ofstream g2_file(temp_path / "group2.bin",
                          std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(g2_file.is_open());
    g2_file.write(reinterpret_cast<const char*>(g2_bytes.data()),
                  g2_bytes.size());
  }

  auto loaded_model = LoadModelFromFile(model_path.string());
  ASSERT_TRUE(loaded_model);

  auto* main_subgraph = loaded_model->get()->MainSubgraph();
  ASSERT_EQ(main_subgraph->Ops().size(), 2u);
  auto& loaded_cst1 = main_subgraph->Op(0).Input(1);
  auto& loaded_cst2 = main_subgraph->Op(1).Input(1);
  EXPECT_TRUE(IsConstant(loaded_cst1));
  EXPECT_TRUE(IsConstant(loaded_cst2));

  EXPECT_THAT(Tensor(&loaded_cst1).WeightsData<float>(),
              IsOkAndHolds(ElementsAreArray(kGroup1Weights)));
  EXPECT_THAT(Tensor(&loaded_cst2).WeightsData<float>(),
              IsOkAndHolds(ElementsAreArray(kGroup2Weights)));
}

TEST(ModelLoadTest, SharedExternalBufferAcrossTensorsAndSubgraphs) {
  auto flatbuffer =
      FlatbufferWrapper::CreateFromTflFile(GetTestFilePath(kCstMultiSubgraph));
  ASSERT_TRUE(flatbuffer);
  auto tfl_model = flatbuffer->get()->Unpack();
  ASSERT_EQ(tfl_model->subgraphs.size(), 2u);

  // Move the shared constant buffer into external_buffer 1 in both subgraphs.
  const int32_t sg0_cst_idx = tfl_model->subgraphs[0]->operators[0]->inputs[1];
  const int32_t sg1_cst_idx = tfl_model->subgraphs[1]->operators[0]->inputs[1];
  tfl_model->buffers[tfl_model->subgraphs[0]->tensors[sg0_cst_idx]->buffer]
      ->data.clear();
  tfl_model->subgraphs[0]->tensors[sg0_cst_idx]->buffer = 0;
  tfl_model->subgraphs[0]->tensors[sg0_cst_idx]->external_buffer = 1;
  tfl_model->subgraphs[1]->tensors[sg1_cst_idx]->buffer = 0;
  tfl_model->subgraphs[1]->tensors[sg1_cst_idx]->external_buffer = 1;

  auto group0 = std::make_unique<tflite::ExternalBufferGroupT>();
  group0->name = "";
  auto group1 = std::make_unique<tflite::ExternalBufferGroupT>();
  group1->name = "shared_ext.bin";
  tfl_model->external_buffer_groups.push_back(std::move(group0));
  tfl_model->external_buffer_groups.push_back(std::move(group1));

  constexpr std::array<float, 4> kSharedWeights = {9.0f, 8.0f, 7.0f, 6.0f};
  auto ext_buf = std::make_unique<tflite::ExternalBufferT>();
  ext_buf->id = 1;
  ext_buf->group = 1;
  ext_buf->offset = 0;
  ext_buf->length = sizeof(kSharedWeights);
  tfl_model->external_buffers.push_back(std::move(ext_buf));

  auto serialized = SerializeFlatbuffer(*tfl_model);
  auto temp_dir = ::litert::testing::UniqueTestDirectory::Create();
  ASSERT_TRUE(temp_dir);
  std::filesystem::path temp_path(std::string(temp_dir->Str()));

  std::filesystem::path model_path = temp_path / "shared_ext.tflite";
  {
    std::ofstream model_file(model_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(model_file.is_open());
    model_file.write(serialized.StrData(), serialized.Size());
  }
  {
    std::ofstream weights_file(temp_path / "shared_ext.bin",
                               std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(weights_file.is_open());
    weights_file.write(reinterpret_cast<const char*>(kSharedWeights.data()),
                       sizeof(kSharedWeights));
  }

  auto loaded_model = LoadModelFromFile(model_path.string());
  ASSERT_TRUE(loaded_model);
  auto& model = **loaded_model;
  ASSERT_EQ(model.NumSubgraphs(), 2u);

  auto& cst0 = model.Subgraph(0).Op(0).Input(1);
  auto& cst1 = model.Subgraph(1).Op(0).Input(1);
  // Both tensors must share the same BufferId in BufferManager.
  EXPECT_EQ(cst0.Weights().GetBufferId(), cst1.Weights().GetBufferId());
  EXPECT_THAT(Tensor(&cst0).WeightsData<float>(),
              IsOkAndHolds(ElementsAreArray(kSharedWeights)));
  EXPECT_THAT(Tensor(&cst1).WeightsData<float>(),
              IsOkAndHolds(ElementsAreArray(kSharedWeights)));

  // Round-trip through SerializeModel and verify both subgraphs still reference
  // external_buffer 1 with only 1 entry in external_buffers.
  auto reserialized = SerializeModel(std::move(model));
  ASSERT_TRUE(reserialized);
  auto fb = FlatbufferWrapper::CreateFromBuffer(*reserialized);
  ASSERT_TRUE(fb);
  auto unpacked = fb->get()->Unpack();
  ASSERT_EQ(unpacked->external_buffers.size(), 1u);
  EXPECT_EQ(unpacked->external_buffers[0]->id, 1u);
  EXPECT_EQ(unpacked->subgraphs[0]
                ->tensors[unpacked->subgraphs[0]->operators[0]->inputs[1]]
                ->external_buffer,
            1u);
  EXPECT_EQ(unpacked->subgraphs[1]
                ->tensors[unpacked->subgraphs[1]->operators[0]->inputs[1]]
                ->external_buffer,
            1u);
}

TEST(ModelLoadTest,
     LoadModelFromFileWithMissingExternalGroupFileSucceedsDeferred) {
  auto flatbuffer =
      FlatbufferWrapper::CreateFromTflFile(GetTestFilePath(kAddCst));
  ASSERT_TRUE(flatbuffer);
  auto tfl_model = flatbuffer->get()->Unpack();

  const int32_t cst_tensor_idx =
      tfl_model->subgraphs[0]->operators[0]->inputs[1];
  auto& tfl_cst_tensor = *tfl_model->subgraphs[0]->tensors[cst_tensor_idx];
  tfl_model->buffers[tfl_cst_tensor.buffer]->data.clear();
  tfl_cst_tensor.buffer = 0;
  tfl_cst_tensor.external_buffer = 1;

  auto group0 = std::make_unique<tflite::ExternalBufferGroupT>();
  group0->name = "";
  auto group1 = std::make_unique<tflite::ExternalBufferGroupT>();
  group1->name = "virtual_runtime_group";
  tfl_model->external_buffer_groups.push_back(std::move(group0));
  tfl_model->external_buffer_groups.push_back(std::move(group1));

  auto ext_buf = std::make_unique<tflite::ExternalBufferT>();
  ext_buf->id = 1;
  ext_buf->group = 1;
  ext_buf->offset = 0;
  ext_buf->length = 16;
  tfl_model->external_buffers.push_back(std::move(ext_buf));

  auto serialized = SerializeFlatbuffer(*tfl_model);
  auto temp_dir = ::litert::testing::UniqueTestDirectory::Create();
  ASSERT_TRUE(temp_dir);
  std::filesystem::path temp_path(std::string(temp_dir->Str()));

  // Write only the .tflite file, NOT "virtual_runtime_group".
  std::filesystem::path model_path = temp_path / "deferred_ext.tflite";
  {
    std::ofstream model_file(model_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(model_file.is_open());
    model_file.write(serialized.StrData(), serialized.Size());
  }

  auto loaded_model = LoadModelFromFile(model_path.string());
  ASSERT_TRUE(loaded_model);

  auto& model = **loaded_model;
  auto& cst_tensor = model.MainSubgraph()->Op(0).Input(1);
  // Buffer is not yet resolved from disk, but external_buffer_id is recorded.
  EXPECT_EQ(cst_tensor.Weights().Buffer().Size(), 0u);
  const auto* ext_buf_meta =
      GetTflExternalBuffer(model, cst_tensor.Weights().GetBufferId());
  ASSERT_NE(ext_buf_meta, nullptr);
  EXPECT_EQ(ext_buf_meta->id(), 1u);

  // Reserializing should still preserve the external_buffers metadata.
  auto reserialized = SerializeModel(std::move(model));
  ASSERT_TRUE(reserialized);
  auto fb = FlatbufferWrapper::CreateFromBuffer(*reserialized);
  ASSERT_TRUE(fb);
  auto unpacked = fb->get()->Unpack();
  ASSERT_EQ(unpacked->external_buffers.size(), 1u);
  EXPECT_EQ(unpacked->external_buffers[0]->id, 1u);
  ASSERT_EQ(unpacked->external_buffer_groups.size(), 2u);
  EXPECT_EQ(unpacked->external_buffer_groups[1]->name, "virtual_runtime_group");
}

TEST(ModelLoadTest, LoadModelFromFileWithInvalidExternalBufferGroupIndexFails) {
  auto temp_dir = ::litert::testing::UniqueTestDirectory::Create();
  ASSERT_TRUE(temp_dir);
  std::filesystem::path temp_path(std::string(temp_dir->Str()));

  auto make_model_with_group = [&](uint32_t group_idx,
                                   absl::string_view group_name) {
    auto flatbuffer =
        FlatbufferWrapper::CreateFromTflFile(GetTestFilePath(kAddCst));
    auto tfl_model = flatbuffer->get()->Unpack();
    const int32_t cst_idx = tfl_model->subgraphs[0]->operators[0]->inputs[1];
    tfl_model->subgraphs[0]->tensors[cst_idx]->buffer = 0;
    tfl_model->subgraphs[0]->tensors[cst_idx]->external_buffer = 1;

    auto group0 = std::make_unique<tflite::ExternalBufferGroupT>();
    group0->name = "";
    auto group1 = std::make_unique<tflite::ExternalBufferGroupT>();
    group1->name = std::string(group_name);
    tfl_model->external_buffer_groups.push_back(std::move(group0));
    tfl_model->external_buffer_groups.push_back(std::move(group1));

    auto ext_buf = std::make_unique<tflite::ExternalBufferT>();
    ext_buf->id = 1;
    ext_buf->group = group_idx;
    ext_buf->offset = 0;
    ext_buf->length = 16;
    tfl_model->external_buffers.push_back(std::move(ext_buf));
    return SerializeFlatbuffer(*tfl_model);
  };

  // Case 1: group_idx == 0 (reserved slot 0)
  {
    auto buf = make_model_with_group(0, "ext_weights.bin");
    auto path = temp_path / "bad_group_0.tflite";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(buf.StrData(), buf.Size());
    f.close();
    EXPECT_THAT(LoadModelFromFile(path.string()),
                IsError(kLiteRtStatusErrorInvalidFlatbuffer));
  }

  // Case 2: group_idx >= external_buffer_groups.size()
  {
    auto buf = make_model_with_group(5, "ext_weights.bin");
    auto path = temp_path / "bad_group_oob.tflite";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(buf.StrData(), buf.Size());
    f.close();
    EXPECT_THAT(LoadModelFromFile(path.string()),
                IsError(kLiteRtStatusErrorInvalidFlatbuffer));
  }

  // Case 3: group_idx == 1 but group name is empty
  {
    auto buf = make_model_with_group(1, "");
    auto path = temp_path / "bad_group_empty_name.tflite";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(buf.StrData(), buf.Size());
    f.close();
    EXPECT_THAT(LoadModelFromFile(path.string()),
                IsError(kLiteRtStatusErrorInvalidFlatbuffer));
  }
}

TEST(ModelLoadTest, LoadModelFromFileWithExternalBufferOffsetOverflowFails) {
  auto flatbuffer =
      FlatbufferWrapper::CreateFromTflFile(GetTestFilePath(kAddCst));
  ASSERT_TRUE(flatbuffer);
  auto tfl_model = flatbuffer->get()->Unpack();

  const int32_t cst_tensor_idx =
      tfl_model->subgraphs[0]->operators[0]->inputs[1];
  tfl_model->subgraphs[0]->tensors[cst_tensor_idx]->buffer = 0;
  tfl_model->subgraphs[0]->tensors[cst_tensor_idx]->external_buffer = 1;

  auto group0 = std::make_unique<tflite::ExternalBufferGroupT>();
  group0->name = "";
  auto group1 = std::make_unique<tflite::ExternalBufferGroupT>();
  group1->name = "ext_weights.bin";
  tfl_model->external_buffer_groups.push_back(std::move(group0));
  tfl_model->external_buffer_groups.push_back(std::move(group1));

  constexpr std::array<float, 4> kExternalWeights = {1.0f, 2.0f, 3.0f, 4.0f};
  auto ext_buf = std::make_unique<tflite::ExternalBufferT>();
  ext_buf->id = 1;
  ext_buf->group = 1;
  ext_buf->offset = std::numeric_limits<size_t>::max();
  ext_buf->length = sizeof(kExternalWeights);
  tfl_model->external_buffers.push_back(std::move(ext_buf));

  auto serialized = SerializeFlatbuffer(*tfl_model);
  auto temp_dir = ::litert::testing::UniqueTestDirectory::Create();
  ASSERT_TRUE(temp_dir);
  std::filesystem::path temp_path(std::string(temp_dir->Str()));

  std::filesystem::path model_path = temp_path / "overflow_ext.tflite";
  {
    std::ofstream model_file(model_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(model_file.is_open());
    model_file.write(serialized.StrData(), serialized.Size());
  }
  {
    std::ofstream weights_file(temp_path / "ext_weights.bin",
                               std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(weights_file.is_open());
    weights_file.write(reinterpret_cast<const char*>(kExternalWeights.data()),
                       sizeof(kExternalWeights));
  }

  EXPECT_THAT(LoadModelFromFile(model_path.string()),
              IsError(kLiteRtStatusErrorInvalidFlatbuffer));
}

TEST(ModelSerializeTest, MixedInlineAndExternalConstantTensors) {
  auto flatbuffer =
      FlatbufferWrapper::CreateFromTflFile(GetTestFilePath(kAddCst));
  ASSERT_TRUE(flatbuffer);
  auto tfl_model = flatbuffer->get()->Unpack();

  auto& sg = *tfl_model->subgraphs[0];
  const int32_t orig_input_idx = sg.operators[0]->inputs[0];
  const int32_t inline_cst_idx = sg.operators[0]->inputs[1];
  const int32_t orig_output_idx = sg.operators[0]->outputs[0];

  // Keep inline_cst_idx as an inline constant ([1, 2, 3, 4]), and add a second
  // op that uses an external constant ([5, 6, 7, 8]).
  auto tmp_tensor = std::make_unique<TflTensor>(*sg.tensors[orig_output_idx]);
  tmp_tensor->name = "add1_out";
  tmp_tensor->buffer = 0;
  tmp_tensor->external_buffer = 0;
  sg.tensors.push_back(std::move(tmp_tensor));
  const int32_t tmp_idx = sg.tensors.size() - 1;

  auto ext_cst_tensor =
      std::make_unique<TflTensor>(*sg.tensors[inline_cst_idx]);
  ext_cst_tensor->name = "ext_cst";
  ext_cst_tensor->buffer = 0;
  ext_cst_tensor->external_buffer = 1;
  sg.tensors.push_back(std::move(ext_cst_tensor));
  const int32_t ext_cst_idx = sg.tensors.size() - 1;

  sg.operators[0]->inputs = {orig_input_idx, inline_cst_idx};
  sg.operators[0]->outputs = {tmp_idx};

  auto op2 = std::make_unique<TflOp>(*sg.operators[0]);
  op2->inputs = {tmp_idx, ext_cst_idx};
  op2->outputs = {orig_output_idx};
  sg.operators.push_back(std::move(op2));

  auto group0 = std::make_unique<tflite::ExternalBufferGroupT>();
  group0->name = "";
  auto group1 = std::make_unique<tflite::ExternalBufferGroupT>();
  group1->name = "ext_weights.bin";
  tfl_model->external_buffer_groups.push_back(std::move(group0));
  tfl_model->external_buffer_groups.push_back(std::move(group1));

  constexpr std::array<float, 4> kExternalWeights = {5.0f, 6.0f, 7.0f, 8.0f};
  auto ext_buf = std::make_unique<tflite::ExternalBufferT>();
  ext_buf->id = 1;
  ext_buf->group = 1;
  ext_buf->offset = 0;
  ext_buf->length = sizeof(kExternalWeights);
  tfl_model->external_buffers.push_back(std::move(ext_buf));

  auto initial_serialized = SerializeFlatbuffer(*tfl_model);
  auto temp_dir = ::litert::testing::UniqueTestDirectory::Create();
  ASSERT_TRUE(temp_dir);
  std::filesystem::path temp_path(std::string(temp_dir->Str()));

  std::filesystem::path model_path = temp_path / "mixed_cst.tflite";
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

  auto loaded_model = LoadModelFromFile(model_path.string());
  ASSERT_TRUE(loaded_model);
  auto& model = **loaded_model;
  EXPECT_THAT(
      Tensor(&model.MainSubgraph()->Op(0).Input(1)).WeightsData<float>(),
      IsOkAndHolds(ElementsAreArray({1.0f, 2.0f, 3.0f, 4.0f})));
  EXPECT_THAT(
      Tensor(&model.MainSubgraph()->Op(1).Input(1)).WeightsData<float>(),
      IsOkAndHolds(ElementsAreArray(kExternalWeights)));

  auto reserialized = SerializeModel(std::move(model));
  ASSERT_TRUE(reserialized);
  auto fb = FlatbufferWrapper::CreateFromBuffer(*reserialized);
  ASSERT_TRUE(fb);
  auto unpacked = fb->get()->Unpack();

  const int32_t out_inline_idx =
      unpacked->subgraphs[0]->operators[0]->inputs[1];
  const int32_t out_ext_idx = unpacked->subgraphs[0]->operators[1]->inputs[1];
  const auto& out_inline_tensor =
      *unpacked->subgraphs[0]->tensors[out_inline_idx];
  const auto& out_ext_tensor = *unpacked->subgraphs[0]->tensors[out_ext_idx];

  // Inline constant tensor keeps its non-zero buffer with data and has
  // external_buffer == 0.
  EXPECT_NE(out_inline_tensor.buffer, 0u);
  EXPECT_EQ(out_inline_tensor.external_buffer, 0u);
  EXPECT_EQ(unpacked->buffers[out_inline_tensor.buffer]->data.size(),
            4 * sizeof(float));

  // External constant tensor keeps buffer == 0 and external_buffer == 1.
  EXPECT_EQ(out_ext_tensor.buffer, 0u);
  EXPECT_EQ(out_ext_tensor.external_buffer, 1u);
  ASSERT_EQ(unpacked->external_buffers.size(), 1u);
  EXPECT_EQ(unpacked->external_buffers[0]->id, 1u);
}

}  // namespace
}  // namespace litert::internal
