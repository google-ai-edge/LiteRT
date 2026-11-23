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

#include "litert/vendors/nvidia/tools/carve_model.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"
#include "litert/vendors/nvidia/tools/carve_pattern.h"
#include "litert/vendors/nvidia/tools/carve_test_util.h"
#include "litert/vendors/nvidia/tools/carve_verify.h"
#include "tflite/schema/schema_generated.h"

namespace litert::tools {
namespace {

using testing::AsBuffer;
using ::testing::ElementsAre;
using testing::FactorizedMlpModelOptions;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::SizeIs;

std::vector<CarvePattern> Builtin() {
  auto patterns = ParseCarvePatterns(BuiltinCarvePatternText());
  EXPECT_TRUE(patterns.ok()) << patterns.status();
  return std::move(*patterns);
}

FactorizedMlpModelOptions Options(int num_mlps, bool appended_buffers) {
  FactorizedMlpModelOptions options;
  options.num_mlps = num_mlps;
  options.appended_buffers = appended_buffers;
  return options;
}

// Carves `source` with the built-in patterns; empty if nothing was carved.
std::string Carve(const std::string& source, CarveReport* report = nullptr) {
  const auto patterns = Builtin();
  CarveReport local;
  auto carved =
      CarveModel(AsBuffer(source), patterns, report ? report : &local);
  EXPECT_TRUE(carved.ok()) << carved.status();
  if (!carved.ok() || !carved->has_value()) return "";
  return testing::ToBytes(**carved);
}

std::unique_ptr<tflite::ModelT> Unpack(const std::string& bytes) {
  return std::unique_ptr<tflite::ModelT>(
      tflite::GetModel(bytes.data())->UnPack());
}

tflite::BuiltinOperator CodeOf(const tflite::ModelT& model,
                               const tflite::OperatorT& op) {
  return model.operator_codes[op.opcode_index]->builtin_code;
}

std::vector<const tflite::OperatorT*> Composites(const tflite::ModelT& model) {
  std::vector<const tflite::OperatorT*> composites;
  for (const auto& op : model.subgraphs[0]->operators) {
    if (CodeOf(model, *op) == tflite::BuiltinOperator_STABLEHLO_COMPOSITE) {
      composites.push_back(op.get());
    }
  }
  return composites;
}

absl::Status Verify(const std::string& original, const std::string& carved) {
  std::string summary;
  return VerifyCarvedStructure(AsBuffer(original), AsBuffer(carved), &summary);
}

class CarveModelTest : public ::testing::TestWithParam<bool> {
 protected:
  bool appended_buffers() const { return GetParam(); }
};

INSTANTIATE_TEST_SUITE_P(Buffers, CarveModelTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "BehindTheFlatbuffer"
                                             : "InTheFlatbuffer";
                         });

TEST_P(CarveModelTest, ReplacesEachMlpByOneComposite) {
  const std::string source =
      testing::BuildFactorizedMlpModel(Options(2, appended_buffers()));
  CarveReport report;
  const std::string carved = Carve(source, &report);
  ASSERT_FALSE(carved.empty());
  EXPECT_EQ(report.num_carved, 2);
  EXPECT_EQ(report.num_subgraphs, 1);
  ASSERT_THAT(report.entries, SizeIs(1));
  EXPECT_EQ(report.entries[0].pattern, "odml.factorized_mlp");
  EXPECT_EQ(report.entries[0].num_carved, 2);

  const auto before = Unpack(source);
  const auto after = Unpack(carved);
  const auto& main = *after->subgraphs[0];
  // CAST, then a composite and an ADD for each MLP, then CAST.
  ASSERT_THAT(main.operators, SizeIs(6));
  EXPECT_EQ(before->subgraphs[0]->operators.size(), 6 + 2 * 27);
  ASSERT_THAT(after->subgraphs, SizeIs(3));
  EXPECT_EQ(after->subgraphs[1]->name, "main_odml.factorized_mlp.impl");
  EXPECT_EQ(after->subgraphs[2]->name, "main_odml.factorized_mlp.impl_0");

  const auto composites = Composites(*after);
  ASSERT_THAT(composites, SizeIs(2));
  for (size_t k = 0; k < 2; ++k) {
    const auto& op = *composites[k];
    const auto* options = op.builtin_options_2.AsStableHLOCompositeOptions();
    ASSERT_NE(options, nullptr);
    EXPECT_EQ(options->name, "odml.factorized_mlp");
    EXPECT_EQ(options->decomposition_subgraph_index, 1 + k);
    const auto attributes =
        flexbuffers::GetRoot(options->composite_attributes).AsMap();
    EXPECT_EQ(attributes["features"].AsInt64(), 16);
    EXPECT_EQ(attributes["activation"].AsString().str(), "silu");
    EXPECT_EQ(attributes["scale_out"].AsFloat(), 0.023f + 0.001f * k);

    // x, then the six weights, which are still constants of the graph with
    // their blockwise quantization and its block shape.
    ASSERT_THAT(op.inputs, SizeIs(7));
    ASSERT_THAT(op.outputs, SizeIs(1));
    EXPECT_EQ(main.tensors[op.inputs[0]]->name, "stream" + std::to_string(k));
    const auto& decomposition = *after->subgraphs[1 + k];
    ASSERT_THAT(decomposition.inputs, ElementsAre(0, 1, 2, 3, 4, 5, 6));
    EXPECT_THAT(decomposition.operators, SizeIs(28));
    for (size_t w = 1; w < 7; ++w) {
      const auto& weights = *main.tensors[op.inputs[w]];
      EXPECT_EQ(weights.type, tflite::TensorType_INT2);
      EXPECT_NE(weights.buffer, 0);
      const auto* blockwise =
          weights.quantization->details.AsBlockwiseQuantization();
      ASSERT_NE(blockwise, nullptr);
      EXPECT_FALSE(blockwise->block_shape.empty());
      EXPECT_EQ(main.tensors[blockwise->scales]->name,
                weights.name + "_scales");

      // In the decomposition the weights are an input without data, whose
      // quantization names constants of the decomposition that share the
      // graph's buffers.
      const auto& argument = *decomposition.tensors[w];
      EXPECT_EQ(argument.name, decomposition.name + "_arg" + std::to_string(w));
      EXPECT_EQ(argument.buffer, 0);
      const auto* inner =
          argument.quantization->details.AsBlockwiseQuantization();
      ASSERT_NE(inner, nullptr);
      EXPECT_EQ(inner->block_shape, blockwise->block_shape);
      EXPECT_EQ(decomposition.tensors[inner->scales]->buffer,
                main.tensors[blockwise->scales]->buffer);
      EXPECT_EQ(decomposition.tensors[inner->zero_points]->buffer,
                main.tensors[blockwise->zero_points]->buffer);
    }
    // The bias the model leaves out is still left out.
    for (const auto& inner : decomposition.operators) {
      if (CodeOf(*after, *inner) == tflite::BuiltinOperator_FULLY_CONNECTED) {
        ASSERT_THAT(inner->inputs, SizeIs(3));
        EXPECT_EQ(inner->inputs[2], -1);
      }
    }
  }
  // The tensors only the MLPs used are gone, and the signature follows the
  // renumbered ones.
  EXPECT_LT(main.tensors.size(), before->subgraphs[0]->tensors.size());
  EXPECT_EQ(
      main.tensors[after->signature_defs[0]->inputs[0]->tensor_index]->name,
      "x");
  EXPECT_EQ(
      main.tensors[after->signature_defs[0]->outputs[0]->tensor_index]->name,
      "y");

  EXPECT_TRUE(Verify(source, carved).ok()) << Verify(source, carved);
}

TEST_P(CarveModelTest, CarvingAgainChangesNothing) {
  const std::string source =
      testing::BuildFactorizedMlpModel(Options(2, appended_buffers()));
  const std::string carved = Carve(source);
  ASSERT_FALSE(carved.empty());
  CarveReport report;
  EXPECT_THAT(Carve(carved, &report), IsEmpty());
  EXPECT_EQ(report.num_carved, 0);
  EXPECT_EQ(report.num_subgraphs, 3);
  EXPECT_EQ(report.num_decompositions_skipped, 2);
}

TEST_P(CarveModelTest, LeavesAModelWithoutRegionsAlone) {
  const std::string source =
      testing::BuildFactorizedMlpModel(Options(0, appended_buffers()));
  EXPECT_THAT(Carve(source), IsEmpty());
}

TEST_P(CarveModelTest, GivesTheSameOutputsOnTheCpu) {
  FactorizedMlpModelOptions options = Options(2, appended_buffers());
  options.tokens = 5;
  const std::string source = testing::BuildFactorizedMlpModel(options);
  const std::string carved = Carve(source);
  ASSERT_FALSE(carved.empty());
  std::string summary;
  const auto status = CompareOnCpu(AsBuffer(source), AsBuffer(carved),
                                   NumericCheckOptions(), &summary);
  EXPECT_TRUE(status.ok()) << status;
  // x [1, 5, 16] in, y of the same shape out, as floats.
  EXPECT_THAT(summary, HasSubstr("1 outputs of 1 signatures (320 bytes)"));
  EXPECT_THAT(summary, ::testing::Not(HasSubstr("NaN")));
}

TEST(CarveModelDataTest, KeepsTheDataBehindTheFlatbufferInPlace) {
  const std::string source = testing::BuildFactorizedMlpModel(Options(2, true));
  const auto patterns = Builtin();
  CarveReport report;
  auto carved = CarveModel(AsBuffer(source), patterns, &report);
  ASSERT_TRUE(carved.ok()) << carved.status();
  ASSERT_TRUE(carved->has_value());
  const CarvedModel& model = **carved;

  // The data is the source's, from its first buffer to its end.
  const auto before = Unpack(source);
  uint64_t first = source.size();
  for (const auto& buffer : before->buffers) {
    if (buffer->offset > 1) first = std::min<uint64_t>(first, buffer->offset);
  }
  EXPECT_EQ(model.appended.Data(), AsBuffer(source).Data() + first);
  EXPECT_EQ(model.appended.Size(), source.size() - first);
  // Every buffer moved by the same whole number of pages.
  const std::string bytes = testing::ToBytes(model);
  const auto after = Unpack(bytes);
  ASSERT_EQ(after->buffers.size(), before->buffers.size());
  const int64_t moved =
      static_cast<int64_t>(model.flatbuffer.Size() + model.gap) -
      static_cast<int64_t>(first);
  EXPECT_EQ(moved % 4096, 0);
  for (size_t i = 0; i < before->buffers.size(); ++i) {
    const auto& x = *before->buffers[i];
    const auto& y = *after->buffers[i];
    EXPECT_EQ(y.size, x.size);
    EXPECT_EQ(y.offset, x.offset > 1 ? x.offset + moved : x.offset);
  }
}

// A constant without elements (an empty shape) whose data is present and
// empty inside the flatbuffer: the schema's objects cannot write that back.
class CarveEmptyConstantTest : public ::testing::TestWithParam<bool> {};

INSTANTIATE_TEST_SUITE_P(Buffers, CarveEmptyConstantTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "WeightsBehindTheFlatbuffer"
                                             : "AllInTheFlatbuffer";
                         });

TEST_P(CarveEmptyConstantTest, StaysAConstant) {
  FactorizedMlpModelOptions options = Options(1, GetParam());
  options.keep_inline_up_to = 16;
  options.mutate = [](tflite::ModelT& model) {
    auto buffer = std::make_unique<tflite::BufferT>();
    buffer->data = {0};
    model.buffers.push_back(std::move(buffer));
    auto tensor = std::make_unique<tflite::TensorT>();
    tensor->name = "empty";
    tensor->type = tflite::TensorType_INT32;
    tensor->shape = {0};
    tensor->has_rank = true;
    tensor->buffer = static_cast<uint32_t>(model.buffers.size() - 1);
    auto& graph = *model.subgraphs[0];
    graph.tensors.push_back(std::move(tensor));
    // Something has to read it. As an input of the last op it makes a graph
    // that cannot run, which the structure of a file does not need.
    graph.operators.back()->inputs.push_back(
        static_cast<int32_t>(graph.tensors.size() - 1));
  };
  std::string source = testing::BuildFactorizedMlpModel(options);
  // The one byte of the last buffer goes: a vector starts with its length.
  auto last_buffer = [](const std::string& bytes) {
    const auto* buffers = tflite::GetModel(bytes.data())->buffers();
    return buffers->Get(buffers->size() - 1);
  };
  ASSERT_NE(last_buffer(source)->data(), nullptr);
  const size_t length_field =
      reinterpret_cast<const char*>(last_buffer(source)->data()) -
      source.data();
  std::memset(&source[length_field], 0, sizeof(uint32_t));
  ASSERT_EQ(last_buffer(source)->data()->size(), 0);

  const std::string carved = Carve(source);
  ASSERT_FALSE(carved.empty());
  // The constant is data of no bytes behind the flatbuffer now, inside the
  // file.
  const auto* buffer = last_buffer(carved);
  EXPECT_EQ(buffer->data(), nullptr);
  EXPECT_GT(buffer->offset(), 1);
  EXPECT_LE(buffer->offset(), carved.size());
  EXPECT_EQ(buffer->size(), 0);
  EXPECT_TRUE(Verify(source, carved).ok()) << Verify(source, carved);

  // LiteRT reads it as the constant it was.
  const std::string* const files[] = {&source, &carved};
  for (const std::string* bytes : files) {
    auto model = litert::internal::LoadModelFromBuffer(AsBuffer(*bytes));
    ASSERT_TRUE(model.HasValue());
    const LiteRtTensorT* empty = nullptr;
    for (const auto* tensor : (*model)->Subgraph(0).Tensors()) {
      if (tensor->Name() == "empty") empty = tensor;
    }
    ASSERT_NE(empty, nullptr);
    EXPECT_TRUE(litert::internal::IsConstant(*empty));
  }
}

TEST(OutlineRegionsTest, RejectsRegionsThatCannotBeOneOp) {
  const std::string source =
      testing::BuildFactorizedMlpModel(Options(1, false));
  auto region_of = [](const tflite::ModelT& model) {
    // The SiLU of the MLP: LOGISTIC and MUL.
    CarveRegionSpec region;
    region.composite_name = "test.silu";
    const auto& main = *model.subgraphs[0];
    for (size_t i = 0; i < main.operators.size(); ++i) {
      const auto code = CodeOf(model, *main.operators[i]);
      if (code == tflite::BuiltinOperator_LOGISTIC) {
        region.ops = {static_cast<int32_t>(i), static_cast<int32_t>(i + 1)};
        region.inputs = {main.operators[i]->inputs[0]};
        region.outputs = {main.operators[i + 1]->outputs[0]};
      }
    }
    return region;
  };
  auto outline = [&](const std::function<void(CarveRegionSpec&)>& change) {
    auto model = Unpack(source);
    CarveRegionSpec region = region_of(*model);
    change(region);
    return OutlineRegions(*model, 0, {region});
  };
  EXPECT_TRUE(outline([](CarveRegionSpec&) {}).ok());
  EXPECT_THAT(outline([](CarveRegionSpec& r) {
                r.ops = {r.ops[1], r.ops[0]};
              }).message(),
              HasSubstr("not ascending"));
  EXPECT_THAT(
      outline([](CarveRegionSpec& r) { r.inputs = r.outputs; }).message(),
      HasSubstr("an input is computed inside the region"));
  EXPECT_THAT(
      outline([](CarveRegionSpec& r) { r.outputs = r.inputs; }).message(),
      HasSubstr("an output is not computed inside the region"));
  EXPECT_THAT(
      outline([](CarveRegionSpec& r) { r.ops.push_back(1 << 20); }).message(),
      HasSubstr("not ascending indices"));

  auto model = Unpack(source);
  const CarveRegionSpec region = region_of(*model);
  EXPECT_THAT(OutlineRegions(*model, 0, {region, region}).message(),
              HasSubstr("overlaps another region"));
  EXPECT_THAT(OutlineRegions(*model, 7, {region}).message(),
              HasSubstr("subgraph index out of range"));
}

// A carved model that was changed afterwards: the structural check names the
// difference.
struct TamperCase {
  const char* name;
  std::function<void(tflite::ModelT&)> mutate;
  const char* message;
};

class CarveTamperTest : public ::testing::TestWithParam<TamperCase> {};

TEST_P(CarveTamperTest, IsFoundByTheStructuralCheck) {
  for (const bool appended_buffers : {false, true}) {
    const std::string source =
        testing::BuildFactorizedMlpModel(Options(2, appended_buffers));
    const std::string carved = Carve(source);
    ASSERT_FALSE(carved.empty());
    const std::string tampered = testing::Mutated(carved, GetParam().mutate);
    ASSERT_FALSE(tampered.empty());
    const auto status = Verify(source, tampered);
    EXPECT_FALSE(status.ok());
    EXPECT_THAT(status.message(), HasSubstr(GetParam().message));
  }
}

tflite::TensorT& TensorNamed(tflite::SubGraphT& subgraph,
                             const std::string& name) {
  for (auto& tensor : subgraph.tensors) {
    if (tensor->name == name) return *tensor;
  }
  ADD_FAILURE() << "no tensor " << name;
  return *subgraph.tensors[0];
}

INSTANTIATE_TEST_SUITE_P(
    Cases, CarveTamperTest,
    ::testing::Values(
        TamperCase{"AScaleInsideADecomposition",
                   [](tflite::ModelT& model) {
                     TensorNamed(*model.subgraphs[1], "mlp0/u0")
                         .quantization->scale = {0.5f};
                   },
                   "what output 0 of subgraph 0 (main) computes"},
        TamperCase{"AnOptionInsideADecomposition",
                   [](tflite::ModelT& model) {
                     for (auto& op : model.subgraphs[2]->operators) {
                       if (auto* options =
                               op->builtin_options.AsConcatenationOptions()) {
                         options->axis = 0;
                       }
                     }
                   },
                   "what output 0 of subgraph 0 (main) computes"},
        TamperCase{"TheBlockShapeOfADecompositionInput",
                   [](tflite::ModelT& model) {
                     // What LiteRT's own serializer does to these weights.
                     model.subgraphs[1]
                         ->tensors[1]
                         ->quantization->details.AsBlockwiseQuantization()
                         ->block_shape.clear();
                   },
                   "differs from tensor 1 (main_odml.factorized_mlp.impl_arg1) "
                   "of its decomposition in type, shape or quantization"},
        TamperCase{"TheScalesOfADecompositionInput",
                   [](tflite::ModelT& model) {
                     // The scales of the first weights name those of the
                     // second.
                     auto& tensors = model.subgraphs[1]->tensors;
                     tensors[1]
                         ->quantization->details.AsBlockwiseQuantization()
                         ->scales =
                         tensors[2]
                             ->quantization->details.AsBlockwiseQuantization()
                             ->scales;
                   },
                   "in type, shape or quantization"},
        TamperCase{"TwoWeightsSwapped",
                   [](tflite::ModelT& model) {
                     for (auto& op : model.subgraphs[0]->operators) {
                       if (op->inputs.size() == 7) {
                         std::swap(op->inputs[1], op->inputs[2]);
                       }
                     }
                   },
                   "in type, shape or quantization"},
        TamperCase{"AnOperatorMovedBeforeItsInput",
                   [](tflite::ModelT& model) {
                     auto& operators = model.subgraphs[0]->operators;
                     std::swap(operators[1], operators[2]);
                   },
                   "is read by operator 1 before it is written"},
        TamperCase{"ASignatureOutputRenamed",
                   [](tflite::ModelT& model) {
                     model.signature_defs[0]->outputs[0]->name = "z";
                   },
                   "signature 0"},
        TamperCase{"ATensorLeftBehind",
                   [](tflite::ModelT& model) {
                     model.subgraphs[0]->tensors.push_back(
                         std::make_unique<tflite::TensorT>());
                   },
                   "leaves 1 tensors unused in subgraph 0"},
        TamperCase{
            "ADecompositionUsedTwice",
            [](tflite::ModelT& model) {
              for (auto& op : model.subgraphs[0]->operators) {
                if (auto* options =
                        op->builtin_options_2.AsStableHLOCompositeOptions()) {
                  options->decomposition_subgraph_index = 1;
                }
              }
            },
            "in type, shape or quantization"},
        TamperCase{"AnExtraSubgraph",
                   [](tflite::ModelT& model) {
                     model.subgraphs.push_back(
                         std::make_unique<tflite::SubGraphT>(
                             *model.subgraphs[2]));
                   },
                   "is the decomposition of 0 composites, not of one"},
        TamperCase{"TheDescription",
                   [](tflite::ModelT& model) { model.description = "other"; },
                   "its version or description"}),
    [](const ::testing::TestParamInfo<TamperCase>& info) {
      return info.param.name;
    });

// A change in the first weights of the first MLP has to pass through every
// stage of both MLPs to reach the output; one in the last weights of the last
// MLP reaches it directly.
class CarveChangedConstantTest
    : public ::testing::TestWithParam<std::tuple<const char*, bool>> {};

INSTANTIATE_TEST_SUITE_P(Weights, CarveChangedConstantTest,
                         ::testing::Combine(::testing::Values("mlp0/w_up1a",
                                                              "mlp1/w_down2b"),
                                            ::testing::Bool()));

TEST_P(CarveChangedConstantTest, IsFoundByBothChecks) {
  const auto [name, appended_buffers] = GetParam();
  FactorizedMlpModelOptions options = Options(2, appended_buffers);
  options.tokens = 5;
  const std::string source = testing::BuildFactorizedMlpModel(options);
  std::string carved = Carve(source);
  ASSERT_FALSE(carved.empty());
  // The first two rows of the weights, negated.
  const auto model = Unpack(carved);
  uint32_t buffer = 0;
  for (const auto& tensor : model->subgraphs[0]->tensors) {
    if (tensor->name == name) buffer = tensor->buffer;
  }
  ASSERT_NE(buffer, 0);
  if (appended_buffers) {
    for (int i = 0; i < 4; ++i) {
      carved[model->buffers[buffer]->offset + i] ^= 0xff;
    }
  } else {
    carved = testing::Mutated(carved, [&](tflite::ModelT& changed) {
      for (int i = 0; i < 4; ++i) changed.buffers[buffer]->data[i] ^= 0xff;
    });
  }
  EXPECT_THAT(Verify(source, carved).message(),
              HasSubstr("what output 0 of subgraph 0 (main) computes"));

  // The same change gives other numbers, which the run on the CPU finds: the
  // test model's signal is alive in every stage.
  std::string summary;
  const auto run = CompareOnCpu(AsBuffer(source), AsBuffer(carved),
                                NumericCheckOptions(), &summary);
  EXPECT_THAT(run.message(), HasSubstr("output 'y' of signature 'main'"));
}

}  // namespace
}  // namespace litert::tools
