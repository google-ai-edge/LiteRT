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

#include "litert/vendors/nvidia/tools/carve_pattern.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/litert_op_code.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"
#include "litert/vendors/nvidia/tools/carve_test_util.h"
#include "tflite/schema/schema_generated.h"

namespace litert::tools {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::SizeIs;

std::vector<CarvePattern> Builtin() {
  auto patterns = ParseCarvePatterns(BuiltinCarvePatternText());
  EXPECT_TRUE(patterns.ok()) << patterns.status();
  return std::move(*patterns);
}

// A loaded test model with the bytes it points into.
struct Loaded {
  std::string bytes;
  std::unique_ptr<LiteRtModelT> model;
  const LiteRtSubgraphT& main() const { return model->Subgraph(0); }
};

// The options of a test model with `num_mlps` MLPs.
testing::FactorizedMlpModelOptions Mlps(int num_mlps) {
  testing::FactorizedMlpModelOptions options;
  options.num_mlps = num_mlps;
  return options;
}

Loaded Load(const testing::FactorizedMlpModelOptions& options) {
  Loaded loaded;
  loaded.bytes = testing::BuildFactorizedMlpModel(options);
  auto model =
      litert::internal::LoadModelFromBuffer(testing::AsBuffer(loaded.bytes));
  EXPECT_TRUE(model.HasValue());
  if (model.HasValue()) loaded.model = std::move(*model);
  return loaded;
}

tflite::TensorT& TensorNamed(tflite::ModelT& model, const std::string& name) {
  for (auto& tensor : model.subgraphs[0]->tensors) {
    if (tensor->name == name) return *tensor;
  }
  ADD_FAILURE() << "no tensor " << name;
  return *model.subgraphs[0]->tensors[0];
}

int32_t TensorIndex(const tflite::ModelT& model, const std::string& name) {
  const auto& tensors = model.subgraphs[0]->tensors;
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i]->name == name) return static_cast<int32_t>(i);
  }
  ADD_FAILURE() << "no tensor " << name;
  return 0;
}

// The operator of the main subgraph that writes tensor `name`.
tflite::OperatorT& ProducerOf(tflite::ModelT& model, const std::string& name) {
  const int32_t index = TensorIndex(model, name);
  for (auto& op : model.subgraphs[0]->operators) {
    if (op->outputs[0] == index) return *op;
  }
  ADD_FAILURE() << "no producer of " << name;
  return *model.subgraphs[0]->operators[0];
}

// Makes tensor `name` an output of the main subgraph and of its signature.
void AddOutput(tflite::ModelT& model, const std::string& name) {
  const int32_t index = TensorIndex(model, name);
  model.subgraphs[0]->outputs.push_back(index);
  auto output = std::make_unique<tflite::TensorMapT>();
  output->name = name;
  output->tensor_index = index;
  model.signature_defs[0]->outputs.push_back(std::move(output));
}

TEST(CarvePatternTest, ParsesTheBuiltinDescription) {
  const auto patterns = Builtin();
  ASSERT_THAT(patterns, SizeIs(1));
  const auto& factorized_mlp = patterns[0];
  EXPECT_EQ(factorized_mlp.name, "odml.factorized_mlp");
  EXPECT_THAT(factorized_mlp.inputs, SizeIs(7));
  EXPECT_THAT(factorized_mlp.outputs, SizeIs(1));
  EXPECT_THAT(factorized_mlp.stmts, SizeIs(28));
  EXPECT_THAT(factorized_mlp.plan, SizeIs(27));
  EXPECT_THAT(factorized_mlp.checks, SizeIs(8));
  EXPECT_THAT(factorized_mlp.attributes, SizeIs(13));
  EXPECT_EQ(factorized_mlp.stmts.back().op_code, kLiteRtOpCodeTflDequantize);
  // Every op is found from a tensor's producer: the pattern has one output
  // and everything leads to it.
  for (const auto& step : factorized_mlp.plan) EXPECT_TRUE(step.via_output);
}

TEST(CarvePatternTest, ReportsWhatIsWrongWithADescription) {
  auto message = [](const char* text) {
    const auto patterns = ParseCarvePatterns(text);
    EXPECT_FALSE(patterns.ok());
    return std::string(patterns.status().message());
  };
  EXPECT_THAT(message(""), HasSubstr("no pattern"));
  EXPECT_THAT(message("in x"), HasSubstr("line 1"));
  EXPECT_THAT(message("pattern p\nout y\ny = NO_SUCH_OP(x)"),
              HasSubstr("line 3: 'NO_SUCH_OP' is not a TFLite builtin"));
  EXPECT_THAT(message("pattern p\nin x\nout y\ny = LOGISTIC(x) : q7"),
              HasSubstr("'q7' is not an element type"));
  EXPECT_THAT(message("pattern p\nin x\ny = LOGISTIC(x)"),
              HasSubstr("no outputs"));
  EXPECT_THAT(message("pattern p\nin x\nout x\nx = LOGISTIC(x)"),
              HasSubstr("an input and the result of an op"));
  EXPECT_THAT(message("pattern p\nin x\nout y\ny = LOGISTIC(x)\ncheck N == 1"),
              HasSubstr("'N' is neither a symbol of a type nor a value"));
  EXPECT_THAT(message("pattern p\nin x\nin z\nout y\nout w\n"
                      "w = LOGISTIC(z)\ny = LOGISTIC(x)"),
              HasSubstr("line 6: the op is not connected"));
  EXPECT_THAT(message("pattern p\nin x\nout y\ny = LOGISTIC(x) {a = }"),
              HasSubstr("expected a value"));
}

TEST(CarvePatternTest, FindsEveryFactorizedMlp) {
  const auto patterns = Builtin();
  const auto loaded = Load(Mlps(3));
  const auto matches = FindCarveRegions(patterns, loaded.main());
  EXPECT_THAT(matches.near_misses, IsEmpty());
  ASSERT_THAT(matches.regions, SizeIs(3));
  for (size_t k = 0; k < 3; ++k) {
    const auto& region = matches.regions[k];
    EXPECT_THAT(region.ops, SizeIs(28));
    ASSERT_THAT(region.inputs, SizeIs(7));
    ASSERT_THAT(region.outputs, SizeIs(1));
    EXPECT_EQ(region.inputs[0]->Name(), absl::StrCat("stream", k));
    const char* const weights[] = {"w_up1a",  "w_up1b",   "w_up2",
                                   "w_down1", "w_down2a", "w_down2b"};
    for (size_t w = 0; w < 6; ++w) {
      EXPECT_EQ(region.inputs[w + 1]->Name(),
                absl::StrCat("mlp", k, "/", weights[w]));
    }
    EXPECT_EQ(region.outputs[0]->Name(), absl::StrCat("mlp", k, "/y"));

    ASSERT_THAT(region.attributes, SizeIs(13));
    auto attribute = [&](const char* key) {
      for (const auto& [name, value] : region.attributes) {
        if (name == key) return value;
      }
      ADD_FAILURE() << "no attribute " << key;
      return CarveValue();
    };
    EXPECT_EQ(attribute("version"), CarveValue(int64_t{1}));
    EXPECT_EQ(attribute("features"), CarveValue(int64_t{16}));
    EXPECT_EQ(attribute("blocks"), CarveValue(int64_t{2}));
    EXPECT_EQ(attribute("groups"), CarveValue(int64_t{2}));
    EXPECT_EQ(attribute("up1_features"), CarveValue(int64_t{8}));
    EXPECT_EQ(attribute("hidden_features"), CarveValue(int64_t{8}));
    EXPECT_EQ(attribute("activation"), CarveValue(std::string("silu")));
    // The scales are the tensors' 32-bit scales, exactly.
    const float in = 0.010f + 0.001f * k;
    const float act = 0.019f + 0.001f * k;
    EXPECT_EQ(attribute("scale_in"), CarveValue(static_cast<double>(in)));
    EXPECT_EQ(attribute("scale_act"), CarveValue(static_cast<double>(act)));
  }
}

TEST(CarvePatternTest, AcceptsBlockWeightsWithALeadingDimension) {
  const auto patterns = Builtin();
  testing::FactorizedMlpModelOptions options;
  options.tokens = 1;
  options.rank4_block_weights = true;
  const auto loaded = Load(options);
  const auto matches = FindCarveRegions(patterns, loaded.main());
  EXPECT_THAT(matches.near_misses, IsEmpty());
  EXPECT_THAT(matches.regions, SizeIs(2));
}

// Each change breaks the first MLP in one way: the second is still found, and
// the first is reported with the reason.
struct NearMissCase {
  const char* name;
  std::function<void(tflite::ModelT&)> mutate;
  const char* reason;
};

class CarveNearMissTest : public ::testing::TestWithParam<NearMissCase> {};

TEST_P(CarveNearMissTest, ReportsTheReason) {
  const auto patterns = Builtin();
  testing::FactorizedMlpModelOptions options;
  options.mutate = GetParam().mutate;
  const auto loaded = Load(options);
  const auto matches = FindCarveRegions(patterns, loaded.main());
  ASSERT_THAT(matches.regions, SizeIs(1));
  EXPECT_EQ(matches.regions[0].outputs[0]->Name(), "mlp1/y");
  ASSERT_THAT(matches.near_misses, SizeIs(1));
  EXPECT_THAT(matches.near_misses[0].reason, HasSubstr(GetParam().reason));
  EXPECT_EQ(matches.near_misses[0].anchor->Output(0).Name(), "mlp0/y");
}

INSTANTIATE_TEST_SUITE_P(
    Cases, CarveNearMissTest,
    ::testing::Values(
        NearMissCase{"SecondHalfStartsAtZero",
                     [](tflite::ModelT& model) {
                       // Both up projections read the first half.
                       ProducerOf(model, "mlp0/h1").inputs[1] =
                           ProducerOf(model, "mlp0/h0").inputs[1];
                     },
                     "check begin1 == [0, H] does not hold"},
        NearMissCase{"StagesOfOneScaleDisagree",
                     [](tflite::ModelT& model) {
                       TensorNamed(model, "mlp0/u1").quantization->scale = {
                           0.5f};
                     },
                     "'u1' has scale 0.5, but Sup1 is"},
        NearMissCase{
            "NonZeroZeroPoint",
            [](tflite::ModelT& model) {
              TensorNamed(model, "mlp0/q1").quantization->zero_point = {3};
            },
            "'q1' is not quantized per tensor with zero point 0"},
        NearMissCase{
            "FusedActivation",
            [](tflite::ModelT& model) {
              ProducerOf(model, "mlp0/f0")
                  .builtin_options.AsFullyConnectedOptions()
                  ->fused_activation_function =
                  tflite::ActivationFunctionType_RELU;
            },
            "option fused_activation_function is \"RELU\", expected \"NONE\""},
        NearMissCase{"WeightsAreNotConstant",
                     [](tflite::ModelT& model) {
                       TensorNamed(model, "mlp0/w_down1").buffer = 0;
                     },
                     "input 'w_down1' is not a constant"},
        NearMissCase{"AnInnerTensorIsReadOutside",
                     [](tflite::ModelT& model) {
                       auto& graph = *model.subgraphs[0];
                       auto tap = std::make_unique<tflite::TensorT>(
                           TensorNamed(model, "mlp0/d0"));
                       tap->name = "tap";
                       graph.tensors.push_back(std::move(tap));
                       auto op = std::make_unique<tflite::OperatorT>(
                           ProducerOf(model, "mlp0/d0"));
                       op->inputs = {TensorIndex(model, "mlp0/r3")};
                       op->outputs = {TensorIndex(model, "tap")};
                       graph.operators.push_back(std::move(op));
                       AddOutput(model, "tap");
                     },
                     "'r3' is also read by DEQUANTIZE outside the region"},
        NearMissCase{
            "AnInnerTensorIsAnOutputOfTheSubgraph",
            [](tflite::ModelT& model) { AddOutput(model, "mlp0/a0"); },
            "'a0' is an output of the subgraph but not of the pattern"},
        NearMissCase{"ADifferentWidth",
                     [](tflite::ModelT& model) {
                       TensorNamed(model, "mlp0/c1").shape = {3, 17};
                     },
                     "'c1' is i8 [3, 17], but F is 16"},
        NearMissCase{"AnOpIsMissing",
                     [](tflite::ModelT& model) {
                       // The activation is the logistic alone, not SiLU.
                       ProducerOf(model, "mlp0/q1").inputs[0] =
                           TensorIndex(model, "mlp0/s0");
                     },
                     "found LOGISTIC after 10 of 28 ops"}),
    [](const ::testing::TestParamInfo<NearMissCase>& info) {
      return info.param.name;
    });

TEST(CarvePatternTest, FollowsUsersAndTriesEachOfThem) {
  // The anchor is the first op of the region here, so the second is found
  // among the users of `d`, of which the anchor itself is one.
  const auto patterns = ParseCarvePatterns(R"(
      pattern test.silu
      in d : f16 [M, D]
      out a
      a = MUL(d, s)
      s = LOGISTIC(d)
      attr width = D
  )");
  ASSERT_TRUE(patterns.ok()) << patterns.status();
  ASSERT_THAT(patterns->at(0).plan, SizeIs(1));
  EXPECT_FALSE(patterns->at(0).plan[0].via_output);

  const auto loaded = Load(Mlps(2));
  const auto matches = FindCarveRegions(*patterns, loaded.main());
  EXPECT_THAT(matches.near_misses, IsEmpty());
  ASSERT_THAT(matches.regions, SizeIs(2));
  const auto& region = matches.regions[0];
  ASSERT_THAT(region.ops, SizeIs(2));
  EXPECT_EQ(region.ops[0]->OpCode(), kLiteRtOpCodeTflLogistic);
  EXPECT_EQ(region.ops[1]->OpCode(), kLiteRtOpCodeTflMul);
  EXPECT_EQ(region.inputs[0]->Name(), "mlp0/d0");
  EXPECT_EQ(region.outputs[0]->Name(), "mlp0/a0");
  EXPECT_THAT(region.attributes,
              ElementsAre(std::make_pair("width", CarveValue(int64_t{8}))));
}

TEST(CarvePatternTest, RejectsARegionWhoseResultIsReadBeforeItEnds) {
  // The two up projections alone: the reshape of the first one's result runs
  // between them, so one op in the place of the second cannot feed it.
  const auto patterns = ParseCarvePatterns(R"(
      pattern test.up
      in q
      in w0 : const
      in w1 : const
      out u0
      out u1
      h0 = SLICE(q, _, _)
      u0 = FULLY_CONNECTED(h0, w0)
      h1 = SLICE(q, begin1, _)
      u1 = FULLY_CONNECTED(h1, w1)
      check begin1 == [0, 8]
  )");
  ASSERT_TRUE(patterns.ok()) << patterns.status();
  const auto loaded = Load(Mlps(1));
  const auto matches = FindCarveRegions(*patterns, loaded.main());
  EXPECT_THAT(matches.regions, IsEmpty());
  // The pattern is tried from each of the two projections: from the second
  // one it fits up to the order of the ops, from the first one the halves are
  // the wrong way round.
  ASSERT_THAT(matches.near_misses, SizeIs(2));
  EXPECT_THAT(matches.near_misses[0].reason,
              HasSubstr("check begin1 == [0, 8] does not hold"));
  EXPECT_THAT(matches.near_misses[1].reason,
              HasSubstr("output 'u0' is read by RESHAPE before the region"));
}

TEST(CarvePatternTest, RegionsDoNotOverlap) {
  // The second pattern's region lies inside the first one's.
  auto text = std::string(BuiltinCarvePatternText());
  text += R"(
      pattern test.silu
      in d
      out a
      s = LOGISTIC(d)
      a = MUL(d, s)
  )";
  const auto patterns = ParseCarvePatterns(text);
  ASSERT_TRUE(patterns.ok()) << patterns.status();
  const auto loaded = Load(Mlps(1));
  const auto matches = FindCarveRegions(*patterns, loaded.main());
  ASSERT_THAT(matches.regions, SizeIs(1));
  EXPECT_EQ(matches.regions[0].pattern->name, "odml.factorized_mlp");
  ASSERT_THAT(matches.near_misses, SizeIs(1));
  EXPECT_THAT(matches.near_misses[0].reason, HasSubstr("overlaps"));
}

TEST(CarvePatternTest, RegionConstantsMustNotBeBlockwiseQuantized) {
  // The weights as a constant of the region instead of an input.
  const auto patterns = ParseCarvePatterns(R"(
      pattern test.fc
      in h
      out u
      u = FULLY_CONNECTED(h, _) : i8 [M, 8]
  )");
  ASSERT_TRUE(patterns.ok()) << patterns.status();
  const auto loaded = Load(Mlps(1));
  const auto matches = FindCarveRegions(*patterns, loaded.main());
  EXPECT_THAT(matches.regions, IsEmpty());
  ASSERT_THAT(matches.near_misses, SizeIs(4));
  EXPECT_THAT(matches.near_misses[0].reason,
              HasSubstr("has blockwise quantization: declare it as a const"));
}

TEST(CarvePatternTest, PacksAttributesAsTheExistingCompositesDo) {
  const std::vector<uint8_t> packed = PackCarveAttributes({
      {"version", CarveValue(int64_t{1})},
      {"hidden_features", CarveValue(int64_t{32768})},
      {"activation", CarveValue(std::string("silu"))},
      {"scale_in", CarveValue(static_cast<double>(0.02716f))},
      {"causal", CarveValue(true)},
      {"axes", CarveValue(std::vector<int64_t>{1, 2})},
  });
  const auto map = flexbuffers::GetRoot(packed).AsMap();
  EXPECT_EQ(map.size(), 6);
  EXPECT_TRUE(map["version"].IsInt());
  EXPECT_EQ(map["version"].AsInt64(), 1);
  EXPECT_EQ(map["hidden_features"].AsInt64(), 32768);
  EXPECT_EQ(map["activation"].AsString().str(), "silu");
  EXPECT_TRUE(map["scale_in"].IsFloat());
  EXPECT_EQ(map["scale_in"].AsFloat(), 0.02716f);
  EXPECT_TRUE(map["causal"].IsBool());
  EXPECT_TRUE(map["causal"].AsBool());
  EXPECT_EQ(map["axes"].AsVector().size(), 2);
  EXPECT_EQ(map["axes"].AsVector()[1].AsInt64(), 2);
}

}  // namespace
}  // namespace litert::tools
