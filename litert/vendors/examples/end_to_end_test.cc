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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_api_types.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_runtime_options.h"
#include "litert/core/model/model.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"

namespace litert::example {
namespace {

using ::litert::testing::GetLiteRtPath;
using ::litert::testing::GetTestFilePath;
using ::testing::ElementsAre;
using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::Pointwise;
using ::testing::SizeIs;

static constexpr absl::string_view kLibsPath = "vendors/examples";
static constexpr absl::string_view kModel = "one_mul.tflite";

TEST(ExampleEndToEndTest, JIT) {
  const auto libs_path = GetLiteRtPath(kLibsPath);
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
          libs_path,
      },
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          libs_path,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, litert::Environment::Create(litert::EnvironmentOptions(
                    absl::MakeConstSpan(environment_options))));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto cm, CompiledModel::Create(env, GetTestFilePath(kModel),
                                     HwAccelerators::kNpu));
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              cm.CreateInputBuffers(cm.DefaultSignatureKey()));
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              cm.CreateOutputBuffers(cm.DefaultSignatureKey()));
  LITERT_ASSERT_OK(input_buffers[0].Write<float>({1.0f, 2.0f, 3.0f, 4.0f}));
  LITERT_ASSERT_OK(input_buffers[1].Write<float>({1.0f, 2.0f, 3.0f, 4.0f}));
  LITERT_ASSERT_OK(
      cm.Run(cm.DefaultSignatureKey(), input_buffers, output_buffers));
  std::vector<float> output(4);
  LITERT_ASSERT_OK(output_buffers[0].Read(absl::MakeSpan(output)));
  EXPECT_THAT(output, ElementsAre(1.0f, 4.0f, 9.0f, 16.0f));
}

TEST(ExampleEndToEndTest, JitHandleExecution) {
  const auto libs_path = GetLiteRtPath(kLibsPath);
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
          libs_path,
      },
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          libs_path,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, litert::Environment::Create(litert::EnvironmentOptions(
                    absl::MakeConstSpan(environment_options))));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto cm, CompiledModel::Create(env, GetTestFilePath(kModel),
                                     HwAccelerators::kNpu));
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              cm.CreateInputBuffers(cm.DefaultSignatureKey()));
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              cm.CreateOutputBuffers(cm.DefaultSignatureKey()));
  LITERT_ASSERT_OK(input_buffers[0].Write<float>({2.0f, 2.0f, 2.0f, 2.0f}));
  LITERT_ASSERT_OK(input_buffers[1].Write<float>({1.0f, 2.0f, 3.0f, 4.0f}));
  LITERT_ASSERT_OK(
      cm.Run(cm.DefaultSignatureKey(), input_buffers, output_buffers));
  std::vector<float> output(4);
  LITERT_ASSERT_OK(output_buffers[0].Read(absl::MakeSpan(output)));
  EXPECT_THAT(output, ElementsAre(2.0f, 4.0f, 6.0f, 8.0f));
}

TEST(ExampleEndToEndTest, SelectedSignaturesWithMultiSubgraphModel) {
  const auto libs_path = GetLiteRtPath(kLibsPath);
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
          libs_path,
      },
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          libs_path,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, litert::Environment::Create(litert::EnvironmentOptions(
                    absl::MakeConstSpan(environment_options))));

  LiteRtModel raw_model = nullptr;
  LITERT_ASSERT_OK(LiteRtCreateModelFromFile(
      env.Get(), GetTestFilePath("cst_multi_subgraph.tflite").c_str(),
      &raw_model));
  char* signatures[] = {const_cast<char*>("sig_0"), const_cast<char*>("sig_1")};
  OwningBufferRef<uint8_t> serialized_model;
  auto [data, size, offset] = serialized_model.GetWeak();
  const LiteRtModelSerializationOptions serialization_options = {
      /*bytecode_alignment=*/64};
  LITERT_ASSERT_OK(LiteRtSerializeModelWithSignatures(
      raw_model, &data, &size, &offset, /*destroy_model=*/true, signatures,
      /*num_signatures=*/2, serialization_options));

  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());
  options.SetHardwareAccelerators(HwAccelerators::kNpu);
  LITERT_ASSERT_OK_AND_ASSIGN(auto& runtime_options,
                              options.GetRuntimeOptions());
  const StringView selected_signatures[] = {"sig_1"};
  LITERT_ASSERT_OK(runtime_options.SetSelectedSignatures(selected_signatures));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto cm, CompiledModel::Create(env, serialized_model, options));
  EXPECT_EQ(cm.GetNumSignatures(), 1);
  LITERT_ASSERT_OK_AND_ASSIGN(auto signature_keys, cm.GetSignatureKeys());
  EXPECT_THAT(signature_keys, ElementsAre("sig_1"));

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              cm.CreateInputBuffers("sig_1"));
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              cm.CreateOutputBuffers("sig_1"));
  LITERT_ASSERT_OK(input_buffers[0].Write<float>({10.0f, 20.0f, 30.0f, 40.0f}));
  LITERT_ASSERT_OK(cm.Run("sig_1", input_buffers, output_buffers));
  std::vector<float> output(4);
  LITERT_ASSERT_OK(output_buffers[0].Read(absl::MakeSpan(output)));
  EXPECT_THAT(output, ElementsAre(10.0f, 40.0f, 90.0f, 160.0f));
}

class CompiledModelPeer : public CompiledModel {
 public:
  using CompiledModel::Create;
};

Expected<Environment> CreateExampleEnvironment() {
  const auto libs_path = GetLiteRtPath(kLibsPath);
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
          libs_path,
      },
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          libs_path,
      },
  };
  return litert::Environment::Create(
      litert::EnvironmentOptions(absl::MakeConstSpan(environment_options)));
}

// Adds a second signature, "unused", that aliases the first one, and returns
// the key of the first one.
std::string AddAliasSignature(LiteRtModelT& model) {
  LiteRtSignatureT& signature = *model.Signatures().front();
  std::vector<LiteRtTensor> inputs;
  std::vector<LiteRtTensor> outputs;
  for (size_t i = 0; i < signature.InputNames().size(); ++i) {
    inputs.push_back(signature.GetInputTensor(i));
  }
  for (size_t i = 0; i < signature.OutputNames().size(); ++i) {
    outputs.push_back(signature.GetOutputTensor(i));
  }
  std::string key(signature.Key());
  model.EmplaceSignature(&signature.GetSubgraph(), signature.InputNames(),
                         inputs, signature.OutputNames(), outputs, "unused");
  return key;
}

// JIT-compiles `model` for the example NPU with only `signature_key` selected.
Expected<CompiledModel> CompileWithSelectedSignature(
    const Environment& env, Model& model, const std::string& signature_key) {
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  options.SetHardwareAccelerators(HwAccelerators::kNpu | HwAccelerators::kCpu);
  LITERT_ASSIGN_OR_RETURN(auto& runtime_options, options.GetRuntimeOptions());
  const StringView selected[] = {signature_key};
  LITERT_RETURN_IF_ERROR(runtime_options.SetSelectedSignatures(selected));
  return CompiledModelPeer::Create(env, model.Get(), options);
}

// 2x2 max pool with stride 2 of a 1x16x16x4 tensor, times `scale`.
std::vector<float> MaxPool2x2Times(const std::vector<float>& x, float scale) {
  constexpr int kH = 16, kW = 16, kC = 4;
  std::vector<float> result;
  for (int h = 0; h < kH / 2; ++h) {
    for (int w = 0; w < kW / 2; ++w) {
      for (int c = 0; c < kC; ++c) {
        float max = x[((2 * h) * kW + 2 * w) * kC + c];
        for (int dh = 0; dh < 2; ++dh) {
          for (int dw = 0; dw < 2; ++dw) {
            max = std::max(max, x[((2 * h + dh) * kW + 2 * w + dw) * kC + c]);
          }
        }
        result.push_back(max * scale);
      }
    }
  }
  return result;
}

// main: REDUCE_WINDOW (2x2 max pool) -> MUL. Signature selection prunes the
// model, the REDUCE_WINDOW and its body stay on CPU, and the MUL is offloaded.
TEST(ExampleEndToEndTest, JitWithSignatureSelectionOnRegionModel) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, CreateExampleEnvironment());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model,
      Model::CreateFromFile(
          env, GetTestFilePath("shlo_reduce_window_region.tflite")));
  LiteRtModelT& model_impl = *model.Get();
  ASSERT_EQ(model_impl.NumSubgraphs(), 2);
  ASSERT_EQ(model_impl.Signatures().size(), 1);
  const std::string signature_key = AddAliasSignature(model_impl);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto cm, CompileWithSelectedSignature(env, model, signature_key));

  // Pruned to the selected signature; the reduction body is kept.
  ASSERT_EQ(model_impl.Signatures().size(), 1);
  EXPECT_EQ(model_impl.Signatures().front()->Key(), signature_key);
  EXPECT_EQ(model_impl.NumSubgraphs(), 2);

  constexpr int kH = 16, kW = 16, kC = 4;
  std::vector<float> x(kH * kW * kC);
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = static_cast<float>((i * 37) % 101) - 50.0f;
  }
  const std::vector<float> scale(kH / 2 * kW / 2 * kC, 2.0f);
  const std::vector<float> expected = MaxPool2x2Times(x, 2.0f);

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              cm.CreateInputBuffers(signature_key));
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              cm.CreateOutputBuffers(signature_key));
  ASSERT_THAT(input_buffers, SizeIs(2));
  LITERT_ASSERT_OK(input_buffers[0].Write<float>(absl::MakeConstSpan(x)));
  LITERT_ASSERT_OK(input_buffers[1].Write<float>(absl::MakeConstSpan(scale)));
  LITERT_ASSERT_OK(cm.Run(signature_key, input_buffers, output_buffers));
  std::vector<float> output(expected.size());
  LITERT_ASSERT_OK(output_buffers[0].Read(absl::MakeSpan(output)));
  EXPECT_THAT(output, Pointwise(FloatEq(), expected));
}

// main: WHILE(cond, body) -> MUL. Signature selection prunes the model and
// keeps both WHILE branches; the MULs in main and in the WHILE body are
// offloaded. The body is active because it is reachable from the selected
// signature, so its dispatch op is prepared, and input x, which feeds both the
// CPU WHILE and the offloaded MUL, is not held locked by the CPU side while
// the dispatch op runs.
TEST(ExampleEndToEndTest, JitWithSignatureSelectionOnControlFlowModel) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, CreateExampleEnvironment());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model,
      Model::CreateFromFile(env, GetTestFilePath("tfl_while_region.tflite")));
  LiteRtModelT& model_impl = *model.Get();
  ASSERT_EQ(model_impl.NumSubgraphs(), 3);
  const std::string signature_key = AddAliasSignature(model_impl);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto cm, CompileWithSelectedSignature(env, model, signature_key));

  ASSERT_EQ(model_impl.Signatures().size(), 1);
  EXPECT_EQ(model_impl.Signatures().front()->Key(), signature_key);
  ASSERT_EQ(model_impl.NumSubgraphs(), 3);
  auto num_dispatch_ops = [](const LiteRtSubgraphT& subgraph) {
    return absl::c_count_if(subgraph.Ops(), [](const LiteRtOp op) {
      return op->OpCode() == kLiteRtOpCodeTflCustom;
    });
  };
  EXPECT_EQ(num_dispatch_ops(model_impl.Subgraph(0)), 1);
  EXPECT_EQ(num_dispatch_ops(model_impl.Subgraph(2)), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              cm.CreateInputBuffers(signature_key));
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              cm.CreateOutputBuffers(signature_key));
  ASSERT_THAT(input_buffers, SizeIs(2));
  LITERT_ASSERT_OK(input_buffers[0].Write<int32_t>({2}));
  LITERT_ASSERT_OK(input_buffers[1].Write<float>({1.0f, 2.0f, 3.0f, 4.0f}));
  // Run twice: buffers are registered and released again on every run.
  for (int run = 0; run < 2; ++run) {
    LITERT_ASSERT_OK(cm.Run(signature_key, input_buffers, output_buffers));
    // Two iterations square x twice; main then multiplies by the original x.
    std::vector<float> output(4);
    LITERT_ASSERT_OK(output_buffers[0].Read(absl::MakeSpan(output)));
    EXPECT_THAT(output, ElementsAre(1.0f, 32.0f, 243.0f, 1024.0f));
  }
}

// main: MUL(x, x) -> ADD(., x) -> REDUCE_WINDOW (2x2 max) -> MUL(., y). Input x
// feeds both an offloaded MUL and the CPU ADD, so its host buffer is
// registered with the dispatch delegate and also used by CPU ops. The CPU side
// must not hold the buffer locked while the dispatch op reads it.
TEST(ExampleEndToEndTest, JitWithInputSharedByCpuAndNpu) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, CreateExampleEnvironment());
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());
  options.SetHardwareAccelerators(HwAccelerators::kNpu | HwAccelerators::kCpu);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto cm,
      CompiledModel::Create(
          env, GetTestFilePath("multi_island_reduce_window.tflite"), options));

  constexpr int kH = 16, kW = 16, kC = 4;
  std::vector<float> x(kH * kW * kC);
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = static_cast<float>((i * 37) % 101) / 10.0f - 5.0f;
  }
  const std::vector<float> y(kH / 2 * kW / 2 * kC, 3.0f);
  std::vector<float> v(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    v[i] = x[i] * x[i] + x[i];
  }
  const std::vector<float> expected = MaxPool2x2Times(v, 3.0f);

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers, cm.CreateInputBuffers());
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers, cm.CreateOutputBuffers());
  ASSERT_THAT(input_buffers, SizeIs(2));
  LITERT_ASSERT_OK(input_buffers[0].Write<float>(absl::MakeConstSpan(x)));
  LITERT_ASSERT_OK(input_buffers[1].Write<float>(absl::MakeConstSpan(y)));
  for (int run = 0; run < 2; ++run) {
    LITERT_ASSERT_OK(cm.Run(input_buffers, output_buffers));
    std::vector<float> output(expected.size());
    LITERT_ASSERT_OK(output_buffers[0].Read(absl::MakeSpan(output)));
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), expected));
  }
}

// Runs `cm` twice on npu_call_region.tflite inputs and checks the output.
void RunNpuCallRegionModel(CompiledModel& cm, absl::string_view signature_key) {
  std::vector<float> x(16 * 16 * 4);
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = static_cast<float>((i * 53) % 97) - 40.0f;
  }
  const std::vector<float> y(8 * 8 * 4, 0.5f);
  const std::vector<float> expected = MaxPool2x2Times(x, 0.5f);
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              cm.CreateInputBuffers(signature_key));
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              cm.CreateOutputBuffers(signature_key));
  ASSERT_THAT(input_buffers, SizeIs(2));
  LITERT_ASSERT_OK(input_buffers[0].Write<float>(absl::MakeConstSpan(x)));
  LITERT_ASSERT_OK(input_buffers[1].Write<float>(absl::MakeConstSpan(y)));
  for (int run = 0; run < 2; ++run) {
    LITERT_ASSERT_OK(cm.Run(signature_key, input_buffers, output_buffers));
    std::vector<float> output(expected.size());
    LITERT_ASSERT_OK(output_buffers[0].Read(absl::MakeSpan(output)));
    EXPECT_THAT(output, Pointwise(FloatEq(), expected));
  }
}

// main: npu_call(body: REDUCE_WINDOW -> MUL). The example plugin is given the
// whole npu_call body, including its REDUCE_WINDOW, but it cannot compile
// REDUCE_WINDOW: compilation fails after the model was partitioned. The
// compiled model must fall back to the original model and run it on CPU.
TEST(ExampleEndToEndTest, JitFallsBackToOriginalModelWhenCompilationFails) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, CreateExampleEnvironment());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model,
      Model::CreateFromFile(env, GetTestFilePath("npu_call_region.tflite")));
  LiteRtModelT& model_impl = *model.Get();
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());
  options.SetHardwareAccelerators(HwAccelerators::kNpu | HwAccelerators::kCpu);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto cm, CompiledModelPeer::Create(env, model.Get(), options));

  // The caller-visible model was partitioned before compilation failed (the
  // npu_call became a dispatch op without byte code), so it only runs because
  // the compiled model fell back to the original model.
  EXPECT_TRUE(absl::c_any_of(model_impl.Subgraph(0).Ops(),
                             [](const LiteRtOp op) {
                               return op->OpCode() == kLiteRtOpCodeTflCustom;
                             }));
  RunNpuCallRegionModel(cm, cm.DefaultSignatureKey());
}

// As above, with one of two signatures selected: the caller-visible model is
// pruned, so the fallback must prune a fresh copy of the original model too.
TEST(ExampleEndToEndTest,
     JitWithSignatureSelectionFallsBackWhenCompilationFails) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, CreateExampleEnvironment());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model,
      Model::CreateFromFile(env, GetTestFilePath("npu_call_region.tflite")));
  LiteRtModelT& model_impl = *model.Get();
  const std::string signature_key = AddAliasSignature(model_impl);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto cm, CompileWithSelectedSignature(env, model, signature_key));

  ASSERT_EQ(model_impl.Signatures().size(), 1);
  EXPECT_EQ(model_impl.Signatures().front()->Key(), signature_key);
  // The caller-visible model was partitioned before compilation failed (the
  // npu_call became a dispatch op without byte code), so it only runs because
  // the compiled model fell back to the original model.
  EXPECT_TRUE(absl::c_any_of(model_impl.Subgraph(0).Ops(),
                             [](const LiteRtOp op) {
                               return op->OpCode() == kLiteRtOpCodeTflCustom;
                             }));
  RunNpuCallRegionModel(cm, signature_key);
}

}  // namespace
}  // namespace litert::example
