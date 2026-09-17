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

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/nvidia/bytecode.h"
#include "litert/vendors/nvidia/cache_layout.h"
#include "tflite/schema/schema_generated.h"

namespace {

class CompilerAotTest : public testing::Test {
 protected:
  void SetUp() override {
    const char* root = std::getenv("TEST_TMPDIR");
    ASSERT_NE(root, nullptr);
    std::string pattern = std::string(root) + "/nvidia-aot-XXXXXX";
    ASSERT_NE(mkdtemp(pattern.data()), nullptr);
    directory_ = pattern;
    source_ = directory_ / "model.bin";
    std::ofstream(source_) << "test model identity";
    ASSERT_EQ(
        setenv("LITERT_NVIDIA_TENSORRT_AOT_CACHE_DIR", directory_.c_str(), 1),
        0);
    ASSERT_EQ(
        setenv("LITERT_NVIDIA_TENSORRT_AOT_MODEL_PATH", source_.c_str(), 1), 0);
    ASSERT_EQ(LiteRtCreateCompilerPlugin(LrtGetCompilerContext(), &plugin_,
                                         nullptr, nullptr),
              kLiteRtStatusOk);
  }

  void TearDown() override {
    if (plugin_ != nullptr) LiteRtDestroyCompilerPlugin(plugin_);
    unsetenv("LITERT_NVIDIA_TENSORRT_AOT_CACHE_DIR");
    unsetenv("LITERT_NVIDIA_TENSORRT_AOT_MODEL_PATH");
    if (!directory_.empty()) std::filesystem::remove_all(directory_);
  }

  LiteRtSubgraphT& AddPartition(LiteRtModelT& model) {
    auto& graph = model.EmplaceSubgraph();
    auto& input = graph.EmplaceTensor();
    input.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {4}));
    graph.Inputs().push_back(&input);
    auto& constant = graph.EmplaceTensor();
    constant.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {4}));
    static const float weights[] = {1, -2, 3, -4};
    SetWeightsFromUnownedBuffer(
        constant.Weights(),
        litert::BufferRef<uint8_t>(reinterpret_cast<const uint8_t*>(weights),
                                   sizeof(weights)));
    auto& output = graph.EmplaceTensor();
    output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {4}));
    graph.Outputs().push_back(&output);
    auto& add = graph.EmplaceOp();
    add.SetOpCode(kLiteRtOpCodeTflAdd);
    tflite::BuiltinOptionsUnion options;
    options.Set(tflite::AddOptionsT{});
    litert::internal::SetTflOptions(add, std::move(options));
    litert::internal::AttachInput(&input, add);
    litert::internal::AttachInput(&constant, add);
    litert::internal::AttachOutput(&output, add);
    return graph;
  }

  size_t CountFilesWithPrefix(const std::string& prefix) const {
    size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory_)) {
      if (entry.path().filename().string().find(prefix) == 0) ++count;
    }
    return count;
  }

  LiteRtCompilerPlugin CreateWithReadOnlyInputs(
      std::vector<std::string> names,
      LiteRtStatus expected_status = kLiteRtStatusOk) {
    std::vector<const char*> pointers;
    for (const auto& name : names) pointers.push_back(name.c_str());
    LiteRtNvidiaReadOnlyValueCacheOptions payload{
        static_cast<uint32_t>(pointers.size()), pointers.data()};
    LiteRtOptions options = nullptr;
    EXPECT_EQ(LiteRtCreateOptions(&options), kLiteRtStatusOk);
    if (options == nullptr) return nullptr;
    LiteRtOpaqueOptions opaque = nullptr;
    EXPECT_EQ(LiteRtCreateOpaqueOptions(
                  LITERT_NVIDIA_READ_ONLY_VALUE_CACHE_OPTIONS_ID, &payload,
                  /*payload_destructor=*/[](void*) {}, &opaque),
              kLiteRtStatusOk);
    if (opaque == nullptr) {
      LiteRtDestroyOptions(options);
      return nullptr;
    }
    EXPECT_EQ(LiteRtSetOpaqueOptionsHash(
                  opaque, litert::nvidia::HashReadOnlyValueCacheOptions),
              kLiteRtStatusOk);
    EXPECT_EQ(LiteRtAddOpaqueOptions(options, opaque), kLiteRtStatusOk);
    LiteRtCompilerPlugin plugin = nullptr;
    EXPECT_EQ(LiteRtCreateCompilerPlugin(LrtGetCompilerContext(), &plugin,
                                         nullptr, options),
              expected_status);
    LiteRtDestroyOptions(options);
    // The returned plugin must own its copy after these names/pointers and
    // the opaque payload go out of scope.
    return plugin;
  }

  LiteRtCompilerPlugin plugin_ = nullptr;
  std::filesystem::path directory_;
  std::filesystem::path source_;
};

TEST_F(CompilerAotTest, PersistsCompletedShardBeforeLaterCompilationFailure) {
  LiteRtModelT model;
  AddPartition(model);
  auto& invalid = model.EmplaceSubgraph();
  invalid.EmplaceOp().SetOpCode(kLiteRtOpCodeTflCustom);
  LiteRtCompiledResult result = nullptr;
  EXPECT_NE(LiteRtCompilerPluginCompile(plugin_, nullptr, &model, &result),
            kLiteRtStatusOk);
  EXPECT_EQ(result, nullptr);
  // Each shard is published before the next partition builds. A failure
  // must never publish a manifest claiming that the model is complete.
  EXPECT_EQ(CountFilesWithPrefix("tensorrt_aot_v"), 1);
  EXPECT_EQ(CountFilesWithPrefix("tensorrt_aot_index_"), 0);
}

TEST_F(CompilerAotTest, ProducesIndependentShardsAndReusesCompleteManifest) {
  LiteRtModelT model;
  AddPartition(model);
  AddPartition(model);
  std::vector<std::vector<uint8_t>> first_locators;
  for (int run = 0; run < 2; ++run) {
    LiteRtCompiledResult raw_result = nullptr;
    ASSERT_EQ(
        LiteRtCompilerPluginCompile(plugin_, nullptr, &model, &raw_result),
        kLiteRtStatusOk);
    std::unique_ptr<LiteRtCompiledResultT,
                    decltype(&LiteRtDestroyCompiledResult)>
        result(raw_result, LiteRtDestroyCompiledResult);
    LiteRtParamIndex modules = 0;
    ASSERT_EQ(LiteRtCompiledResultNumByteCodeModules(result.get(), &modules),
              kLiteRtStatusOk);
    ASSERT_EQ(modules, 2);
    for (LiteRtParamIndex i = 0; i < modules; ++i) {
      const void* data = nullptr;
      size_t size = 0;
      ASSERT_EQ(LiteRtGetCompiledResultByteCode(result.get(), i, &data, &size),
                kLiteRtStatusOk);
      const auto* bytes = static_cast<const uint8_t*>(data);
      std::vector<uint8_t> locator_bytes(bytes, bytes + size);
      if (run == 0)
        first_locators.push_back(locator_bytes);
      else
        EXPECT_EQ(locator_bytes, first_locators[i]);
      auto locator = litert::nvidia::TryParseTensorRtAotLocator(bytes, size);
      ASSERT_TRUE(locator.HasValue());
      ASSERT_TRUE(locator->has_value());
      std::ifstream file((*locator)->path, std::ios::binary);
      std::vector<uint8_t> artifact((std::istreambuf_iterator<char>(file)), {});
      const std::string function = "tensorrt_partition_" + std::to_string(i);
      auto parsed = litert::nvidia::ParseTensorRtBytecode(
          artifact.data(), artifact.size(), function.c_str());
      ASSERT_TRUE(parsed.HasValue());
      EXPECT_EQ(parsed->function_name, function);
      EXPECT_FALSE(parsed->engine_size == 0);
      EXPECT_FALSE(parsed->refit_weights.empty());
      auto other = litert::nvidia::ParseTensorRtBytecode(
          artifact.data(), artifact.size(), "tensorrt_partition_999");
      EXPECT_FALSE(other.HasValue());
    }
    EXPECT_EQ(CountFilesWithPrefix("tensorrt_aot_index_"), 1);
  }
}

TEST_F(CompilerAotTest, NonAotStillSharesOneWeightStoreAcrossEngines) {
  ASSERT_EQ(unsetenv("LITERT_NVIDIA_TENSORRT_AOT_CACHE_DIR"), 0);
  LiteRtModelT model;
  AddPartition(model);
  AddPartition(model);
  LiteRtCompiledResult raw_result = nullptr;
  ASSERT_EQ(LiteRtCompilerPluginCompile(plugin_, nullptr, &model, &raw_result),
            kLiteRtStatusOk);
  std::unique_ptr<LiteRtCompiledResultT, decltype(&LiteRtDestroyCompiledResult)>
      result(raw_result, LiteRtDestroyCompiledResult);
  LiteRtParamIndex modules = 0;
  ASSERT_EQ(LiteRtCompiledResultNumByteCodeModules(result.get(), &modules),
            kLiteRtStatusOk);
  ASSERT_EQ(modules, 1);
  const void* data = nullptr;
  size_t size = 0;
  ASSERT_EQ(LiteRtGetCompiledResultByteCode(result.get(), 0, &data, &size),
            kLiteRtStatusOk);
  auto first =
      litert::nvidia::ParseTensorRtBytecode(data, size, "tensorrt_partition_0");
  auto second =
      litert::nvidia::ParseTensorRtBytecode(data, size, "tensorrt_partition_1");
  ASSERT_TRUE(first.HasValue());
  ASSERT_TRUE(second.HasValue());
  ASSERT_EQ(first->refit_weights.size(), 1);
  ASSERT_EQ(second->refit_weights.size(), 1);
  EXPECT_EQ(first->refit_weights[0].data, second->refit_weights[0].data);
  EXPECT_EQ(CountFilesWithPrefix("tensorrt_aot_"), 0);
}

TEST_F(CompilerAotTest, ReadOnlyValueCacheLayoutSeparatesDirectAotCache) {
  LiteRtModelT model;
  auto& graph = model.EmplaceSubgraph();
  auto& cache = graph.EmplaceTensor();
  cache.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16, {1, 2, 8, 16}));
  cache.SetName("cache");
  graph.Inputs().push_back(&cache);
  auto& lhs = graph.EmplaceTensor();
  lhs.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16, {1, 2, 1, 16}));
  graph.Inputs().push_back(&lhs);
  auto& output = graph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16, {1, 2, 1, 8}));
  graph.Outputs().push_back(&output);
  auto& matmul = graph.EmplaceOp();
  matmul.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  tflite::BatchMatMulOptionsT matmul_options;
  matmul_options.adj_y = true;
  tflite::BuiltinOptionsUnion options;
  options.Set(std::move(matmul_options));
  litert::internal::SetTflOptions(matmul, std::move(options));
  litert::internal::AttachInput(&lhs, matmul);
  litert::internal::AttachInput(&cache, matmul);
  litert::internal::AttachOutput(&output, matmul);

  std::unique_ptr<LiteRtCompilerPluginT, decltype(&LiteRtDestroyCompilerPlugin)>
      transposed_plugin(CreateWithReadOnlyInputs({"cache"}),
                        LiteRtDestroyCompilerPlugin);
  ASSERT_NE(transposed_plugin, nullptr);
  std::vector<std::vector<uint8_t>> first_locators(2);
  for (int run = 0; run < 4; ++run) {
    const bool transposed = run % 2 != 0;
    SCOPED_TRACE(run);
    LiteRtCompiledResult raw_result = nullptr;
    ASSERT_EQ(LiteRtCompilerPluginCompile(
                  transposed ? transposed_plugin.get() : plugin_, nullptr,
                  &model, &raw_result),
              kLiteRtStatusOk);
    std::unique_ptr<LiteRtCompiledResultT,
                    decltype(&LiteRtDestroyCompiledResult)>
        result(raw_result, LiteRtDestroyCompiledResult);
    const void* data = nullptr;
    size_t size = 0;
    ASSERT_EQ(LiteRtGetCompiledResultByteCode(result.get(), 0, &data, &size),
              kLiteRtStatusOk);
    const auto* bytes = static_cast<const uint8_t*>(data);
    const std::vector<uint8_t> locator_bytes(bytes, bytes + size);
    if (run < 2)
      first_locators[transposed] = locator_bytes;
    else
      EXPECT_EQ(locator_bytes, first_locators[transposed]);
    auto locator = litert::nvidia::TryParseTensorRtAotLocator(bytes, size);
    ASSERT_TRUE(locator.HasValue());
    ASSERT_TRUE(locator->has_value());
    std::ifstream file((*locator)->path, std::ios::binary);
    const std::vector<uint8_t> artifact((std::istreambuf_iterator<char>(file)),
                                        {});
    auto parsed = litert::nvidia::ParseTensorRtBytecode(
        artifact.data(), artifact.size(), "tensorrt_partition_0");
    ASSERT_TRUE(parsed.HasValue());
    ASSERT_FALSE(parsed->input_names.empty());
    EXPECT_EQ(
        litert::nvidia::IsTransposedValueCacheTensor(parsed->input_names[0]),
        transposed);
    EXPECT_EQ(CountFilesWithPrefix("tensorrt_aot_index_"), run == 0 ? 1 : 2);
  }
  EXPECT_NE(first_locators[0], first_locators[1]);
}

TEST_F(CompilerAotTest, ReadOnlyValueCacheOptionsAreCanonicalAndValidated) {
  std::vector<std::string> sdk_versions;
  for (const auto& names : std::vector<std::vector<std::string>>{
           {}, {"a"}, {"b"}, {"a", "b"}, {"b", "a"}}) {
    std::unique_ptr<LiteRtCompilerPluginT,
                    decltype(&LiteRtDestroyCompilerPlugin)>
        plugin(CreateWithReadOnlyInputs(names), LiteRtDestroyCompilerPlugin);
    ASSERT_NE(plugin, nullptr);
    const char* sdk = nullptr;
    ASSERT_EQ(LiteRtGetCompilerPluginSDKVersion(plugin.get(), &sdk),
              kLiteRtStatusOk);
    sdk_versions.emplace_back(sdk);
  }
  EXPECT_NE(sdk_versions[0], sdk_versions[1]);
  EXPECT_NE(sdk_versions[1], sdk_versions[2]);
  EXPECT_NE(sdk_versions[1], sdk_versions[3]);
  EXPECT_EQ(sdk_versions[3], sdk_versions[4]);
  EXPECT_EQ(
      CreateWithReadOnlyInputs({"a", "a"}, kLiteRtStatusErrorInvalidArgument),
      nullptr);
  EXPECT_EQ(CreateWithReadOnlyInputs({""}, kLiteRtStatusErrorInvalidArgument),
            nullptr);
}

}  // namespace
