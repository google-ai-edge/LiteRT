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

#include "litert/vendors/mediatek/schema/schema_resolver.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/vendors/mediatek/schema/neuron_schema_generated.h"

namespace neuron {
namespace {

using Buffer = std::pair<const void*, size_t>;

std::vector<int8_t> Contents(const Buffer& buffer) {
  const auto* data = static_cast<const int8_t*>(buffer.first);
  return std::vector<int8_t>(data, data + buffer.second);
}

int32_t AddSharedWeight(BytecodeBuilder& builder, const std::string& name,
                        const std::vector<int8_t>& data) {
  return builder.AddSharedWeightBuffer(name, data.data(), data.size());
}

// Finishes `builder` and initializes `resolver` with the resulting bytecode.
void FinishAndResolve(BytecodeBuilder& builder, SchemaResolver& resolver) {
  ASSERT_TRUE(builder.Finish());
  auto [data, size] = builder.GetBytecode();
  auto initialized = resolver.Initialize(data, size);
  ASSERT_TRUE(initialized.HasValue());
  ASSERT_TRUE(initialized.Value());
}

// Builds bytecode with one one-byte buffer and one subgraph "g" whose
// `weight_share_index` has the given union types and values. The result is not
// guaranteed to pass flatbuffer verification.
std::vector<uint8_t> BuildGraphs(
    const std::vector<uint8_t>& types,
    const std::vector<::flatbuffers::Offset<void>>& values,
    ::flatbuffers::FlatBufferBuilder& fb) {
  const std::vector<int8_t> data = {42};
  std::vector<::flatbuffers::Offset<NeuronSchema::Buffer>> buffers = {
      NeuronSchema::CreateBufferDirect(fb, "dla", &data)};
  std::vector<::flatbuffers::Offset<NeuronSchema::Subgraph>> subgraphs = {
      NeuronSchema::CreateSubgraph(
          fb, fb.CreateString("g"), NeuronSchema::CompiledType_AdapterCache,
          NeuronSchema::BufferIndicate_Index,
          NeuronSchema::CreateIndex(fb, 0).Union(), fb.CreateVector(types),
          fb.CreateVector(values))};
  fb.Finish(NeuronSchema::CreateGraphsDirect(fb, 1, &subgraphs, &buffers));
  return std::vector<uint8_t>(fb.GetBufferPointer(),
                              fb.GetBufferPointer() + fb.GetSize());
}

TEST(BytecodeBuilderTest, SharedWeightsAreStoredOncePerDistinctContents) {
  const std::vector<int8_t> dla = {1, 2, 3, 4};
  const std::vector<int8_t> weight_a = {10, 20, 30};
  // Same contents as `weight_a` at a different address.
  const std::vector<int8_t> weight_a_copy = weight_a;
  // Same size as `weight_a` but different contents.
  const std::vector<int8_t> weight_b = {10, 20, 31};
  // A prefix of `weight_a`.
  const std::vector<int8_t> weight_c = {10, 20};

  BytecodeBuilder builder;
  const int32_t dla_index = builder.AddBuffer("dla", dla);
  const int32_t a = AddSharedWeight(builder, "a", weight_a);
  const int32_t b = AddSharedWeight(builder, "b", weight_b);
  const int32_t a_copy = AddSharedWeight(builder, "a_copy", weight_a_copy);
  const int32_t c = AddSharedWeight(builder, "c", weight_c);

  EXPECT_EQ(a_copy, a);
  EXPECT_NE(b, a);
  EXPECT_NE(c, a);
  EXPECT_NE(c, b);
  EXPECT_NE(dla_index, a);

  builder.AddCompiledNetwork("g0", NeuronSchema::CompiledType_AdapterCache,
                             dla_index, {a, b});
  builder.AddCompiledNetwork("g1", NeuronSchema::CompiledType_AdapterCache,
                             dla_index, {a_copy, c});
  SchemaResolver resolver;
  ASSERT_NO_FATAL_FAILURE(FinishAndResolve(builder, resolver));

  auto g0 = resolver.GetCompiledGraph("g0");
  ASSERT_TRUE(g0.has_value());
  auto g0_weights = g0->GetWeightShareBuffers();
  ASSERT_TRUE(g0_weights.HasValue());
  ASSERT_EQ(g0_weights->size(), 2);
  EXPECT_EQ(Contents((*g0_weights)[0]), weight_a);
  EXPECT_EQ(Contents((*g0_weights)[1]), weight_b);

  auto g1 = resolver.GetCompiledGraph("g1");
  ASSERT_TRUE(g1.has_value());
  auto g1_weights = g1->GetWeightShareBuffers();
  ASSERT_TRUE(g1_weights.HasValue());
  ASSERT_EQ(g1_weights->size(), 2);
  // Both subgraphs point at the same stored copy of `weight_a`.
  EXPECT_EQ((*g1_weights)[0].first, (*g0_weights)[0].first);
  EXPECT_EQ(Contents((*g1_weights)[1]), weight_c);
}

TEST(BytecodeBuilderTest, EmptySharedWeightsAreStoredOnce) {
  const int8_t unused = 0;
  BytecodeBuilder builder;
  const int32_t first = builder.AddSharedWeightBuffer("e0", &unused, 0);
  const int32_t second = builder.AddSharedWeightBuffer("e1", &unused, 0);
  EXPECT_EQ(second, first);
}

TEST(BytecodeBuilderTest, AddBufferDoesNotDeduplicate) {
  const std::vector<int8_t> data = {1, 2, 3};
  BytecodeBuilder builder;
  const int32_t first = builder.AddBuffer("x", data.data(), data.size());
  const int32_t second = builder.AddBuffer("y", data.data(), data.size());
  EXPECT_NE(second, first);
}

TEST(CompiledGraphTest, SubgraphWithoutSharedWeightsHasNoWeightBuffers) {
  const std::vector<int8_t> dla = {1, 2, 3, 4};
  BytecodeBuilder builder;
  const int32_t dla_index = builder.AddBuffer("dla", dla);
  builder.AddCompiledNetwork("g", NeuronSchema::CompiledType_AdapterCache,
                             dla_index);
  SchemaResolver resolver;
  ASSERT_NO_FATAL_FAILURE(FinishAndResolve(builder, resolver));

  auto graph = resolver.GetCompiledGraph("g");
  ASSERT_TRUE(graph.has_value());
  EXPECT_EQ(graph->GetCompiledType(), NeuronSchema::CompiledType_AdapterCache);
  auto network = graph->GetCompiledNetwork();
  ASSERT_TRUE(network.HasValue());
  EXPECT_EQ(Contents(*network), dla);
  auto weights = graph->GetWeightShareBuffers();
  ASSERT_TRUE(weights.HasValue());
  EXPECT_TRUE(weights->empty());
}

TEST(CompiledGraphTest, OutOfBoundsWeightIndexIsAnError) {
  const std::vector<int8_t> dla = {1, 2, 3, 4};
  BytecodeBuilder builder;
  const int32_t dla_index = builder.AddBuffer("dla", dla);
  builder.AddCompiledNetwork("past_end",
                             NeuronSchema::CompiledType_AdapterCache, dla_index,
                             {dla_index + 1});
  builder.AddCompiledNetwork(
      "negative", NeuronSchema::CompiledType_AdapterCache, dla_index, {-1});
  SchemaResolver resolver;
  ASSERT_NO_FATAL_FAILURE(FinishAndResolve(builder, resolver));

  for (const char* name : {"past_end", "negative"}) {
    SCOPED_TRACE(name);
    auto graph = resolver.GetCompiledGraph(name);
    ASSERT_TRUE(graph.has_value());
    auto weights = graph->GetWeightShareBuffers();
    ASSERT_FALSE(weights.HasValue());
    EXPECT_EQ(weights.Error().Status(), kLiteRtStatusErrorIndexOOB);
  }
}

TEST(CompiledGraphTest, NonIndexWeightReferenceIsUnsupported) {
  ::flatbuffers::FlatBufferBuilder fb;
  const auto identifier =
      NeuronSchema::CreateIdentifier(fb, fb.CreateString("dla")).Union();
  const std::vector<uint8_t> bytecode =
      BuildGraphs({NeuronSchema::BufferIndicate_Identifier}, {identifier}, fb);

  SchemaResolver resolver;
  auto initialized = resolver.Initialize(bytecode.data(), bytecode.size());
  ASSERT_TRUE(initialized.HasValue());
  ASSERT_TRUE(initialized.Value());
  auto graph = resolver.GetCompiledGraph("g");
  ASSERT_TRUE(graph.has_value());
  auto weights = graph->GetWeightShareBuffers();
  ASSERT_FALSE(weights.HasValue());
  EXPECT_EQ(weights.Error().Status(), kLiteRtStatusErrorUnsupported);
}

TEST(CompiledGraphTest, MismatchedWeightReferenceVectorsAreAnError) {
  // Flatbuffer verification rejects this, so bypass `SchemaResolver`.
  ::flatbuffers::FlatBufferBuilder fb;
  const auto index = NeuronSchema::CreateIndex(fb, 0).Union();
  const std::vector<uint8_t> bytecode = BuildGraphs(
      {NeuronSchema::BufferIndicate_Index, NeuronSchema::BufferIndicate_Index},
      {index}, fb);
  ASSERT_FALSE(IsNeuronSchema(bytecode.data(), bytecode.size()));

  const auto* graphs = NeuronSchema::GetGraphs(bytecode.data());
  CompiledGraph graph(*graphs, *graphs->subgraphs()->Get(0));
  auto weights = graph.GetWeightShareBuffers();
  ASSERT_FALSE(weights.HasValue());
  EXPECT_EQ(weights.Error().Status(), kLiteRtStatusErrorInvalidFlatbuffer);
}

TEST(SchemaResolverTest, UnknownEntryPointIsNotFound) {
  const std::vector<int8_t> dla = {1};
  BytecodeBuilder builder;
  builder.AddCompiledNetwork("g", NeuronSchema::CompiledType_AdapterCache,
                             builder.AddBuffer("dla", dla));
  SchemaResolver resolver;
  ASSERT_NO_FATAL_FAILURE(FinishAndResolve(builder, resolver));
  EXPECT_FALSE(resolver.GetCompiledGraph("other").has_value());
}

}  // namespace
}  // namespace neuron
