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

#include "litert/vendors/intel_openvino/compiler/global_graph.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace litert {
namespace openvino {
namespace {

// Builds a container with two shared buffers and two subgraphs whose const_maps
// reference those buffers.
OpenVinoGlobalGraph MakeSample() {
  OpenVinoGlobalGraph graph;
  graph.buffers[0] = std::string("\x01\x02\x03\x04", 4);
  graph.buffers[7] = std::string(10, '\xAB');  // non-contiguous id, larger buffer

  OpenVinoGlobalGraph::Subgraph prefill;
  prefill.name = "Partition_0";
  prefill.device = 2;  // e.g. GPU enum
  prefill.const_map = {{"weight_a", 0u}, {"weight_b", 7u}};
  prefill.payload = "prefill-blob-bytes";

  OpenVinoGlobalGraph::Subgraph decode;
  decode.name = "Partition_1";
  decode.device = 2;
  decode.const_map = {{"weight_a", 7u}};
  decode.payload = std::string("\x00\x00\xFF", 3);  // embedded NULs must survive

  graph.subgraphs[prefill.name] = prefill;
  graph.subgraphs[decode.name] = decode;
  return graph;
}

// Serialize -> Parse reproduces the buffer pool, subgraph topology, const_maps,
// device, and payloads exactly (including embedded NUL bytes).
TEST(GlobalGraphTest, RoundTrip) {
  const OpenVinoGlobalGraph in = MakeSample();
  const std::string blob = in.Serialize();

  ASSERT_TRUE(OpenVinoGlobalGraph::HasMagic(
      reinterpret_cast<const uint8_t*>(blob.data()), blob.size()));

  auto parsed = OpenVinoGlobalGraph::Parse(
      reinterpret_cast<const uint8_t*>(blob.data()), blob.size());
  ASSERT_TRUE(parsed.HasValue());
  const OpenVinoGlobalGraph& out = parsed.Value();

  // Buffer pool.
  ASSERT_EQ(out.buffers.size(), in.buffers.size());
  for (const auto& [id, bytes] : in.buffers) {
    ASSERT_TRUE(out.buffers.count(id));
    EXPECT_EQ(out.buffers.at(id), bytes);
  }
  EXPECT_EQ(out.BankBytes(), in.BankBytes());

  // Subgraphs.
  ASSERT_EQ(out.subgraphs.size(), in.subgraphs.size());
  for (const auto& [name, in_subgraph] : in.subgraphs) {
    ASSERT_TRUE(out.subgraphs.count(name));
    const auto& out_subgraph = out.subgraphs.at(name);
    EXPECT_EQ(out_subgraph.name, in_subgraph.name);
    EXPECT_EQ(out_subgraph.device, in_subgraph.device);
    EXPECT_EQ(out_subgraph.payload, in_subgraph.payload);
    EXPECT_EQ(out_subgraph.const_map, in_subgraph.const_map);
  }
}

// BankBytes sums the deduplicated buffer pool.
TEST(GlobalGraphTest, BankBytesSumsPool) {
  const OpenVinoGlobalGraph graph = MakeSample();
  EXPECT_EQ(graph.BankBytes(), 4u + 10u);
}

// An empty container round-trips (magic + zero counts).
TEST(GlobalGraphTest, EmptyRoundTrips) {
  OpenVinoGlobalGraph in;
  const std::string blob = in.Serialize();
  auto parsed = OpenVinoGlobalGraph::Parse(
      reinterpret_cast<const uint8_t*>(blob.data()), blob.size());
  ASSERT_TRUE(parsed.HasValue());
  EXPECT_TRUE(parsed.Value().buffers.empty());
  EXPECT_TRUE(parsed.Value().subgraphs.empty());
}

// HasMagic rejects non-container / short input.
TEST(GlobalGraphTest, HasMagicRejectsBadInput) {
  EXPECT_FALSE(OpenVinoGlobalGraph::HasMagic(nullptr, 0));
  const std::string notmagic = "NOTMAGIC.....";
  EXPECT_FALSE(OpenVinoGlobalGraph::HasMagic(
      reinterpret_cast<const uint8_t*>(notmagic.data()), notmagic.size()));
  const std::string tooshort = "OVG";
  EXPECT_FALSE(OpenVinoGlobalGraph::HasMagic(
      reinterpret_cast<const uint8_t*>(tooshort.data()), tooshort.size()));
}

// Parse errors (does not crash / over-read) on bad magic.
TEST(GlobalGraphTest, ParseRejectsBadMagic) {
  const std::string junk = "not-an-ovglobal-container-blob";
  auto parsed = OpenVinoGlobalGraph::Parse(
      reinterpret_cast<const uint8_t*>(junk.data()), junk.size());
  EXPECT_FALSE(parsed.HasValue());
}

// Parse errors on a blob whose version byte does not match kVersion, rather
// than misparsing a future/unknown layout.
TEST(GlobalGraphTest, ParseRejectsUnknownVersion) {
  std::string blob = MakeSample().Serialize();
  // The version is a little-endian uint16 immediately after the 8-byte magic.
  ASSERT_GT(blob.size(), 9u);
  blob[8] = static_cast<char>(OpenVinoGlobalGraph::kVersion + 1);
  auto parsed = OpenVinoGlobalGraph::Parse(
      reinterpret_cast<const uint8_t*>(blob.data()), blob.size());
  EXPECT_FALSE(parsed.HasValue());
}

// Parse errors on a truncated (mid-buffer) blob rather than over-reading.
TEST(GlobalGraphTest, ParseRejectsTruncated) {
  const std::string blob = MakeSample().Serialize();
  for (size_t cut : {blob.size() / 2, blob.size() - 1}) {
    auto parsed = OpenVinoGlobalGraph::Parse(
        reinterpret_cast<const uint8_t*>(blob.data()), cut);
    EXPECT_FALSE(parsed.HasValue()) << "cut=" << cut;
  }
}

}  // namespace
}  // namespace openvino
}  // namespace litert
