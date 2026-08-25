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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl

namespace litert {
namespace openvino {
namespace {

absl::Span<const uint8_t> AsBytes(const std::string& s) {
  return absl::MakeConstSpan(reinterpret_cast<const uint8_t*>(s.data()),
                             s.size());
}

// A sample container plus the storage its spans borrow from. Because the
// unified OpenVinoGlobalGraph carries borrowed spans (not owned strings), the
// backing bytes must outlive the graph; this holder owns both. Two shared
// buffers (non-contiguous ids) and two subgraphs whose const_maps reference
// them, with an embedded-NUL payload.
struct Sample {
  std::string buf0 = "\x01\x02\x03\x04";
  std::string buf7 = std::string(10, '\xAB');  // larger, non-contiguous id
  std::string payload0 = "prefill-blob-bytes";
  std::string payload1 = std::string("\x00\x00\xFF", 3);  // embedded NULs
  OpenVinoGlobalGraph graph;

  Sample() {
    // Ascending buffer_id, pool_offset == running byte sum (the invariant
    // Serialize() enforces).
    graph.buffers.push_back({0u, 0u, AsBytes(buf0)});
    graph.buffers.push_back({7u, buf0.size(), AsBytes(buf7)});

    OpenVinoGlobalGraph::Subgraph prefill;
    prefill.name = "Partition_0";
    prefill.device = 2;  // e.g. GPU enum
    prefill.const_map = {{"weight_a", 0u}, {"weight_b", 7u}};
    prefill.payload = AsBytes(payload0);
    graph.subgraphs["Partition_0"] = prefill;

    OpenVinoGlobalGraph::Subgraph decode;
    decode.name = "Partition_1";
    decode.device = 2;
    decode.const_map = {{"weight_a", 7u}};
    decode.payload = AsBytes(payload1);
    graph.subgraphs["Partition_1"] = decode;
  }
};

// Serialize -> Parse reproduces the buffer pool, subgraph topology, const_maps,
// device, and payloads exactly (including embedded NUL bytes), and Parse
// locates them zero-copy (spans alias the blob).
TEST(GlobalGraphTest, RoundTrip) {
  Sample sample;
  const OpenVinoGlobalGraph& in = sample.graph;
  const std::string blob = in.Serialize();
  const auto* base = reinterpret_cast<const uint8_t*>(blob.data());

  ASSERT_TRUE(OpenVinoGlobalGraph::HasMagic(base, blob.size()));

  auto parsed = OpenVinoGlobalGraph::Parse(base, blob.size());
  ASSERT_TRUE(parsed.HasValue());
  const OpenVinoGlobalGraph& out = parsed.Value();

  // Pool locus: the pool span aliases the blob, in-bounds, sized to the sum.
  EXPECT_EQ(out.pool.size(), in.BankBytes());
  ASSERT_EQ(out.pool.data(), base + out.pool_data_offset);
  EXPECT_GE(out.pool_data_offset, 8u + 2u);  // after magic + version
  EXPECT_LE(out.pool_data_offset + out.pool.size(), blob.size());
  EXPECT_EQ(out.BankBytes(), in.BankBytes());

  // Buffer pool: same ascending-id entries, offsets, and bytes; the bytes are
  // aliased in place (== pool.data() + pool_offset).
  ASSERT_EQ(out.buffers.size(), in.buffers.size());
  for (size_t i = 0; i < in.buffers.size(); ++i) {
    const auto& a = in.buffers[i];
    const auto& b = out.buffers[i];
    EXPECT_EQ(b.id, a.id);
    EXPECT_EQ(b.pool_offset, a.pool_offset);
    ASSERT_EQ(b.bytes.size(), a.bytes.size()) << "buffer id=" << a.id;
    EXPECT_EQ(b.bytes.data(), out.pool.data() + b.pool_offset);
    EXPECT_EQ(0, std::string(reinterpret_cast<const char*>(b.bytes.data()),
                             b.bytes.size())
                     .compare(std::string(
                         reinterpret_cast<const char*>(a.bytes.data()),
                         a.bytes.size())))
        << "buffer id=" << a.id;
    // FindBuffer resolves by id.
    const auto* found = out.FindBuffer(a.id);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->pool_offset, a.pool_offset);
  }

  // Subgraphs: topology, device, const_map, and aliased payloads match.
  ASSERT_EQ(out.subgraphs.size(), in.subgraphs.size());
  for (const auto& [name, in_sg] : in.subgraphs) {
    ASSERT_TRUE(out.subgraphs.count(name));
    const auto& out_sg = out.subgraphs.at(name);
    EXPECT_EQ(out_sg.name, in_sg.name);
    EXPECT_EQ(out_sg.device, in_sg.device);
    EXPECT_EQ(out_sg.const_map, in_sg.const_map);
    const std::string got(reinterpret_cast<const char*>(out_sg.payload.data()),
                          out_sg.payload.size());
    const std::string want(reinterpret_cast<const char*>(in_sg.payload.data()),
                           in_sg.payload.size());
    EXPECT_EQ(got, want) << "subgraph " << name;
  }
}

// BankBytes sums the deduplicated buffer pool.
TEST(GlobalGraphTest, BankBytesSumsPool) {
  Sample sample;
  EXPECT_EQ(sample.graph.BankBytes(), 4u + 10u);
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
  std::string blob = Sample().graph.Serialize();
  // The version is a little-endian uint16 immediately after the 8-byte magic.
  ASSERT_GT(blob.size(), 9u);
  blob[8] = static_cast<char>(OpenVinoGlobalGraph::kVersion + 1);
  auto parsed = OpenVinoGlobalGraph::Parse(
      reinterpret_cast<const uint8_t*>(blob.data()), blob.size());
  EXPECT_FALSE(parsed.HasValue());
}

// Parse errors on a truncated (mid-buffer) blob rather than over-reading.
TEST(GlobalGraphTest, ParseRejectsTruncated) {
  const std::string blob = Sample().graph.Serialize();
  for (size_t cut : {blob.size() / 2, blob.size() - 1}) {
    auto parsed = OpenVinoGlobalGraph::Parse(
        reinterpret_cast<const uint8_t*>(blob.data()), cut);
    EXPECT_FALSE(parsed.HasValue()) << "cut=" << cut;
  }
}

}  // namespace
}  // namespace openvino
}  // namespace litert
