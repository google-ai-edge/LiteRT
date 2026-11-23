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

#include "litert/vendors/nvidia/tools/carve_container.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "litert/cc/litert_buffer_ref.h"
#include "litert/vendors/nvidia/tools/carve_model.h"
#include "litert/vendors/nvidia/tools/carve_test_util.h"

namespace litert::tools {
namespace {

using testing::AsBuffer;
using ::testing::HasSubstr;
using ::testing::SizeIs;

constexpr uint64_t kBlock = 16 * 1024;

std::string Hash(const std::string& bytes) {
  Sha256 hash;
  hash.Update(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
  return hash.HexDigest();
}

std::string ReadFile(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  std::stringstream bytes;
  bytes << stream.rdbuf();
  return bytes.str();
}

struct TestSection {
  uint8_t data_type = 0;
  std::string model_type;
  std::string bytes;
};

// A container as LiteRT-LM's builder writes it. The header is built with the
// low-level flatbuffer API by the field numbers of
// litertlm_header_schema.fbs, the way the tool reads it.
std::string BuildContainer(const std::vector<TestSection>& sections) {
  using Table = flatbuffers::Offset<flatbuffers::Table>;
  auto header = [&](const std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
    flatbuffers::FlatBufferBuilder builder;
    std::vector<Table> objects;
    for (size_t i = 0; i < sections.size(); ++i) {
      std::vector<Table> items;
      auto item = [&](const char* key, const std::string& value) {
        const auto key_offset = builder.CreateString(key);
        const auto text = builder.CreateString(value);
        auto start = builder.StartTable();
        builder.AddOffset(4, text);
        const Table string_value(builder.EndTable(start));
        start = builder.StartTable();
        builder.AddOffset(4, key_offset);
        builder.AddElement<uint8_t>(6, 9, 0);
        builder.AddOffset(8, string_value);
        items.push_back(Table(builder.EndTable(start)));
      };
      if (!sections[i].model_type.empty()) {
        item("model_type", sections[i].model_type);
      }
      item("sha256", Hash(sections[i].bytes));
      const auto items_vector = builder.CreateVector(items);
      const auto start = builder.StartTable();
      builder.AddOffset(4, items_vector);
      builder.AddElement<uint64_t>(6, ranges[i].first, 0);
      builder.AddElement<uint64_t>(8, ranges[i].second, 0);
      builder.AddElement<uint8_t>(10, sections[i].data_type, 0);
      objects.push_back(Table(builder.EndTable(start)));
    }
    const auto objects_vector = builder.CreateVector(objects);
    auto start = builder.StartTable();
    builder.AddOffset(4, objects_vector);
    const Table section_metadata(builder.EndTable(start));
    start = builder.StartTable();
    builder.AddOffset(6, section_metadata);
    builder.Finish(Table(builder.EndTable(start)));
    return std::string(
        reinterpret_cast<const char*>(builder.GetBufferPointer()),
        builder.GetSize());
  };

  // The offsets are fixed-width fields: a header with any offsets has the
  // size of the final one.
  std::vector<std::pair<uint64_t, uint64_t>> ranges(sections.size(), {1, 1});
  const uint64_t header_end = 32 + header(ranges).size();
  uint64_t position = header_end;
  for (size_t i = 0; i < sections.size(); ++i) {
    position = (position + kBlock - 1) / kBlock * kBlock;
    ranges[i] = {position, position + sections[i].bytes.size()};
    position = ranges[i].second;
  }

  std::string file = "LITERTLM";
  const uint32_t version[] = {1, 9, 0, 0};
  file.append(reinterpret_cast<const char*>(version), sizeof(version));
  file.append(reinterpret_cast<const char*>(&header_end), sizeof(header_end));
  file += header(ranges);
  for (size_t i = 0; i < sections.size(); ++i) {
    file.resize(ranges[i].first);
    file += sections[i].bytes;
  }
  return file;
}

std::string Filled(size_t size, char seed) {
  std::string bytes(size, 0);
  for (size_t i = 0; i < size; ++i) bytes[i] = static_cast<char>(seed + i * 7);
  return bytes;
}

std::vector<TestSection> TestSections() {
  testing::FactorizedMlpModelOptions options;
  options.appended_buffers = true;
  return {
      {5, "", Filled(109, 1)},
      {3, "tf_lite_embedder", Filled(kBlock + 5, 2)},
      {3, "tf_lite_prefill_decode", testing::BuildFactorizedMlpModel(options)},
      {7, "tf_lite_prefill_decode", Filled(3 * kBlock, 3)},
      {4, "", Filled(1000, 4)},
  };
}

std::string SectionBytes(const std::string& file,
                         const ContainerSection& section) {
  return file.substr(section.begin, section.end - section.begin);
}

TEST(Sha256Test, MatchesTheStandardVectors) {
  EXPECT_EQ(Hash(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(Hash("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(Hash("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  EXPECT_EQ(Hash(std::string(1000000, 'a')),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(Sha256Test, TakesTheMessageInPieces) {
  const std::string message = Filled(1000, 9);
  for (const size_t piece : {1, 55, 56, 63, 64, 65, 119, 128, 999}) {
    Sha256 hash;
    for (size_t i = 0; i < message.size(); i += piece) {
      hash.Update(reinterpret_cast<const uint8_t*>(message.data()) + i,
                  std::min(piece, message.size() - i));
    }
    EXPECT_EQ(hash.HexDigest(), Hash(message)) << piece;
  }
}

TEST(CarveContainerTest, ReadsTheSections) {
  const auto expected = TestSections();
  const std::string file = BuildContainer(expected);
  ASSERT_TRUE(IsContainer(AsBuffer(file)));
  const auto sections = ParseContainer(AsBuffer(file));
  ASSERT_TRUE(sections.ok()) << sections.status();
  ASSERT_THAT(*sections, SizeIs(expected.size()));
  for (size_t i = 0; i < expected.size(); ++i) {
    const auto& section = (*sections)[i];
    EXPECT_EQ(section.data_type, expected[i].data_type);
    EXPECT_EQ(section.model_type, expected[i].model_type);
    EXPECT_EQ(section.begin % kBlock, 0);
    EXPECT_EQ(SectionBytes(file, section), expected[i].bytes);
    // The fields are where the header holds the numbers and the checksum.
    uint64_t begin;
    std::memcpy(&begin, file.data() + section.begin_field, sizeof(begin));
    EXPECT_EQ(begin, section.begin);
    uint64_t end;
    std::memcpy(&end, file.data() + section.end_field, sizeof(end));
    EXPECT_EQ(end, section.end);
    EXPECT_EQ(file.substr(section.sha256_field, 64), Hash(expected[i].bytes));
  }
}

TEST(CarveContainerTest, RejectsWhatIsNotAContainer) {
  const std::string model =
      testing::BuildFactorizedMlpModel(testing::FactorizedMlpModelOptions());
  EXPECT_FALSE(IsContainer(AsBuffer(model)));
  EXPECT_FALSE(ParseContainer(AsBuffer(model)).ok());

  std::string file = BuildContainer(TestSections());
  // A section that ends behind the end of the file.
  const std::string cut = file.substr(0, file.size() - 1);
  EXPECT_THAT(ParseContainer(AsBuffer(cut)).status().message(),
              HasSubstr("not in file order inside the file"));
  // A header that ends behind the end of the file.
  EXPECT_THAT(ParseContainer(AsBuffer(file.substr(0, 40))).status().message(),
              HasSubstr("header is cut off"));
  // Another major version.
  file[8] = 2;
  EXPECT_THAT(ParseContainer(AsBuffer(file)).status().message(),
              HasSubstr("version 2"));
}

TEST(CarveContainerTest, ReplacesASectionAndKeepsTheRest) {
  const auto contents = TestSections();
  const std::string file = BuildContainer(contents);
  const auto sections = ParseContainer(AsBuffer(file));
  ASSERT_TRUE(sections.ok()) << sections.status();

  // The model of section 2, carved.
  const auto patterns = ParseCarvePatterns(BuiltinCarvePatternText());
  ASSERT_TRUE(patterns.ok());
  CarveReport report;
  auto carved = CarveModel(AsBuffer(contents[2].bytes), *patterns, &report);
  ASSERT_TRUE(carved.ok()) << carved.status();
  ASSERT_TRUE(carved->has_value());
  const std::string carved_bytes = testing::ToBytes(**carved);
  ASSERT_NE(carved_bytes.size(), contents[2].bytes.size());

  std::vector<std::optional<CarvedModel>> replacements(sections->size());
  replacements[2] = **carved;
  const std::string path = ::testing::TempDir() + "/carved.litertlm";
  const auto written =
      WriteContainer(AsBuffer(file), *sections, replacements, path);
  ASSERT_TRUE(written.ok()) << written.status();

  const std::string output = ReadFile(path);
  const auto reread = ParseContainer(AsBuffer(output));
  ASSERT_TRUE(reread.ok()) << reread.status();
  ASSERT_THAT(*reread, SizeIs(sections->size()));
  for (size_t i = 0; i < reread->size(); ++i) {
    const auto& section = (*reread)[i];
    EXPECT_EQ(section.begin, (*written)[i].begin);
    EXPECT_EQ(section.end, (*written)[i].end);
    EXPECT_EQ(section.data_type, contents[i].data_type);
    EXPECT_EQ(section.model_type, contents[i].model_type);
    const std::string& expected = i == 2 ? carved_bytes : contents[i].bytes;
    EXPECT_EQ(SectionBytes(output, section), expected) << i;
    // The checksum is that of the section as it is now.
    EXPECT_EQ(output.substr(section.sha256_field, 64), Hash(expected)) << i;
    // Up to the replaced section nothing moved; behind it the sections
    // follow each other on block boundaries.
    if (i <= 2) {
      EXPECT_EQ(section.begin, (*sections)[i].begin);
    } else {
      EXPECT_EQ(section.begin,
                ((*reread)[i - 1].end + kBlock - 1) / kBlock * kBlock);
    }
  }
  // The header kept its size, and the file ends with its last section.
  EXPECT_EQ(output.substr(0, 24), file.substr(0, 24));
  EXPECT_EQ(output.substr(24, 8), file.substr(24, 8));
  EXPECT_EQ(output.size(), reread->back().end);
}

TEST(CarveContainerTest, MovesNothingWhenTheSizeStays) {
  const auto contents = TestSections();
  const std::string file = BuildContainer(contents);
  const auto sections = ParseContainer(AsBuffer(file));
  ASSERT_TRUE(sections.ok()) << sections.status();

  const std::string replacement = Filled(contents[1].bytes.size(), 77);
  std::vector<std::optional<CarvedModel>> replacements(sections->size());
  replacements[1].emplace();
  replacements[1]->flatbuffer = litert::OwningBufferRef<uint8_t>(
      reinterpret_cast<const uint8_t*>(replacement.data()), replacement.size());
  const std::string path = ::testing::TempDir() + "/same_size.litertlm";
  ASSERT_TRUE(
      WriteContainer(AsBuffer(file), *sections, replacements, path).ok());

  // Only the section and its checksum differ.
  std::string expected = file;
  expected.replace((*sections)[1].begin, replacement.size(), replacement);
  expected.replace((*sections)[1].sha256_field, 64, Hash(replacement));
  EXPECT_EQ(ReadFile(path), expected);
}

TEST(CarveContainerTest, WritesAModelFile) {
  CarvedModel model;
  const std::string flatbuffer = Filled(100, 5);
  const std::string appended = Filled(300, 6);
  model.flatbuffer = litert::OwningBufferRef<uint8_t>(
      reinterpret_cast<const uint8_t*>(flatbuffer.data()), flatbuffer.size());
  model.gap = 28;
  model.appended = AsBuffer(appended);
  const std::string path = ::testing::TempDir() + "/model.tflite";
  ASSERT_TRUE(WriteModel(model, path).ok());
  EXPECT_EQ(ReadFile(path), flatbuffer + std::string(28, 0) + appended);
  EXPECT_EQ(model.Size(), 428);

  const std::string copy = ::testing::TempDir() + "/copy.bin";
  ASSERT_TRUE(WriteFile(AsBuffer(appended), copy).ok());
  EXPECT_EQ(ReadFile(copy), appended);
  EXPECT_FALSE(WriteFile(AsBuffer(appended), "/nonexistent/dir/file").ok());
}

}  // namespace
}  // namespace litert::tools
