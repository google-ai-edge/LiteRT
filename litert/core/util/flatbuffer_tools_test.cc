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

#include "litert/core/util/flatbuffer_tools.h"

#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT
#include <fstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_macros.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Lt;

FlatbufferWrapper::Ptr TestFlatbuffer(
    absl::string_view filename = "one_mul.tflite") {
  const auto tfl_path = testing::GetTestFilePath(filename);
  LITERT_ASSIGN_OR_ABORT(auto ptr,
                         FlatbufferWrapper::CreateFromTflFile(tfl_path));
  return ptr;
}

static const absl::string_view kKey = "MyKey";
static const absl::string_view kData = "MyData";

TEST(FlatbufferToolsTest, Metadata) {
  auto flatbuffer = TestFlatbuffer();
  ASSERT_NE(flatbuffer, nullptr);
  auto tfl_model = flatbuffer->Unpack();

  LITERT_ASSERT_OK(PushMetadata(
      kKey, *tfl_model, BufferRef<uint8_t>(kData.data(), kData.size())));

  auto metadata = GetMetadata(kKey, *tfl_model);
  ASSERT_TRUE(metadata);
  EXPECT_EQ(metadata->StrView(), kData);
}

TEST(FlatbufferToolsTest, GetMetadataNotFound) {
  auto flatbuffer = TestFlatbuffer();
  auto tfl_model = flatbuffer->Unpack();
  ASSERT_NE(flatbuffer, nullptr);
  EXPECT_FALSE(GetMetadata(kKey, *tfl_model));
}

TEST(FlatbufferToolsTest, TflBuffer) {
  auto flatbuffer = TestFlatbuffer();
  ASSERT_NE(flatbuffer, nullptr);
  auto tfl_model = flatbuffer->Unpack();

  auto ind = PushTflBuffer((*tfl_model),
                           BufferRef<uint8_t>(kData.data(), kData.size()));
  ASSERT_TRUE(ind);

  auto buf = GetTflBuffer((*tfl_model), *ind);
  ASSERT_TRUE(buf);
  ASSERT_EQ(buf->StrView(), kData);
}

TEST(FlatbufferToolsTest, GetTflBufferNotFound) {
  auto flatbuffer = TestFlatbuffer();
  ASSERT_NE(flatbuffer, nullptr);
  auto tfl_model = flatbuffer->Unpack();

  auto buf = GetTflBuffer((*tfl_model), 100);
  ASSERT_FALSE(buf);
}

TEST(FlatbufferToolsTest, GetTflOpCode) {
  auto flatbuffer = TestFlatbuffer();
  ASSERT_NE(flatbuffer, nullptr);
  auto tfl_model = flatbuffer->Unpack();

  auto op_code = GetTflOpCode((*tfl_model), 0);
  ASSERT_TRUE(op_code);
}

TEST(FlatbufferToolsTest, GetTflOpCodeNotFound) {
  auto flatbuffer = TestFlatbuffer();
  ASSERT_NE(flatbuffer, nullptr);
  auto tfl_model = flatbuffer->Unpack();

  auto op_code = GetTflOpCode((*tfl_model), 100);
  ASSERT_FALSE(op_code);
}

TEST(FlatbufferToolsTest, StaticTensorTypeTest) {
  auto flatbuffer = TestFlatbuffer();
  auto tfl_model = flatbuffer->Unpack();
  auto& tensor = tfl_model->subgraphs.front()->tensors.front();

  TflShapeInfo shape(*tensor);

  ASSERT_TRUE(IsRankedTensorType(shape));
  ASSERT_TRUE(IsStaticTensorType(shape));

  auto static_shape = AsStaticShape(shape);

  ASSERT_TRUE(static_shape);
  ASSERT_THAT(*static_shape, ElementsAreArray({2, 2}));
}

TEST(FlatbufferToolsTest, UnrankedTensorTypeTest) {
  auto flatbuffer = TestFlatbuffer("unranked_tensor.tflite");
  auto tfl_model = flatbuffer->Unpack();
  auto& tensor = tfl_model->subgraphs.front()->tensors.front();

  TflShapeInfo shape(*tensor);

  ASSERT_FALSE(IsRankedTensorType(shape));
}

TEST(FlatbufferToolsTest, RankedDynamicTensorTypeTest) {
  auto flatbuffer = TestFlatbuffer("dynamic_shape_tensor.tflite");
  auto tfl_model = flatbuffer->Unpack();
  auto& tensor = tfl_model->subgraphs.front()->tensors.front();

  TflShapeInfo shape(*tensor);

  ASSERT_TRUE(IsRankedTensorType(shape));
  ASSERT_FALSE(IsStaticTensorType(shape));

  auto dyn_shape = AsDynamicShape(shape);

  ASSERT_TRUE(dyn_shape);
  ASSERT_THAT(*dyn_shape, ElementsAre(Lt(0), 2));
}

TEST(FlatbufferToolsTest, PerTensorQuantizedTest) {
  auto flatbuffer =
      TestFlatbuffer("single_add_default_a16w8_recipe_quantized.tflite");
  auto tfl_model = flatbuffer->Unpack();
  auto& tensor = tfl_model->subgraphs.front()->tensors.front();

  const auto* const q_parms = tensor->quantization.get();

  ASSERT_TRUE(IsQuantized(q_parms));
  EXPECT_TRUE(IsPerTensorQuantized(q_parms));

  auto per_tensor = AsPerTensorQparams(q_parms);
  ASSERT_TRUE(per_tensor);
}

TEST(FlatbufferToolsTest, PerChannelQuantizedTest) {
  auto flatbuffer = TestFlatbuffer("static_w8_a16_quantized_k_einsum.tflite");
  auto tfl_model = flatbuffer->Unpack();
  auto& tensor = tfl_model->subgraphs.front()->tensors[1];

  const auto* const q_parms = tensor->quantization.get();

  ASSERT_TRUE(IsQuantized(q_parms));
  EXPECT_TRUE(IsPerChannelQuantized(q_parms));

  auto per_channel = AsPerChannelQparams(q_parms);
  ASSERT_TRUE(per_channel);
}

TEST(FlatbufferToolsTest, CopyModelMetadataFromFileReadsRootOnly) {
  static constexpr absl::string_view kMetadataKey = "DispatchManifest";
  static constexpr absl::string_view kMetadataData = "manifest_payload";
  static constexpr absl::string_view kTrailingPayload = "TRAILING_BYTECODE";

  auto flatbuffer = TestFlatbuffer();
  ASSERT_NE(flatbuffer, nullptr);
  auto model = flatbuffer->Unpack();
  ASSERT_EQ(PushMetadata(
                kMetadataKey, *model,
                BufferRef<uint8_t>(kMetadataData.data(), kMetadataData.size())),
            kLiteRtStatusOk);
  auto serialized = SerializeFlatbuffer(*model);

  const size_t root_size = serialized.Size();
  OwningBufferRef<uint8_t> model_with_trailing(root_size +
                                               kTrailingPayload.size());
  std::memcpy(model_with_trailing.Data(), serialized.Data(), root_size);
  std::memcpy(model_with_trailing.Data() + root_size, kTrailingPayload.data(),
              kTrailingPayload.size());

  const std::filesystem::path path =
      std::filesystem::path(::testing::TempDir()) /
      "metadata_only_copy_with_trailing.tflite";
  std::ofstream ofs(path, std::ios::binary);
  ASSERT_TRUE(ofs.good());
  ofs.write(reinterpret_cast<const char*>(model_with_trailing.Data()),
            model_with_trailing.Size());
  ofs.close();

  LITERT_ASSERT_OK_AND_ASSIGN(auto computed_root_size,
                              GetFlatbufferRootSizeFromFile(path.string()));
  EXPECT_EQ(computed_root_size, root_size);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto metadata_copy,
      CopyModelMetadataFromFile(path.string(), kMetadataKey));
  EXPECT_EQ(metadata_copy.StrView(), kMetadataData);
}

TEST(FlatbufferToolsTest, CopyModelMetadataFromFileWithTruncatedModelFails) {
  auto flatbuffer = TestFlatbuffer();
  ASSERT_NE(flatbuffer, nullptr);
  auto serialized = SerializeFlatbuffer(*flatbuffer);
  ASSERT_GT(serialized.Size(), 32);

  const std::filesystem::path path =
      std::filesystem::path(::testing::TempDir()) / "truncated_model.tflite";
  std::ofstream ofs(path, std::ios::binary);
  ASSERT_TRUE(ofs.good());
  ofs.write(reinterpret_cast<const char*>(serialized.Data()), 32);
  ofs.close();

  EXPECT_FALSE(CopyModelMetadataFromFile(path.string(), kKey));
}

TEST(FlatbufferToolsTest,
     MetadataOnlyFileLoadFailsWithExternalTensorBufferRanges) {
  auto flatbuffer = TestFlatbuffer("add_cst.tflite");
  ASSERT_NE(flatbuffer, nullptr);
  auto tfl_model = flatbuffer->Unpack();

  bool patched_external_range = false;
  for (auto& subgraph : tfl_model->subgraphs) {
    if (subgraph == nullptr) {
      continue;
    }
    for (auto& tensor : subgraph->tensors) {
      if (tensor == nullptr) {
        continue;
      }
      const auto buffer_index = tensor->buffer;
      if (buffer_index == 0 || buffer_index >= tfl_model->buffers.size()) {
        continue;
      }
      auto* buffer = tfl_model->buffers[buffer_index].get();
      if (buffer == nullptr) {
        continue;
      }
      buffer->offset = 16;
      buffer->size = 4;
      patched_external_range = true;
      break;
    }
    if (patched_external_range) {
      break;
    }
  }
  ASSERT_TRUE(patched_external_range);

  auto serialized = SerializeFlatbuffer(*tfl_model);
  const std::filesystem::path path =
      std::filesystem::path(::testing::TempDir()) /
      "metadata_only_external_ranges.tflite";
  std::ofstream ofs(path, std::ios::binary);
  ASSERT_TRUE(ofs.good());
  ofs.write(reinterpret_cast<const char*>(serialized.Data()),
            serialized.Size());
  ofs.close();

  FlatbufferWrapper::FileLoadOptions options;
  options.load_mode = FlatbufferWrapper::FileLoadMode::kMetadataOnlyForFileCopy;
  auto model_with_metadata_only =
      FlatbufferWrapper::CreateFromTflFile(path.string(), options);
  ASSERT_FALSE(model_with_metadata_only);
  EXPECT_EQ(model_with_metadata_only.Error().Status(),
            kLiteRtStatusErrorInvalidArgument);
}

}  // namespace
}  // namespace litert::internal
