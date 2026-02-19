// Copyright 2025 Google LLC.
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

#include "weight_loader/external_weight_loader_litert.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers  // IWYU pragma: keep
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/internal/scoped_file.h"
#include "litert/cc/internal/scoped_weight_source.h"
#include "tflite/schema/schema_generated.h"

namespace weight_loader {
namespace {

using litert::ScopedFile;
using litert::ScopedWeightSection;
using litert::ScopedWeightSource;

constexpr uint32_t kExternalBufferId = 1;
constexpr uint32_t kGroupId = 1;

// Vector tensor for testing.
constexpr uint32_t kTensorElementCount = 8;
constexpr uint64_t kSliceOffset = 0;
constexpr uint64_t kSliceLengthBytes = kTensorElementCount;

struct ModelBuffer {
  std::vector<uint8_t> data;

  const tflite::Model* model() const { return tflite::GetModel(data.data()); }
};

// Builds a TfLite model. The model contains a single subgraph with one UINT8
// tensor. This tensor is linked to an external buffer with `kExternalBufferId`
// and belongs to an external buffer group identified by `group_name` and
// `kGroupId`.
ModelBuffer BuildModel(absl::string_view group_name) {
  flatbuffers::FlatBufferBuilder builder;

  const std::vector<int32_t> tensor_shape = {
      static_cast<int32_t>(kTensorElementCount)};
  auto tensor = tflite::CreateTensor(
      builder, builder.CreateVector(tensor_shape), tflite::TensorType_UINT8,
      /*buffer=*/0, builder.CreateString("external_tensor"),
      /*quantization=*/0,
      /*is_variable=*/false,
      /*sparsity=*/0,
      /*shape_signature=*/0,
      /*has_rank=*/false,
      /*variant_tensors=*/0, kExternalBufferId);

  auto tensors_vec = builder.CreateVector(
      std::vector<flatbuffers::Offset<tflite::Tensor>>{tensor});
  auto empty_int_vec = builder.CreateVector<int32_t>(std::vector<int32_t>{});
  auto empty_op_vec = builder.CreateVector(
      std::vector<flatbuffers::Offset<tflite::Operator>>{});

  auto subgraph =
      tflite::CreateSubGraph(builder, tensors_vec, empty_int_vec, empty_int_vec,
                             empty_op_vec, builder.CreateString("main"));
  auto subgraphs_vec = builder.CreateVector(
      std::vector<flatbuffers::Offset<tflite::SubGraph>>{subgraph});

  auto buffer = tflite::CreateBuffer(builder);
  auto buffers_vec = builder.CreateVector(
      std::vector<flatbuffers::Offset<tflite::Buffer>>{buffer});

  // Group id 0 is reserved by the runtime; insert a dummy entry at slot 0 so
  // the real external buffer (group 1) resolves correctly during parsing.
  auto placeholder_group = tflite::CreateExternalBufferGroupDirect(builder, "");
  auto group = tflite::CreateExternalBufferGroupDirect(
      builder, std::string(group_name).c_str());
  auto groups_vec = builder.CreateVector(
      std::vector<flatbuffers::Offset<tflite::ExternalBufferGroup>>{
          placeholder_group, group});

  auto ext_buffer = tflite::CreateExternalBuffer(
      builder, kExternalBufferId, kGroupId, kSliceOffset, kSliceLengthBytes,
      builder.CreateString(""));
  auto ext_buffers_vec = builder.CreateVector(
      std::vector<flatbuffers::Offset<tflite::ExternalBuffer>>{ext_buffer});

  auto model =
      tflite::CreateModel(builder, /*version=*/3,
                          /*operator_codes=*/0, subgraphs_vec,
                          builder.CreateString("test_model"), buffers_vec,
                          /*metadata_buffer=*/0, /*metadata=*/0,
                          /*signature_defs=*/0, groups_vec, ext_buffers_vec);
  tflite::FinishModelBuffer(builder, model);

  ModelBuffer result;
  result.data.assign(builder.GetBufferPointer(),
                     builder.GetBufferPointer() + builder.GetSize());
  return result;
}

std::string WriteWeightsFile(absl::string_view filename,
                             std::string_view payload) {
  std::string path =
      std::string(::testing::TempDir()) + "/" + std::string(filename);
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  EXPECT_TRUE(file.is_open());
  file.write(payload.data(), payload.size());
  file.close();
  return path;
}

std::vector<uint8_t> ExpectedSlice(std::string_view payload) {
  return std::vector<uint8_t>(
      payload.begin() + kSliceOffset,
      payload.begin() + kSliceOffset + kSliceLengthBytes);
}

void ExpectHostBufferEquals(const WeightAccess* access,
                            absl::Span<const uint8_t> expected) {
  ASSERT_NE(access, nullptr);
  const auto host_buffer = access->GetHostBuffer();
  void* host_mem_addr;
  ASSERT_EQ(LiteRtLockTensorBuffer(host_buffer, &host_mem_addr,
                                   kLiteRtTensorBufferLockModeRead),
            kLiteRtStatusOk);
  auto actual = absl::MakeSpan(static_cast<const uint8_t*>(host_mem_addr),
                               expected.size());
  EXPECT_EQ(actual, expected);
}

const WeightInfo& GetSingleWeightInfo(const WeightLoader& loader) {
  auto infos = loader.GetWeightInfo();
  EXPECT_EQ(infos.size(), 1);
  return infos[0];
}

void ExpectWeightInfo(const WeightInfo& info) {
  EXPECT_EQ(info.external_buffer_id, kExternalBufferId);
  EXPECT_EQ(info.packing, "");
}

void ExpectHostBufferMetadata(const WeightAccess* access) {
  ASSERT_NE(access, nullptr);
  LiteRtTensorBuffer host_buffer = access->GetHostBuffer();

  LiteRtRankedTensorType tensor_type;
  ASSERT_EQ(LiteRtGetTensorBufferTensorType(host_buffer, &tensor_type),
            kLiteRtStatusOk);

  ASSERT_EQ(tensor_type.element_type, kLiteRtElementTypeUInt8);

  LiteRtLayout layout = tensor_type.layout;
  ASSERT_EQ(layout.rank, 1);
  ASSERT_EQ(layout.dimensions[0], kTensorElementCount);

  size_t packed_size;
  ASSERT_EQ(LiteRtGetTensorBufferPackedSize(host_buffer, &packed_size),
            kLiteRtStatusOk);
  EXPECT_EQ(packed_size, kSliceLengthBytes);
}

TEST(ExternalWeightLoaderTest, LoadsWeightsFromFilesystemPath) {
  constexpr absl::string_view kGroupName = "weights.bin";
  const std::string payload = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ4545454545";
  auto model = BuildModel(kGroupName);
  WriteWeightsFile(kGroupName, payload);

  auto loader = CreateLiteRtWeightLoader(
      model.model(),
      /*model_directory=*/std::string(::testing::TempDir()),
      /*scoped_weight_source=*/nullptr);
  ASSERT_NE(loader, nullptr);
  const auto& weight_info = GetSingleWeightInfo(*loader);
  ExpectWeightInfo(weight_info);
  WeightAccessRequest request;
  request.cpu = true;
  absl::Status status = loader->PrepareAccess(request, /*env=*/nullptr);
  ASSERT_TRUE(status.ok()) << status.message();

  const auto* access =
      loader->GetExternalWeightByBuffer(weight_info.external_buffer_id);
  ExpectHostBufferMetadata(access);
  auto expected = ExpectedSlice(payload);
  ExpectHostBufferEquals(access, expected);
}

TEST(ExternalWeightLoaderTest, LoadsWeightsFromScopedFile) {
  constexpr absl::string_view kGroupName = "scoped_group";
  const std::string payload = "ZYXWVUTSRQPONMLKJIHGFEDCBA987654321045454545";
  auto model = BuildModel(kGroupName);
  const std::string weights_path = WriteWeightsFile("scoped.bin", payload);

  auto scoped_file_or = ScopedFile::Open(weights_path);
  ASSERT_TRUE(scoped_file_or.ok());

  absl::flat_hash_map<std::string, ScopedWeightSection> sections;
  sections.emplace(
      std::string(kGroupName),
      ScopedWeightSection{.offset = 0,
                          .length = static_cast<uint64_t>(payload.size())});
  auto scoped_source = std::make_unique<ScopedWeightSource>(
      std::move(*scoped_file_or), std::move(sections));

  auto loader = CreateLiteRtWeightLoader(model.model(),
                                         /*model_directory=*/std::nullopt,
                                         std::move(scoped_source));
  ASSERT_NE(loader, nullptr);
  const auto& weight_info = GetSingleWeightInfo(*loader);
  ExpectWeightInfo(weight_info);
  WeightAccessRequest request;
  request.cpu = true;
  absl::Status status = loader->PrepareAccess(request, /*env=*/nullptr);
  ASSERT_TRUE(status.ok()) << status.message();

  const auto* access =
      loader->GetExternalWeightByBuffer(weight_info.external_buffer_id);
  ExpectHostBufferMetadata(access);
  auto expected = ExpectedSlice(payload);
  ExpectHostBufferEquals(access, expected);
}

}  // namespace
}  // namespace weight_loader
