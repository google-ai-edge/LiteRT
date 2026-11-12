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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/internal/scoped_file.h"
#include "litert/cc/internal/scoped_weight_source.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/core/options.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "weight_loader/external_weight_loader_litert.h"
#include "tflite/schema/schema_generated.h"

namespace {

using ::testing::ElementsAre;

TEST(JitCompilation, JitBypassesCaching) {
  const auto litert_libs_path =
      litert::testing::GetLiteRtPath("vendors/examples");
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
          litert_libs_path,
      },
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          litert_libs_path,
      },
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kCompilerCacheDir,
          // Use a dummy cache dir to force the caching logic to trigger.
          "/tmp/dummy_cache_dir_for_test",
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, litert::Environment::Create(litert::EnvironmentOptions(
                    absl::MakeConstSpan(environment_options))));

  std::string path = litert::testing::GetTestFilePath("one_mul.tflite");

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      litert::CompiledModel::Create(env, path, litert::HwAccelerators::kNpu));

  // Verify it actually succeeded and loaded.
  auto num_signatures = compiled_model.GetNumSignatures();
  ASSERT_EQ(num_signatures, 1);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffers,
      compiled_model.CreateInputBuffers(/*signature_index=*/0));
  EXPECT_EQ(input_buffers.size(), 2);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto output_buffers,
      compiled_model.CreateOutputBuffers(/*signature_index=*/0));
  EXPECT_EQ(output_buffers.size(), 1);

  LITERT_ASSERT_OK(input_buffers[0].Write<float>({2.0f, 2.0f, 2.0f, 2.0f}));
  LITERT_ASSERT_OK(input_buffers[1].Write<float>({1.0f, 2.0f, 3.0f, 4.0f}));

  // Execute model.
  const size_t signature_index = 0;
  compiled_model.Run(/*signature_index=*/signature_index, input_buffers,
                     output_buffers);

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], litert::TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, 4);
    EXPECT_THAT(output, ElementsAre(2.0f, 4.0f, 6.0f, 8.0f));
  }
}

// Helper to construct a 2-op model:
//   tmp = mul(arg0, ext_cst_1) [compiled by example_plugin onto NPU]
//   out = add(tmp, ext_cst_2)  [executed on CPU]
// where ext_cst_1 and ext_cst_2 reference external_buffer 1 and 2 in group 1.
litert::OwningBufferRef<uint8_t> BuildExternalWeightMulAddModel(
    absl::string_view group_name) {
  auto flatbuffer = litert::internal::FlatbufferWrapper::CreateFromTflFile(
      litert::testing::GetTestFilePath("add_cst.tflite"));
  auto tfl_model = flatbuffer->get()->Unpack();

  auto mul_opcode = std::make_unique<litert::internal::TflOpCode>();
  mul_opcode->builtin_code = ::tflite::BuiltinOperator_MUL;
  tfl_model->operator_codes.push_back(std::move(mul_opcode));
  const uint32_t mul_opcode_idx = tfl_model->operator_codes.size() - 1;

  auto& sg = *tfl_model->subgraphs[0];
  const int32_t orig_input_idx = sg.operators[0]->inputs[0];
  const int32_t cst1_idx = sg.operators[0]->inputs[1];
  const int32_t orig_output_idx = sg.operators[0]->outputs[0];

  tfl_model->buffers[sg.tensors[cst1_idx]->buffer]->data.clear();
  sg.tensors[cst1_idx]->buffer = 0;
  sg.tensors[cst1_idx]->external_buffer = 1;

  auto tmp_tensor = std::make_unique<litert::internal::TflTensor>(
      *sg.tensors[orig_output_idx]);
  tmp_tensor->name = "mul_out";
  tmp_tensor->buffer = 0;
  tmp_tensor->external_buffer = 0;
  sg.tensors.push_back(std::move(tmp_tensor));
  const int32_t tmp_tensor_idx = sg.tensors.size() - 1;

  auto cst2_tensor =
      std::make_unique<litert::internal::TflTensor>(*sg.tensors[cst1_idx]);
  cst2_tensor->name = "ext_cst_2";
  cst2_tensor->buffer = 0;
  cst2_tensor->external_buffer = 2;
  sg.tensors.push_back(std::move(cst2_tensor));
  const int32_t cst2_idx = sg.tensors.size() - 1;

  sg.operators[0]->opcode_index = mul_opcode_idx;
  sg.operators[0]->inputs = {orig_input_idx, cst1_idx};
  sg.operators[0]->outputs = {tmp_tensor_idx};

  auto add_op = std::make_unique<litert::internal::TflOp>();
  add_op->opcode_index = 0;  // ADD
  add_op->inputs = {tmp_tensor_idx, cst2_idx};
  add_op->outputs = {orig_output_idx};
  sg.operators.push_back(std::move(add_op));

  auto group0 = std::make_unique<tflite::ExternalBufferGroupT>();
  group0->name = "";
  auto group1 = std::make_unique<tflite::ExternalBufferGroupT>();
  group1->name = std::string(group_name);
  tfl_model->external_buffer_groups.push_back(std::move(group0));
  tfl_model->external_buffer_groups.push_back(std::move(group1));

  // Offset of each external buffer in the file must be aligned to
  // LITERT_HOST_MEMORY_BUFFER_ALIGNMENT (64 bytes) for zero-copy mmap host
  // tensor buffers.
  constexpr uint64_t kExtBuf2Offset = 64;
  auto ext_buf1 = std::make_unique<tflite::ExternalBufferT>();
  ext_buf1->id = 1;
  ext_buf1->group = 1;
  ext_buf1->offset = 0;
  ext_buf1->length = 4 * sizeof(float);
  auto ext_buf2 = std::make_unique<tflite::ExternalBufferT>();
  ext_buf2->id = 2;
  ext_buf2->group = 1;
  ext_buf2->offset = kExtBuf2Offset;
  ext_buf2->length = 4 * sizeof(float);
  tfl_model->external_buffers.push_back(std::move(ext_buf1));
  tfl_model->external_buffers.push_back(std::move(ext_buf2));

  return litert::internal::SerializeFlatbuffer(*tfl_model);
}

std::vector<uint8_t> BuildAlignedExternalWeightPayload() {
  std::vector<uint8_t> payload(128, 0);
  constexpr std::array<float, 4> kExtWeights1 = {2.0f, 3.0f, 4.0f, 5.0f};
  constexpr std::array<float, 4> kExtWeights2 = {10.0f, 20.0f, 30.0f, 40.0f};
  const auto* weights1_bytes =
      reinterpret_cast<const uint8_t*>(kExtWeights1.data());
  const auto* weights2_bytes =
      reinterpret_cast<const uint8_t*>(kExtWeights2.data());
  std::copy(weights1_bytes, weights1_bytes + sizeof(kExtWeights1),
            payload.begin());
  std::copy(weights2_bytes, weights2_bytes + sizeof(kExtWeights2),
            payload.begin() + 64);
  return payload;
}

TEST(JitCompilation, ExternalWeightsFromFileNpuAndCpuEndToEnd) {
  const auto litert_libs_path =
      litert::testing::GetLiteRtPath("vendors/examples");
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
          litert_libs_path,
      },
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          litert_libs_path,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, litert::Environment::Create(litert::EnvironmentOptions(
                    absl::MakeConstSpan(environment_options))));

  auto temp_dir = litert::testing::UniqueTestDirectory::Create();
  ASSERT_TRUE(temp_dir);
  std::filesystem::path temp_path(std::string(temp_dir->Str()));

  auto serialized = BuildExternalWeightMulAddModel("ext_weights.bin");
  std::filesystem::path model_path = temp_path / "ext_mul_add.tflite";
  {
    std::ofstream model_file(model_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(model_file.is_open());
    model_file.write(serialized.StrData(), serialized.Size());
  }

  const auto payload = BuildAlignedExternalWeightPayload();
  std::filesystem::path weights_path = temp_path / "ext_weights.bin";
  {
    std::ofstream weights_file(weights_path,
                               std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(weights_file.is_open());
    weights_file.write(reinterpret_cast<const char*>(payload.data()),
                       payload.size());
  }

  LITERT_ASSERT_OK_AND_ASSIGN(auto options, litert::Options::Create());
  LITERT_ASSERT_OK(options.SetHardwareAccelerators(
      litert::HwAccelerators::kNpu | litert::HwAccelerators::kCpu));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      litert::CompiledModel::Create(env, model_path.string(), options));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffers,
      compiled_model.CreateInputBuffers(/*signature_index=*/0));
  ASSERT_EQ(input_buffers.size(), 1u);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto output_buffers,
      compiled_model.CreateOutputBuffers(/*signature_index=*/0));
  ASSERT_EQ(output_buffers.size(), 1u);

  LITERT_ASSERT_OK(input_buffers[0].Write<float>({1.0f, 2.0f, 3.0f, 4.0f}));
  LITERT_ASSERT_OK(compiled_model.Run(input_buffers, output_buffers));

  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], litert::TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, 4);
    // arg0 * [2, 3, 4, 5] + [10, 20, 30, 40] = [12, 26, 42, 60]
    EXPECT_THAT(output, ElementsAre(12.0f, 26.0f, 42.0f, 60.0f));
  }
}

TEST(JitCompilation, ExternalWeightsFromMemoryBufferAndScopedWeightSource) {
  const auto litert_libs_path =
      litert::testing::GetLiteRtPath("vendors/examples");
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
          litert_libs_path,
      },
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          litert_libs_path,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, litert::Environment::Create(litert::EnvironmentOptions(
                    absl::MakeConstSpan(environment_options))));

  auto temp_dir = litert::testing::UniqueTestDirectory::Create();
  ASSERT_TRUE(temp_dir);
  std::filesystem::path temp_path(std::string(temp_dir->Str()));

  // Load the model from an in-memory buffer so LoadModelFromFile does not
  // resolve external weights from disk; instead, ScopedWeightSource in
  // LiteRtOptionsT must resolve them for both JIT NPU compilation and CPU
  // fallback execution.
  auto serialized = BuildExternalWeightMulAddModel("scoped_group");

  const auto payload = BuildAlignedExternalWeightPayload();
  std::filesystem::path weights_path = temp_path / "scoped_weights.bin";
  {
    std::ofstream weights_file(weights_path,
                               std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(weights_file.is_open());
    weights_file.write(reinterpret_cast<const char*>(payload.data()),
                       payload.size());
  }

  LITERT_ASSERT_OK_AND_ASSIGN(auto scoped_file,
                              litert::ScopedFile::Open(weights_path.string()));

  LITERT_ASSERT_OK_AND_ASSIGN(auto options, litert::Options::Create());
  LITERT_ASSERT_OK(options.SetHardwareAccelerators(
      litert::HwAccelerators::kNpu | litert::HwAccelerators::kCpu));

  litert::Options::ScopedWeightSectionMap sections;
  sections.emplace(
      "scoped_group",
      litert::ScopedWeightSection{
          .offset = 0, .length = static_cast<uint64_t>(payload.size())});
  LITERT_ASSERT_OK(
      options.SetExternalWeightScopedFile(scoped_file, std::move(sections)));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      litert::CompiledModel::Create(env, serialized, options));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffers,
      compiled_model.CreateInputBuffers(/*signature_index=*/0));
  ASSERT_EQ(input_buffers.size(), 1u);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto output_buffers,
      compiled_model.CreateOutputBuffers(/*signature_index=*/0));
  ASSERT_EQ(output_buffers.size(), 1u);

  LITERT_ASSERT_OK(input_buffers[0].Write<float>({1.0f, 2.0f, 3.0f, 4.0f}));
  LITERT_ASSERT_OK(compiled_model.Run(input_buffers, output_buffers));

  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], litert::TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, 4);
    EXPECT_THAT(output, ElementsAre(12.0f, 26.0f, 42.0f, 60.0f));
  }
}

TEST(JitCompilation, ExternalWeightsFromWeightInMemoryMap) {
  const auto litert_libs_path =
      litert::testing::GetLiteRtPath("vendors/examples");
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
          litert_libs_path,
      },
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          litert_libs_path,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, litert::Environment::Create(litert::EnvironmentOptions(
                    absl::MakeConstSpan(environment_options))));

  auto serialized = BuildExternalWeightMulAddModel("in_memory_group");

  alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) std::array<std::byte, 128>
      aligned_payload{};
  const auto raw_payload = BuildAlignedExternalWeightPayload();
  const auto* raw_bytes =
      reinterpret_cast<const std::byte*>(raw_payload.data());
  std::copy(raw_bytes, raw_bytes + raw_payload.size(), aligned_payload.begin());

  absl::flat_hash_map<std::string, absl::Span<const std::byte>> weight_map;
  weight_map.emplace("in_memory_group", absl::MakeConstSpan(aligned_payload));

  LITERT_ASSERT_OK_AND_ASSIGN(auto options, litert::Options::Create());
  LITERT_ASSERT_OK(options.SetHardwareAccelerators(
      litert::HwAccelerators::kNpu | litert::HwAccelerators::kCpu));
  LITERT_ASSERT_OK(options.SetWeightInMemoryMap(&weight_map));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      litert::CompiledModel::Create(env, serialized, options));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffers,
      compiled_model.CreateInputBuffers(/*signature_index=*/0));
  ASSERT_EQ(input_buffers.size(), 1u);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto output_buffers,
      compiled_model.CreateOutputBuffers(/*signature_index=*/0));
  ASSERT_EQ(output_buffers.size(), 1u);

  LITERT_ASSERT_OK(input_buffers[0].Write<float>({1.0f, 2.0f, 3.0f, 4.0f}));
  LITERT_ASSERT_OK(compiled_model.Run(input_buffers, output_buffers));

  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], litert::TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, 4);
    EXPECT_THAT(output, ElementsAre(12.0f, 26.0f, 42.0f, 60.0f));
  }
}

class TrackingWeightLoader : public weight_loader::WeightLoader {
 public:
  TrackingWeightLoader() {
    infos_.push_back(
        weight_loader::WeightInfo{.external_buffer_id = 1, .packing = ""});
    infos_.push_back(
        weight_loader::WeightInfo{.external_buffer_id = 2, .packing = ""});

    const auto raw_payload = BuildAlignedExternalWeightPayload();
    absl::c_copy(raw_payload, aligned_payload_.begin());
  }

  absl::Span<const weight_loader::WeightInfo> GetWeightInfo() const override {
    return absl::MakeConstSpan(infos_);
  }

  const weight_loader::WeightInfo* FindWeightInfoByBuffer(
      uint32_t external_buffer_id) const override {
    for (const auto& info : infos_) {
      if (info.external_buffer_id == external_buffer_id) {
        return &info;
      }
    }
    return nullptr;
  }

  uint32_t GetCanonicalExternalBufferId(
      uint32_t external_buffer_id) const override {
    return external_buffer_id;
  }

  LiteRtStatus PrepareAccess(const weight_loader::WeightAccessRequest& request,
                             LiteRtEnvironmentT* env) override {
    for (const auto& info : infos_) {
      LiteRtStatus status =
          PrepareAccessForBuffer(info.external_buffer_id, request, env);
      if (status != kLiteRtStatusOk) {
        return status;
      }
    }
    return kLiteRtStatusOk;
  }

  LiteRtStatus PrepareAccessForBuffer(
      uint32_t external_buffer_id,
      const weight_loader::WeightAccessRequest& request,
      LiteRtEnvironmentT* env) override {
    if (external_buffer_id != 1 && external_buffer_id != 2) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    if (access_map_.contains(external_buffer_id)) {
      return kLiteRtStatusOk;
    }
    const size_t offset = (external_buffer_id == 1) ? 0 : 64;
    litert::RankedTensorType tensor_type(litert::ElementType::Float32,
                                         litert::Layout(litert::Dimensions{4}));
    auto tb = litert::TensorBuffer::CreateFromHostMemory(
        tensor_type, aligned_payload_.data() + offset, 4 * sizeof(float));
    if (!tb) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    owned_tensor_buffers_[external_buffer_id] = std::move(*tb);
    weight_loader::WeightAccess access;
    access.SetHostBuffer(weight_loader::LiteRtTensorBufferPtr(
        owned_tensor_buffers_[external_buffer_id].Get(),
        weight_loader::LiteRtTensorBufferDeleter{nullptr}));
    access_map_[external_buffer_id] = std::move(access);
    prepared_ids_.insert(external_buffer_id);
    return kLiteRtStatusOk;
  }

  LiteRtStatus SetExternalWeightByBuffer(
      uint32_t external_buffer_id,
      weight_loader::WeightAccess access) override {
    access_map_[external_buffer_id] = std::move(access);
    return kLiteRtStatusOk;
  }

  const weight_loader::WeightAccess* GetExternalWeightByBuffer(
      uint32_t external_buffer_id) const override {
    auto it = access_map_.find(external_buffer_id);
    return it != access_map_.end() ? &it->second : nullptr;
  }

  LiteRtStatus DiscardExternalWeightByBuffer(
      uint32_t external_buffer_id) override {
    return kLiteRtStatusOk;
  }

  LiteRtStatus ReleaseExternalWeightByBuffer(
      uint32_t external_buffer_id) override {
    access_map_.erase(external_buffer_id);
    owned_tensor_buffers_.erase(external_buffer_id);
    released_ids_.insert(external_buffer_id);
    return kLiteRtStatusOk;
  }

  const absl::flat_hash_set<uint32_t>& prepared_ids() const {
    return prepared_ids_;
  }
  const absl::flat_hash_set<uint32_t>& released_ids() const {
    return released_ids_;
  }

 private:
  alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT)
      std::array<uint8_t, 128> aligned_payload_{};
  std::vector<weight_loader::WeightInfo> infos_;
  absl::flat_hash_map<uint32_t, litert::TensorBuffer> owned_tensor_buffers_;
  absl::flat_hash_map<uint32_t, weight_loader::WeightAccess> access_map_;
  absl::flat_hash_set<uint32_t> prepared_ids_;
  absl::flat_hash_set<uint32_t> released_ids_;
};

TEST(JitCompilation,
     ExternalWeightsWithClientWeightLoaderReleasesNpuConsumedWeights) {
  const auto litert_libs_path =
      litert::testing::GetLiteRtPath("vendors/examples");
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
          litert_libs_path,
      },
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          litert_libs_path,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, litert::Environment::Create(litert::EnvironmentOptions(
                    absl::MakeConstSpan(environment_options))));

  auto serialized = BuildExternalWeightMulAddModel("custom_loader_group");
  TrackingWeightLoader client_loader;

  LITERT_ASSERT_OK_AND_ASSIGN(auto options, litert::Options::Create());
  LITERT_ASSERT_OK(options.SetHardwareAccelerators(
      litert::HwAccelerators::kNpu | litert::HwAccelerators::kCpu));
  LITERT_ASSERT_OK(options.SetWeightLoader(&client_loader));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      litert::CompiledModel::Create(env, serialized, options));

  // External buffer 1 was prepared for JIT compilation, consumed by the NPU
  // partition, and then released from the client's WeightLoader; external
  // buffer 2 remains active for the CPU add op.
  EXPECT_TRUE(client_loader.prepared_ids().contains(1u));
  EXPECT_TRUE(client_loader.prepared_ids().contains(2u));
  EXPECT_TRUE(client_loader.released_ids().contains(1u));
  EXPECT_FALSE(client_loader.released_ids().contains(2u));
  EXPECT_EQ(client_loader.GetExternalWeightByBuffer(1), nullptr);
  EXPECT_NE(client_loader.GetExternalWeightByBuffer(2), nullptr);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffers,
      compiled_model.CreateInputBuffers(/*signature_index=*/0));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto output_buffers,
      compiled_model.CreateOutputBuffers(/*signature_index=*/0));

  LITERT_ASSERT_OK(input_buffers[0].Write<float>({1.0f, 2.0f, 3.0f, 4.0f}));
  LITERT_ASSERT_OK(compiled_model.Run(input_buffers, output_buffers));

  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], litert::TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, 4);
    EXPECT_THAT(output, ElementsAre(12.0f, 26.0f, 42.0f, 60.0f));
  }
}

TEST(JitCompilation, ExternalWeightsFullyCompiledOnNpuOnly) {
  const auto litert_libs_path =
      litert::testing::GetLiteRtPath("vendors/examples");
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
          litert_libs_path,
      },
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          litert_libs_path,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, litert::Environment::Create(litert::EnvironmentOptions(
                    absl::MakeConstSpan(environment_options))));

  auto flatbuffer = litert::internal::FlatbufferWrapper::CreateFromTflFile(
      litert::testing::GetTestFilePath("add_cst.tflite"));
  ASSERT_TRUE(flatbuffer);
  auto tfl_model = flatbuffer->get()->Unpack();

  // Convert the single op to MUL so 100% of the graph and its external weight
  // are compiled onto NPU.
  tfl_model->operator_codes[0]->builtin_code = ::tflite::BuiltinOperator_MUL;
  const int32_t cst_idx = tfl_model->subgraphs[0]->operators[0]->inputs[1];
  tfl_model->buffers[tfl_model->subgraphs[0]->tensors[cst_idx]->buffer]
      ->data.clear();
  tfl_model->subgraphs[0]->tensors[cst_idx]->buffer = 0;
  tfl_model->subgraphs[0]->tensors[cst_idx]->external_buffer = 1;

  auto group0 = std::make_unique<tflite::ExternalBufferGroupT>();
  group0->name = "";
  auto group1 = std::make_unique<tflite::ExternalBufferGroupT>();
  group1->name = "ext_weights.bin";
  tfl_model->external_buffer_groups.push_back(std::move(group0));
  tfl_model->external_buffer_groups.push_back(std::move(group1));

  constexpr std::array<float, 4> kExtWeights = {2.0f, 3.0f, 4.0f, 5.0f};
  auto ext_buf = std::make_unique<tflite::ExternalBufferT>();
  ext_buf->id = 1;
  ext_buf->group = 1;
  ext_buf->offset = 0;
  ext_buf->length = sizeof(kExtWeights);
  tfl_model->external_buffers.push_back(std::move(ext_buf));

  auto serialized = litert::internal::SerializeFlatbuffer(*tfl_model);
  auto temp_dir = litert::testing::UniqueTestDirectory::Create();
  ASSERT_TRUE(temp_dir);
  std::filesystem::path temp_path(std::string(temp_dir->Str()));

  std::filesystem::path model_path = temp_path / "ext_mul_only.tflite";
  {
    std::ofstream model_file(model_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(model_file.is_open());
    model_file.write(serialized.StrData(), serialized.Size());
  }
  {
    std::ofstream weights_file(temp_path / "ext_weights.bin",
                               std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(weights_file.is_open());
    weights_file.write(reinterpret_cast<const char*>(kExtWeights.data()),
                       sizeof(kExtWeights));
  }

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      litert::CompiledModel::Create(env, model_path.string(),
                                    litert::HwAccelerators::kNpu));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffers,
      compiled_model.CreateInputBuffers(/*signature_index=*/0));
  ASSERT_EQ(input_buffers.size(), 1u);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto output_buffers,
      compiled_model.CreateOutputBuffers(/*signature_index=*/0));
  ASSERT_EQ(output_buffers.size(), 1u);

  LITERT_ASSERT_OK(input_buffers[0].Write<float>({1.0f, 2.0f, 3.0f, 4.0f}));
  LITERT_ASSERT_OK(compiled_model.Run(input_buffers, output_buffers));

  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], litert::TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, 4);
    EXPECT_THAT(output, ElementsAre(2.0f, 6.0f, 12.0f, 20.0f));
  }
}

TEST(JitCompilation, SharedExternalWeightAcrossNpuMulAndCpuAdd) {
  const auto litert_libs_path =
      litert::testing::GetLiteRtPath("vendors/examples");
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
          litert_libs_path,
      },
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          litert_libs_path,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, litert::Environment::Create(litert::EnvironmentOptions(
                    absl::MakeConstSpan(environment_options))));

  auto flatbuffer = litert::internal::FlatbufferWrapper::CreateFromTflFile(
      litert::testing::GetTestFilePath("add_cst.tflite"));
  ASSERT_TRUE(flatbuffer);
  auto tfl_model = flatbuffer->get()->Unpack();

  auto mul_opcode = std::make_unique<litert::internal::TflOpCode>();
  mul_opcode->builtin_code = ::tflite::BuiltinOperator_MUL;
  tfl_model->operator_codes.push_back(std::move(mul_opcode));
  const uint32_t mul_opcode_idx = tfl_model->operator_codes.size() - 1;

  auto& sg = *tfl_model->subgraphs[0];
  const int32_t orig_input_idx = sg.operators[0]->inputs[0];
  const int32_t shared_cst_idx = sg.operators[0]->inputs[1];
  const int32_t orig_output_idx = sg.operators[0]->outputs[0];

  tfl_model->buffers[sg.tensors[shared_cst_idx]->buffer]->data.clear();
  sg.tensors[shared_cst_idx]->buffer = 0;
  sg.tensors[shared_cst_idx]->external_buffer = 1;

  auto tmp_tensor = std::make_unique<litert::internal::TflTensor>(
      *sg.tensors[orig_output_idx]);
  tmp_tensor->name = "mul_out";
  tmp_tensor->buffer = 0;
  tmp_tensor->external_buffer = 0;
  sg.tensors.push_back(std::move(tmp_tensor));
  const int32_t tmp_tensor_idx = sg.tensors.size() - 1;

  sg.operators[0]->opcode_index = mul_opcode_idx;
  sg.operators[0]->inputs = {orig_input_idx, shared_cst_idx};
  sg.operators[0]->outputs = {tmp_tensor_idx};

  auto add_op = std::make_unique<litert::internal::TflOp>();
  add_op->opcode_index = 0;  // ADD
  add_op->inputs = {tmp_tensor_idx, shared_cst_idx};
  add_op->outputs = {orig_output_idx};
  sg.operators.push_back(std::move(add_op));

  auto group0 = std::make_unique<tflite::ExternalBufferGroupT>();
  group0->name = "";
  auto group1 = std::make_unique<tflite::ExternalBufferGroupT>();
  group1->name = "ext_weights.bin";
  tfl_model->external_buffer_groups.push_back(std::move(group0));
  tfl_model->external_buffer_groups.push_back(std::move(group1));

  constexpr std::array<float, 4> kSharedWeights = {2.0f, 3.0f, 4.0f, 5.0f};
  auto ext_buf = std::make_unique<tflite::ExternalBufferT>();
  ext_buf->id = 1;
  ext_buf->group = 1;
  ext_buf->offset = 0;
  ext_buf->length = sizeof(kSharedWeights);
  tfl_model->external_buffers.push_back(std::move(ext_buf));

  auto serialized = litert::internal::SerializeFlatbuffer(*tfl_model);
  auto temp_dir = litert::testing::UniqueTestDirectory::Create();
  ASSERT_TRUE(temp_dir);
  std::filesystem::path temp_path(std::string(temp_dir->Str()));

  std::filesystem::path model_path = temp_path / "shared_ext_mul_add.tflite";
  {
    std::ofstream model_file(model_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(model_file.is_open());
    model_file.write(serialized.StrData(), serialized.Size());
  }
  {
    std::ofstream weights_file(temp_path / "ext_weights.bin",
                               std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(weights_file.is_open());
    weights_file.write(reinterpret_cast<const char*>(kSharedWeights.data()),
                       sizeof(kSharedWeights));
  }

  LITERT_ASSERT_OK_AND_ASSIGN(auto options, litert::Options::Create());
  LITERT_ASSERT_OK(options.SetHardwareAccelerators(
      litert::HwAccelerators::kNpu | litert::HwAccelerators::kCpu));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      litert::CompiledModel::Create(env, model_path.string(), options));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffers,
      compiled_model.CreateInputBuffers(/*signature_index=*/0));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto output_buffers,
      compiled_model.CreateOutputBuffers(/*signature_index=*/0));

  LITERT_ASSERT_OK(input_buffers[0].Write<float>({1.0f, 2.0f, 3.0f, 4.0f}));
  LITERT_ASSERT_OK(compiled_model.Run(input_buffers, output_buffers));

  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], litert::TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, 4);
    // arg0 * [2, 3, 4, 5] + [2, 3, 4, 5] = [4, 9, 16, 25]
    EXPECT_THAT(output, ElementsAre(4.0f, 9.0f, 16.0f, 25.0f));
  }
}

}  // namespace
