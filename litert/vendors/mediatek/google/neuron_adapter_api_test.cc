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

#include "third_party/odml/litert/litert/vendors/mediatek/neuron_adapter_api.h"

#include <cstdint>
#include <filesystem>  // NOLINT: open source test
#include <optional>
#include <string>

#include "neuron/api/NeuronAdapter.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/odml/litert/litert/test/common.h"
#include "third_party/odml/litert/litert/test/matchers.h"

namespace litert::mediatek {
namespace {

using ::litert::testing::UniqueTestDirectory;

// Test that the shared libraries can be found and properly loaded.

TEST(NeuronAdapterApiTest, CreateWithDefaultSoPath) {
  auto adapter = NeuronAdapterApi::Create(/*shared_library_dir=*/std::nullopt);
  LITERT_ASSERT_OK(adapter);
}

TEST(NeuronAdapterApiTest, CompileToyModel) {
  static constexpr absl::string_view kModelName = "test_model";
  static constexpr absl::string_view kCompileOptions = "";
  static constexpr absl::string_view kPlatform = "mt6989";
  static constexpr absl::string_view kDLAPrefix = "test_";

  // 2x2 tensor
  static constexpr uint32_t kDims[] = {2, 2};
  static constexpr uint32_t kDimCount = 2;

  // qi8
  static constexpr int32_t kType = NEURON_TENSOR_QUANT8_ASYMM;
  static constexpr float kScale = 1.0f;
  static constexpr int32_t kZeroPoint = 0;

  // single abs model
  static constexpr NeuronOperationType kOperationType = NEURON_ABS;
  static constexpr uint32_t kInputInd = 0;
  static constexpr uint32_t kOutputInd = 1;
  static constexpr uint32_t kOpInputCount = 1;
  static constexpr uint32_t kOpInputIndices[] = {kInputInd};
  static constexpr uint32_t kOpOutputCount = 1;
  static constexpr uint32_t kOpOutputIndices[] = {kOutputInd};
  static constexpr uint32_t kModelInputCount = 1;
  static constexpr uint32_t kModelInputIndices[] = {kInputInd};
  static constexpr uint32_t kModelOutputCount = 1;
  static constexpr uint32_t kModelOutputIndices[] = {kOutputInd};

  setenv("MTKNN_ADAPTER_DLA_PLATFORM", kPlatform.data(), 1);
  setenv("MTKNN_ADAPTER_DLA_PREFIX", kDLAPrefix.data(), 1);

  auto dir = UniqueTestDirectory::Create();
  const std::string dir_string(dir->Str());
  setenv("MTKNN_ADAPTER_DLA_DIR", dir_string.c_str(), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(auto adapter,
                              NeuronAdapterApi::Create(std::nullopt));
  auto& api = *adapter;

  LITERT_ASSERT_OK_AND_ASSIGN(auto model, api.CreateModel());
  ASSERT_TRUE(model);

  ASSERT_EQ(api.api().model_set_name(model.get(), kModelName.data()),
            ::NEURON_NO_ERROR);

  {
    NeuronOperandType type;
    type.type = kType;
    type.scale = kScale;
    type.zeroPoint = kZeroPoint;
    type.dimensions = kDims;
    type.dimensionCount = kDimCount;

    ASSERT_EQ(api.api().model_add_operand(model.get(), &type),
              ::NEURON_NO_ERROR);
  }

  {
    NeuronOperandType type;
    type.type = kType;
    type.scale = kScale;
    type.zeroPoint = kZeroPoint;
    type.dimensions = kDims;
    type.dimensionCount = kDimCount;

    ASSERT_EQ(api.api().model_add_operand(model.get(), &type),
              ::NEURON_NO_ERROR);
  }

  {
    const auto status = api.api().model_add_operation(
        model.get(), kOperationType, kOpInputCount, kOpInputIndices,
        kOpOutputCount, kOpOutputIndices);
    ASSERT_EQ(status, ::NEURON_NO_ERROR);
  }

  {
    const auto status = api.api().model_identify_inputs_and_outputs(
        model.get(), kModelInputCount, kModelInputIndices, kModelOutputCount,
        kModelOutputIndices);
    ASSERT_EQ(status, ::NEURON_NO_ERROR);
  }

  ASSERT_EQ(api.api().model_finish(model.get()), ::NEURON_NO_ERROR);

  const auto compile_options = std::string(kCompileOptions);
  auto compilation = api.CreateCompilation(model.get(), compile_options);
  ASSERT_TRUE(compilation);

  ASSERT_EQ(api.api().compilation_finish(compilation->get()),
            ::NEURON_NO_ERROR);

  for (auto p : std::filesystem::directory_iterator(dir_string)) {
    ASSERT_EQ(p.path().extension().string(), ".dla");
    ASSERT_TRUE(p.is_regular_file());
    ASSERT_GT(p.file_size(), 0);
    break;
  }
}

}  // namespace
}  // namespace litert::mediatek
