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
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_scheduling_info.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_compiled_model_next.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"

#if !defined(LITERT_WINDOWS_OS)
#include <dlfcn.h>
#include <unistd.h>
#endif  // !defined(LITERT_WINDOWS_OS)

namespace litert {
namespace {

#if !defined(LITERT_WINDOWS_OS)
using ClearLastSchedulingInfoFn = LiteRtStatus (*)();
using GetLastSchedulingInfoFn = LiteRtStatus (*)(LiteRtSchedulingInfo*);

struct ExampleDispatchTestSymbols {
  void* handle;
  ClearLastSchedulingInfoFn clear_last_scheduling_info;
  GetLastSchedulingInfoFn get_last_scheduling_info;
};

ExampleDispatchTestSymbols LoadExampleDispatchTestSymbols(
    const std::string& litert_libs_path) {
  std::string dispatch_so_path =
      litert_libs_path + "/libLiteRtDispatch_Example.so";
  void* handle = dlopen(dispatch_so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    return ExampleDispatchTestSymbols{
        .handle = nullptr,
        .clear_last_scheduling_info = nullptr,
        .get_last_scheduling_info = nullptr,
    };
  }

  auto clear = reinterpret_cast<ClearLastSchedulingInfoFn>(
      dlsym(handle, "LiteRtDispatchExampleClearLastSchedulingInfo"));

  auto get = reinterpret_cast<GetLastSchedulingInfoFn>(
      dlsym(handle, "LiteRtDispatchExampleGetLastSchedulingInfo"));

  return ExampleDispatchTestSymbols{
      .handle = handle,
      .clear_last_scheduling_info = clear,
      .get_last_scheduling_info = get,
  };
}
#endif  // !defined(LITERT_WINDOWS_OS)

TEST(DispatchDelegateSchedulingInfoTest,
     PlumbsDefaultModelAndRequestSchedulingInfo) {
#if defined(LITERT_WINDOWS_OS)
  GTEST_SKIP() << "Example dispatch test helpers use dlopen/dlsym.";
#else
  // 1) Load test hooks from the example dispatch library.
  const auto litert_libs_path =
      litert::testing::GetLiteRtPath("vendors/examples");
  auto symbols = LoadExampleDispatchTestSymbols(litert_libs_path);
  const char* dl_error = dlerror();
  ASSERT_NE(symbols.handle, nullptr)
      << "dlopen failed: " << (dl_error ? dl_error : "(unknown)");
  ASSERT_NE(symbols.clear_last_scheduling_info, nullptr);
  ASSERT_NE(symbols.get_last_scheduling_info, nullptr);
  ASSERT_EQ(symbols.clear_last_scheduling_info(), kLiteRtStatusOk);

  // 2) Setup Environment and CompiledModel.
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          litert_libs_path,
      },
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
          litert_libs_path,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, litert::Environment::Create(litert::EnvironmentOptions(
                    absl::MakeConstSpan(environment_options))));

  std::string model_path = litert::testing::GetTestFilePath("one_mul.tflite");

  LITERT_ASSERT_OK_AND_ASSIGN(auto compilation_options, Options::Create());
  LITERT_ASSERT_OK(compilation_options.SetHardwareAccelerators(
      litert::HwAccelerators::kNpu));

  auto compiled_model_result =
      CompiledModelNext::Create(env, model_path, compilation_options);
  if (!compiled_model_result) {
    GTEST_SKIP() << compiled_model_result.Error().Message();
  }
  auto compiled_model = std::move(*compiled_model_result);

  // 3) Prepare for execution.
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names,
                              compiled_model.GetSignatureInputNames());
  ASSERT_FALSE(input_names.empty());

  absl::flat_hash_map<absl::string_view, TensorBuffer> input_map;
  for (auto input_name : input_names) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffer,
                                compiled_model.CreateInputBuffer(input_name));
    LITERT_ASSERT_OK(input_buffer.Clear());
    input_map[input_name] = std::move(input_buffer);
  }

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_names,
                              compiled_model.GetSignatureOutputNames());
  absl::flat_hash_map<absl::string_view, TensorBuffer> output_map;
  for (auto output_name : output_names) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffer,
                                compiled_model.CreateOutputBuffer(output_name));
    LITERT_ASSERT_OK(output_buffer.Clear());
    output_map[output_name] = std::move(output_buffer);
  }

  // Helper for expected fields mask.
  constexpr uint32_t kAllFieldsMask =
      static_cast<uint32_t>(kLiteRtSchedulingInfoFieldOriginalUid) |
      static_cast<uint32_t>(kLiteRtSchedulingInfoFieldDebugFeatureId) |
      static_cast<uint32_t>(kLiteRtSchedulingInfoFieldJobPriority) |
      static_cast<uint32_t>(kLiteRtSchedulingInfoFieldGroupId);
  constexpr uint8_t kRequestGroupId[kLiteRtSchedulingInfoGroupIdSize] = {
      0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
      0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
  constexpr uint8_t kEmptyGroupId[kLiteRtSchedulingInfoGroupIdSize] = {};

  // 4) Default scheduling info (user does not specify).
  ASSERT_EQ(symbols.clear_last_scheduling_info(), kLiteRtStatusOk);
  LITERT_ASSERT_OK(compiled_model.Run(input_map, output_map));

  LiteRtSchedulingInfo observed{};
  ASSERT_EQ(symbols.get_last_scheduling_info(&observed), kLiteRtStatusOk);
  EXPECT_EQ(observed.fields_mask, kAllFieldsMask);
  EXPECT_EQ(observed.original_uid, static_cast<int32_t>(getuid()));
  ASSERT_NE(observed.debug_feature_id, nullptr);
  EXPECT_STREQ(observed.debug_feature_id, "");
  EXPECT_EQ(observed.job_priority, 0);
  EXPECT_EQ(memcmp(observed.group_id, kEmptyGroupId,
                   kLiteRtSchedulingInfoGroupIdSize),
            0);

  // 5) Model-level scheduling override (user specifies once per model).
  LiteRtSchedulingInfo model_info{};
  model_info.fields_mask =
      static_cast<uint32_t>(kLiteRtSchedulingInfoFieldJobPriority);
  model_info.job_priority = 7;
  LITERT_ASSERT_OK(compiled_model.SetSchedulingInfo(model_info));

  ASSERT_EQ(symbols.clear_last_scheduling_info(), kLiteRtStatusOk);
  LITERT_ASSERT_OK(compiled_model.Run(input_map, output_map));
  ASSERT_EQ(symbols.get_last_scheduling_info(&observed), kLiteRtStatusOk);
  EXPECT_EQ(observed.fields_mask, kAllFieldsMask);
  EXPECT_EQ(observed.original_uid, static_cast<int32_t>(getuid()));
  ASSERT_NE(observed.debug_feature_id, nullptr);
  EXPECT_STREQ(observed.debug_feature_id, "");
  EXPECT_EQ(observed.job_priority, 7);
  EXPECT_EQ(memcmp(observed.group_id, kEmptyGroupId,
                   kLiteRtSchedulingInfoGroupIdSize),
            0);

  // 6) Per-request scheduling override (user specifies once per request).
  LiteRtSchedulingInfo request_info{};
  request_info.fields_mask =
      static_cast<uint32_t>(kLiteRtSchedulingInfoFieldDebugFeatureId) |
      static_cast<uint32_t>(kLiteRtSchedulingInfoFieldGroupId);
  request_info.debug_feature_id = "com.android.aicore.text_summarization";
  memcpy(request_info.group_id, kRequestGroupId,
         kLiteRtSchedulingInfoGroupIdSize);

  ASSERT_EQ(symbols.clear_last_scheduling_info(), kLiteRtStatusOk);
  LITERT_ASSERT_OK(compiled_model.Run(input_map, output_map, request_info));
  ASSERT_EQ(symbols.get_last_scheduling_info(&observed), kLiteRtStatusOk);
  EXPECT_EQ(observed.fields_mask, kAllFieldsMask);
  EXPECT_EQ(observed.original_uid, static_cast<int32_t>(getuid()));
  ASSERT_NE(observed.debug_feature_id, nullptr);
  EXPECT_STREQ(observed.debug_feature_id,
               "com.android.aicore.text_summarization");
  EXPECT_EQ(observed.job_priority, 7);
  EXPECT_EQ(memcmp(observed.group_id, kRequestGroupId,
                   kLiteRtSchedulingInfoGroupIdSize),
            0);
#endif  // defined(LITERT_WINDOWS_OS)
}

}  // namespace
}  // namespace litert
