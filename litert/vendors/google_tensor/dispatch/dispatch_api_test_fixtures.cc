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

#include "litert/vendors/google_tensor/dispatch/dispatch_api_test_fixtures.h"

#include <array>
#include <string>

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_options.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/filesystem.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#if defined(__ANDROID__)
#include "litert/test/testdata/simple_model_test_vectors.h"
#endif
#include "litert/vendors/c/litert_dispatch.h"

namespace litert::google_tensor::testing {

void DispatchApiTest::SetUp() {
  static const absl::NoDestructor<std::string> dispatch_library_dir(
#if defined(__ANDROID__)
      "/data/local/tmp"
#else
      litert::testing::GetLiteRtPath("vendors/google_tensor/dispatch")
#endif
  );

  std::array env_options_for_create = {
      LiteRtEnvOption{
          kLiteRtEnvOptionTagDispatchLibraryDir,
          LiteRtAny{.type = kLiteRtAnyTypeString,
                    .str_value = dispatch_library_dir->c_str()},
      },
  };
  LITERT_ASSERT_OK(LiteRtCreateEnvironment(
      env_options_for_create.size(), env_options_for_create.data(), &env_));

  LiteRtOptions options;
  LITERT_ASSERT_OK(LiteRtCreateOptions(&options));

  LITERT_ASSERT_OK(LiteRtDispatchInitialize(env_, options));

  const char* vendor_id;
  LITERT_ASSERT_OK(LiteRtDispatchGetVendorId(&vendor_id));
  LITERT_LOG(LITERT_INFO, "Vendor ID: %s", vendor_id);

  const char* build_id;
  LITERT_ASSERT_OK(LiteRtDispatchGetBuildId(&build_id));
  LITERT_LOG(LITERT_INFO, "Build ID: %s", build_id);

  LiteRtApiVersion api_version;
  LITERT_ASSERT_OK(LiteRtDispatchGetApiVersion(&api_version));
  LITERT_LOG(LITERT_INFO, "API version: %d.%d.%d", api_version.major,
             api_version.minor, api_version.patch);

  LITERT_ASSERT_OK(
      LiteRtDispatchDeviceContextCreate(options, &device_context_));
  LiteRtDestroyOptions(options);
}

void DispatchApiTest::TearDown() {
  LITERT_ASSERT_OK(LiteRtDispatchDeviceContextDestroy(device_context_));

  LiteRtDestroyEnvironment(env_);
}

void SimpleModelTest::SetUp() {
  DispatchApiTest::SetUp();

  std::string model_file_path =
#if defined(__ANDROID__)
      litert::testing::GetTestFilePath(kGoogleTensorModelFileName);
#else
      litert::testing::GetLiteRtPath(
          "vendors/google_tensor/dispatch/"
          "simple_model_reference_google_tensor.bin");
#endif
  LITERT_ASSERT_OK_AND_ASSIGN(
      model_, litert::internal::LoadBinaryFile(model_file_path));

  LITERT_LOG(LITERT_INFO, "Loaded model '%s': %zu bytes",
             model_file_path.c_str(), model_.Size());

  model_bytecode_ = {/*.fd=*/-1,
                     /*.base_addr=*/model_.Data(),
                     /*.offset=*/0,
                     /*.size=*/model_.Size()};
}

}  // namespace litert::google_tensor::testing
