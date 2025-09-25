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

#include "litert/runtime/magic_number_utils.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/core/environment.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"

using ::testing::litert::IsError;
using ::testing::litert::IsOkAndHolds;

namespace litert::internal {
namespace {

// Built by full_model_magic_test.textproto with random weights.
constexpr absl::string_view kTestModelPath = "model_magic_test.tflite";

class MagicNumberUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto model = LoadModelFromFile(testing::GetTestFilePath(kTestModelPath),
                                   /*allow_motifications=*/true);
    ASSERT_TRUE(model);
    model_ = std::move(*model);
  }

  LiteRtEnvironmentT env_;
  std::unique_ptr<LiteRtModelT> model_;
};

TEST_F(MagicNumberUtilsTest, ReplaceMagicNumbersIfAny_NoMagicNumberConfigs) {
  EXPECT_THAT(ReplaceMagicNumbersIfAny(env_, *model_), IsOkAndHolds(0));
}

TEST_F(MagicNumberUtilsTest,
       ReplaceMagicNumbersIfAny_SuccessWithVerifications_LongContext) {
  LiteRtMagicNumberConfigs* magic_number_configs =
      reinterpret_cast<LiteRtMagicNumberConfigs*>(
          alloca(sizeof(LiteRtMagicNumberConfigs) +
                 sizeof(LiteRtMagicNumberConfig) * 3));
  magic_number_configs->num_configs = 3;
  magic_number_configs->configs[0].magic_number = 8209;
  magic_number_configs->configs[0].target_number = 8192;
  magic_number_configs->configs[0].signature_prefix = nullptr;
  magic_number_configs->configs[1].magic_number = 4099;
  magic_number_configs->configs[1].target_number = 4096;
  magic_number_configs->configs[1].signature_prefix = "prefill";
  magic_number_configs->configs[2].magic_number = 11;
  magic_number_configs->configs[2].target_number = 10;
  magic_number_configs->configs[2].signature_prefix = "decode";

  LiteRtMagicNumberVerifications* verifications =
      reinterpret_cast<LiteRtMagicNumberVerifications*>(
          alloca(sizeof(LiteRtMagicNumberVerifications) +
                 sizeof(LiteRtMagicNumberVerification) * 2));
  verifications->num_verifications = 2;
  verifications->verifications[0].signature = "prefill";
  verifications->verifications[0].test_signature = "test_prefill_8192";
  verifications->verifications[0].is_superset = false;
  verifications->verifications[1].signature = "decode";
  verifications->verifications[1].test_signature = "test_decode_8192";
  verifications->verifications[1].is_superset = false;

  LiteRtEnvOption magic_number_options[] = {
      {.tag = LiteRtEnvOptionTag::kLiteRtEnvOptionTagMagicNumberConfigs,
       .value = LiteRtAny{.type = kLiteRtAnyTypeVoidPtr,
                          .ptr_value = magic_number_configs}},
      {.tag = LiteRtEnvOptionTag::kLiteRtEnvOptionTagMagicNumberVerifications,
       .value = LiteRtAny{.type = kLiteRtAnyTypeVoidPtr,
                          .ptr_value = verifications}}};
  LITERT_EXPECT_OK(env_.AddOptions(absl::MakeConstSpan(magic_number_options)));

  EXPECT_THAT(ReplaceMagicNumbersIfAny(env_, *model_), IsOkAndHolds(3206));
}

TEST_F(MagicNumberUtilsTest,
       ReplaceMagicNumbersIfAny_SuccessWithVerifications_MediumContext) {
  LiteRtMagicNumberConfigs* magic_number_configs =
      reinterpret_cast<LiteRtMagicNumberConfigs*>(
          alloca(sizeof(LiteRtMagicNumberConfigs) +
                 sizeof(LiteRtMagicNumberConfig) * 3));
  magic_number_configs->num_configs = 3;
  magic_number_configs->configs[0].magic_number = 8209;
  magic_number_configs->configs[0].target_number = 1280;
  magic_number_configs->configs[0].signature_prefix = nullptr;
  magic_number_configs->configs[1].magic_number = 4099;
  magic_number_configs->configs[1].target_number = 1024;
  magic_number_configs->configs[1].signature_prefix = "prefill";
  magic_number_configs->configs[2].magic_number = 11;
  magic_number_configs->configs[2].target_number = 5;
  magic_number_configs->configs[2].signature_prefix = "decode";

  LiteRtMagicNumberVerifications* verifications =
      reinterpret_cast<LiteRtMagicNumberVerifications*>(
          alloca(sizeof(LiteRtMagicNumberVerifications) +
                 sizeof(LiteRtMagicNumberVerification) * 2));
  verifications->num_verifications = 2;
  verifications->verifications[0].signature = "prefill";
  verifications->verifications[0].test_signature = "test_prefill_1280";
  verifications->verifications[0].is_superset = false;
  verifications->verifications[1].signature = "decode";
  verifications->verifications[1].test_signature = "test_decode_1280";
  verifications->verifications[1].is_superset = false;

  LiteRtEnvOption magic_number_options[] = {
      {.tag = LiteRtEnvOptionTag::kLiteRtEnvOptionTagMagicNumberConfigs,
       .value = LiteRtAny{.type = kLiteRtAnyTypeVoidPtr,
                          .ptr_value = magic_number_configs}},
      {.tag = LiteRtEnvOptionTag::kLiteRtEnvOptionTagMagicNumberVerifications,
       .value = LiteRtAny{.type = kLiteRtAnyTypeVoidPtr,
                          .ptr_value = verifications}}};
  LITERT_EXPECT_OK(env_.AddOptions(absl::MakeConstSpan(magic_number_options)));

  EXPECT_THAT(ReplaceMagicNumbersIfAny(env_, *model_), IsOkAndHolds(3206));
}

TEST_F(MagicNumberUtilsTest,
       ReplaceMagicNumbersIfAny_FailureWithVerifications_ShortContext) {
  LiteRtMagicNumberConfigs* magic_number_configs =
      reinterpret_cast<LiteRtMagicNumberConfigs*>(
          alloca(sizeof(LiteRtMagicNumberConfigs) +
                 sizeof(LiteRtMagicNumberConfig) * 3));
  magic_number_configs->num_configs = 3;
  magic_number_configs->configs[0].magic_number = 8209;
  magic_number_configs->configs[0].target_number = 128;
  magic_number_configs->configs[0].signature_prefix = nullptr;
  magic_number_configs->configs[1].magic_number = 4099;
  magic_number_configs->configs[1].target_number = 64;
  magic_number_configs->configs[1].signature_prefix = "prefill";
  magic_number_configs->configs[2].magic_number = 11;
  magic_number_configs->configs[2].target_number = 3;
  magic_number_configs->configs[2].signature_prefix = "decode";

  LiteRtMagicNumberVerifications* verifications =
      reinterpret_cast<LiteRtMagicNumberVerifications*>(
          alloca(sizeof(LiteRtMagicNumberVerifications) +
                 sizeof(LiteRtMagicNumberVerification) * 2));
  verifications->num_verifications = 2;
  verifications->verifications[0].signature = "prefill";
  verifications->verifications[0].test_signature = "test_prefill_128";
  verifications->verifications[0].is_superset = false;
  verifications->verifications[1].signature = "decode";
  verifications->verifications[1].test_signature = "test_decode_128";
  verifications->verifications[1].is_superset = false;

  LiteRtEnvOption magic_number_options[] = {
      {.tag = LiteRtEnvOptionTag::kLiteRtEnvOptionTagMagicNumberConfigs,
       .value = LiteRtAny{.type = kLiteRtAnyTypeVoidPtr,
                          .ptr_value = magic_number_configs}},
      {.tag = LiteRtEnvOptionTag::kLiteRtEnvOptionTagMagicNumberVerifications,
       .value = LiteRtAny{.type = kLiteRtAnyTypeVoidPtr,
                          .ptr_value = verifications}}};
  LITERT_EXPECT_OK(env_.AddOptions(absl::MakeConstSpan(magic_number_options)));

  EXPECT_THAT(ReplaceMagicNumbersIfAny(env_, *model_),
              IsError(kLiteRtStatusErrorUnknown));
}

TEST_F(MagicNumberUtilsTest,
       ReplaceMagicNumbersIfAny_SuccessWithSupersetVerifications_ShortContext) {
  LiteRtMagicNumberConfigs* magic_number_configs =
      reinterpret_cast<LiteRtMagicNumberConfigs*>(
          alloca(sizeof(LiteRtMagicNumberConfigs) +
                 sizeof(LiteRtMagicNumberConfig) * 3));
  magic_number_configs->num_configs = 3;
  magic_number_configs->configs[0].magic_number = 8209;
  magic_number_configs->configs[0].target_number = 128;
  magic_number_configs->configs[0].signature_prefix = nullptr;
  magic_number_configs->configs[1].magic_number = 4099;
  magic_number_configs->configs[1].target_number = 64;
  magic_number_configs->configs[1].signature_prefix = "prefill";
  magic_number_configs->configs[2].magic_number = 11;
  magic_number_configs->configs[2].target_number = 3;
  magic_number_configs->configs[2].signature_prefix = "decode";

  LiteRtMagicNumberVerifications* verifications =
      reinterpret_cast<LiteRtMagicNumberVerifications*>(
          alloca(sizeof(LiteRtMagicNumberVerifications) +
                 sizeof(LiteRtMagicNumberVerification) * 2));
  verifications->num_verifications = 2;
  verifications->verifications[0].signature = "prefill";
  verifications->verifications[0].test_signature = "test_prefill_128";
  verifications->verifications[0].is_superset = true;
  verifications->verifications[1].signature = "decode";
  verifications->verifications[1].test_signature = "test_decode_128";
  verifications->verifications[1].is_superset = true;

  LiteRtEnvOption magic_number_options[] = {
      {.tag = LiteRtEnvOptionTag::kLiteRtEnvOptionTagMagicNumberConfigs,
       .value = LiteRtAny{.type = kLiteRtAnyTypeVoidPtr,
                          .ptr_value = magic_number_configs}},
      {.tag = LiteRtEnvOptionTag::kLiteRtEnvOptionTagMagicNumberVerifications,
       .value = LiteRtAny{.type = kLiteRtAnyTypeVoidPtr,
                          .ptr_value = verifications}}};
  LITERT_EXPECT_OK(env_.AddOptions(absl::MakeConstSpan(magic_number_options)));

  EXPECT_THAT(ReplaceMagicNumbersIfAny(env_, *model_), IsOkAndHolds(3206));
}

TEST_F(MagicNumberUtilsTest,
       ReplaceMagicNumbersIfAny_SuccessWithoutVerifications) {
  LiteRtMagicNumberConfigs* magic_number_configs =
      reinterpret_cast<LiteRtMagicNumberConfigs*>(
          alloca(sizeof(LiteRtMagicNumberConfigs) +
                 sizeof(LiteRtMagicNumberConfig) * 3));
  magic_number_configs->num_configs = 3;
  magic_number_configs->configs[0].magic_number = 8209;
  magic_number_configs->configs[0].target_number = 1280;
  magic_number_configs->configs[0].signature_prefix = nullptr;
  magic_number_configs->configs[1].magic_number = 4099;
  magic_number_configs->configs[1].target_number = 1024;
  magic_number_configs->configs[1].signature_prefix = "prefill";
  magic_number_configs->configs[2].magic_number = 11;
  magic_number_configs->configs[2].target_number = 5;
  magic_number_configs->configs[2].signature_prefix = "decode";

  LiteRtEnvOption magic_number_options[] = {
      {.tag = LiteRtEnvOptionTag::kLiteRtEnvOptionTagMagicNumberConfigs,
       .value = LiteRtAny{.type = kLiteRtAnyTypeVoidPtr,
                          .ptr_value = magic_number_configs}}};
  LITERT_EXPECT_OK(env_.AddOptions(absl::MakeConstSpan(magic_number_options)));

  EXPECT_THAT(ReplaceMagicNumbersIfAny(env_, *model_), IsOkAndHolds(3206));
}

TEST_F(MagicNumberUtilsTest,
       ReplaceMagicNumbersIfAny_FailureWithWrongTestSignatures) {
  LiteRtMagicNumberConfigs* magic_number_configs =
      reinterpret_cast<LiteRtMagicNumberConfigs*>(
          alloca(sizeof(LiteRtMagicNumberConfigs) +
                 sizeof(LiteRtMagicNumberConfig) * 3));
  magic_number_configs->num_configs = 3;
  magic_number_configs->configs[0].magic_number = 8209;
  magic_number_configs->configs[0].target_number = 1280;
  magic_number_configs->configs[0].signature_prefix = nullptr;
  magic_number_configs->configs[1].magic_number = 4099;
  magic_number_configs->configs[1].target_number = 1024;
  magic_number_configs->configs[1].signature_prefix = "prefill";
  magic_number_configs->configs[2].magic_number = 11;
  magic_number_configs->configs[2].target_number = 5;
  magic_number_configs->configs[2].signature_prefix = "decode";

  LiteRtMagicNumberVerifications* verifications =
      reinterpret_cast<LiteRtMagicNumberVerifications*>(
          alloca(sizeof(LiteRtMagicNumberVerifications) +
                 sizeof(LiteRtMagicNumberVerification) * 2));
  verifications->num_verifications = 2;
  verifications->verifications[0].signature = "prefill";
  verifications->verifications[0].test_signature = "test_prefill_1280";
  verifications->verifications[0].is_superset = false;
  verifications->verifications[1].signature = "decode";
  verifications->verifications[1].test_signature = "wrong_test_decode";
  verifications->verifications[1].is_superset = false;

  LiteRtEnvOption magic_number_options[] = {
      {.tag = LiteRtEnvOptionTag::kLiteRtEnvOptionTagMagicNumberConfigs,
       .value = LiteRtAny{.type = kLiteRtAnyTypeVoidPtr,
                          .ptr_value = magic_number_configs}},
      {.tag = LiteRtEnvOptionTag::kLiteRtEnvOptionTagMagicNumberVerifications,
       .value = LiteRtAny{.type = kLiteRtAnyTypeVoidPtr,
                          .ptr_value = verifications}}};
  LITERT_EXPECT_OK(env_.AddOptions(absl::MakeConstSpan(magic_number_options)));

  EXPECT_THAT(ReplaceMagicNumbersIfAny(env_, *model_),
              IsError(kLiteRtStatusErrorUnknown));
}

TEST_F(MagicNumberUtilsTest,
       ReplaceMagicNumbersIfAny_FailedWithWrongContextLength) {
  LiteRtMagicNumberConfigs* magic_number_configs =
      reinterpret_cast<LiteRtMagicNumberConfigs*>(
          alloca(sizeof(LiteRtMagicNumberConfigs) +
                 sizeof(LiteRtMagicNumberConfig) * 3));
  magic_number_configs->num_configs = 3;
  magic_number_configs->configs[0].magic_number = 8209;
  magic_number_configs->configs[0].target_number = 1240;
  magic_number_configs->configs[0].signature_prefix = nullptr;
  magic_number_configs->configs[1].magic_number = 4099;
  magic_number_configs->configs[1].target_number = 1024;
  magic_number_configs->configs[1].signature_prefix = "prefill";
  magic_number_configs->configs[2].magic_number = 11;
  magic_number_configs->configs[2].target_number = 5;
  magic_number_configs->configs[2].signature_prefix = "decode";

  LiteRtMagicNumberVerifications* verifications =
      reinterpret_cast<LiteRtMagicNumberVerifications*>(
          alloca(sizeof(LiteRtMagicNumberVerifications) +
                 sizeof(LiteRtMagicNumberVerification) * 2));
  verifications->num_verifications = 2;
  verifications->verifications[0].signature = "prefill";
  verifications->verifications[0].test_signature = "test_prefill_1280";
  verifications->verifications[0].is_superset = false;
  verifications->verifications[1].signature = "decode";
  verifications->verifications[1].test_signature = "test_decode_1280";
  verifications->verifications[1].is_superset = false;

  LiteRtEnvOption magic_number_options[] = {
      {.tag = LiteRtEnvOptionTag::kLiteRtEnvOptionTagMagicNumberConfigs,
       .value = LiteRtAny{.type = kLiteRtAnyTypeVoidPtr,
                          .ptr_value = magic_number_configs}},
      {.tag = LiteRtEnvOptionTag::kLiteRtEnvOptionTagMagicNumberVerifications,
       .value = LiteRtAny{.type = kLiteRtAnyTypeVoidPtr,
                          .ptr_value = verifications}}};
  LITERT_EXPECT_OK(env_.AddOptions(absl::MakeConstSpan(magic_number_options)));

  EXPECT_THAT(ReplaceMagicNumbersIfAny(env_, *model_),
              IsError(kLiteRtStatusErrorUnknown));
}

TEST_F(MagicNumberUtilsTest,
       ReplaceMagicNumbersIfAny_FailedWithWrongPrefillLength) {
  LiteRtMagicNumberConfigs* magic_number_configs =
      reinterpret_cast<LiteRtMagicNumberConfigs*>(
          alloca(sizeof(LiteRtMagicNumberConfigs) +
                 sizeof(LiteRtMagicNumberConfig) * 3));
  magic_number_configs->num_configs = 3;
  magic_number_configs->configs[0].magic_number = 8209;
  magic_number_configs->configs[0].target_number = 1280;
  magic_number_configs->configs[0].signature_prefix = nullptr;
  magic_number_configs->configs[1].magic_number = 4099;
  magic_number_configs->configs[1].target_number = 512;
  magic_number_configs->configs[1].signature_prefix = "prefill";
  magic_number_configs->configs[2].magic_number = 11;
  magic_number_configs->configs[2].target_number = 5;
  magic_number_configs->configs[2].signature_prefix = "decode";

  LiteRtMagicNumberVerifications* verifications =
      reinterpret_cast<LiteRtMagicNumberVerifications*>(
          alloca(sizeof(LiteRtMagicNumberVerifications) +
                 sizeof(LiteRtMagicNumberVerification) * 2));
  verifications->num_verifications = 2;
  verifications->verifications[0].signature = "prefill";
  verifications->verifications[0].test_signature = "test_prefill_1280";
  verifications->verifications[0].is_superset = false;
  verifications->verifications[1].signature = "decode";
  verifications->verifications[1].test_signature = "test_decode_1280";
  verifications->verifications[1].is_superset = false;

  LiteRtEnvOption magic_number_options[] = {
      {.tag = LiteRtEnvOptionTag::kLiteRtEnvOptionTagMagicNumberConfigs,
       .value = LiteRtAny{.type = kLiteRtAnyTypeVoidPtr,
                          .ptr_value = magic_number_configs}},
      {.tag = LiteRtEnvOptionTag::kLiteRtEnvOptionTagMagicNumberVerifications,
       .value = LiteRtAny{.type = kLiteRtAnyTypeVoidPtr,
                          .ptr_value = verifications}}};
  LITERT_EXPECT_OK(env_.AddOptions(absl::MakeConstSpan(magic_number_options)));

  EXPECT_THAT(ReplaceMagicNumbersIfAny(env_, *model_),
              IsError(kLiteRtStatusErrorUnknown));
}

TEST_F(MagicNumberUtilsTest,
       ReplaceMagicNumbersIfAny_FailureWithWrongDecodeBatchSize) {
  LiteRtMagicNumberConfigs* magic_number_configs =
      reinterpret_cast<LiteRtMagicNumberConfigs*>(
          alloca(sizeof(LiteRtMagicNumberConfigs) +
                 sizeof(LiteRtMagicNumberConfig) * 3));
  magic_number_configs->num_configs = 3;
  magic_number_configs->configs[0].magic_number = 8209;
  magic_number_configs->configs[0].target_number = 1280;
  magic_number_configs->configs[0].signature_prefix = nullptr;
  magic_number_configs->configs[1].magic_number = 4099;
  magic_number_configs->configs[1].target_number = 1024;
  magic_number_configs->configs[1].signature_prefix = "prefill";
  magic_number_configs->configs[2].magic_number = 11;
  magic_number_configs->configs[2].target_number = 7;
  magic_number_configs->configs[2].signature_prefix = "decode";

  LiteRtMagicNumberVerifications* verifications =
      reinterpret_cast<LiteRtMagicNumberVerifications*>(
          alloca(sizeof(LiteRtMagicNumberVerifications) +
                 sizeof(LiteRtMagicNumberVerification) * 2));
  verifications->num_verifications = 2;
  verifications->verifications[0].signature = "prefill";
  verifications->verifications[0].test_signature = "test_prefill_1280";
  verifications->verifications[0].is_superset = false;
  verifications->verifications[1].signature = "decode";
  verifications->verifications[1].test_signature = "test_decode_1280";
  verifications->verifications[1].is_superset = false;

  LiteRtEnvOption magic_number_options[] = {
      {.tag = LiteRtEnvOptionTag::kLiteRtEnvOptionTagMagicNumberConfigs,
       .value = LiteRtAny{.type = kLiteRtAnyTypeVoidPtr,
                          .ptr_value = magic_number_configs}},
      {.tag = LiteRtEnvOptionTag::kLiteRtEnvOptionTagMagicNumberVerifications,
       .value = LiteRtAny{.type = kLiteRtAnyTypeVoidPtr,
                          .ptr_value = verifications}}};
  LITERT_EXPECT_OK(env_.AddOptions(absl::MakeConstSpan(magic_number_options)));

  EXPECT_THAT(ReplaceMagicNumbersIfAny(env_, *model_),
              IsError(kLiteRtStatusErrorUnknown));
}

}  // namespace
}  // namespace litert::internal
