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

#include <cstddef>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/ats/compile_fixture.h"
#include "litert/ats/configure.h"
#include "litert/ats/inference_fixture.h"
#include "litert/ats/register_composite_ops.h"
#include "litert/ats/register_core_ops.h"

namespace litert::testing {
namespace {

struct RegistrationSpan {
  size_t start_id;
  size_t end_id;
  std::string expected_prefix;
  std::string fixture_kind;
};

std::vector<RegistrationSpan>& GetSpans() {
  static auto* spans = new std::vector<RegistrationSpan>();
  return *spans;
}

TEST(RegisterOpsContractTest, AllCoreOpsHaveCoreSingleOpPrefix) {
  const auto* unit_test = ::testing::UnitTest::GetInstance();
  size_t verified_count = 0;

  for (int i = 0; i < unit_test->total_test_suite_count(); ++i) {
    const auto* suite = unit_test->GetTestSuite(i);
    absl::string_view name = suite->name();
    if (absl::StartsWith(name, "CoreSingleOp_")) {
      EXPECT_THAT(name, ::testing::Not(::testing::StartsWith("ats_")))
          << "GTest suite " << name
          << " must be grouped by family_logic without ats_<id>_ prefix";
      ++verified_count;
    }
  }

  EXPECT_GT(verified_count, 0) << "Expected at least one CoreSingleOp test "
                                  "suite to be registered and verified";
}

TEST(RegisterOpsContractTest, AllCompositeOpsHaveCompositeOpPrefix) {
  const auto* unit_test = ::testing::UnitTest::GetInstance();
  size_t verified_count = 0;

  for (int i = 0; i < unit_test->total_test_suite_count(); ++i) {
    const auto* suite = unit_test->GetTestSuite(i);
    absl::string_view name = suite->name();
    if (absl::StartsWith(name, "CompositeOp_")) {
      EXPECT_THAT(name, ::testing::Not(::testing::StartsWith("ats_")))
          << "GTest suite " << name
          << " must be grouped by family_logic without ats_<id>_ prefix";
      ++verified_count;
    }
  }

  EXPECT_GT(verified_count, 0) << "Expected at least one CompositeOp test "
                                  "suite to be registered and verified";
}

TEST(RegisterOpsContractTest,
     TestMethodNamesAreNormalizedAndDeduplicatedForTestGrid) {
  const auto* unit_test = ::testing::UnitTest::GetInstance();
  size_t verified_count = 0;

  for (const auto& span : GetSpans()) {
    const std::string suite_prefix =
        absl::StrFormat("%s_%s_", span.expected_prefix, span.fixture_kind);
    for (int i = 0; i < unit_test->total_test_suite_count(); ++i) {
      const auto* suite = unit_test->GetTestSuite(i);
      absl::string_view suite_name = suite->name();
      if (!absl::StartsWith(suite_name, suite_prefix)) {
        continue;
      }
      EXPECT_GT(suite->total_test_count(), 0);
      std::vector<std::string> seen_names;
      for (int j = 0; j < suite->total_test_count(); ++j) {
        absl::string_view test_name = suite->GetTestInfo(j)->name();
        EXPECT_THAT(test_name,
                    ::testing::Not(::testing::StartsWith(suite_prefix)))
            << "Test method name " << test_name
            << " must not redundantly repeat the suite prefix " << suite_prefix;
        EXPECT_THAT(test_name, ::testing::Not(::testing::HasSubstr("<")))
            << "Test method name " << test_name
            << " must not contain concrete random dimensions '<...>'";
        EXPECT_THAT(seen_names, ::testing::Not(::testing::Contains(test_name)))
            << "Duplicate GTest test method name " << test_name
            << " in suite " << suite_name
            << "; random shape iterations must be grouped into 1 GTest test";
        seen_names.emplace_back(test_name);
        ++verified_count;
      }
    }
  }

  EXPECT_GT(verified_count, 0);
}

}  // namespace
}  // namespace litert::testing

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  auto options = litert::testing::AtsConf::ParseFlagsAndDoSetup();
  ABSL_CHECK(options.HasValue())
      << "Failed to parse ATS options: " << options.Error().Message();

  size_t test_id = 0;
  litert::testing::AtsInferenceTest::Capture i_cap;
  litert::testing::AtsCompileTest::Capture c_cap;

  // 1. Register Core Ops (Inference)
  size_t core_inf_start = test_id;
  litert::testing::RegisterCoreOps(*options, test_id, i_cap);
  litert::testing::GetSpans().push_back(
      {core_inf_start, test_id, "CoreSingleOp", "inference"});

  // 2. Register Core Ops (Compile)
  size_t core_comp_start = test_id;
  litert::testing::RegisterCoreOps(*options, test_id, c_cap);
  litert::testing::GetSpans().push_back(
      {core_comp_start, test_id, "CoreSingleOp", "compile"});

  // 3. Register Composite Ops (Inference)
  size_t comp_inf_start = test_id;
  litert::testing::RegisterCompositeOps(*options, test_id, i_cap);
  litert::testing::GetSpans().push_back(
      {comp_inf_start, test_id, "CompositeOp", "inference"});

  // 4. Register Composite Ops (Compile)
  size_t comp_comp_start = test_id;
  litert::testing::RegisterCompositeOps(*options, test_id, c_cap);
  litert::testing::GetSpans().push_back(
      {comp_comp_start, test_id, "CompositeOp", "compile"});

  // Filter GoogleTest execution to ONLY run the contract validation tests.
  // This prevents the actual generated models from executing during the
  // contract check.
  ::testing::GTEST_FLAG(filter) = "RegisterOpsContractTest.*";

  return RUN_ALL_TESTS();
}
