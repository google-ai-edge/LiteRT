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
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/ats/common.h"
#include "litert/ats/compile_fixture.h"
#include "litert/ats/configure.h"
#include "litert/ats/inference_fixture.h"
#include "litert/ats/register_composite_ops.h"
#include "litert/ats/register_single_ops.h"
#include "litert/c/litert_op_code.h"

namespace litert::testing {
namespace {

using ::testing::Contains;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::StartsWith;

std::vector<const ::testing::TestSuite*>& SingleOpSuites() {
  static auto* suites = new std::vector<const ::testing::TestSuite*>();
  return *suites;
}

std::vector<const ::testing::TestSuite*>& CompositeOpSuites() {
  static auto* suites = new std::vector<const ::testing::TestSuite*>();
  return *suites;
}

std::vector<const ::testing::TestSuite*> CollectNewSuites(int& next_suite_idx) {
  const auto* unit_test = ::testing::UnitTest::GetInstance();
  std::vector<const ::testing::TestSuite*> suites;
  for (; next_suite_idx < unit_test->total_test_suite_count();
       ++next_suite_idx) {
    suites.push_back(unit_test->GetTestSuite(next_suite_idx));
  }
  return suites;
}

// Returns the set of formatted op names (e.g. "tfl.add") that satisfy
// IsCoreSingleOp(op_code).
const absl::flat_hash_set<std::string>& CoreSingleOpNames() {
  static const auto* names = [] {
    auto* set = new absl::flat_hash_set<std::string>();
    for (int code = 0; code <= kLiteRtOpCodeShloComposite; ++code) {
      const auto op_code = static_cast<LiteRtOpCode>(code);
      if (IsCoreSingleOp(op_code)) {
        set->insert(absl::StrFormat("%v", op_code));
      }
    }
    return set;
  }();
  return *names;
}

// Extracts the op name before '{' or '(' from a test name like
// "tfl.add{fused_activation_function=NONE}(2d_f32,2d_f32)->(2d_f32)".
absl::string_view ExtractOpName(absl::string_view test_name) {
  const size_t end = test_name.find_first_of("{(");
  return end == absl::string_view::npos ? test_name : test_name.substr(0, end);
}

void VerifySuiteTestsNormalizedAndDeduplicated(
    const ::testing::TestSuite* suite) {
  EXPECT_GT(suite->total_test_count(), 0);
  absl::flat_hash_set<absl::string_view> seen_names;
  for (int i = 0; i < suite->total_test_count(); ++i) {
    absl::string_view test_name = suite->GetTestInfo(i)->name();
    EXPECT_THAT(test_name, Not(StartsWith(suite->name())))
        << "Test method name " << test_name
        << " must not redundantly repeat the suite name " << suite->name();
    EXPECT_THAT(test_name, Not(HasSubstr("<")))
        << "Test method name " << test_name
        << " must not contain concrete random dimensions '<...>'";
    EXPECT_THAT(seen_names, Not(Contains(test_name)))
        << "Duplicate GTest test method name " << test_name << " in suite "
        << suite->name();
    seen_names.insert(test_name);
  }
}

TEST(RegisterOpsContractTest, SingleOpsUseCoreOrStandardSingleOpPrefix) {
  size_t core_suite_count = 0;
  size_t single_suite_count = 0;

  for (const auto* suite : SingleOpSuites()) {
    absl::string_view suite_name = suite->name();
    const bool is_core = absl::StartsWith(suite_name, "CoreSingleOp_");
    const bool is_single = absl::StartsWith(suite_name, "SingleOp_");
    ASSERT_TRUE(is_core || is_single)
        << "Suite registered by RegisterSingleOps has unexpected prefix: "
        << suite_name;

    if (is_core) {
      ++core_suite_count;
    } else {
      ++single_suite_count;
    }

    for (int i = 0; i < suite->total_test_count(); ++i) {
      absl::string_view test_name = suite->GetTestInfo(i)->name();
      absl::string_view op_name = ExtractOpName(test_name);
      EXPECT_EQ(is_core, CoreSingleOpNames().contains(op_name))
          << "Suite " << suite_name << " has mismatched prefix for op "
          << op_name << " (test: " << test_name << ")";
    }
  }

  EXPECT_GT(core_suite_count, 0);
  EXPECT_GT(single_suite_count, 0);
}

TEST(RegisterOpsContractTest, CompositeOpsUseCompositeOpPrefix) {
  EXPECT_FALSE(CompositeOpSuites().empty());
  for (const auto* suite : CompositeOpSuites()) {
    EXPECT_THAT(suite->name(), StartsWith("CompositeOp_"));
  }
}

TEST(RegisterOpsContractTest,
     TestMethodNamesAreNormalizedAndDeduplicatedForTestGrid) {
  for (const auto* suite : SingleOpSuites()) {
    VerifySuiteTestsNormalizedAndDeduplicated(suite);
  }
  for (const auto* suite : CompositeOpSuites()) {
    VerifySuiteTestsNormalizedAndDeduplicated(suite);
  }
}

}  // namespace
}  // namespace litert::testing

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  auto options = litert::testing::AtsConf::ParseFlagsAndDoSetup();
  ABSL_CHECK(options.HasValue())
      << "Failed to parse ATS options: " << options.Error().Message();

  int next_suite_idx =
      ::testing::UnitTest::GetInstance()->total_test_suite_count();
  size_t test_id = 0;
  litert::testing::AtsInferenceTest::Capture i_cap;
  litert::testing::AtsCompileTest::Capture c_cap;

  litert::testing::RegisterSingleOps(*options, test_id, i_cap);
  litert::testing::RegisterSingleOps(*options, test_id, c_cap);
  litert::testing::SingleOpSuites() =
      litert::testing::CollectNewSuites(next_suite_idx);

  litert::testing::RegisterCompositeOps(*options, test_id, i_cap);
  litert::testing::RegisterCompositeOps(*options, test_id, c_cap);
  litert::testing::CompositeOpSuites() =
      litert::testing::CollectNewSuites(next_suite_idx);

  // Filter GoogleTest execution to ONLY run the contract validation tests.
  // This prevents the actual generated models from executing during the
  // contract check.
  ::testing::GTEST_FLAG(filter) = "RegisterOpsContractTest.*";

  return RUN_ALL_TESTS();
}
