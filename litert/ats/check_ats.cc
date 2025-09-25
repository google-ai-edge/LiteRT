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

#include <cstddef>
#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/ats/configure.h"
#include "litert/ats/fixture.h"
#include "litert/ats/register.h"
#include "litert/cc/litert_detail.h"
#include "litert/cc/litert_expected.h"
#include "litert/test/common.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/no_op.h"

// Simple validatino logic for the registration of ATS tests. We cannot use
// gtest constructs for this.

namespace litert::testing {
namespace {

int CheckAts() {
  const std::string do_register = "(ats_1_|ats_2_)";
  absl::SetFlag(&FLAGS_do_register, do_register);
  const std::string extra_models_path = GetLiteRtPath("test/testdata/");
  absl::SetFlag(&FLAGS_extra_models, extra_models_path);
  size_t test_id = 0;
  auto options = litert::testing::AtsConf::ParseFlagsAndDoSetup();
  ABSL_CHECK(options);
  RegisterExtraModels<AtsTest>(test_id, *options);
  static constexpr auto kIters = 1;
  RegisterCombinations<AtsTest, NoOp, SizeListC<1>, TypeList<float, int32_t>>(
      kIters, test_id, *options);
  const auto* unit_test = ::testing::UnitTest::GetInstance();
  ABSL_CHECK_EQ(unit_test->total_test_count(), 3);
  return 0;
}

}  // namespace
}  // namespace litert::testing

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return litert::testing::CheckAts();
}
