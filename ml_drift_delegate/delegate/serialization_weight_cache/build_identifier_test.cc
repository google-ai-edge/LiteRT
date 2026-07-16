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

#include "ml_drift_delegate/delegate/serialization_weight_cache/build_identifier.h"

#include <cstdint>
#include <vector>

#include "testing/base/public/gunit.h"
#include "absl/types/span.h"  // from @com_google_absl

namespace ml_drift {
namespace {

TEST(BuildIdentifierTest, BuildIdentifierSize) {
  EXPECT_EQ(ml_drift::GetBuildIdentifier().size(), 32);
}

TEST(BuildIdentifierTest, CheckBuildIdentifierSuccess) {
  absl::Span<const uint8_t> build_identifier = ml_drift::GetBuildIdentifier();
  EXPECT_EQ(ml_drift::CheckBuildIdentifier(build_identifier), true);
}

TEST(BuildIdentifierTest, CheckBuildIdentifierFailure) {
  absl::Span<const uint8_t> build_identifier = ml_drift::GetBuildIdentifier();
  // Double check that the build identifier is the expected size.
  EXPECT_EQ(build_identifier.size(), 32);

  // Create a copy of the build identifier and modify the first byte of it.
  std::vector<uint8_t> build_identifier_data(build_identifier.begin(),
                                             build_identifier.end());
  build_identifier_data[0] += 1;
  absl::Span<uint8_t> build_identifier_copy = absl::MakeSpan(
      build_identifier_data.data(), build_identifier_data.size());

  // Check that the build identifier no longer matches.
  EXPECT_EQ(ml_drift::CheckBuildIdentifier(build_identifier_copy), false);
}

}  // namespace

}  // namespace ml_drift
