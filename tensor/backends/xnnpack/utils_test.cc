/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/backends/xnnpack/utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xnnpack.h"  // from @XNNPACK
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/types/source_location.h"  // from @com_google_absl
#include "tensor/utils/macros.h"

namespace litert::tensor {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Property;

TEST(XnnStatusToAbslTest, RecordsCallerLocation) {
  const int line = __LINE__ + 1;
  const absl::Status status = XnnStatusToAbsl(xnn_status_invalid_state, "op");

  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInternal, HasSubstr("op")));
  EXPECT_THAT(status.GetSourceLocations(),
              ElementsAre(Property(&absl::SourceLocation::line, line)));
}

TEST(XnnStatusToAbslTest, SuccessIsOk) {
  EXPECT_THAT(XnnStatusToAbsl(xnn_status_success, "op"), IsOk());
}

TEST(XnnStatusErrorConversionTest, MacroErrorIsCreatedAtCallSite) {
  const int line = __LINE__ + 2;
  auto propagate = []() -> absl::Status {
    LRT_TENSOR_RETURN_IF_ERROR(xnn_status_invalid_parameter);
    return absl::OkStatus();
  };
  const absl::Status status = propagate();

  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(status.GetSourceLocations(),
              ElementsAre(Property(&absl::SourceLocation::line, line),
                          Property(&absl::SourceLocation::line, line)));
}

TEST(XnnStatusErrorConversionTest, ConvertsStatusToXnnStatus) {
  auto convert = [](absl::Status error) -> xnn_status {
    LRT_TENSOR_RETURN_IF_ERROR(error);
    return xnn_status_success;
  };
  EXPECT_EQ(convert(absl::OkStatus()), xnn_status_success);
  EXPECT_EQ(convert(absl::InvalidArgumentError("x")),
            xnn_status_invalid_parameter);
  EXPECT_EQ(convert(absl::UnimplementedError("x")),
            xnn_status_unsupported_parameter);
  EXPECT_EQ(convert(absl::ResourceExhaustedError("x")),
            xnn_status_out_of_memory);
  EXPECT_EQ(convert(absl::InternalError("x")), xnn_status_invalid_state);
}

TEST(XnnStatusErrorConversionTest, ConvertsAnnotatedStatusToXnnStatus) {
  auto convert = []() -> xnn_status {
    LRT_TENSOR_RETURN_IF_ERROR(absl::InternalError("x"))
            .SetCode(absl::StatusCode::kInvalidArgument)
        << "Extra.";
    return xnn_status_success;
  };
  EXPECT_EQ(convert(), xnn_status_invalid_parameter);
}

}  // namespace
}  // namespace litert::tensor
