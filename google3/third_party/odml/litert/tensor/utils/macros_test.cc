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

#include "third_party/odml/litert/tensor/utils/macros.h"

#include <sstream>
#include <utility>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl

namespace litert {
namespace {

using testing::AllOf;
using testing::HasSubstr;

TEST(LiteRtReturnIfErrorTest, ConvertsResultToStatus) {
  EXPECT_THAT(
      []() -> absl::Status {
        LRT_TENSOR_RETURN_IF_ERROR(
            absl::StatusOr<int>(absl::NotFoundError("")));
        return absl::OkStatus();
      }(),
      absl_testing::StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(
      []() -> absl::Status {
        LRT_TENSOR_RETURN_IF_ERROR(absl::NotFoundError(""));
        return absl::OkStatus();
      }(),
      absl_testing::StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(
      []() -> absl::Status {
        LRT_TENSOR_RETURN_IF_ERROR(absl::NotFoundError(""));
        return absl::OkStatus();
      }(),
      absl_testing::StatusIs(absl::StatusCode::kNotFound));
  EXPECT_EQ(
      []() -> absl::Status {
        LRT_TENSOR_RETURN_IF_ERROR(true);
        return absl::OkStatus();
      }(),
      absl::OkStatus());
  EXPECT_THAT(
      []() -> absl::Status {
        LRT_TENSOR_RETURN_IF_ERROR(false);
        return absl::OkStatus();
      }(),
      absl_testing::StatusIs(absl::StatusCode::kUnknown));
}

TEST(LiteRtReturnIfErrorTest, ConvertsResultToExpectedHoldingAnError) {
  EXPECT_THAT(
      []() -> absl::StatusOr<int> {
        LRT_TENSOR_RETURN_IF_ERROR(
            absl::StatusOr<int>(absl::NotFoundError("")));
        return 1;
      }(),
      absl_testing::StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(
      []() -> absl::StatusOr<int> {
        LRT_TENSOR_RETURN_IF_ERROR(true);
        return 1;
      }(),
      absl_testing::IsOkAndHolds(1));
  EXPECT_THAT(
      []() -> absl::StatusOr<int> {
        LRT_TENSOR_RETURN_IF_ERROR(false);
        return 1;
      }(),
      absl_testing::StatusIs(absl::StatusCode::kUnknown));
  EXPECT_THAT(
      []() -> absl::StatusOr<int> {
        LRT_TENSOR_RETURN_IF_ERROR(false) << "Extra message";
        return 1;
      }(),
      absl_testing::StatusIs(absl::StatusCode::kUnknown,
                             HasSubstr("Extra message")));
}

TEST(LiteRtReturnIfErrorTest, DoesntReturnOnSuccess) {
  int canary_value = 0;
  auto ReturnExpectedIfError = [&canary_value]() -> absl::StatusOr<int> {
    LRT_TENSOR_RETURN_IF_ERROR(absl::OkStatus());
    canary_value = 1;
    return 1;
  };
  EXPECT_THAT(ReturnExpectedIfError(), absl_testing::IsOk());
  EXPECT_EQ(canary_value, 1);

  EXPECT_THAT(
      [&canary_value]() -> absl::Status {
        LRT_TENSOR_RETURN_IF_ERROR(absl::OkStatus());
        canary_value = 2;
        return absl::OkStatus();
      }(),
      absl_testing::IsOk());
  EXPECT_EQ(canary_value, 2);
}

TEST(LiteRtReturnIfErrorTest, ExtraLoggingWorks) {
  int canary_value = 0;
  EXPECT_THAT(
      [&canary_value]() -> absl::Status {
        LRT_TENSOR_RETURN_IF_ERROR(false)
            << "Successful default level logging.";
        canary_value = 2;
        return absl::OkStatus();
      }(),
      absl_testing::StatusIs(absl::StatusCode::kUnknown,
                             HasSubstr("Successful default level logging.")));
  EXPECT_EQ(canary_value, 0);
}

TEST(LiteRtAssignOrReturnTest, VariableAssignmentWorks) {
  int canary_value = 0;
  auto ChangeCanaryValue = [&canary_value]() -> absl::Status {
    LRT_TENSOR_ASSIGN_OR_RETURN(canary_value, absl::StatusOr<int>(1));
    return absl::OkStatus();
  };
  EXPECT_EQ(ChangeCanaryValue(), absl::OkStatus());
  EXPECT_EQ(canary_value, 1);
}

TEST(LiteRtAssignOrReturnTest, StatusOrHoldingReferenceWorks) {
  int origin_value = 1;
  int canary_value = 0;
  auto ChangeCanaryValue = [&canary_value, &origin_value]() -> absl::Status {
    LRT_TENSOR_ASSIGN_OR_RETURN(int& assigned_value,
                                absl::StatusOr<int&>(origin_value));
    canary_value = assigned_value;
    assigned_value = 2;
    return absl::OkStatus();
  };
  EXPECT_EQ(ChangeCanaryValue(), absl::OkStatus());
  EXPECT_EQ(canary_value, 1);
  EXPECT_EQ(origin_value, 2);
}

TEST(LiteRtAssignOrReturnTest, MoveOnlyVariableAssignmentWorks) {
  struct MoveOnly {
    explicit MoveOnly(int val) : val(val) {};
    MoveOnly(const MoveOnly&) = delete;
    MoveOnly& operator=(const MoveOnly&) = delete;
    MoveOnly(MoveOnly&&) = default;
    MoveOnly& operator=(MoveOnly&&) = default;
    int val = 1;
  };

  MoveOnly canary_value{0};
  auto ChangeCanaryValue = [&canary_value]() -> absl::Status {
    LRT_TENSOR_ASSIGN_OR_RETURN(canary_value, absl::StatusOr<MoveOnly>(1));
    return absl::OkStatus();
  };
  EXPECT_EQ(ChangeCanaryValue(), absl::OkStatus());
  EXPECT_EQ(canary_value.val, 1);
}

TEST(LiteRtAssignOrReturnTest, ReturnsOnFailure) {
  absl::StatusOr<int> kInvalidArgumentError = absl::InvalidArgumentError("");

  int canary_value = 0;
  auto ErrorWithStatus = [&]() -> absl::Status {
    LRT_TENSOR_ASSIGN_OR_RETURN(canary_value, kInvalidArgumentError);
    return absl::OkStatus();
  };
  EXPECT_THAT(ErrorWithStatus(),
              absl_testing::StatusIs(kInvalidArgumentError.status().code()));
  EXPECT_EQ(canary_value, 0);

  auto ErrorWithCustomStatus = [&]() -> int {
    LRT_TENSOR_ASSIGN_OR_RETURN(canary_value, kInvalidArgumentError, 42);
    return 1;
  };
  EXPECT_EQ(ErrorWithCustomStatus(), 42);
  EXPECT_EQ(canary_value, 0);

  auto ErrorWithExpected = [&]() -> absl::StatusOr<int> {
    LRT_TENSOR_ASSIGN_OR_RETURN(canary_value, kInvalidArgumentError);
    return 1;
  };
  auto expected_return = ErrorWithExpected();
  ASSERT_FALSE(expected_return.ok());
  EXPECT_THAT(expected_return,
              absl_testing::StatusIs(kInvalidArgumentError.status().code()));
  EXPECT_EQ(canary_value, 0);
}

TEST(LiteRtAssignOrReturnTest, AllowsStructuredBindings) {
  const std::pair p(1, "a");
  absl::StatusOr<decltype(p)> e(p);
  auto Function = [&]() -> absl::StatusOr<std::pair<int, const char*>> {
    LRT_TENSOR_ASSIGN_OR_RETURN((auto [i, c]), e);
    EXPECT_EQ(i, p.first);
    EXPECT_EQ(c, p.second);
    return e;
  };
  EXPECT_THAT(Function(), absl_testing::IsOk());
}

TEST(LiteRtAbortIfErrorTest, DoesntDieWithSuccessValues) {
  LRT_TENSOR_ABORT_IF_ERROR(absl::OkStatus());
  LRT_TENSOR_ABORT_IF_ERROR(true);
}

TEST(LiteRtAbortIfErrorTest, DiesWithErrorValue) {
  absl::StatusOr<int> InvalidArgumentError =
      absl::InvalidArgumentError("Unexpected message");
  EXPECT_DEATH(
      LRT_TENSOR_ABORT_IF_ERROR(InvalidArgumentError) << "Error abort log",
#ifndef NDEBUG
      AllOf(HasSubstr("Error abort log"), HasSubstr("Unexpected message"))
#else
      ""
#endif
  );
}

TEST(LiteRtAssignOrAbortTest, WorksWithValidExpected) {
  LRT_TENSOR_ASSIGN_OR_ABORT(int v, absl::StatusOr<int>(3));
  EXPECT_EQ(v, 3);
}

TEST(LiteRtAssignOrAbortTest, AllowsStructuredBindings) {
  const std::pair p(1, "a");
  absl::StatusOr<decltype(p)> e(p);
  LRT_TENSOR_ASSIGN_OR_ABORT((auto [i, c]), e);
  EXPECT_EQ(i, p.first);
  EXPECT_EQ(c, p.second);
}

TEST(LiteRtAssignOrAbortTest, DiesWithError) {
  absl::StatusOr<int> InvalidArgumentError =
      absl::InvalidArgumentError("Unexpected message");
  EXPECT_DEATH(
      LRT_TENSOR_ASSIGN_OR_ABORT([[maybe_unused]] int v, InvalidArgumentError),
#ifndef NDEBUG
      "Unexpected message"
#else
      ""
#endif
  );
}

TEST(LiteRtAssignOrAbortTest, DiesWithErrorAndCustomMessage) {
  absl::StatusOr<int> InvalidArgumentError =
      absl::InvalidArgumentError("Unexpected message");
  EXPECT_DEATH(
      LRT_TENSOR_ASSIGN_OR_ABORT([[maybe_unused]] int v, InvalidArgumentError,
                                 _ << "Error abort log"),
#ifndef NDEBUG
      AllOf(HasSubstr("Error abort log"), HasSubstr("Unexpected message"))
#else
      ""
#endif
  );
}

TEST(LiteRtErrorStatusBuilderTest, BacktraceWorks) {
  const int error_1_line = __LINE__ + 2;
  auto error_1 = []() -> absl::StatusOr<int> {
    LRT_TENSOR_RETURN_IF_ERROR(absl::UnknownError("An error message."));
    return 1;
  };

  const int error_2_line = __LINE__ + 2;
  auto error_2 = [&]() -> absl::StatusOr<int> {
    LRT_TENSOR_RETURN_IF_ERROR(error_1());
    return 1;
  };

  const int error_3_line = __LINE__ + 2;
  auto error_3 = [&]() -> absl::StatusOr<int> {
    LRT_TENSOR_RETURN_IF_ERROR(error_2()) << "An extra message.";
    return 1;
  };

  const absl::StatusOr<int> res = error_3();
  ASSERT_FALSE(res.ok());
  std::stringstream error_message_builder;
  error_message_builder.str("");
  error_message_builder << "[" << __FILE__ << ":" << error_1_line << "]";
  EXPECT_THAT(res.status().message(), HasSubstr(error_message_builder.str()));

  error_message_builder.str("");
  error_message_builder << "[" << __FILE__ << ":" << error_2_line << "]";
  EXPECT_THAT(res.status().message(), HasSubstr(error_message_builder.str()));

  error_message_builder.str("");
  error_message_builder << "[" << __FILE__ << ":" << error_3_line
                        << "] An extra message.";
  EXPECT_THAT(res.status().message(), HasSubstr(error_message_builder.str()));

  EXPECT_THAT(res.status().message(), HasSubstr("An error message."));
}

}  // namespace
}  // namespace litert
