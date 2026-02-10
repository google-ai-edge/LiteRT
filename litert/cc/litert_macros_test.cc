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

#include "litert/cc/litert_macros.h"

#include <sstream>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_logging.h"
#include "litert/cc/litert_expected.h"
#include "litert/test/matchers.h"

namespace litert {
namespace {

using testing::AllOf;
using testing::HasSubstr;
using testing::Property;
using testing::litert::IsError;

TEST(LiteRtReturnIfErrorTest, ConvertsResultToLiteRtStatus) {
  EXPECT_EQ(
      []() -> LiteRtStatus {
        LITERT_RETURN_IF_ERROR(
            Expected<int>(Unexpected(kLiteRtStatusErrorNotFound)));
        return kLiteRtStatusOk;
      }(),
      kLiteRtStatusErrorNotFound);
  EXPECT_EQ(
      []() -> LiteRtStatus {
        LITERT_RETURN_IF_ERROR(Unexpected(kLiteRtStatusErrorNotFound));
        return kLiteRtStatusOk;
      }(),
      kLiteRtStatusErrorNotFound);
  EXPECT_EQ(
      []() -> LiteRtStatus {
        LITERT_RETURN_IF_ERROR(kLiteRtStatusErrorNotFound);
        return kLiteRtStatusOk;
      }(),
      kLiteRtStatusErrorNotFound);
  EXPECT_EQ(
      []() -> LiteRtStatus {
        LITERT_RETURN_IF_ERROR(true);
        return kLiteRtStatusOk;
      }(),
      kLiteRtStatusOk);
  EXPECT_EQ(
      []() -> LiteRtStatus {
        LITERT_RETURN_IF_ERROR(false);
        return kLiteRtStatusOk;
      }(),
      kLiteRtStatusErrorUnknown);
}

TEST(LiteRtReturnIfErrorTest, ConvertsResultToExpectedHoldingAnError) {
  EXPECT_THAT(
      []() -> Expected<void> {
        LITERT_RETURN_IF_ERROR(
            Expected<void>(Unexpected(kLiteRtStatusErrorNotFound)));
        return {};
      }(),
      AllOf(Property(&Expected<void>::HasValue, false),
            Property(&Expected<void>::Error,
                     Property(&Error::Status, kLiteRtStatusErrorNotFound))));
  EXPECT_THAT(
      []() -> Expected<void> {
        LITERT_RETURN_IF_ERROR(Unexpected(kLiteRtStatusErrorNotFound));
        return {};
      }(),
      AllOf(Property(&Expected<void>::HasValue, false),
            Property(&Expected<void>::Error,
                     Property(&Error::Status, kLiteRtStatusErrorNotFound))));
  EXPECT_THAT(
      []() -> Expected<void> {
        LITERT_RETURN_IF_ERROR(kLiteRtStatusErrorNotFound);
        return {};
      }(),
      AllOf(Property(&Expected<void>::HasValue, false),
            Property(&Expected<void>::Error,
                     Property(&Error::Status, kLiteRtStatusErrorNotFound))));
  EXPECT_THAT(
      []() -> Expected<void> {
        LITERT_RETURN_IF_ERROR(true);
        return {};
      }(),
      Property(&Expected<void>::HasValue, true));
  EXPECT_THAT(
      []() -> Expected<void> {
        LITERT_RETURN_IF_ERROR(false);
        return {};
      }(),
      AllOf(Property(&Expected<void>::HasValue, false),
            Property(&Expected<void>::Error,
                     Property(&Error::Status, kLiteRtStatusErrorUnknown))));
  EXPECT_THAT(
      []() -> Expected<void> {
        LITERT_RETURN_IF_ERROR(false) << "Extra message";
        return {};
      }(),
      AllOf(Property(&Expected<void>::HasValue, false),
            Property(&Expected<void>::Error,
                     Property(&Error::Status, kLiteRtStatusErrorUnknown)),
            Property(&Expected<void>::Error,
                     Property(&Error::Message, HasSubstr("Extra message")))));
}

TEST(LiteRtReturnIfErrorTest, DoesntReturnOnSuccess) {
  int canary_value = 0;
  auto ReturnExpectedIfError = [&canary_value]() -> Expected<void> {
    LITERT_RETURN_IF_ERROR(Expected<void>());
    canary_value = 1;
    return {};
  };
  EXPECT_THAT(ReturnExpectedIfError(),
              Property(&Expected<void>::HasValue, true));
  EXPECT_EQ(canary_value, 1);

  [&canary_value]() -> LiteRtStatus {
    LITERT_RETURN_IF_ERROR(kLiteRtStatusOk);
    canary_value = 2;
    return kLiteRtStatusOk;
  }();
  EXPECT_EQ(canary_value, 2);
}

TEST(LiteRtReturnIfErrorTest, ExtraLoggingWorks) {
  int canary_value = 0;
  [&canary_value]() -> LiteRtStatus {
    LITERT_RETURN_IF_ERROR(false) << "Successful default level logging.";
    canary_value = 2;
    return kLiteRtStatusOk;
  }();
  EXPECT_EQ(canary_value, 0);

  canary_value = 0;
  [&canary_value]() -> LiteRtStatus {
    LITERT_RETURN_IF_ERROR(false).LogVerbose() << "Successful verbose logging.";
    canary_value = 2;
    return kLiteRtStatusOk;
  }();
  EXPECT_EQ(canary_value, 0);

  canary_value = 0;
  [&canary_value]() -> LiteRtStatus {
    LITERT_RETURN_IF_ERROR(false).LogInfo() << "Successful info logging.";
    canary_value = 2;
    return kLiteRtStatusOk;
  }();
  EXPECT_EQ(canary_value, 0);

  canary_value = 0;
  [&canary_value]() -> LiteRtStatus {
    LITERT_RETURN_IF_ERROR(false).LogWarning() << "Successful warning logging.";
    canary_value = 2;
    return kLiteRtStatusOk;
  }();
  EXPECT_EQ(canary_value, 0);

  canary_value = 0;
  [&canary_value]() -> LiteRtStatus {
    LITERT_RETURN_IF_ERROR(false).LogError() << "Successful error logging.";
    canary_value = 2;
    return kLiteRtStatusOk;
  }();
  EXPECT_EQ(canary_value, 0);

  canary_value = 0;
  [&canary_value]() -> LiteRtStatus {
    LITERT_RETURN_IF_ERROR(false).NoLog() << "This should never be printed";
    canary_value = 2;
    return kLiteRtStatusOk;
  }();
  EXPECT_EQ(canary_value, 0);
}

TEST(LiteRtAssignOrReturnTest, VariableAssignmentWorks) {
  int canary_value = 0;
  auto ChangeCanaryValue = [&canary_value]() -> LiteRtStatus {
    LITERT_ASSIGN_OR_RETURN(canary_value, Expected<int>(1));
    return kLiteRtStatusOk;
  };
  EXPECT_EQ(ChangeCanaryValue(), kLiteRtStatusOk);
  EXPECT_EQ(canary_value, 1);
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
  auto ChangeCanaryValue = [&canary_value]() -> LiteRtStatus {
    LITERT_ASSIGN_OR_RETURN(canary_value, Expected<MoveOnly>(1));
    return kLiteRtStatusOk;
  };
  EXPECT_EQ(ChangeCanaryValue(), kLiteRtStatusOk);
  EXPECT_EQ(canary_value.val, 1);
}

TEST(LiteRtAssignOrReturnTest, ReturnsOnFailure) {
  Expected<int> InvalidArgumentError =
      Expected<int>(Unexpected(kLiteRtStatusErrorInvalidArgument));

  int canary_value = 0;
  auto ErrorWithStatus = [&]() -> LiteRtStatus {
    LITERT_ASSIGN_OR_RETURN(canary_value, InvalidArgumentError);
    return kLiteRtStatusOk;
  };
  EXPECT_EQ(ErrorWithStatus(), kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(canary_value, 0);

  auto ErrorWithCustomStatus = [&]() -> int {
    LITERT_ASSIGN_OR_RETURN(canary_value, InvalidArgumentError, 42);
    return 1;
  };
  EXPECT_EQ(ErrorWithCustomStatus(), 42);
  EXPECT_EQ(canary_value, 0);

  auto ErrorWithExpected = [&]() -> Expected<void> {
    LITERT_ASSIGN_OR_RETURN(canary_value, InvalidArgumentError);
    return {};
  };
  auto expected_return = ErrorWithExpected();
  ASSERT_FALSE(expected_return.HasValue());
  EXPECT_EQ(expected_return.Error().Status(),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(canary_value, 0);
}

TEST(LiteRtAssignOrReturnTest, AllowsStructuredBindings) {
  Expected<std::pair<int, const char*>> e(std::pair(1, "a"));
  auto Function = [&]() -> Expected<std::pair<int, const char*>> {
    LITERT_ASSIGN_OR_RETURN((auto [i, c]), e);
    EXPECT_EQ(i, e.Value().first);
    EXPECT_EQ(c, e.Value().second);
    return e;
  };
  LITERT_EXPECT_OK(Function());
}

TEST(LiteRtAbortIfErrorTest, DoesntDieWithSuccessValues) {
  LITERT_ABORT_IF_ERROR(kLiteRtStatusOk);
  LITERT_ABORT_IF_ERROR(true);
}

TEST(LiteRtAbortIfErrorTest, DiesWithErrorValue) {
  Expected<int> InvalidArgumentError = Expected<int>(
      Unexpected(kLiteRtStatusErrorInvalidArgument, "Unexpected message"));
  EXPECT_DEATH(
      LITERT_ABORT_IF_ERROR(InvalidArgumentError) << "Error abort log",
#ifndef NDEBUG
      AllOf(HasSubstr("Error abort log"), HasSubstr("Unexpected message"))
#else
      ""
#endif
  );
}

TEST(LiteRtAssignOrAbortTest, WorksWithValidExpected) {
  LITERT_ASSIGN_OR_ABORT(int v, Expected<int>(3));
  EXPECT_EQ(v, 3);
}

TEST(LiteRtAssignOrAbortTest, AllowsStructuredBindings) {
  Expected<std::pair<int, const char*>> e(std::pair(1, "a"));
  LITERT_ASSIGN_OR_ABORT((auto [i, c]), e);
  EXPECT_EQ(i, e.Value().first);
  EXPECT_EQ(c, e.Value().second);
}

TEST(LiteRtAssignOrAbortTest, DiesWithError) {
  Expected<int> InvalidArgumentError = Expected<int>(
      Unexpected(kLiteRtStatusErrorInvalidArgument, "Unexpected message"));
  EXPECT_DEATH(
      LITERT_ASSIGN_OR_ABORT([[maybe_unused]] int v, InvalidArgumentError),
#ifndef NDEBUG
      "Unexpected message"
#else
      ""
#endif
  );
}

TEST(LiteRtAssignOrAbortTest, DiesWithErrorAndCustomMessage) {
  Expected<int> InvalidArgumentError = Expected<int>(
      Unexpected(kLiteRtStatusErrorInvalidArgument, "Unexpected message"));
  EXPECT_DEATH(
      LITERT_ASSIGN_OR_ABORT([[maybe_unused]] int v, InvalidArgumentError,
                             _ << "Error abort log"),
#ifndef NDEBUG
      AllOf(HasSubstr("Error abort log"), HasSubstr("Unexpected message"))
#else
      ""
#endif
  );
}

TEST(LiteRtErrorStatusBuilderTest, BacktraceWorks) {
  const int error_1_line = __LINE__ + 3;
  auto error_1 = []() -> Expected<void> {
    LITERT_RETURN_IF_ERROR(
        Unexpected(kLiteRtStatusErrorUnknown, "An error message."));
    return {};
  };

  const int error_2_line = __LINE__ + 2;
  auto error_2 = [&]() -> Expected<void> {
    LITERT_RETURN_IF_ERROR(error_1());
    return {};
  };

  const int error_3_line = __LINE__ + 2;
  auto error_3 = [&]() -> Expected<void> {
    LITERT_RETURN_IF_ERROR(error_2()) << "An extra message.";
    return {};
  };

  const Expected<void> res = error_3();
  ASSERT_THAT(res, IsError(kLiteRtStatusErrorUnknown));
  std::stringstream error_message_builder;
  error_message_builder.str("");
  error_message_builder << "ERROR: [" << __FILE__ << ":" << error_1_line << "]";
  EXPECT_THAT(res.Error().Message(), HasSubstr(error_message_builder.str()));

  error_message_builder.str("");
  error_message_builder << "ERROR: [" << __FILE__ << ":" << error_2_line << "]";
  EXPECT_THAT(res.Error().Message(), HasSubstr(error_message_builder.str()));

  error_message_builder.str("");
  error_message_builder << "ERROR: [" << __FILE__ << ":" << error_3_line
                        << "] An extra message.";
  EXPECT_THAT(res.Error().Message(), HasSubstr(error_message_builder.str()));

  EXPECT_THAT(res.Error().Message(), HasSubstr("An error message."));
}

TEST(LiteRtErrorStatusBuilderTest, CastToLiteRtStatusLogsError) {
#ifdef NDEBUG
  GTEST_SKIP() << "Logging is disabled in optimized builds";
#endif
  litert::InterceptLogs log_interceptor;
  auto error_1 = []() -> LiteRtStatus {
    LITERT_RETURN_IF_ERROR(
        Unexpected(kLiteRtStatusErrorUnknown, "An error message."))
        << "Failed a subcall.";
    return kLiteRtStatusOk;
  };
  EXPECT_THAT(error_1(), IsError(kLiteRtStatusErrorUnknown));
  std::stringstream intercepted_logs;
  intercepted_logs << log_interceptor;
  EXPECT_THAT(intercepted_logs.str(), HasSubstr("Failed a subcall."));
  EXPECT_THAT(intercepted_logs.str(), HasSubstr("An error message."));
}

TEST(LiteRtErrorStatusBuilderTest, ConvertToAbslStatus) {
  auto error = []() -> absl::Status {
    LITERT_RETURN_IF_ERROR(
        Unexpected(kLiteRtStatusErrorInvalidArgument, "An error message."));
    return absl::OkStatus();
  }();
  EXPECT_FALSE(error.ok());
  EXPECT_EQ(error.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(error.message(), HasSubstr("An error message."));
}

TEST(LiteRtErrorStatusBuilderTest, ConvertToAbslStatusOr) {
  auto error_1 = []() -> absl::StatusOr<int> {
    LITERT_RETURN_IF_ERROR(
        Unexpected(kLiteRtStatusErrorInvalidArgument, "An error message."));
    return 1;
  }();
  EXPECT_FALSE(error_1.ok());
  EXPECT_EQ(error_1.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(error_1.status().message(), HasSubstr("An error message."));

  auto error_2 = []() -> absl::StatusOr<int> {
    LITERT_ASSIGN_OR_RETURN(
        auto v, Expected<int>(Unexpected(kLiteRtStatusErrorInvalidArgument,
                                         "An error message.")));
    return v;
  }();
  EXPECT_FALSE(error_2.ok());
  EXPECT_EQ(error_2.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(error_2.status().message(), HasSubstr("An error message."));
}

}  // namespace
}  // namespace litert
