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

#include "tensor/utils/macros.h"

#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_builder.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/cord.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/source_location.h"  // from @com_google_absl

namespace litert::tensor {
namespace {

// Error type used to test `ErrorStatusBuilder::ErrorConversion`.
enum class TestCode { kOk, kInvalidArgument, kOther };

}  // namespace

template <>
struct ErrorStatusBuilder::ErrorConversion<TestCode> {
  static bool IsError(TestCode code) { return code != TestCode::kOk; }
  static absl::Status AsError(TestCode /*code*/) {
    return absl::InternalError("TestCode error");
  }
  static TestCode FromError(const absl::Status& status, absl::SourceLocation) {
    return status.code() == absl::StatusCode::kInvalidArgument
               ? TestCode::kInvalidArgument
               : TestCode::kOther;
  }
};

}  // namespace litert::tensor

namespace litert {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::litert::tensor::TestCode;
using testing::AllOf;
using testing::ElementsAre;
using testing::EndsWith;
using testing::HasSubstr;

constexpr char kPayloadUrl[] = "type.googleapis.com/litert.test";

// Returns the status source location trace as `file:line` strings.
std::vector<std::string> Trace(const absl::Status& status) {
  std::vector<std::string> trace;
  for (const absl::SourceLocation& loc : status.GetSourceLocations()) {
    trace.push_back(absl::StrCat(loc.file_name(), ":", loc.line()));
  }
  return trace;
}

// Returns the `file:line` string for a line of this file.
std::string At(int line) { return absl::StrCat(__FILE__, ":", line); }

TEST(LiteRtReturnIfErrorTest, ConvertsResultToStatus) {
  EXPECT_THAT(
      []() -> absl::Status {
        LRT_TENSOR_RETURN_IF_ERROR(
            absl::StatusOr<int>(absl::NotFoundError("")));
        return absl::OkStatus();
      }(),
      StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(
      []() -> absl::Status {
        LRT_TENSOR_RETURN_IF_ERROR(absl::NotFoundError(""));
        return absl::OkStatus();
      }(),
      StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(
      []() -> absl::Status {
        LRT_TENSOR_RETURN_IF_ERROR(absl::NotFoundError(""));
        return absl::OkStatus();
      }(),
      StatusIs(absl::StatusCode::kNotFound));
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
      StatusIs(absl::StatusCode::kUnknown));
}

TEST(LiteRtReturnIfErrorTest, ConvertsResultToExpectedHoldingAnError) {
  EXPECT_THAT(
      []() -> absl::StatusOr<int> {
        LRT_TENSOR_RETURN_IF_ERROR(
            absl::StatusOr<int>(absl::NotFoundError("")));
        return 1;
      }(),
      StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(
      []() -> absl::StatusOr<int> {
        LRT_TENSOR_RETURN_IF_ERROR(true);
        return 1;
      }(),
      IsOkAndHolds(1));
  EXPECT_THAT(
      []() -> absl::StatusOr<int> {
        LRT_TENSOR_RETURN_IF_ERROR(false);
        return 1;
      }(),
      StatusIs(absl::StatusCode::kUnknown));
  EXPECT_THAT(
      []() -> absl::StatusOr<int> {
        LRT_TENSOR_RETURN_IF_ERROR(false) << "Extra message";
        return 1;
      }(),
      StatusIs(absl::StatusCode::kUnknown, HasSubstr("Extra message")));
}

TEST(LiteRtReturnIfErrorTest, DoesntReturnOnSuccess) {
  int canary_value = 0;
  auto ReturnExpectedIfError = [&canary_value]() -> absl::StatusOr<int> {
    LRT_TENSOR_RETURN_IF_ERROR(absl::OkStatus());
    canary_value = 1;
    return 1;
  };
  EXPECT_THAT(ReturnExpectedIfError(), IsOk());
  EXPECT_EQ(canary_value, 1);

  EXPECT_THAT(
      [&canary_value]() -> absl::Status {
        LRT_TENSOR_RETURN_IF_ERROR(absl::OkStatus());
        canary_value = 2;
        return absl::OkStatus();
      }(),
      IsOk());
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
      StatusIs(absl::StatusCode::kUnknown,
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
              StatusIs(kInvalidArgumentError.status().code()));
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
  EXPECT_THAT(expected_return, StatusIs(kInvalidArgumentError.status().code()));
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
  EXPECT_THAT(Function(), IsOk());
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
  // The error is created and propagated on the same line, so the line appears
  // twice.
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
  EXPECT_EQ(res.status().message(), "An error message.; An extra message.");
  EXPECT_THAT(Trace(res.status()),
              ElementsAre(At(error_1_line), At(error_1_line), At(error_2_line),
                          At(error_3_line)));
}

TEST(LiteRtErrorStatusBuilderTest, AnnotationMatchesAbslStatusBuilder) {
  absl::Status original = absl::InvalidArgumentError("Original message.");
  original.SetPayload(kPayloadUrl, absl::Cord("payload"));

  const int line = __LINE__ + 2;
  auto propagate = [&]() -> absl::Status {
    LRT_TENSOR_RETURN_IF_ERROR(original) << "Extra " << 42;
    return absl::OkStatus();
  };
  const absl::Status res = propagate();
  const absl::Status expected = absl::StatusBuilder(original) << "Extra " << 42;

  EXPECT_EQ(res.code(), expected.code());
  EXPECT_EQ(res.message(), expected.message());
  EXPECT_EQ(res.GetPayload(kPayloadUrl), expected.GetPayload(kPayloadUrl));
  std::vector<std::string> expected_trace = Trace(original);
  expected_trace.push_back(At(line));
  EXPECT_EQ(Trace(res), expected_trace);
}

TEST(LiteRtAssignOrReturnTest, AddsOneLocationPerHop) {
  const int origin_line = __LINE__ + 1;
  const absl::Status original = absl::NotFoundError("Not found.");

  const int hop_1_line = __LINE__ + 2;
  auto hop_1 = [&]() -> absl::StatusOr<int> {
    LRT_TENSOR_ASSIGN_OR_RETURN(int value, absl::StatusOr<int>(original));
    return value;
  };

  const int hop_2_line = __LINE__ + 2;
  auto hop_2 = [&]() -> absl::Status {
    LRT_TENSOR_ASSIGN_OR_RETURN([[maybe_unused]] int value, hop_1());
    return absl::OkStatus();
  };

  const absl::Status res = hop_2();
  EXPECT_THAT(res, StatusIs(absl::StatusCode::kNotFound, "Not found."));
  EXPECT_THAT(Trace(res),
              ElementsAre(At(origin_line), At(hop_1_line), At(hop_2_line)));
}

TEST(LiteRtReturnIfErrorTest, PreservesPayloads) {
  absl::Status original = absl::InternalError("Internal.");
  original.SetPayload(kPayloadUrl, absl::Cord("payload"));

  auto without_message = [&]() -> absl::Status {
    LRT_TENSOR_RETURN_IF_ERROR(original);
    return absl::OkStatus();
  };
  auto with_message = [&]() -> absl::Status {
    LRT_TENSOR_RETURN_IF_ERROR(original) << "Extra message.";
    return absl::OkStatus();
  };

  EXPECT_EQ(without_message().GetPayload(kPayloadUrl), absl::Cord("payload"));
  EXPECT_EQ(with_message().GetPayload(kPayloadUrl), absl::Cord("payload"));
}

TEST(LiteRtReturnIfErrorTest, ConditionErrorIsReportedAtCallSite) {
  const int bool_line = __LINE__ + 2;
  auto from_bool = []() -> absl::Status {
    LRT_TENSOR_RETURN_IF_ERROR(false);
    return absl::OkStatus();
  };
  const absl::Status bool_res = from_bool();
  EXPECT_THAT(bool_res, StatusIs(absl::StatusCode::kUnknown, "Check failed"));
  EXPECT_THAT(Trace(bool_res), ElementsAre(At(bool_line), At(bool_line)));

  const int pointer_line = __LINE__ + 3;
  auto from_pointer = []() -> absl::Status {
    int* ptr = nullptr;
    LRT_TENSOR_RETURN_IF_ERROR(ptr) << "Missing pointer.";
    return absl::OkStatus();
  };
  const absl::Status pointer_res = from_pointer();
  EXPECT_THAT(pointer_res, StatusIs(absl::StatusCode::kUnknown,
                                    "Check failed; Missing pointer."));
  EXPECT_THAT(Trace(pointer_res),
              ElementsAre(At(pointer_line), At(pointer_line)));
}

TEST(LiteRtReturnIfErrorTest, ConvertsCustomErrorTypes) {
  auto from_test_code = []() -> absl::Status {
    LRT_TENSOR_RETURN_IF_ERROR(TestCode::kOther);
    return absl::OkStatus();
  };
  EXPECT_THAT(from_test_code(),
              StatusIs(absl::StatusCode::kInternal, "TestCode error"));

  auto to_test_code = []() -> TestCode {
    LRT_TENSOR_RETURN_IF_ERROR(absl::InvalidArgumentError("Invalid."));
    return TestCode::kOk;
  };
  EXPECT_EQ(to_test_code(), TestCode::kInvalidArgument);

  auto annotated_to_test_code = []() -> TestCode {
    LRT_TENSOR_RETURN_IF_ERROR(absl::InternalError("Internal.")) << "Extra.";
    return TestCode::kOk;
  };
  EXPECT_EQ(annotated_to_test_code(), TestCode::kOther);
}

TEST(LiteRtReturnIfErrorTest, AbslStatusBuilderMethodsAreAvailable) {
  auto propagate = []() -> absl::Status {
    LRT_TENSOR_RETURN_IF_ERROR(absl::InternalError("Internal."))
            .SetCode(absl::StatusCode::kAborted)
        << "Extra message.";
    return absl::OkStatus();
  };
  EXPECT_THAT(propagate(),
              StatusIs(absl::StatusCode::kAborted, EndsWith("Extra message.")));
}

// A return type that can be constructed from an `absl::Status`.
struct StatusHolder {
  explicit StatusHolder(absl::Status status) : status(std::move(status)) {}
  absl::Status status;
};

TEST(LiteRtReturnIfErrorTest, AbslStatusBuilderMethodsKeepCustomConversions) {
  auto set_code = []() -> TestCode {
    LRT_TENSOR_RETURN_IF_ERROR(absl::InternalError("Internal."))
        .SetCode(absl::StatusCode::kInvalidArgument);
    return TestCode::kOk;
  };
  EXPECT_EQ(set_code(), TestCode::kInvalidArgument);

  auto log_then_annotate = []() -> TestCode {
    LRT_TENSOR_RETURN_IF_ERROR(absl::InvalidArgumentError("Invalid."))
            .LogError()
        << "Extra.";
    return TestCode::kOk;
  };
  EXPECT_EQ(log_then_annotate(), TestCode::kInvalidArgument);

  auto prepend = []() -> StatusHolder {
    LRT_TENSOR_RETURN_IF_ERROR(absl::InternalError("Internal.")).SetPrepend()
        << "Prefix: ";
    return StatusHolder(absl::OkStatus());
  };
  EXPECT_THAT(prepend().status,
              StatusIs(absl::StatusCode::kInternal, "Prefix: Internal."));

  auto policy = []() -> TestCode {
    LRT_TENSOR_RETURN_IF_ERROR(absl::InternalError("Internal."))
        .With([](absl::StatusBuilder builder) -> absl::StatusBuilder {
          return std::move(builder).SetCode(absl::StatusCode::kInvalidArgument);
        });
    return TestCode::kOk;
  };
  EXPECT_EQ(policy(), TestCode::kInvalidArgument);

  auto terminal = []() -> int {
    LRT_TENSOR_RETURN_IF_ERROR(absl::InternalError("Internal."))
        .With([](const absl::Status&) { return 42; });
    return 0;
  };
  EXPECT_EQ(terminal(), 42);
}

TEST(LiteRtAssignOrReturnTest, AbslStatusBuilderMethodsKeepCustomConversions) {
  auto set_code = []() -> TestCode {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        [[maybe_unused]] int value,
        absl::StatusOr<int>(absl::InternalError("Internal.")),
        _.SetCode(absl::StatusCode::kInvalidArgument) << "Extra.");
    return TestCode::kOk;
  };
  EXPECT_EQ(set_code(), TestCode::kInvalidArgument);

  auto custom_value = []() -> TestCode {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        [[maybe_unused]] int value,
        absl::StatusOr<int>(absl::InternalError("Internal.")),
        TestCode::kOther);
    return TestCode::kOk;
  };
  EXPECT_EQ(custom_value(), TestCode::kOther);
}

}  // namespace
}  // namespace litert
