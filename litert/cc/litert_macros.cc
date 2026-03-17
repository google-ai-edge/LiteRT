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
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_source_location.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_expected.h"

namespace litert {

namespace {
LiteRtStatus ToLiteRtStatus(const absl::StatusCode& code) {
  switch (code) {
    case absl::StatusCode::kOk:
      return kLiteRtStatusOk;
    case absl::StatusCode::kCancelled:
      return kLiteRtStatusErrorTimeoutExpired;
    case absl::StatusCode::kUnknown:
      return kLiteRtStatusErrorUnknown;
    case absl::StatusCode::kInvalidArgument:
      return kLiteRtStatusErrorInvalidArgument;
    case absl::StatusCode::kDeadlineExceeded:
      return kLiteRtStatusErrorTimeoutExpired;
    case absl::StatusCode::kNotFound:
      return kLiteRtStatusErrorNotFound;
    case absl::StatusCode::kAlreadyExists:
      return kLiteRtStatusErrorRuntimeFailure;
    case absl::StatusCode::kPermissionDenied:
      return kLiteRtStatusErrorRuntimeFailure;
    case absl::StatusCode::kResourceExhausted:
      return kLiteRtStatusErrorRuntimeFailure;
    case absl::StatusCode::kFailedPrecondition:
      return kLiteRtStatusErrorRuntimeFailure;
    case absl::StatusCode::kAborted:
      return kLiteRtStatusErrorRuntimeFailure;
    case absl::StatusCode::kOutOfRange:
      return kLiteRtStatusErrorIndexOOB;
    case absl::StatusCode::kUnimplemented:
      return kLiteRtStatusErrorUnsupported;
    case absl::StatusCode::kInternal:
      return kLiteRtStatusErrorUnknown;
    case absl::StatusCode::kUnavailable:
      return kLiteRtStatusErrorNotFound;
    case absl::StatusCode::kDataLoss:
      return kLiteRtStatusErrorRuntimeFailure;
    case absl::StatusCode::kUnauthenticated:
      return kLiteRtStatusErrorRuntimeFailure;
    default:
      return kLiteRtStatusErrorUnknown;
  }
  return kLiteRtStatusErrorUnknown;
}
}  // namespace

litert::Error ErrorStatusBuilder::ErrorConversion<absl::Status>::AsError(
    const absl::Status& value) {
  return litert::Error(ToLiteRtStatus(value.code()),
                       std::string(value.message()));
}

absl::Status ErrorStatusBuilder::ToAbslStatus() const noexcept {
  switch (error_.StatusCC()) {
    case Status::kOk:
      return absl::OkStatus();
    case Status::kErrorInvalidArgument:
      return absl::InvalidArgumentError(LogMessage());
    case Status::kErrorMemoryAllocationFailure:
      return absl::ResourceExhaustedError(LogMessage());
    case Status::kErrorRuntimeFailure:
      return absl::InternalError(LogMessage());
    case Status::kErrorMissingInputTensor:
      return absl::InvalidArgumentError(LogMessage());
    case Status::kErrorUnsupported:
      return absl::UnimplementedError(LogMessage());
    case Status::kErrorNotFound:
      return absl::NotFoundError(LogMessage());
    case Status::kErrorTimeoutExpired:
      return absl::DeadlineExceededError(LogMessage());
    case Status::kCancelled:
      return absl::CancelledError(LogMessage());
    case Status::kErrorWrongVersion:
      return absl::FailedPreconditionError(LogMessage());
    case Status::kErrorUnknown:
      return absl::UnknownError(LogMessage());
    case Status::kErrorAlreadyExists:
      return absl::AlreadyExistsError(LogMessage());
    case Status::kErrorFileIO:
      return absl::UnavailableError(LogMessage());
    case Status::kErrorInvalidFlatbuffer:
      return absl::InvalidArgumentError(LogMessage());
    case Status::kErrorDynamicLoading:
      return absl::UnavailableError(LogMessage());
    case Status::kErrorSerialization:
      return absl::InternalError(LogMessage());
    case Status::kErrorCompilation:
      return absl::InternalError(LogMessage());
    case Status::kErrorIndexOOB:
      return absl::OutOfRangeError(LogMessage());
    case Status::kErrorInvalidIrType:
      return absl::InvalidArgumentError(LogMessage());
    case Status::kErrorInvalidGraphInvariant:
      return absl::InvalidArgumentError(LogMessage());
    case Status::kErrorGraphModification:
      return absl::InternalError(LogMessage());
    case Status::kErrorInvalidToolConfig:
      return absl::InvalidArgumentError(LogMessage());
    case Status::kLegalizeNoMatch:
      return absl::NotFoundError(LogMessage());
    case Status::kErrorInvalidLegalization:
      return absl::InvalidArgumentError(LogMessage());
    case Status::kPatternNoMatch:
      return absl::NotFoundError(error_.Message());
    case Status::kInvalidTransformation:
      return absl::InvalidArgumentError(error_.Message());
    case Status::kErrorUnsupportedRuntimeVersion:
      return absl::FailedPreconditionError(LogMessage());
    case Status::kErrorUnsupportedCompilerVersion:
      return absl::FailedPreconditionError(LogMessage());
    case Status::kErrorIncompatibleByteCodeVersion:
      return absl::FailedPreconditionError(LogMessage());
  }
}

std::string ErrorStatusBuilder::LogMessage() const {
  LiteRtLogger logger = LiteRtGetDefaultLogger();
  LiteRtLogSeverity min_severity;
  if (LiteRtGetMinLoggerSeverity(logger, &min_severity) != kLiteRtStatusOk) {
    min_severity = kLiteRtLogSeverityVerbose;
  }
  if (log_level_ >= min_severity) {
    std::stringstream sstr;
    sstr << LiteRtGetLogSeverityName(log_level_) << ": [" << loc_.file_name()
         << ':' << loc_.line() << ']';
    if (extra_log_) {
      sstr << ' ' << extra_log_->str();
    }
    if (!error_.Message().empty()) {
      sstr << "\n└ " << error_.Message();
    }
    return sstr.str();
  }
  return "";
}

}  // namespace litert
