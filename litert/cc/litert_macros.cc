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
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_source_location.h"

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

ErrorStatusBuilder::ErrorStatusBuilder(absl::Status&& status,
                                       litert::SourceLocation loc)
    : error_(ToLiteRtStatus(status.code()), std::string(status.message())),
      loc_(loc) {}

absl::Status ErrorStatusBuilder::ToAbslStatus() const noexcept {
  PrintLog();
  switch (error_.Status()) {
    case kLiteRtStatusOk:
      return absl::OkStatus();
    case kLiteRtStatusErrorInvalidArgument:
      return absl::InvalidArgumentError(error_.Message());
    case kLiteRtStatusErrorMemoryAllocationFailure:
      return absl::ResourceExhaustedError(error_.Message());
    case kLiteRtStatusErrorRuntimeFailure:
      return absl::InternalError(error_.Message());
    case kLiteRtStatusErrorMissingInputTensor:
      return absl::InvalidArgumentError(error_.Message());
    case kLiteRtStatusErrorUnsupported:
      return absl::UnimplementedError(error_.Message());
    case kLiteRtStatusErrorNotFound:
      return absl::NotFoundError(error_.Message());
    case kLiteRtStatusErrorTimeoutExpired:
      return absl::DeadlineExceededError(error_.Message());
    case kLiteRtStatusErrorWrongVersion:
      return absl::FailedPreconditionError(error_.Message());
    case kLiteRtStatusErrorUnknown:
      return absl::UnknownError(error_.Message());
    case kLiteRtStatusErrorAlreadyExists:
      return absl::AlreadyExistsError(error_.Message());
    case kLiteRtStatusErrorFileIO:
      return absl::UnavailableError(error_.Message());
    case kLiteRtStatusErrorInvalidFlatbuffer:
      return absl::InvalidArgumentError(error_.Message());
    case kLiteRtStatusErrorDynamicLoading:
      return absl::UnavailableError(error_.Message());
    case kLiteRtStatusErrorSerialization:
      return absl::InternalError(error_.Message());
    case kLiteRtStatusErrorCompilation:
      return absl::InternalError(error_.Message());
    case kLiteRtStatusErrorIndexOOB:
      return absl::OutOfRangeError(error_.Message());
    case kLiteRtStatusErrorInvalidIrType:
      return absl::InvalidArgumentError(error_.Message());
    case kLiteRtStatusErrorInvalidGraphInvariant:
      return absl::InvalidArgumentError(error_.Message());
    case kLiteRtStatusErrorGraphModification:
      return absl::InternalError(error_.Message());
    case kLiteRtStatusErrorInvalidToolConfig:
      return absl::InvalidArgumentError(error_.Message());
    case kLiteRtStatusLegalizeNoMatch:
      return absl::NotFoundError(error_.Message());
    case kLiteRtStatusErrorInvalidLegalization:
      return absl::InvalidArgumentError(error_.Message());
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
