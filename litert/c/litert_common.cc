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

#include "litert/c/litert_common.h"

#include <array>
#include <string_view>
#include <tuple>

namespace {

// Compile-time status string lookup
struct StatusStringEntry {
  LiteRtStatus status;
  std::string_view name;
};

constexpr std::array<StatusStringEntry, 22> kStatusStrings = {{
    {kLiteRtStatusOk, "kLiteRtStatusOk"},
    {kLiteRtStatusErrorInvalidArgument, "kLiteRtStatusErrorInvalidArgument"},
    {kLiteRtStatusErrorMemoryAllocationFailure,
     "kLiteRtStatusErrorMemoryAllocationFailure"},
    {kLiteRtStatusErrorRuntimeFailure, "kLiteRtStatusErrorRuntimeFailure"},
    {kLiteRtStatusErrorMissingInputTensor,
     "kLiteRtStatusErrorMissingInputTensor"},
    {kLiteRtStatusErrorUnsupported, "kLiteRtStatusErrorUnsupported"},
    {kLiteRtStatusErrorNotFound, "kLiteRtStatusErrorNotFound"},
    {kLiteRtStatusErrorTimeoutExpired, "kLiteRtStatusErrorTimeoutExpired"},
    {kLiteRtStatusErrorWrongVersion, "kLiteRtStatusErrorWrongVersion"},
    {kLiteRtStatusErrorUnknown, "kLiteRtStatusErrorUnknown"},
    {kLiteRtStatusErrorFileIO, "kLiteRtStatusErrorFileIO"},
    {kLiteRtStatusErrorInvalidFlatbuffer,
     "kLiteRtStatusErrorInvalidFlatbuffer"},
    {kLiteRtStatusErrorDynamicLoading, "kLiteRtStatusErrorDynamicLoading"},
    {kLiteRtStatusErrorSerialization, "kLiteRtStatusErrorSerialization"},
    {kLiteRtStatusErrorCompilation, "kLiteRtStatusErrorCompilation"},
    {kLiteRtStatusErrorIndexOOB, "kLiteRtStatusErrorIndexOOB"},
    {kLiteRtStatusErrorInvalidIrType, "kLiteRtStatusErrorInvalidIrType"},
    {kLiteRtStatusErrorInvalidGraphInvariant,
     "kLiteRtStatusErrorInvalidGraphInvariant"},
    {kLiteRtStatusErrorGraphModification,
     "kLiteRtStatusErrorGraphModification"},
    {kLiteRtStatusErrorInvalidToolConfig,
     "kLiteRtStatusErrorInvalidToolConfig"},
    {kLiteRtStatusLegalizeNoMatch, "kLiteRtStatusLegalizeNoMatch"},
    {kLiteRtStatusErrorInvalidLegalization,
     "kLiteRtStatusErrorInvalidLegalization"},
}};

constexpr std::string_view kUnknownStatus = "Unknown";

// Compile-time binary search for status string
constexpr std::string_view FindStatusString(LiteRtStatus status) {
  for (const auto& entry : kStatusStrings) {
    if (entry.status == status) {
      return entry.name;
    }
  }
  return kUnknownStatus;
}

}  // namespace

extern "C" {

const char* LiteRtGetStatusString(LiteRtStatus status) {
  // Use compile-time lookup for known statuses
  constexpr auto max_known_status =
      static_cast<LiteRtStatus>(kLiteRtStatusErrorInvalidLegalization);

  if (status >= kLiteRtStatusOk && status <= max_known_status) {
    for (const auto& entry : kStatusStrings) {
      if (entry.status == status) {
        return entry.name.data();
      }
    }
  }

  return kUnknownStatus.data();
}

int LiteRtCompareApiVersion(LiteRtApiVersion version,
                            LiteRtApiVersion reference) {
  const auto v_tuple = std::tie(version.major, version.minor, version.patch);
  const auto r_tuple =
      std::tie(reference.major, reference.minor, reference.patch);

  if (v_tuple > r_tuple) return 1;
  if (v_tuple < r_tuple) return -1;
  return 0;
}

}  // extern "C"
