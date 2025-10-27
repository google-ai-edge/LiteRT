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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_LOGGING_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_LOGGING_H_

#include <cstddef>
#include <ostream>
#include <string>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"

namespace litert {

// Indicates specialization of absl::Stringify has yet to be implemented.
static constexpr auto kNoPrinterTag = "!no_printer";

// Detects whether the current build is debug or not.
inline constexpr bool IsDbg() {
#ifdef NDEBUG
  return false;
#else
  return true;
#endif
}

// Sets up log interception and reverts it upon destruction.
//
// Useful to catch logs automatically dumped when the C API is called.
//
// ```cpp
// Expected<int> DoSomething() {
//   InterceptLogs intercepted_logs;
//   LITERT_RETURN_IF_ERROR(LiteRtCApiFunction()) << intercepted_logs;
//   LITERT_RETURN_IF_ERROR(litert::CppApiFunction()) << intercepted_logs;
//   return 2;
// }
// ```
class InterceptLogs {
 public:
  InterceptLogs() {
    original_logger_ = LiteRtGetDefaultLogger();
    LiteRtUseSinkLogger();
    ClearLogs();
  }

  ~InterceptLogs() { LiteRtSetDefaultLogger(original_logger_); }

  // Clears the current default sink logger. No-op if the logger isn't a sink
  // logger.
  static void ClearLogs() {
    LiteRtLogger logger = LiteRtGetDefaultLogger();
    LiteRtClearSinkLogger(logger);
  }

  // Streams the intercepted logs to an output stream.
  friend std::ostream& operator<<(std::ostream& os, const InterceptLogs&) {
    LiteRtLogger logger = LiteRtGetDefaultLogger();
    size_t sz = 0;
    auto status = LiteRtGetSinkLoggerSize(logger, &sz);
    if (status != kLiteRtStatusOk) {
      return os << "Couldn't get intercepted logs count.";
    }
    const char* sep = "";
    for (size_t i = 0; i < sz; ++i) {
      const char* msg = nullptr;
      status = LiteRtGetSinkLoggerMessage(logger, i, &msg);
      if (status != kLiteRtStatusOk || msg == nullptr) {
        os << sep << "Couldn't get log message number " << i;
      } else {
        os << sep << msg;
      }
      sep = "\n";
    }
    return os;
  }

  LiteRtLogger original_logger_;
};

// Returns a human readable string representing the given number of bytes.
inline std::string HumanReadableSize(size_t bytes) {
  static constexpr auto kGb = 1024 * 1024 * 1024;
  static constexpr auto kMb = 1024 * 1024;
  static constexpr auto kKb = 1024;
  if (bytes >= kGb) return std::to_string((float)bytes / kGb) + "GB";
  if (bytes >= kMb) return std::to_string((float)bytes / kMb) + "MB";
  if (bytes >= kKb) return std::to_string((float)bytes / kKb) + "kB";
  return std::to_string(bytes) + "B";
}

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_LOGGING_H_
