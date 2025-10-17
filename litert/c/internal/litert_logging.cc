// Copyright 2024 Google LLC.
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

#include "litert/c/internal/litert_logging.h"

#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "tflite/core/c/common.h"
#include "tflite/logger.h"
#include "tflite/minimal_logging.h"

struct LiteRtLoggerT {
  virtual ~LiteRtLoggerT() = default;
  virtual void(Log)(LiteRtLogSeverity severity, const char* format,
                    va_list args) = 0;
  virtual LiteRtLogSeverity(GetMinSeverity)() = 0;
  virtual void(SetMinSeverity)(LiteRtLogSeverity severity) = 0;
  virtual const char* GetIdentifier() const = 0;
};

class LiteRtStandardLoggerT final : public LiteRtLoggerT {
 public:
  LiteRtLogSeverity GetMinSeverity() override {
    return ConvertSeverity(
        tflite::logging_internal::MinimalLogger::GetMinimumLogSeverity());
  }

  void SetMinSeverity(LiteRtLogSeverity severity) override {
    tflite::logging_internal::MinimalLogger::SetMinimumLogSeverity(
        ConvertSeverity(severity));
  }

  void Log(LiteRtLogSeverity severity, const char* format,
           va_list args) override {
    tflite::logging_internal::MinimalLogger::LogFormatted(
        ConvertSeverity(severity), format, args);
  }

  const char* GetIdentifier() const override { return "LiteRtDefaultLogger"; }

 private:
  static tflite::LogSeverity ConvertSeverity(LiteRtLogSeverity severity) {
    return static_cast<tflite::LogSeverity>(severity);
  }

  static LiteRtLogSeverity ConvertSeverity(tflite::LogSeverity severity) {
    return static_cast<LiteRtLogSeverity>(severity);
  }
};

class LiteRtSinkLoggerT final : public LiteRtLoggerT {
 public:
  static constexpr absl::string_view kIdentifier = "LiteRtSinkLogger";
  LiteRtLogSeverity GetMinSeverity() override { return min_severity_; }

  void SetMinSeverity(LiteRtLogSeverity severity) override {
    min_severity_ = severity;
  }

  void Log(LiteRtLogSeverity severity, const char* format,
           va_list args) override {
    va_list args2;
    va_copy(args2, args);
    logs_.emplace_back(LiteRtGetLogSeverityName(severity));
    std::string& log = logs_.back();
    const int print_start = log.size();
    const int len = vsnprintf(nullptr, 0, format, args);
    if (len > 0) {
      // Initial log severity string + separator + message + null terminator.
      log.resize(print_start + 2 + len + 1);
      log[print_start] = ':';
      log[print_start + 1] = ' ';
      vsnprintf(log.data() + print_start + 2, len + 1, format, args2);
    }
    va_end(args2);
  }

  const char* GetIdentifier() const override { return kIdentifier.data(); }

  std::vector<std::string>& Logs() { return logs_; }

 private:
  LiteRtLogSeverity min_severity_ = kLiteRtLogSeverityInfo;
  std::vector<std::string> logs_;
};

const char* LiteRtGetLogSeverityName(LiteRtLogSeverity severity) {
  switch (severity) {
    case kLiteRtLogSeverityVerbose:
      return "VERBOSE";
    case kLiteRtLogSeverityInfo:
      return "INFO";
    case kLiteRtLogSeverityWarning:
      return "WARNING";
    case kLiteRtLogSeverityError:
      return "ERROR";
    case kLiteRtLogSeveritySilent:
      return "SILENT";
    case kLiteRtLogSeverityDebug:
      TFL_UNREACHABLE();
  }
  return "UNKNOWN";
}

LiteRtStatus LiteRtCreateLogger(LiteRtLogger* logger) {
  if (!logger) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *logger = new LiteRtStandardLoggerT;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCreateSinkLogger(LiteRtLogger* logger) {
  if (!logger) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *logger = new LiteRtSinkLoggerT;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetMinLoggerSeverity(LiteRtLogger logger,
                                        LiteRtLogSeverity* min_severity) {
  if (!logger || !min_severity) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *min_severity = logger->GetMinSeverity();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetMinLoggerSeverity(LiteRtLogger logger,
                                        LiteRtLogSeverity min_severity) {
  if (!logger) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  logger->SetMinSeverity(min_severity);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtLoggerLog(LiteRtLogger logger, LiteRtLogSeverity severity,
                             const char* format, ...) {
  if (!logger || !format) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  va_list args;
  va_start(args, format);
  logger->Log(severity, format, args);
  va_end(args);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetLoggerIdentifier(LiteRtLoggerConst logger,
                                       const char** identifier) {
  if (!logger || !identifier) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *identifier = logger->GetIdentifier();
  return kLiteRtStatusOk;
}

void LiteRtDestroyLogger(LiteRtLogger logger) {
  if (logger != nullptr) {
    delete logger;
  }
}

namespace {

// Cast to a sink logger if possible.
//
// Note: RTTI may be disabled so we want use dynamic cast to do this natively.
LiteRtSinkLoggerT* AsSinkLogger(LiteRtLogger logger) {
  if (logger && logger->GetIdentifier() == LiteRtSinkLoggerT::kIdentifier) {
    return static_cast<LiteRtSinkLoggerT*>(logger);
  }
  return nullptr;
}
}  // namespace

// Returns the number of log calls that were done to the sink logger.
LiteRtStatus LiteRtGetSinkLoggerSize(LiteRtLogger logger, size_t* size) {
  LiteRtSinkLoggerT* sink = AsSinkLogger(logger);
  if (!sink || !size) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *size = sink->Logs().size();
  return kLiteRtStatusOk;
}

// Returns the idx_th log in the sink logger.
LiteRtStatus LiteRtGetSinkLoggerMessage(LiteRtLogger logger, size_t idx,
                                        const char** message) {
  LiteRtSinkLoggerT* sink = AsSinkLogger(logger);
  if (!sink || !message) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (idx >= sink->Logs().size()) {
    return kLiteRtStatusErrorNotFound;
  }
  *message = sink->Logs()[idx].c_str();
  return kLiteRtStatusOk;
}

// Clears the sink logger.
LiteRtStatus LiteRtClearSinkLogger(LiteRtLogger logger) {
  LiteRtSinkLoggerT* sink = AsSinkLogger(logger);
  if (!sink) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  sink->Logs().clear();
  return kLiteRtStatusOk;
}

namespace {

LiteRtLogger GetStaticStandardLogger() {
  // This is a static variable that will be destroyed at the same time as the
  // process.
  //
  // See
  // https://google.github.io/styleguide/cppguide.html#Static_and_Global_Variables.
  static LiteRtLogger static_logger = new LiteRtStandardLoggerT();
  return static_logger;
}

LiteRtLogger GetStaticSinkLogger() {
  // This is a static variable that will be destroyed at the same time as the
  // process.
  //
  // See
  // https://google.github.io/styleguide/cppguide.html#Static_and_Global_Variables.
  static LiteRtLogger static_logger = new LiteRtSinkLoggerT();
  return static_logger;
}

LiteRtLogger& GetDefaultLogger() {
  static LiteRtLogger default_logger = GetStaticStandardLogger();
  return default_logger;
}

}  // namespace

LiteRtStatus LiteRtSetDefaultLogger(LiteRtLogger logger) {
  if (!logger) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  GetDefaultLogger() = logger;
  return kLiteRtStatusOk;
}

LiteRtLogger LiteRtGetDefaultLogger() { return GetDefaultLogger(); }

LiteRtStatus LiteRtUseStandardLogger() {
  return LiteRtSetDefaultLogger(GetStaticStandardLogger());
}

LiteRtStatus LiteRtUseSinkLogger() {
  return LiteRtSetDefaultLogger(GetStaticSinkLogger());
}
