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

#include "litert/c/litert_error_reporter.h"

#include <cstdarg>
#include <memory>
#include <string>

#include "litert/core/buffer_error_reporter.h"
#include "litert/core/error_reporter.h"
#include "litert/core/stateful_error_reporter.h"

namespace {

struct LiteRtErrorReporterT {
  std::unique_ptr<litert::ErrorReporter> reporter;
  std::string last_message;
};

}  // namespace

extern "C" {

LiteRtStatus LiteRtCreateStderrErrorReporter(LiteRtErrorReporter* reporter) {
  if (!reporter) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  
  auto* impl = new LiteRtErrorReporterT;
  impl->reporter = std::make_unique<litert::StderrReporter>();
  *reporter = impl;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCreateBufferErrorReporter(LiteRtErrorReporter* reporter) {
  if (!reporter) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  
  auto* impl = new LiteRtErrorReporterT;
  impl->reporter = std::make_unique<litert::BufferErrorReporter>();
  *reporter = impl;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtErrorReporterReport(LiteRtErrorReporter reporter,
                                       const char* format, ...) {
  if (!reporter || !format) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  
  va_list args;
  va_start(args, format);
  reporter->reporter->Report(format, args);
  va_end(args);
  
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtErrorReporterGetMessage(LiteRtErrorReporter reporter,
                                           const char** message) {
  if (!reporter || !message) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  
  auto* stateful = dynamic_cast<litert::StatefulErrorReporter*>(
      reporter->reporter.get());
  if (!stateful) {
    return kLiteRtStatusErrorUnsupported;
  }
  
  reporter->last_message = stateful->message();
  *message = reporter->last_message.c_str();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtErrorReporterClear(LiteRtErrorReporter reporter) {
  if (!reporter) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  
  auto* buffer = dynamic_cast<litert::BufferErrorReporter*>(
      reporter->reporter.get());
  if (!buffer) {
    return kLiteRtStatusErrorUnsupported;
  }
  
  buffer->Clear();
  return kLiteRtStatusOk;
}

LiteRtErrorReporter LiteRtGetDefaultErrorReporter() {
  static LiteRtErrorReporterT* default_reporter = []() {
    auto* reporter = new LiteRtErrorReporterT;
    reporter->reporter.reset(litert::DefaultErrorReporter());
    return reporter;
  }();
  return default_reporter;
}

void LiteRtDestroyErrorReporter(LiteRtErrorReporter reporter) {
  if (reporter && reporter != LiteRtGetDefaultErrorReporter()) {
    delete reporter;
  }
}

}  // extern "C"