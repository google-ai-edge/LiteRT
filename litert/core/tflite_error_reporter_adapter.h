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

#ifndef ODML_LITERT_LITERT_CORE_TFLITE_ERROR_REPORTER_ADAPTER_H_
#define ODML_LITERT_LITERT_CORE_TFLITE_ERROR_REPORTER_ADAPTER_H_

#include <cstdarg>

#include "litert/core/error_reporter.h"
#include "tflite/core/api/error_reporter.h"

namespace litert {

class TfliteErrorReporterAdapter : public ::tflite::ErrorReporter {
 public:
  explicit TfliteErrorReporterAdapter(ErrorReporter* litert_reporter)
      : litert_reporter_(litert_reporter) {}

  int Report(const char* format, va_list args) override {
    if (litert_reporter_) {
      return litert_reporter_->Report(format, args);
    }
    return 0;
  }

 private:
  ErrorReporter* litert_reporter_;  // Not owned
};

class LiteRtErrorReporterAdapter : public ErrorReporter {
 public:
  explicit LiteRtErrorReporterAdapter(::tflite::ErrorReporter* tflite_reporter)
      : tflite_reporter_(tflite_reporter) {}

  int Report(const char* format, va_list args) override {
    if (tflite_reporter_) {
      return tflite_reporter_->Report(format, args);
    }
    return 0;
  }

 private:
  ::tflite::ErrorReporter* tflite_reporter_;  // Not owned
};

inline ::tflite::ErrorReporter* GetTfliteCompatibleErrorReporter() {
  static TfliteErrorReporterAdapter adapter(DefaultErrorReporter());
  return &adapter;
}

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CORE_TFLITE_ERROR_REPORTER_ADAPTER_H_