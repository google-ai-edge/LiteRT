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

#ifndef ODML_LITERT_LITERT_CORE_ERROR_REPORTER_H_
#define ODML_LITERT_LITERT_CORE_ERROR_REPORTER_H_

#include <cstdarg>

#include "tflite/core/api/error_reporter.h"

namespace litert {

// Default implementation that logs errors to stderr using LITERT_LOG
class StderrReporter : public ::tflite::ErrorReporter {
 public:
  // Inherit variadic version.
  using ErrorReporter::Report;

  // Report an error message.
  int Report(const char* format, va_list args) override;
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CORE_ERROR_REPORTER_H_
