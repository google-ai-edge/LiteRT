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

namespace litert {

// Abstract base class for error reporting.
class ErrorReporter {
 public:
  virtual ~ErrorReporter() = default;

  // Reports an error message using the given format string and arguments.
  // Returns the number of characters written or a negative value if an error
  // occurred.
  virtual int Report(const char* format, va_list args) = 0;

  // Reports an error message using the given format string and arguments.
  // This is a variadic version of `Report`.
  int Report(const char* format, ...);
};

// An error reporter that writes to standard error.
struct StderrReporter : ErrorReporter {
  // Reports an error message to standard error using the given format string and arguments.
  // Returns the number of characters written or a negative value if an error
  // occurred.
  int Report(const char* format, va_list args) override;
  using ErrorReporter::Report;  // Inherit the variadic version
};

// Returns a default error reporter, which is a StderrReporter.
ErrorReporter* DefaultErrorReporter();

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CORE_ERROR_REPORTER_H_
