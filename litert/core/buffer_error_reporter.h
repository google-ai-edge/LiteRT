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

#ifndef ODML_LITERT_LITERT_CORE_BUFFER_ERROR_REPORTER_H_
#define ODML_LITERT_LITERT_CORE_BUFFER_ERROR_REPORTER_H_

#include <cstdarg>
#include <cstddef>
#include <sstream>
#include <string>

#include "tflite/core/api/error_reporter.h"

namespace litert {

// A custom error reporter that stores error messages in a buffer.
class BufferErrorReporter : public ::tflite::ErrorReporter {
 public:
  BufferErrorReporter() = default;

  // Report an error message.
  int Report(const char* format, va_list args) override;

  // Inherit the variadic version.
  using ErrorReporter::Report;

  // Get the accumulated error messages.
  std::string message();

  // Clear the accumulated error messages.
  void Clear();

  // Get the number of errors reported.
  size_t NumErrors() const { return num_errors_; }

 private:
  std::stringstream buffer_;
  size_t num_errors_ = 0;
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CORE_BUFFER_ERROR_REPORTER_H_
