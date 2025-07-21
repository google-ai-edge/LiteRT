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

#include "litert/core/error_reporter.h"
#include "litert/core/stateful_error_reporter.h"

namespace litert {

// A concrete implementation of `StatefulErrorReporter` that stores
// reported errors in a memory buffer.
class BufferErrorReporter : public StatefulErrorReporter {
 public:
  BufferErrorReporter() = default;

  // Reports an error message using a va_list for the arguments.
  // Returns the number of characters written or a negative value if an error
  // occurred.
  int Report(const char* format, va_list args) override;
  
  // Inherit the variadic version of Report from the base class.
  using ErrorReporter::Report;

  // Returns the accumulated error messages as a string.
  std::string message() override;

  // Clears the internal buffer, removing all stored error messages.
  void Clear();

  // Returns the number of errors reported since the last clear.
  size_t NumErrors() const { return num_errors_; }

 private:
  std::stringstream buffer_;
  size_t num_errors_ = 0;
};
}  // namespace litert

#endif  // ODML_LITERT_LITERT_CORE_BUFFER_ERROR_REPORTER_H_
