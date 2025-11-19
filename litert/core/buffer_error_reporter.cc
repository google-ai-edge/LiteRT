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

#include "litert/core/buffer_error_reporter.h"

#include <cstdarg>
#include <cstdio>
#include <string>

namespace litert {

int BufferErrorReporter::Report(const char* format, va_list args) {
  char buf[1024];
  int formatted = vsnprintf(buf, sizeof(buf), format, args);
  buffer_ << buf;
  if (!buffer_.str().empty() && buffer_.str().back() != '\n') {
    buffer_ << '\n';
  }
  ++num_errors_;
  return formatted;
}

std::string BufferErrorReporter::message() {
  std::string value = buffer_.str();
  Clear();
  return value;
}

void BufferErrorReporter::Clear() {
  buffer_.str("");
  buffer_.clear();
  num_errors_ = 0;
}

}  // namespace litert
