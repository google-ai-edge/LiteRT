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

#include "litert/core/error_reporter.h"

#include <cstdarg>
#include <cstdio>

#include "litert/c/internal/litert_logging.h"

namespace litert {

int StderrReporter::Report(const char* format, va_list args) {
  char buffer[1024];
  int result = vsnprintf(buffer, sizeof(buffer), format, args);
  LITERT_LOG(LITERT_ERROR, "%s", buffer);
  return result;
}

}  // namespace litert
