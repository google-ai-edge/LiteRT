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

#ifndef ODML_LITERT_LITERT_CORE_STATEFUL_ERROR_REPORTER_H_
#define ODML_LITERT_LITERT_CORE_STATEFUL_ERROR_REPORTER_H_

#include <string>

#include "litert/core/error_reporter.h"

namespace litert {

class StatefulErrorReporter : public ErrorReporter {
 public:
  virtual std::string message() = 0;
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CORE_STATEFUL_ERROR_REPORTER_H_