// Copyright 2026 Google LLC.
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

#include "litert/test/load_test_model.h"

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_macros.h"
#include "litert/test/common.h"

namespace litert::testing {

ExtendedModel LoadTestFileModel(absl::string_view filename) {
  LITERT_ASSIGN_OR_ABORT(
      auto model, ExtendedModel::CreateFromFile(GetTestFilePath(filename)));
  return model;
}

}  // namespace litert::testing
