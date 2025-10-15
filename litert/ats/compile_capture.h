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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_ATS_COMPILE_CAPTURE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_ATS_COMPILE_CAPTURE_H_

#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/ats/print.h"

namespace litert::testing {

struct CompileCaptureEntry : public PrintableRow<> {
  CompileCaptureEntry() = default;

 private:
  Printables GetPrintables() const override { return Printables{}; }

  std::string Name() const override { return "CompileCapture"; }
};

// TODO: lukeboyer - Implement this and subclasses.
class CompileCapture : public PrintableCollection<CompileCaptureEntry> {
 public:
  using Entry = CompileCaptureEntry;

 private:
  absl::string_view Name() const override { return "Ats Compile Results"; }
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_ATS_COMPILE_CAPTURE_H_
