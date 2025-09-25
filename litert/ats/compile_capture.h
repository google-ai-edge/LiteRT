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

#include <chrono>  // NOLINT
#include <functional>
#include <limits>
#include <ratio>  // NOLINT
#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/ats/capture_common.h"
#include "litert/ats/common.h"
#include "litert/ats/print.h"

namespace litert::testing {

// Information about the time taken to compile the model.
class CompilationTime : public Printable<Nanoseconds> {
 public:
  // Start timing.
  TimePoint Start() const { return Clock::now(); }

  // Stop timing and record the latency.
  void Stop(const TimePoint& start) {
    std::chrono::duration<Nanoseconds, std::nano> nano = Clock::now() - start;
    nanos_ = nano.count();
  }

  Nanoseconds Nanos() const { return nanos_; }

  CompilationTime() : Printable("CompilationTime", "compile_time(ns)") {}

 private:
  Fields GetFields() const override { return Fields{nanos_}; }

  Nanoseconds nanos_ = std::numeric_limits<Nanoseconds>::max();
};

// Type to hold all of the capturable information related compilation test
// case..
struct CompileCaptureEntry
    : public PrintableRow<ModelDetail, CompilationDetail, AcceleratorDetail,
                          CompilationTime> {
  CompileCaptureEntry() = default;

  ModelDetail model = {};
  CompilationDetail compilation_detail = {};
  AcceleratorDetail accelerator = {};
  CompilationTime compilation_time = {};

 private:
  Printables GetPrintables() const override {
    return Printables{std::cref(model), std::cref(compilation_detail),
                      std::cref(accelerator), std::cref(compilation_time)};
  }

  std::string Name() const override { return model.name; }
};

// Contains a collection of CompileCaptureEntry.
class CompileCapture : public PrintableCollection<CompileCaptureEntry> {
 public:
  using Entry = CompileCaptureEntry;

 private:
  absl::string_view Name() const override { return "Ats Compile Results"; }
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_ATS_COMPILE_CAPTURE_H_
