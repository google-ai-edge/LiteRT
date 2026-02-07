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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_ATS_INFERENCE_CAPTURE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_ATS_INFERENCE_CAPTURE_H_

#include <algorithm>
#include <chrono>  // NOLINT
#include <functional>
#include <optional>
#include <ratio>  // NOLINT
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/ats/capture_common.h"
#include "litert/ats/common.h"
#include "litert/ats/print.h"
#include "litert/cc/internal/litert_detail.h"

namespace litert::testing {

// Information about the latency of the execution.
class Latency
    : public Printable<Microseconds, Microseconds, Microseconds, size_t> {
 public:
  using Ref = std::reference_wrapper<Latency>;

  // Syntactic sugar for optional collection.
  static std::optional<TimePoint> Start(std::optional<Ref> s) {
    if (s) {
      return s->get().Start();
    }
    return std::nullopt;
  }
  static void Stop(std::optional<Ref> s,
                   const std::optional<TimePoint>& start) {
    if (s && start) {
      s->get().Stop(*start);
    }
  }

  // Start timing.
  TimePoint Start() const { return Clock::now(); }

  // Stop timing and record the latency.
  void Stop(const TimePoint& start) {
    const auto micro = std::chrono::duration_cast<
        std::chrono::duration<Microseconds, std::micro>>(Clock::now() - start);
    latencies_.push_back(micro.count());
  }

  // Average latency.
  Microseconds Avg() const {
    return ::litert::Avg(latencies_.cbegin(), latencies_.cend());
  }

  // Maximum latency.
  Microseconds Max() const {
    if (latencies_.empty()) {
      return 0;
    }
    return *std::max_element(latencies_.begin(), latencies_.end());
  }

  // Minimum latency.
  Microseconds Min() const {
    if (latencies_.empty()) {
      return 0;
    }
    return *std::min_element(latencies_.begin(), latencies_.end());
  }

  // Number of samples.
  size_t NumSamples() const { return latencies_.size(); }

  Latency()
      : Printable("Latency", "avg_latency(us)", "max_latency(us)",
                  "min_latency(us)", "num_samples") {}

 private:
  Fields GetFields() const override {
    return Fields{Avg(), Max(), Min(), NumSamples()};
  }

  std::vector<Microseconds> latencies_;
};

// Information about the numerics of the execution.
class Numerics : public Printable<ReferenceType, double> {
 public:
  // Average mean squared error across all the runs.
  double AvgMse() const { return Avg(mses_.cbegin(), mses_.cend()); }

  // The type of reference used to validate against.
  ReferenceType reference_type = ReferenceType::kNone;  // NOLINT

  // Add a new value.
  void NewMse(double mse) { mses_.push_back(mse); }

  Numerics() : Printable("Numerics", "reference_type", "avg_mse") {}

 private:
  Fields GetFields() const override { return Fields{reference_type, AvgMse()}; }

  std::vector<double> mses_;
};

// Information about the execution of the model.
struct RunDetail : public Printable<size_t, RunStatus> {
  // The number of iterations the model was run.
  size_t num_iterations = 1;

  // The status of the run.
  RunStatus status = RunStatus::kUnknown;

  RunDetail() : Printable("Run", "num_iterations", "status") {}

 private:
  Fields GetFields() const override { return Fields{num_iterations, status}; }
};

// Type to hold all of the capturable information related to a single test case.
struct InferenceCaptureEntry
    : public PrintableRow<ModelDetail, AcceleratorDetail, Latency, Numerics,
                          RunDetail, CompilationDetail> {
  InferenceCaptureEntry() = default;

  ModelDetail model = {};
  AcceleratorDetail accelerator = {};
  Latency latency = {};
  Numerics numerics = {};
  RunDetail run = {};
  CompilationDetail compilation = {};

 private:
  Printables GetPrintables() const override {
    return Printables{std::cref(model),   std::cref(accelerator),
                      std::cref(latency), std::cref(numerics),
                      std::cref(run),     std::cref(compilation)};
  }

  std::string Name() const override { return model.name; }
};

// Contains a collection of AtsCaptureEntry.
class InferenceCapture : public PrintableCollection<InferenceCaptureEntry> {
 public:
  using Entry = InferenceCaptureEntry;

 private:
  absl::string_view Name() const override { return "Ats Results"; }
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_ATS_INFERENCE_CAPTURE_H_
