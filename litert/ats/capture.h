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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_ATS_CAPTURE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_ATS_CAPTURE_H_

#include <algorithm>
#include <chrono>  // NOLINT
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <functional>
#include <list>
#include <optional>
#include <ratio>  // NOLINT
#include <string>
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/ats/configure.h"
#include "litert/cc/litert_detail.h"

namespace litert::testing {

// Type to hold all of the capturable information related to a single test case.
struct AtsCaptureEntry {
 public:
  using Ref = std::reference_wrapper<AtsCaptureEntry>;

  static AtsCaptureEntry Create() {
    AtsCaptureEntry entry;
    entry.start_time = Latency::Clock::now();
    return entry;
  }

  // Information about the input model.
  struct ModelDetail {
    // File name, if in memeory only graph, an identifier of the graph.
    std::string name = "";
    // Optional description or representation of the model.
    std::string desc = "";
    // Was the input model precompiled offline?
    bool precompiled = false;

    static constexpr absl::string_view kCols = "name,desc,precompiled";

    template <typename Sink>
    void Row(Sink& sink) const {
      sink << absl::StrFormat("%s,%s,%s", name, desc,
                              precompiled ? "true" : "false");
    }
  };

  // Information about the accelerator used if any.
  struct AcceleratorDetail {
    using AcceleratorType = AtsConf::ExecutionBackend;

    // The type of accelerator used.
    AcceleratorType a_type = AcceleratorType::kCpu;

    // Only applicable in the NPU case.
    std::string soc_man = "n/a";
    std::string soc_model = "n/a";

    // Were all the ops offloaded to the accelerator?
    bool is_fully_accelerated = false;

    static constexpr absl::string_view kCols =
        "a_type,soc_man,soc_model,is_fully_accelerated";

    template <typename Sink>
    void Row(Sink& sink) const {
      sink << absl::StrFormat("%v,%s,%s,%s", a_type, soc_man, soc_model,
                              is_fully_accelerated ? "true" : "false");
    }
  };

  // Information about the latency of the execution.
  class Latency {
   public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    using Nanoseconds = uint64_t;
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
      std::chrono::duration<Nanoseconds, std::nano> nano = Clock::now() - start;
      latencies_.push_back(nano.count());
    }

    // Average latency.
    Nanoseconds Avg() const {
      return ::litert::Avg(latencies_.cbegin(), latencies_.cend());
    }

    // Maximum latency.
    Nanoseconds Max() const {
      if (latencies_.empty()) {
        return 0;
      }
      return *std::max_element(latencies_.begin(), latencies_.end());
    }

    // Minimum latency.
    Nanoseconds Min() const {
      if (latencies_.empty()) {
        return 0;
      }
      return *std::min_element(latencies_.begin(), latencies_.end());
    }

    // Number of samples.
    size_t NumSamples() const { return latencies_.size(); }

    template <typename Sink>
    friend void AbslStringify(Sink& sink, const Latency& stats) {
      absl::Format(
          &sink, "\tAvg: %ens\n \tMax: %ens\n \tMin: %ens\n \tNumSamples: %lu",
          stats.Avg(), stats.Max(), stats.Min(), stats.NumSamples());
    }

    static constexpr absl::string_view kCols =
        "avg_latency(ns),max_latency(ns),min_latency(ns),num_samples";

    template <typename Sink>
    void Row(Sink& sink) const {
      sink << absl::StrFormat("%e,%e,%e,%lu", Avg(), Max(), Min(),
                              NumSamples());
    }

   private:
    std::vector<Nanoseconds> latencies_;
  };

  // Information about the numerics of the execution.
  class Numerics {
   public:
    // The type of reference used to validate against.
    enum class ReferenceType {
      kNone,
      // Standard CPU inference.
      kCpu,
      // Custom c++ reference.
      kCustom,
    };

    // Average mean squared error accross all the runs.
    double AvgMse() const {
      return ::litert::Avg(mses_.cbegin(), mses_.cend());
    }

    // The type of reference used to validate against.
    ReferenceType reference_type = ReferenceType::kNone;  // NOLINT

    // Add a new value.
    void NewMse(double mse) { mses_.push_back(mse); }

    static constexpr absl::string_view kCols = "reference_type,avg_mse";

    template <typename Sink>
    void Row(Sink& sink) const {
      sink << absl::StrFormat(
          "%s,%e", reference_type == ReferenceType::kCpu ? "cpu" : "custom",
          AvgMse());
    }

   private:
    std::vector<double> mses_;
  };

  // Information about the execution of the model.
  struct RunDetail {
    enum class Status {
      // End never recorded.
      kUnknown,
      // The runs completed successfully.
      kOk,
      // The runs failed due to an error.
      kError,
      // The runs failed due to timeout.
      kTimeout,
    };

    // The number of iterations the model was run.
    size_t num_iterations = 1;

    // The status of the run.
    Status status = Status::kUnknown;

    static constexpr absl::string_view kCols = "num_iterations,status";

    template <typename Sink>
    void Row(Sink& sink) const {
      absl::string_view status_str;
      switch (status) {
        case Status::kUnknown:
          status_str = "unknown";
          break;
        case Status::kOk:
          status_str = "ok";
          break;
        case Status::kError:
          status_str = "error";
          break;
        case Status::kTimeout:
          status_str = "timeout";
          break;
      }

      sink << absl::StrFormat("%lu,%s", num_iterations, status_str);
    }
  };

  template <typename Sink>
  void Row(Sink& sink) const {
    model.Row(sink);
    sink << ",";
    accelerator.Row(sink);
    sink << ",";
    latency.Row(sink);
    sink << ",";
    numerics.Row(sink);
    sink << ",";
    run.Row(sink);
  }

  AtsCaptureEntry() = default;

  ModelDetail model = {};
  AcceleratorDetail accelerator = {};
  Latency latency = {};
  Numerics numerics = {};
  RunDetail run = {};

 private:
  Latency::TimePoint start_time;
};

// Contains a collection of AtsCaptureEntry.
class AtsCapture {
 public:
  using Ref = std::reference_wrapper<AtsCapture>;

  // Start a new entry.
  AtsCaptureEntry& NewEntry() {
    entries_.push_back(AtsCaptureEntry::Create());
    return entries_.back();
  }

  // Syntactic sugar for optional collection.
  static std::optional<AtsCaptureEntry::Ref> NewEntry(
      std::optional<AtsCapture::Ref> cap) {
    if (cap) {
      return cap->get().NewEntry();
    }
    return std::nullopt;
  }

  // Get the entries.
  const std::list<AtsCaptureEntry>& Entries() const { return entries_; }

  // Print the entries in CSV format.
  template <typename Sink>
  void Csv(Sink& sink) const {
    static constexpr absl::string_view kCols[] = {
        AtsCaptureEntry::ModelDetail::kCols,
        AtsCaptureEntry::AcceleratorDetail::kCols,
        AtsCaptureEntry::Latency::kCols, AtsCaptureEntry::Numerics::kCols,
        AtsCaptureEntry::RunDetail::kCols};
    sink << absl::StrJoin(kCols, ",");
    sink << "\n";
    for (const auto& entry : entries_) {
      entry.Row(sink);
      sink << "\n";
    }
  }

  // Print the entries in human readable format.
  template <typename Sink>
  void Print(Sink& sink) const {
    // TODO: Human readable dump.
  }

  // Print just the latency information in a human readable format.
  template <typename Sink>
  void PrintLatency(Sink& sink) const {
    sink << kPrintHeader << "\n";
    for (const auto& entry : entries_) {
      sink << absl::StrFormat("%s : %s\n%v\n\n", entry.model.name,
                              entry.model.desc, entry.latency);
    }
    sink << kPrintFooter << "\n";
  }

 private:
  static constexpr absl::string_view kPrintHeader =
      "========== Ats Results ==========";
  static constexpr absl::string_view kPrintFooter =
      "=================================";

  std::list<AtsCaptureEntry> entries_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_ATS_CAPTURE_H_
