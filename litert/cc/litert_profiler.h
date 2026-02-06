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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_PROFILER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_PROFILER_H_
#include <cstdlib>
#include <string>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_profiler_event.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

/// @file
/// @brief Defines the C++ wrapper for the LiteRT profiler.

namespace litert {

class Profiler : public internal::BaseHandle<LiteRtProfiler> {
 public:
  Profiler() = delete;

  /// @brief Constructs a `Profiler` object.
  /// @param env The `EnvironmentHolder` object used to create the profiler.
  /// @param profiler The `LiteRtProfiler` handle.
  /// @param owned Indicates if the created `Profiler` object should take
  /// ownership of the provided `profiler` handle.
  explicit Profiler(internal::EnvironmentHolder& env, LiteRtProfiler profiler,
                    OwnHandle owned)
      : internal::BaseHandle<LiteRtProfiler>(
            profiler,
            [runtime = env.runtime](LiteRtProfiler profiler) {
              runtime->DestroyProfiler(profiler);
            },
            owned),
        env_(env) {}

  /// @brief Get the number of events.
  Expected<int> GetNumEvents() const {
    int num_events = -1;
    LITERT_RETURN_IF_ERROR(
        env_.runtime->GetNumProfilerEvents(Get(), &num_events));
    return num_events;
  };

  /// @brief Get the profiled events.
  ///
  /// The caller owns the returned vector. `ProfiledEventData` is a struct that
  /// contains the event name, event type, start timestamp, end timestamp, and
  /// event source.
  Expected<std::vector<ProfiledEventData>> GetEvents() const {
    LITERT_ASSIGN_OR_RETURN(int num_events, GetNumEvents());
    if (num_events == 0) {
      return std::vector<ProfiledEventData>();
    }

    std::vector<ProfiledEventData> events(num_events);
    LITERT_RETURN_IF_ERROR(
        env_.runtime->GetProfilerEvents(Get(), num_events, events.data()));
    return events;
  }

  /// @brief Reset the profiler.
  Expected<void> Reset() {
    LITERT_RETURN_IF_ERROR(env_.runtime->ResetProfiler(Get()));
    return {};
  }

  /// @brief Start profiling.
  Expected<void> StartProfiling() {
    LITERT_RETURN_IF_ERROR(env_.runtime->StartProfiler(Get()));
    return {};
  }

  /// @brief Stop profiling.
  Expected<void> StopProfiling() {
    LITERT_RETURN_IF_ERROR(env_.runtime->StopProfiler(Get()));
    return {};
  }

  /// @brief Get the profile summary.
  Expected<std::string> GetProfileSummary(LiteRtCompiledModel compiled_model) {
    const char* summary = nullptr;
    LITERT_RETURN_IF_ERROR(
        env_.runtime->GetProfileSummary(Get(), compiled_model, &summary));
    std::string result(summary);
    free(const_cast<char*>(summary));
    return result;
  }

  /// @brief Set the current event source.
  ///
  /// `ProfiledEventSource` is used to determine the source of the event
  /// (e.g., LiteRT, TFLite delegate, TFLite interpreter).
  Expected<void> SetCurrentEventSource(ProfiledEventSource event_source) {
    LITERT_RETURN_IF_ERROR(
        env_.runtime->SetProfilerCurrentEventSource(Get(), event_source));
    return {};
  }

 private:
  internal::EnvironmentHolder env_;
};
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_PROFILER_H_
