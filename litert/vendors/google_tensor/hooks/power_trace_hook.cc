#include "litert/vendors/google_tensor/hooks/power_trace_hook.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "platforms/darwinn/devtools/power_stats/power_stats.h"
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/vendors/google_tensor/hooks/hooks_utils.h"

namespace litert::google_tensor {

struct PowerTraceContext {
  std::unique_ptr<platforms::darwinn::devtools::PowerStats> power_stats;
  std::optional<uint64_t> start_energy = std::nullopt;
  std::optional<std::string> power_dump_path = std::nullopt;
  bool dump_power_metrics = false;

  int inference_count = 0;
  uint64_t total_inference_energy_uws = 0;
  bool initialized = false;
};

namespace {

constexpr absl::string_view kDumpPowerMetricsTrue = "dump_power_metrics:true";
constexpr absl::string_view kPowerDumpPathKey = "power_dump_path:\"";

}  // namespace

PowerTraceContext* CreatePowerTraceContext() { return new PowerTraceContext(); }

void DestroyPowerTraceContext(PowerTraceContext* context) { delete context; }

void ParsePowerTraceConfig(absl::string_view input,
                           PowerTraceContext* context) {
  if (absl::StrContains(input, kDumpPowerMetricsTrue)) {
    context->dump_power_metrics = true;
  }

  auto path_pos = input.find(kPowerDumpPathKey);
  if (path_pos != absl::string_view::npos) {
    auto start = path_pos + kPowerDumpPathKey.length();
    auto end = input.find('"', start);
    if (end != absl::string_view::npos) {
      context->power_dump_path = std::string(input.substr(start, end - start));
    }
  }
}

void InitializePowerStats(PowerTraceContext* context) {
  if (context->dump_power_metrics && !context->power_stats) {
    auto stats_or = platforms::darwinn::devtools::PowerStats::Create();
    if (stats_or) {
      context->power_stats = std::move(stats_or);
    }
  }
}

// Starts power trace collection before inference begins.
void HandlePowerRuntimeStart(PowerTraceContext* context) {
  if (!context) return;
  context->inference_count++;

  if (!context->initialized) {
    context->initialized = true;
    std::string input = GetVendorHookArgsConfig();
    if (!input.empty()) {
      ParsePowerTraceConfig(input, context);
    }
    InitializePowerStats(context);
  }

  if (context->power_stats) {
    context->start_energy = std::nullopt;
    auto energy_or = context->power_stats->GetEnergyConsumedUWs(
        platforms::darwinn::devtools::power_stats::SUBSYSTEM_TPU);
    if (energy_or.ok()) {
      context->start_energy = *energy_or;
    }
  }
}

// Concludes the metrics profiling block after inference completes.
void HandlePowerRuntimeStop(PowerTraceContext* context) {
  if (!context) return;
  if (context->power_stats && context->start_energy.has_value()) {
    auto energy_or = context->power_stats->GetEnergyConsumedUWs(
        platforms::darwinn::devtools::power_stats::SUBSYSTEM_TPU);
    if (energy_or.ok()) {
      uint64_t end_energy = *energy_or;
      if (end_energy >= *context->start_energy) {
        context->total_inference_energy_uws +=
            (end_energy - *context->start_energy);
      }
    }
  }
  context->start_energy = std::nullopt;
}

// Triggers a final aggregated statistics dump and performs full resource
// pickup.
void HandlePowerStopAndProcess(PowerTraceContext* context) {
  if (!context) return;
  if (context->dump_power_metrics) {
    uint64_t diff_energy = context->total_inference_energy_uws;
    int count = context->inference_count;
    double avg_energy_uj =
        (count > 0) ? static_cast<double>(diff_energy) / count : 0;

    LITERT_LOG(LITERT_INFO,
               "Power hook has generated following data:\n"
               "Average TPU Energy consumed per inference(uJ): %.3f\n"
               "Total Energy (uJ): %lu\n"
               "Inference count: %d",
               avg_energy_uj, diff_energy, count);

    if (context->power_dump_path.has_value() &&
        !context->power_dump_path->empty()) {
      std::string output = absl::StrCat(
          "Average TPU Energy consumed per inference(uJ): ", avg_energy_uj,
          "\n", "Total Energy (uJ): ", diff_energy, "\n",
          "Inference count: ", count, "\n");
      std::ofstream outfile(*context->power_dump_path);
      if (outfile.is_open()) {
        outfile << output;
        outfile.close();
        LITERT_LOG(LITERT_INFO, "Google TensorHook: Power metrics dumped to %s",
                   context->power_dump_path->c_str());
      }
    }
  }

  // State reset for this session context
  context->power_stats.reset();
  context->total_inference_energy_uws = 0;
  context->inference_count = 0;
  context->power_dump_path = std::nullopt;
  context->dump_power_metrics = false;
  context->initialized = false;
}

}  // namespace litert::google_tensor
