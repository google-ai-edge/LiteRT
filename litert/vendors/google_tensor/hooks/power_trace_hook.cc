#include "litert/vendors/google_tensor/hooks/power_trace_hook.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>

#include "platforms/darwinn/devtools/power_stats/power_stats.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
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
  bool initialized = false;
};

namespace {

constexpr absl::string_view kDumpPowerMetricsTrue = "dump_power_metrics:true";
constexpr absl::string_view kPowerDumpPathKey = "power_dump_path:\"";

void ParsePowerTraceConfig(absl::string_view input,
                           PowerTraceContext* context) {
  if (absl::StrContains(input, kDumpPowerMetricsTrue)) {
    context->dump_power_metrics = true;
  }

  const size_t path_pos = input.find(kPowerDumpPathKey);
  if (path_pos != absl::string_view::npos) {
    const size_t start = path_pos + kPowerDumpPathKey.length();
    const size_t end = input.find('"', start);
    if (end != absl::string_view::npos) {
      context->power_dump_path = std::string(input.substr(start, end - start));
    }
  }
}

// Initializes PowerStats and captures the baseline TPU energy reading once per
// session.
absl::Status InitializeSessionBaselineEnergy(PowerTraceContext* context) {
  context->power_stats = platforms::darwinn::devtools::PowerStats::Create();
  if (!context->power_stats) {
    return absl::InternalError("Failed to create PowerStats");
  }
  absl::StatusOr<uint64_t> start_energy =
      context->power_stats->GetEnergyConsumedUWs(
          platforms::darwinn::devtools::power_stats::SUBSYSTEM_TPU);
  if (!start_energy.ok()) {
    return start_energy.status();
  }
  context->start_energy = *start_energy;
  return absl::OkStatus();
}

// Logs and optionally writes the aggregated TPU energy metrics to disk.
void DumpPowerMetrics(const PowerTraceContext& context,
                      uint64_t total_energy_uj) {
  const int count = context.inference_count;
  const double avg_energy_uj = static_cast<double>(total_energy_uj) / count;

  const std::string output = absl::StrFormat(
      "Average TPU Energy consumed per inference(uJ): %.3f\n"
      "Total Energy (uJ): %llu\n"
      "Inference count: %d\n",
      avg_energy_uj, total_energy_uj, count);

  LITERT_LOG(LITERT_INFO, "Power hook has generated following data:\n%s",
             output.c_str());

  if (context.power_dump_path.has_value() &&
      !context.power_dump_path->empty()) {
    std::ofstream outfile(*context.power_dump_path);
    if (outfile.is_open()) {
      outfile << output;
      outfile.close();
      LITERT_LOG(LITERT_INFO, "Google TensorHook: Power metrics dumped to %s",
                 context.power_dump_path->c_str());
    }
  }
}

}  // namespace

PowerTraceContext* CreatePowerTraceContext() { return new PowerTraceContext(); }

void DestroyPowerTraceContext(PowerTraceContext* context) { delete context; }

void HandlePowerRuntimeStart(PowerTraceContext* context) {
  if (!context || context->initialized) return;

  context->initialized = true;
  const std::string input = GetVendorHookArgsConfig();
  if (!input.empty()) {
    ParsePowerTraceConfig(input, context);
  }
  if (!context->dump_power_metrics) return;

  if (absl::Status status = InitializeSessionBaselineEnergy(context);
      !status.ok()) {
    LITERT_LOG(LITERT_ERROR,
               "Google TensorHook failed HandlePowerRuntimeStart: %s",
               std::string(status.message()).c_str());
  }
}

void HandlePowerRuntimeStop(PowerTraceContext* context) {
  if (!context || !context->dump_power_metrics) return;

  if (context->start_energy.has_value()) {
    context->inference_count++;
  }
}

void HandlePowerStopAndProcess(PowerTraceContext* context) {
  if (!context) return;
  if (context->dump_power_metrics) {
    if (context->start_energy.has_value() && context->inference_count > 0 &&
        context->power_stats) {
      absl::StatusOr<uint64_t> end_energy =
          context->power_stats->GetEnergyConsumedUWs(
              platforms::darwinn::devtools::power_stats::SUBSYSTEM_TPU);
      if (!end_energy.ok()) {
        LITERT_LOG(LITERT_ERROR,
                   "Google TensorHook failed final ODPM energy read: %s",
                   std::string(end_energy.status().message()).c_str());
      } else if (*end_energy < *context->start_energy) {
        LITERT_LOG(
            LITERT_ERROR,
            "Google TensorHook: Final energy (%llu) is less than start energy "
            "(%llu)",
            static_cast<unsigned long long>(*end_energy),
            static_cast<unsigned long long>(*context->start_energy));
      } else {
        DumpPowerMetrics(*context, *end_energy - *context->start_energy);
      }
    } else {
      LITERT_LOG(LITERT_WARNING,
                 "Google TensorHook: Skipping power metrics dump due to failed "
                 "ODPM energy reads or zero recorded inferences (count=%d).",
                 context->inference_count);
    }
  }

  // Resets session state on the context.
  context->power_stats.reset();
  context->start_energy = std::nullopt;
  context->inference_count = 0;
  context->power_dump_path = std::nullopt;
  context->dump_power_metrics = false;
  context->initialized = false;
}

}  // namespace litert::google_tensor
