#include "litert/vendors/google_tensor/hooks/tpu_tile_hook.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>

#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_metrics.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_invocation_context.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_metrics.h"
#include "litert/vendors/google_tensor/hooks/hooks_utils.h"

namespace litert::google_tensor {

struct TpuTileTimeContext {
  bool dump_tpu_tile_time_metrics = false;
  std::optional<std::string> tpu_metrics_dump_path = std::nullopt;
  bool initialized = false;
  uint64_t total_tpu_tile_time_us = 0;
  int inference_count = 0;
};

namespace {
constexpr absl::string_view kDumpTpuMetricsTrue =
    "dump_tpu_tile_time_metrics:true";
constexpr absl::string_view kTpuMetricsDumpPathKey =
    "tpu_tile_time_metrics_dump_path:\"";
}  // namespace

TpuTileTimeContext* CreateTpuTileTimeContext() {
  return new TpuTileTimeContext();
}

void DestroyTpuTileTimeContext(TpuTileTimeContext* context) { delete context; }

void ParseTpuTileTimeConfig(absl::string_view input,
                            TpuTileTimeContext* context) {
  if (absl::StrContains(input, kDumpTpuMetricsTrue)) {
    context->dump_tpu_tile_time_metrics = true;
  }

  auto tpu_path_pos = input.find(kTpuMetricsDumpPathKey);
  if (tpu_path_pos != absl::string_view::npos) {
    auto start = tpu_path_pos + kTpuMetricsDumpPathKey.length();
    auto end = input.find('"', start);
    if (end != absl::string_view::npos) {
      context->tpu_metrics_dump_path =
          std::string(input.substr(start, end - start));
    }
  }
}

void HandleTpuTileTimeRuntimeStart(TpuTileTimeContext* context,
                                   LiteRtDispatchInvocationContext icontext) {
  if (!context) return;

  context->inference_count++;

  if (!context->initialized) {
    context->initialized = true;
    std::string input = GetVendorHookArgsConfig();
    if (!input.empty()) {
      ParseTpuTileTimeConfig(input, context);
    }
  }

  if (context->dump_tpu_tile_time_metrics && icontext) {
    icontext->StartMetricsCollection(1);
  }
}

void HandleTpuTileTimeRuntimeStop(TpuTileTimeContext* context,
                                  LiteRtDispatchInvocationContext icontext) {
  if (!context || !context->dump_tpu_tile_time_metrics || !icontext) return;

  LiteRtDispatchMetrics metrics_raw = nullptr;
  auto status = icontext->StopMetricsCollection(metrics_raw);
  if (status == kLiteRtStatusOk && metrics_raw != nullptr) {
    int num_metrics = metrics_raw->GetNumMetrics();
    uint64_t current_time_us = 0;

    for (int i = 0; i < num_metrics; ++i) {
      LiteRtMetric metric;
      metric.name = nullptr;
      if (metrics_raw->GetMetric(i, metric) != kLiteRtStatusOk) {
        continue;
      }

      if (metric.name == nullptr) {
        continue;
      }

      if (absl::string_view(metric.name) == "hardware_execution_time_us") {
        if (metric.value.type == kLiteRtAnyTypeInt) {
          current_time_us = metric.value.int_value;
        } else if (metric.value.type == kLiteRtAnyTypeReal) {
          current_time_us = static_cast<uint64_t>(metric.value.real_value);
        }
        break;
      }
    }

    context->total_tpu_tile_time_us += current_time_us;

    std::unique_ptr<LiteRtDispatchMetricsT> metrics_deleter(metrics_raw);
  }
}

void HandleTpuTileTimeStopAndProcess(TpuTileTimeContext* context) {
  if (!context) return;

  if (context->dump_tpu_tile_time_metrics && context->inference_count > 0) {
    uint64_t avg_tpu_tile_time_us =
        context->total_tpu_tile_time_us / context->inference_count;
    std::string output = absl::StrCat(
        "==== Google Tensor TPU Tile Time Metrics ====\n"
        "Total inferences: ",
        context->inference_count,
        "\n"
        "Total TPU tile time (us): ",
        context->total_tpu_tile_time_us,
        "\n"
        "Average TPU tile time (us): ",
        avg_tpu_tile_time_us,
        "\n"
        "==================================");
    LITERT_LOG(LITERT_INFO, "%s", output.c_str());

    if (context->tpu_metrics_dump_path.has_value() &&
        !context->tpu_metrics_dump_path->empty()) {
      std::ofstream outfile(*context->tpu_metrics_dump_path);
      if (outfile.is_open()) {
        outfile << output << "\n";
        outfile.close();
        LITERT_LOG(LITERT_INFO, "Google TensorHook: TPU metrics dumped to %s",
                   context->tpu_metrics_dump_path->c_str());
      }
    }
  }

  context->total_tpu_tile_time_us = 0;
  context->inference_count = 0;
  context->tpu_metrics_dump_path = std::nullopt;
  context->dump_tpu_tile_time_metrics = false;
  context->initialized = false;
}

}  // namespace litert::google_tensor
