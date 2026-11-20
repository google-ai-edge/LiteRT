/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensor/backends/ml_drift/ml_drift_conversion.h"

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "tensor/backends/ml_drift/arithmetic_ml_drift.h"
#include "tensor/backends/ml_drift/ml_drift_helpers.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/graph_traversal.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {

namespace {

using ::ml_drift::GpuModelBuilder;

}  // namespace

absl::StatusOr<
    absl::flat_hash_map<graph::Tensor, GpuModelBuilder::TensorHandle>>
AppendGpuModelBuilder(MlDriftBuildContext& ctx,
                      std::vector<TensorHandle> outputs) {
  LRT_TENSOR_ASSIGN_OR_RETURN(
      std::vector<const graph::Operation*> execution_plan,
      GetExecutionPlan(outputs));

  // `LRT_COMPILE_TIMING=1`: logs the conversion time per op type.
  const bool timing = std::getenv("LRT_COMPILE_TIMING") != nullptr;
  absl::flat_hash_map<std::string, std::pair<int, absl::Duration>> op_times;
  for (const graph::Operation* op : execution_plan) {
    auto mld_op = op->GetExtension<MlDriftOperation>();
    if (mld_op == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Operation ", op->GetName(),
                       " does not implement MlDriftOperation."));
    }
    const absl::Time start = timing ? absl::Now() : absl::InfinitePast();
    LRT_TENSOR_RETURN_IF_ERROR(mld_op->ToMlDrift(*op, ctx));
    if (timing) {
      auto& entry = op_times[op->GetName()];
      ++entry.first;
      entry.second += absl::Now() - start;
    }
  }
  for (const auto& [name, entry] : op_times) {
    ABSL_LOG(INFO) << "LRT_COMPILE_OP " << name << " n=" << entry.first
                   << " ms=" << absl::ToDoubleMilliseconds(entry.second);
  }
  return std::move(ctx.graph_tensor_to_handle);
}

}  // namespace litert::tensor
