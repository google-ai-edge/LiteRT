// Copyright 2026 The ML Drift Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_RUNTIME_BATCHED_MATMUL_PARSER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_RUNTIME_BATCHED_MATMUL_PARSER_H_

#include <optional>

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

constexpr const char kRuntimeBatchedMatMulType[] = "runtime_batched_matmul";

// Runtime check params used by FullyConnected and BatchedMatMul ops.
// See ConvRuntimeCheckDesc in common/task/gpu_operation.h for more details.
struct RuntimeCheckParams {
  std::optional<int> src_end_ch_index;
  std::optional<int> dst_end_ch_index;
};

struct ExternalWeightsAttributes {
  ::ml_drift::WeightsDescription desc;
  ::ml_drift::OHWI weights_shape;
};

struct RuntimeBatchedMatMulAttributes {
  RuntimeCheckParams runtime_check;
  // BMM case
  std::optional<bool> transpose_left = false;
  std::optional<bool> transpose_right = false;
  // FC case
  std::optional<ExternalWeightsAttributes> external_weights;
  // quantized case
  std::optional<
      ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>>
      scale;
};

class RuntimeBatchedMatMulOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration*) final;

  void Parse(const TfLiteNode* tflite_node, const TfLiteRegistration*,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_RUNTIME_BATCHED_MATMUL_PARSER_H_
