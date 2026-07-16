// Copyright 2025 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_MODEL_BUILDER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_MODEL_BUILDER_H_

#include <limits>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "ml_drift_delegate/tflite/shared_const_tensor_map.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/core/api/op_resolver.h"
#include "tflite/model_builder.h"

namespace litert::ml_drift {

// Options for parsing tflite model with model builder.
struct ModelBuilderOptions {
  // Set to enforce capping inf/-inf to max float values for the softmax input
  // and padding.
  bool enable_infinite_float_capping = false;
  // Set to enforce max f16 for the max float value when capping softmax input
  // and padding.
  bool enable_reduced_precision = false;
  // Set to propagate pointers to the tflite weights instead of copying them to
  // GraphFloat32 operations.
  bool enable_raw_weights_propagation = false;
  // Set to avoid copying model weights to the GraphFloat32 attributes.
  bool enable_spanned_weights = false;
  // Set to allow/disallow boolean tensors.
  // TODO - b/483403743: Remove this once OpenGL supports boolean tensors.
  bool allow_bool_tensors = true;
};

// Checks if the given TFLiteNode is compatible with GPU kernels.
// 'allow_quant_ops', whether to allow quantization or not.
// 'excluded_ops', if not null, specifies a set of ops that should not be
// run with GPU kernels.
absl::Status CheckIfSupportedNode(
    const TfLiteContext* context, const TfLiteNode* node,
    const TfLiteRegistration* registration, bool allow_quant_ops = false,
    const ModelBuilderOptions& options = {},
    const absl::flat_hash_set<TfLiteBuiltinOperator>* excluded_ops = nullptr,
    TFLiteStablehloCompositeParserFactory* composite_parser_factory = nullptr);

// Validates which operations are supported and returns array of operations to
// replace with GPU kernels. The caller must free the pointer on TfLiteIntArray.
// 'max_delegated_partitions' limits the maximum number of partitions to
// delegate as a graph could possibly have multiple partitions (each partition
// consists of a subset of ops) to be replaced.
// 'excluded_ops', if not null, specifies a set of ops that should not be
// replaced with GPU kernels.
TfLiteIntArray* GetOpsToReplace(
    TfLiteContext* context, bool allow_quant_ops = false,
    int max_delegated_partitions = 1,
    const absl::flat_hash_set<TfLiteBuiltinOperator>* excluded_ops = nullptr,
    int start_node_index = 0,
    int end_node_index = std::numeric_limits<int>::max(),
    TFLiteStablehloCompositeParserFactory* composite_parser_factory = nullptr);

// Same as GetOpsToReplace, but allows passing options to control the
// model building process.
TfLiteIntArray* GetOpsToReplaceWithOptions(
    TfLiteContext* context, bool allow_quant_ops,
    const ModelBuilderOptions& options, int max_delegated_partitions,
    const absl::flat_hash_set<TfLiteBuiltinOperator>* excluded_ops = nullptr,
    int start_node_index = 0,
    int end_node_index = std::numeric_limits<int>::max(),
    TFLiteStablehloCompositeParserFactory* composite_parser_factory = nullptr);

// Extracts TFLite delegate execution plan from the input TFLite context and
// converts it into generic graph format.
//
// If model is quantized, quant_conversion_map maps the dequantized tensor
// (floating-point) to the original tensor (fixed-point) & vice-versa.
// NOTE: Not all of these new tensors will have any data and need memory
// allocated for them. We need to do that only for the overall GPU graph inputs
// & outputs. This should be done by the delegate, by setting the appropriate
// TfLiteNode->temporaries.
// 'tensor_to_buffer_id_map' maps TFLite tensor indices to ML Drift buffer IDs.
// 'tensor_to_external_buffer_id_map' maps TFLite tensor indices to external
// buffer IDs. The external buffer ID is used to identify a weight buffer that
// is managed externally. This is introduced to reduce externalize the weight
// buffer from TFLite model file.
absl::Status BuildModel(
    TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
    const ModelBuilderOptions& options, ::ml_drift::GraphFloat32* graph,
    absl::flat_hash_map<int, int>* quant_conversion_map = nullptr,
    SharedConstTensorsMap* shared_tensors = nullptr,
    const TensorIndexToBufferIdMap* tensor_to_buffer_id_map = nullptr,
    const TensorIndexToExternalBufferIdMap* tensor_to_external_buffer_id_map =
        nullptr,
    TFLiteStablehloCompositeParserFactory* composite_parser_factory = nullptr);

// Same as BuildModel, but enforces user-provided input/output indices instead
// of using delegate_params->inputs and delegate_params->outputs for
// inputs/outputs preallocating.
absl::Status BuildModelEnforceIO(
    TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
    const ModelBuilderOptions& options, const std::vector<int>& input_ids,
    const std::vector<int>& output_ids, ::ml_drift::GraphFloat32* graph,
    absl::flat_hash_map<int, int>* quant_conversion_map = nullptr,
    SharedConstTensorsMap* shared_tensors = nullptr,
    const TensorIndexToBufferIdMap* tensor_to_buffer_id_map = nullptr,
    const TensorIndexToExternalBufferIdMap* tensor_to_external_buffer_id_map =
        nullptr,
    TFLiteStablehloCompositeParserFactory* composite_parser_factory = nullptr);

// Same as above but also apply all transformations on the final graph.
// Prefer using this method instead of BuildModel.
//
// If model is quantized, quant_conversion_map maps the dequantized tensor
// (floating-point) to the original TFLite tensor (fixed-point) & vice-versa.
// NOTE: Not all of these new tensors will have any data and need memory
// allocated for them. We need to do that only for the overall GPU graph inputs
// & outputs. This should be done by the delegate, by setting the appropriate
// TfLiteNode->temporaries.
absl::Status BuildFinalModel(
    TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
    const ModelBuilderOptions& options, ::ml_drift::GraphFloat32* graph,
    absl::flat_hash_map<int, int>* quant_conversion_map = nullptr,
    SharedConstTensorsMap* shared_tensors = nullptr,
    const TensorIndexToBufferIdMap* tensor_to_buffer_id_map = nullptr,
    const TensorIndexToExternalBufferIdMap* tensor_to_external_buffer_id_map =
        nullptr,
    TFLiteStablehloCompositeParserFactory* composite_parser_factory = nullptr);

// Convenience wrapper that builds a GraphFloat32 from the provided
// FlatBufferModel.
// TODO: b/378522761 - Merge these two calls together, move allow_quant_ops
// param to ModelBuilderOptions.
absl::Status BuildFromFlatBuffer(const tflite::FlatBufferModel& flatbuffer,
                                 const tflite::OpResolver& op_resolver,
                                 ::ml_drift::GraphFloat32* graph,
                                 bool allow_quant_ops = false,
                                 bool apply_model_transformations = true);
absl::Status BuildFromFlatBuffer(const tflite::FlatBufferModel& flatbuffer,
                                 const tflite::OpResolver& op_resolver,
                                 const ModelBuilderOptions& options,
                                 ::ml_drift::GraphFloat32* graph,
                                 bool allow_quant_ops = false,
                                 bool apply_model_transformations = true);

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_MODEL_BUILDER_H_
