// Copyright 2026 Google LLC.
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

#ifndef ODML_LITERT_LITERT_COMPILER_CC_LITERT_OP_OPTIONS_H_
#define ODML_LITERT_LITERT_COMPILER_CC_LITERT_OP_OPTIONS_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_tfl_types.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert::compiler {

/// @brief Base struct for operator options.
struct OpOptions {
  virtual LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                                  LiteRtOp op) = 0;
  virtual ~OpOptions() = default;
};

/// @brief Struct to hold options for LiteRT composite ops.
struct CompositeOptions : public OpOptions {
  /// Name for special composites representing manual partitions.
  static constexpr absl::string_view kNpuCall = "odml.npu_call";
  static constexpr absl::string_view kCpuCall = "odml.cpu_call";
  static constexpr absl::string_view kRmsNorm = "odml.rms_norm";
  static constexpr absl::string_view kL2Norm = "odml.l2_norm";
  static constexpr absl::string_view kGroupNorm = "odml.group_norm";

  /// The root op.
  LiteRtOp op;
  /// Decomposition subgraph.
  int subgraph;
  /// The name of the composite op (stored in model).
  absl::string_view name;
  /// The version of the composite op.
  int32_t version;
  /// The attributes of the composite op.
  std::optional<flexbuffers::Map> attributes_map;

  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_shlo_composite_op_name == nullptr ||
        ctx->get_shlo_composite_op_decomposition_subgraph_index == nullptr ||
        ctx->get_shlo_composite_op_version == nullptr ||
        ctx->get_shlo_composite_op_attributes == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }

    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeShloComposite) {
      return kLiteRtStatusErrorInvalidArgument;
    }

    const char* op_name;
    LITERT_RETURN_IF_ERROR(ctx->get_shlo_composite_op_name(op, &op_name));
    name = op_name;

    LITERT_RETURN_IF_ERROR(
        ctx->get_shlo_composite_op_decomposition_subgraph_index(op, &subgraph));
    LITERT_RETURN_IF_ERROR(ctx->get_shlo_composite_op_version(op, &version));

    const uint8_t* impl_attributes = nullptr;
    int32_t impl_attributes_size = 0;
    LITERT_RETURN_IF_ERROR(ctx->get_shlo_composite_op_attributes(
        op, &impl_attributes, &impl_attributes_size));

    if (impl_attributes_size < 0) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    if (impl_attributes_size > 0) {
      if (impl_attributes == nullptr ||
          !flexbuffers::VerifyBuffer(
              impl_attributes, static_cast<size_t>(impl_attributes_size))) {
        return kLiteRtStatusErrorInvalidArgument;
      }
      auto root = flexbuffers::GetRoot(impl_attributes, impl_attributes_size);
      if (!root.IsMap()) {
        return kLiteRtStatusErrorInvalidArgument;
      }
      attributes_map = root.AsMap();
    }
    this->op = op;

    return kLiteRtStatusOk;
  }
};

struct RmsNormOpts : public CompositeOptions {
  /// The epsilon composite attribute of the RMS norm.
  float epsilon;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp litert_op) override {
    LITERT_RETURN_IF_ERROR(CompositeOptions::InitFromOp(ctx, litert_op));
    if (!attributes_map.has_value()) {
      return kLiteRtStatusErrorInvalidArgument;
    }

    constexpr char kEpsilonKey[] = "epsilon";
    flexbuffers::Reference raw_epsilon = attributes_map.value()[kEpsilonKey];
    if (raw_epsilon.IsNull()) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    epsilon = raw_epsilon.AsFloat();
    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT Add op.
struct AddOptions : public OpOptions {
  LiteRtOp op;
  ActivationFunction fused_activation_function;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_add_fused_activation_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflAdd) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(
        ctx->get_add_fused_activation_option(op, &fused_activation_function));

    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT BatchMatmul op.
struct BatchMatmulOptions : public OpOptions {
  LiteRtOp op;
  bool adj_x;
  bool adj_y;
  bool asymmetric_quantize_input;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_batch_matmul_adj_x_option == nullptr ||
        ctx->get_batch_matmul_adj_y_option == nullptr ||
        ctx->get_batch_matmul_asymmetric_quantize_input_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflBatchMatmul) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(ctx->get_batch_matmul_adj_x_option(op, &adj_x));
    LITERT_RETURN_IF_ERROR(ctx->get_batch_matmul_adj_y_option(op, &adj_y));
    LITERT_RETURN_IF_ERROR(
        ctx->get_batch_matmul_asymmetric_quantize_input_option(
            op, &asymmetric_quantize_input));

    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT Concatenation op.
struct ConcatenationOptions : public OpOptions {
  LiteRtOp op;
  ActivationFunction fused_activation_function;
  int32_t axis;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_concatenation_axis_option == nullptr ||
        ctx->get_concatenation_fused_activation_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflConcatenation) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(ctx->get_concatenation_axis_option(op, &axis));
    LITERT_RETURN_IF_ERROR(ctx->get_concatenation_fused_activation_option(
        op, &fused_activation_function));

    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT Div op.
struct DivOptions : public OpOptions {
  LiteRtOp op;
  ActivationFunction fused_activation_function;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_div_fused_activation_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflDiv) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(
        ctx->get_div_fused_activation_option(op, &fused_activation_function));

    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT FullyConnected op.
struct FullyConnectedOptions : public OpOptions {
  LiteRtOp op;
  ActivationFunction fused_activation_function = 1;
  FullyConnectedOptionsWeightsFormat weights_format;
  bool keep_num_dims;
  LiteRtElementType quantized_bias_type;
  bool asymmetric_quantize_input;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_fully_connected_fused_activation_option == nullptr ||
        ctx->get_fully_connected_weights_format_option == nullptr ||
        ctx->get_fully_connected_keep_num_dims_option == nullptr ||
        ctx->get_fully_connected_quantized_bias_type_option == nullptr ||
        ctx->get_fully_connected_asymmetric_quantize_input_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflFullyConnected) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(ctx->get_fully_connected_fused_activation_option(
        op, &fused_activation_function));
    LITERT_RETURN_IF_ERROR(
        ctx->get_fully_connected_weights_format_option(op, &weights_format));
    LITERT_RETURN_IF_ERROR(
        ctx->get_fully_connected_keep_num_dims_option(op, &keep_num_dims));
    uint32_t retrieved_quantized_bias_type;
    LITERT_RETURN_IF_ERROR(ctx->get_fully_connected_quantized_bias_type_option(
        op, &retrieved_quantized_bias_type));
    quantized_bias_type = GetElementType(retrieved_quantized_bias_type);
    LITERT_RETURN_IF_ERROR(
        ctx->get_fully_connected_asymmetric_quantize_input_option(
            op, &asymmetric_quantize_input));

    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT Mul op.
struct MulOptions : public OpOptions {
  LiteRtOp op;
  ActivationFunction fused_activation_function;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_mul_fused_activation_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflMul) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(
        ctx->get_mul_fused_activation_option(op, &fused_activation_function));

    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT Softmax op.
struct SoftmaxOptions : public OpOptions {
  LiteRtOp op;
  float beta;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_softmax_beta_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflSoftmax) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(ctx->get_softmax_beta_option(op, &beta));

    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT StridedSlice op.
struct StridedSliceOptions : public OpOptions {
  LiteRtOp op;
  int32_t begin_mask;
  int32_t end_mask;
  int32_t ellipsis_mask;
  int32_t new_axis_mask;
  int32_t shrink_axis_mask;
  bool offset;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_strided_slice_begin_mask_option == nullptr ||
        ctx->get_strided_slice_end_mask_option == nullptr ||
        ctx->get_strided_slice_ellipsis_mask_option == nullptr ||
        ctx->get_strided_slice_new_axis_mask_option == nullptr ||
        ctx->get_strided_slice_shrink_axis_mask_option == nullptr ||
        ctx->get_strided_slice_offset_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflStridedSlice) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(
        ctx->get_strided_slice_begin_mask_option(op, &begin_mask));
    LITERT_RETURN_IF_ERROR(
        ctx->get_strided_slice_end_mask_option(op, &end_mask));
    LITERT_RETURN_IF_ERROR(
        ctx->get_strided_slice_ellipsis_mask_option(op, &ellipsis_mask));
    LITERT_RETURN_IF_ERROR(
        ctx->get_strided_slice_new_axis_mask_option(op, &new_axis_mask));
    LITERT_RETURN_IF_ERROR(
        ctx->get_strided_slice_shrink_axis_mask_option(op, &shrink_axis_mask));
    LITERT_RETURN_IF_ERROR(ctx->get_strided_slice_offset_option(op, &offset));

    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT Sub op.
struct SubOptions : public OpOptions {
  LiteRtOp op;
  ActivationFunction fused_activation_function;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_sub_fused_activation_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflSub) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(
        ctx->get_sub_fused_activation_option(op, &fused_activation_function));

    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT Reshape op.
struct ReshapeOptions : public OpOptions {
  LiteRtOp op;
  std::vector<int32_t> new_shape;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_reshape_new_shape_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflReshape) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    const int32_t* new_shape_data;
    int32_t new_shape_size;
    LITERT_RETURN_IF_ERROR(ctx->get_reshape_new_shape_option(
        op, &new_shape_data, &new_shape_size));
    new_shape.assign(new_shape_data, new_shape_data + new_shape_size);

    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT Sum op.
struct SumOptions : public OpOptions {
  LiteRtOp op;
  bool keep_dims;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_sum_keep_dims_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflSum) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(ctx->get_sum_keep_dims_option(op, &keep_dims));

    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT ReduceMax op.
struct ReduceMaxOptions : public OpOptions {
  LiteRtOp op;
  bool keep_dims;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_reduce_max_keep_dims_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflReduceMax) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(
        ctx->get_reduce_max_keep_dims_option(op, &keep_dims));

    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT ReduceMin op.
struct ReduceMinOptions : public OpOptions {
  LiteRtOp op;
  bool keep_dims;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_reduce_min_keep_dims_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflReduceMin) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(
        ctx->get_reduce_min_keep_dims_option(op, &keep_dims));

    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT ReduceAny op.
struct ReduceAnyOptions : public OpOptions {
  LiteRtOp op;
  bool keep_dims;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_reduce_any_keep_dims_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflReduceAny) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(
        ctx->get_reduce_any_keep_dims_option(op, &keep_dims));

    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT ReduceAll op.
struct ReduceAllOptions : public OpOptions {
  LiteRtOp op;
  bool keep_dims;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_reduce_all_keep_dims_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflReduceAll) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(
        ctx->get_reduce_all_keep_dims_option(op, &keep_dims));

    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT Pack op.
struct PackOptions : public OpOptions {
  LiteRtOp op;
  int32_t axis;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_pack_axis_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflPack) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(ctx->get_pack_axis_option(op, &axis));
    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT Unpack op.
struct UnpackOptions : public OpOptions {
  LiteRtOp op;
  int32_t axis;
  int32_t num;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_unpack_axis_option == nullptr ||
        ctx->get_unpack_num_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflUnpack) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(ctx->get_unpack_axis_option(op, &axis));
    LITERT_RETURN_IF_ERROR(ctx->get_unpack_num_option(op, &num));
    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT Gather op.
struct GatherOptions : public OpOptions {
  LiteRtOp op;
  int32_t axis;
  int32_t batch_dims;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_gather_axis_option == nullptr ||
        ctx->get_gather_batch_dims_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflGather) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(ctx->get_gather_axis_option(op, &axis));
    LITERT_RETURN_IF_ERROR(ctx->get_gather_batch_dims_option(op, &batch_dims));
    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT Mean op.
struct MeanOptions : public OpOptions {
  LiteRtOp op;
  bool keep_dims;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_mean_keep_dims_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflMean) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(ctx->get_mean_keep_dims_option(op, &keep_dims));
    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT Split op.
struct SplitOptions : public OpOptions {
  LiteRtOp op;
  int32_t num_splits;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_split_num_splits_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflSplit) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(ctx->get_split_num_splits_option(op, &num_splits));
    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT Conv2d op.
struct Conv2dOptions : public OpOptions {
  LiteRtOp op;
  Padding padding;
  int32_t stride_w;
  int32_t stride_h;
  int32_t dilation_w_factor;
  int32_t dilation_h_factor;
  ActivationFunction fused_activation_function;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_conv_2d_padding_option == nullptr ||
        ctx->get_conv_2d_stride_w_option == nullptr ||
        ctx->get_conv_2d_stride_h_option == nullptr ||
        ctx->get_conv_2d_dilation_w_option == nullptr ||
        ctx->get_conv_2d_dilation_h_option == nullptr ||
        ctx->get_conv_2d_fused_activation_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflConv2d) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(ctx->get_conv_2d_padding_option(op, &padding));
    LITERT_RETURN_IF_ERROR(ctx->get_conv_2d_stride_w_option(op, &stride_w));
    LITERT_RETURN_IF_ERROR(ctx->get_conv_2d_stride_h_option(op, &stride_h));
    LITERT_RETURN_IF_ERROR(
        ctx->get_conv_2d_dilation_w_option(op, &dilation_w_factor));
    LITERT_RETURN_IF_ERROR(
        ctx->get_conv_2d_dilation_h_option(op, &dilation_h_factor));
    LITERT_RETURN_IF_ERROR(ctx->get_conv_2d_fused_activation_option(
        op, &fused_activation_function));
    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT Conv3d op.
struct Conv3dOptions : public OpOptions {
  LiteRtOp op;
  Padding padding;
  int32_t stride_w;
  int32_t stride_h;
  int32_t stride_d;
  int32_t dilation_w_factor;
  int32_t dilation_h_factor;
  int32_t dilation_d_factor;
  ActivationFunction fused_activation_function;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_conv_3d_padding_option == nullptr ||
        ctx->get_conv_3d_stride_w_option == nullptr ||
        ctx->get_conv_3d_stride_h_option == nullptr ||
        ctx->get_conv_3d_stride_d_option == nullptr ||
        ctx->get_conv_3d_dilation_w_option == nullptr ||
        ctx->get_conv_3d_dilation_h_option == nullptr ||
        ctx->get_conv_3d_dilation_d_option == nullptr ||
        ctx->get_conv_3d_fused_activation_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflConv3d) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(ctx->get_conv_3d_padding_option(op, &padding));
    LITERT_RETURN_IF_ERROR(ctx->get_conv_3d_stride_w_option(op, &stride_w));
    LITERT_RETURN_IF_ERROR(ctx->get_conv_3d_stride_h_option(op, &stride_h));
    LITERT_RETURN_IF_ERROR(ctx->get_conv_3d_stride_d_option(op, &stride_d));
    LITERT_RETURN_IF_ERROR(
        ctx->get_conv_3d_dilation_w_option(op, &dilation_w_factor));
    LITERT_RETURN_IF_ERROR(
        ctx->get_conv_3d_dilation_h_option(op, &dilation_h_factor));
    LITERT_RETURN_IF_ERROR(
        ctx->get_conv_3d_dilation_d_option(op, &dilation_d_factor));
    LITERT_RETURN_IF_ERROR(ctx->get_conv_3d_fused_activation_option(
        op, &fused_activation_function));
    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT DepthwiseConv2d op.
struct DepthwiseConv2dOptions : public OpOptions {
  LiteRtOp op;
  Padding padding;
  int32_t stride_w;
  int32_t stride_h;
  int32_t depth_multiplier;
  ActivationFunction fused_activation_function;
  int32_t dilation_w_factor;
  int32_t dilation_h_factor;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_depthwise_conv_2d_padding_option == nullptr ||
        ctx->get_depthwise_conv_2d_stride_w_option == nullptr ||
        ctx->get_depthwise_conv_2d_stride_h_option == nullptr ||
        ctx->get_depthwise_conv_2d_depth_multiplier_option == nullptr ||
        ctx->get_depthwise_conv_2d_fused_activation_option == nullptr ||
        ctx->get_depthwise_conv_2d_dilation_w_option == nullptr ||
        ctx->get_depthwise_conv_2d_dilation_h_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflDepthwiseConv2d) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(
        ctx->get_depthwise_conv_2d_padding_option(op, &padding));
    LITERT_RETURN_IF_ERROR(
        ctx->get_depthwise_conv_2d_stride_w_option(op, &stride_w));
    LITERT_RETURN_IF_ERROR(
        ctx->get_depthwise_conv_2d_stride_h_option(op, &stride_h));
    LITERT_RETURN_IF_ERROR(ctx->get_depthwise_conv_2d_depth_multiplier_option(
        op, &depth_multiplier));
    LITERT_RETURN_IF_ERROR(ctx->get_depthwise_conv_2d_fused_activation_option(
        op, &fused_activation_function));
    LITERT_RETURN_IF_ERROR(
        ctx->get_depthwise_conv_2d_dilation_w_option(op, &dilation_w_factor));
    LITERT_RETURN_IF_ERROR(
        ctx->get_depthwise_conv_2d_dilation_h_option(op, &dilation_h_factor));
    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT TransposeConv op.
struct TransposeConvOptions : public OpOptions {
  LiteRtOp op;
  Padding padding;
  int32_t stride_w;
  int32_t stride_h;
  ActivationFunction fused_activation_function;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_transpose_conv_padding_option == nullptr ||
        ctx->get_transpose_conv_stride_w_option == nullptr ||
        ctx->get_transpose_conv_stride_h_option == nullptr ||
        ctx->get_transpose_conv_fused_activation_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflTransposeConv) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(
        ctx->get_transpose_conv_padding_option(op, &padding));
    LITERT_RETURN_IF_ERROR(
        ctx->get_transpose_conv_stride_w_option(op, &stride_w));
    LITERT_RETURN_IF_ERROR(
        ctx->get_transpose_conv_stride_h_option(op, &stride_h));
    LITERT_RETURN_IF_ERROR(ctx->get_transpose_conv_fused_activation_option(
        op, &fused_activation_function));
    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT AveragePool2d op.
struct AveragePool2dOptions : public OpOptions {
  LiteRtOp op;
  Padding padding;
  int32_t stride_w;
  int32_t stride_h;
  int32_t filter_width;
  int32_t filter_height;
  ActivationFunction fused_activation_function;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_average_pool_2d_padding_option == nullptr ||
        ctx->get_average_pool_2d_stride_w_option == nullptr ||
        ctx->get_average_pool_2d_stride_h_option == nullptr ||
        ctx->get_average_pool_2d_filter_width_option == nullptr ||
        ctx->get_average_pool_2d_filter_height_option == nullptr ||
        ctx->get_average_pool_2d_fused_activation_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflAveragePool2d) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(
        ctx->get_average_pool_2d_padding_option(op, &padding));
    LITERT_RETURN_IF_ERROR(
        ctx->get_average_pool_2d_stride_w_option(op, &stride_w));
    LITERT_RETURN_IF_ERROR(
        ctx->get_average_pool_2d_stride_h_option(op, &stride_h));
    LITERT_RETURN_IF_ERROR(
        ctx->get_average_pool_2d_filter_width_option(op, &filter_width));
    LITERT_RETURN_IF_ERROR(
        ctx->get_average_pool_2d_filter_height_option(op, &filter_height));
    LITERT_RETURN_IF_ERROR(ctx->get_average_pool_2d_fused_activation_option(
        op, &fused_activation_function));
    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT MaxPool2d op.
struct MaxPool2dOptions : public OpOptions {
  LiteRtOp op;
  Padding padding;
  int32_t stride_w;
  int32_t stride_h;
  int32_t filter_width;
  int32_t filter_height;
  ActivationFunction fused_activation_function;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_max_pool_2d_padding_option == nullptr ||
        ctx->get_max_pool_2d_stride_w_option == nullptr ||
        ctx->get_max_pool_2d_stride_h_option == nullptr ||
        ctx->get_max_pool_2d_filter_width_option == nullptr ||
        ctx->get_max_pool_2d_filter_height_option == nullptr ||
        ctx->get_max_pool_2d_fused_activation_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflMaxPool2d) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(ctx->get_max_pool_2d_padding_option(op, &padding));
    LITERT_RETURN_IF_ERROR(ctx->get_max_pool_2d_stride_w_option(op, &stride_w));
    LITERT_RETURN_IF_ERROR(ctx->get_max_pool_2d_stride_h_option(op, &stride_h));
    LITERT_RETURN_IF_ERROR(
        ctx->get_max_pool_2d_filter_width_option(op, &filter_width));
    LITERT_RETURN_IF_ERROR(
        ctx->get_max_pool_2d_filter_height_option(op, &filter_height));
    LITERT_RETURN_IF_ERROR(ctx->get_max_pool_2d_fused_activation_option(
        op, &fused_activation_function));
    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT L2Pool2d op.
struct L2Pool2dOptions : public OpOptions {
  LiteRtOp op;
  Padding padding;
  int32_t stride_w;
  int32_t stride_h;
  int32_t filter_width;
  int32_t filter_height;
  ActivationFunction fused_activation_function;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_l2_pool_2d_padding_option == nullptr ||
        ctx->get_l2_pool_2d_stride_w_option == nullptr ||
        ctx->get_l2_pool_2d_stride_h_option == nullptr ||
        ctx->get_l2_pool_2d_filter_width_option == nullptr ||
        ctx->get_l2_pool_2d_filter_height_option == nullptr ||
        ctx->get_l2_pool_2d_fused_activation_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflL2Pool2d) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(ctx->get_l2_pool_2d_padding_option(op, &padding));
    LITERT_RETURN_IF_ERROR(ctx->get_l2_pool_2d_stride_w_option(op, &stride_w));
    LITERT_RETURN_IF_ERROR(ctx->get_l2_pool_2d_stride_h_option(op, &stride_h));
    LITERT_RETURN_IF_ERROR(
        ctx->get_l2_pool_2d_filter_width_option(op, &filter_width));
    LITERT_RETURN_IF_ERROR(
        ctx->get_l2_pool_2d_filter_height_option(op, &filter_height));
    LITERT_RETURN_IF_ERROR(ctx->get_l2_pool_2d_fused_activation_option(
        op, &fused_activation_function));
    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT ResizeBilinear op.
struct ResizeBilinearOptions : public OpOptions {
  LiteRtOp op;
  bool align_corners;
  bool half_pixel_centers;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_resize_bilinear_align_corners_option == nullptr ||
        ctx->get_resize_bilinear_half_pixel_center_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflResizeBilinear) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(
        ctx->get_resize_bilinear_align_corners_option(op, &align_corners));
    LITERT_RETURN_IF_ERROR(ctx->get_resize_bilinear_half_pixel_center_option(
        op, &half_pixel_centers));
    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT LeakyRelu op.
struct LeakyReluOptions : public OpOptions {
  LiteRtOp op;
  float alpha;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_leaky_relu_alpha_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflLeakyRelu) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(ctx->get_leaky_relu_alpha_option(op, &alpha));
    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT SpaceToDepth op.
struct SpaceToDepthOptions : public OpOptions {
  LiteRtOp op;
  int32_t block_size;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_space_to_depth_block_size_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflSpaceToDepth) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(
        ctx->get_space_to_depth_block_size_option(op, &block_size));
    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT DepthToSpace op.
struct DepthToSpaceOptions : public OpOptions {
  LiteRtOp op;
  int32_t block_size;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_depth_to_space_block_size_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflDepthToSpace) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(
        ctx->get_depth_to_space_block_size_option(op, &block_size));
    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT ResizeNearestNeighbor op.
struct ResizeNearestNeighborOptions : public OpOptions {
  LiteRtOp op;
  bool align_corners;
  bool half_pixel_centers;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_resize_nearest_neighbor_align_corners_option == nullptr ||
        ctx->get_resize_nearest_neighbor_half_pixel_center_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflResizeNearestNeighbor) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(
        ctx->get_resize_nearest_neighbor_align_corners_option(op,
                                                              &align_corners));
    LITERT_RETURN_IF_ERROR(
        ctx->get_resize_nearest_neighbor_half_pixel_center_option(
            op, &half_pixel_centers));
    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT CumSum op.
struct CumSumOptions : public OpOptions {
  LiteRtOp op;
  bool exclusive;
  bool reverse;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_cumsum_exclusive_option == nullptr ||
        ctx->get_cumsum_reverse_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflCumsum) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(ctx->get_cumsum_exclusive_option(op, &exclusive));
    LITERT_RETURN_IF_ERROR(ctx->get_cumsum_reverse_option(op, &reverse));
    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT Gelu op.
struct GeluOptions : public OpOptions {
  LiteRtOp op;
  bool approximate;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_gelu_approximate_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflGelu) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(ctx->get_gelu_approximate_option(op, &approximate));
    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT MirrorPad op.
struct MirrorPadOptions : public OpOptions {
  LiteRtOp op;
  MirrorPadMode mode;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_mirror_pad_mode_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflMirrorPad) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(ctx->get_mirror_pad_mode_option(op, &mode));
    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT Squeeze op.
struct SqueezeOptions : public OpOptions {
  LiteRtOp op;
  std::vector<int32_t> squeeze_dims;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_squeeze_dims_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflSqueeze) {
      return kLiteRtStatusErrorInvalidArgument;
    }

    const int32_t* squeeze_dims_data;
    int32_t num_squeeze_dims;
    LITERT_RETURN_IF_ERROR(ctx->get_squeeze_dims_option(op, &squeeze_dims_data,
                                                        &num_squeeze_dims));
    squeeze_dims.assign(squeeze_dims_data,
                        squeeze_dims_data + num_squeeze_dims);

    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Struct to hold options for the LiteRT OneHot op.
struct OneHotOptions : public OpOptions {
  LiteRtOp op;
  int32_t axis;
  LiteRtStatus InitFromOp(const LiteRtCompilerContext* ctx,
                          LiteRtOp op) override {
    if (ctx == nullptr || ctx->get_op_code == nullptr ||
        ctx->get_one_hot_axis_option == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtOpCode opcode;
    LITERT_RETURN_IF_ERROR(ctx->get_op_code(op, &opcode));
    if (opcode != kLiteRtOpCodeTflOneHot) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(ctx->get_one_hot_axis_option(op, &axis));
    this->op = op;

    return kLiteRtStatusOk;
  }
};

/// @brief Returns the composite info for the given op if it is a composite op.
template <typename OptionsT>
Expected<OptionsT> GetOptionsAs(const LiteRtCompilerContext* ctx, LiteRtOp op) {
  OptionsT options;
  auto status = options.InitFromOp(ctx, op);
  if (status != kLiteRtStatusOk) {
    return Unexpected(ToStatus(status));
  }
  return options;
}

}  // namespace litert::compiler

#endif  // ODML_LITERT_LITERT_COMPILER_CC_LITERT_OP_OPTIONS_H_
