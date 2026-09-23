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

#include "litert/vendors/qualcomm/compiler/transformations/entry_embedding_transformation.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_matchers.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/compiler/cc/litert_op_options.h"
using litert::BuildLayout;
using litert::ElementType;
using litert::Layout;
using litert::RankedTensorType;
using litert::compiler::AddOptions;
using litert::compiler::Builder;
using litert::compiler::GatherOptions;
using litert::compiler::m_AllOf;
using litert::compiler::m_Any;
using litert::compiler::m_AnyOf;
using litert::compiler::m_CaptureOrSameAs;
using litert::compiler::m_ElementType;
using litert::compiler::m_IsConstant;
using litert::compiler::m_Op;
using litert::compiler::m_Shape;
using litert::compiler::Match;
using litert::compiler::Op;
using litert::compiler::RankedTensorSpecBuilder;
using litert::compiler::Tensor;

namespace {

constexpr int32_t kBatchSize = 1;
constexpr int32_t kSeqLen = 630;
constexpr int32_t kEmbeddingDim = 768;
constexpr int32_t kCoordDepth = 10240;

Tensor FindCoordTensorFallback(const Tensor& fc_in) {
  auto def_op = fc_in.GetDefiningOp();
  if (!def_op) return Tensor();
  if (def_op->Code() == kLiteRtOpCodeTflOneHot) {
    if (!def_op->Inputs().empty()) {
      return def_op->Inputs()[0];
    }
  }
  for (const auto& in : def_op->Inputs()) {
    auto in_def_op = in.GetDefiningOp();
    if (in_def_op && in_def_op->Code() == kLiteRtOpCodeTflOneHot) {
      if (!in_def_op->Inputs().empty()) {
        return in_def_op->Inputs()[0];
      }
    }
  }
  return Tensor();
}

}  // namespace

extern "C" {

LiteRtStatus EntryEmbeddingTransformation(const LiteRtCompilerContext* context,
                                          LiteRtBuilder builder_ptr,
                                          LiteRtOp op) {
  Builder builder(context, builder_ptr);
  Op root_op(context, op);

  Op concat_op(context, nullptr);
  Op reshape_x_op(context, nullptr);
  Op reshape_y_op(context, nullptr);
  Op fc_x_op(context, nullptr);
  Op fc_y_op(context, nullptr);
  Tensor fc_in_x(context, nullptr);
  Tensor fc_in_y(context, nullptr);
  Tensor coord_x(context, nullptr);
  Tensor coord_y(context, nullptr);
  Tensor w_x(context, nullptr);
  Tensor w_y(context, nullptr);

  auto one_hot_pattern = [](Tensor* coord) {
    return m_Op<kLiteRtOpCodeTflOneHot>(
        m_CaptureOrSameAs(coord, m_Shape({kBatchSize, kSeqLen})), m_Any(),
        m_Any(), m_Any());
  };

  auto fc_in_pattern = [&](Tensor* in, Tensor* coord) {
    return m_CaptureOrSameAs(
        in, m_AnyOf(one_hot_pattern(coord),
                    m_Op<kLiteRtOpCodeTflSelectV2>(
                        m_Any(), one_hot_pattern(coord), m_Any()),
                    m_Op<kLiteRtOpCodeTflSelectV2>(m_Any(), m_Any(),
                                                   one_hot_pattern(coord)),
                    m_Op<kLiteRtOpCodeTflSelect>(
                        m_Any(), one_hot_pattern(coord), m_Any()),
                    m_Op<kLiteRtOpCodeTflSelect>(m_Any(), m_Any(),
                                                 one_hot_pattern(coord)),
                    m_Any()));
  };

  auto fc_pattern = [&](Op* fc_op, Tensor* in, Tensor* coord, Tensor* w) {
    auto w_matcher = m_CaptureOrSameAs(
        w, m_AllOf(m_IsConstant(), m_Shape({kEmbeddingDim, kCoordDepth})));
    return m_CaptureOrSameAs(
        fc_op, m_AnyOf(m_Op<kLiteRtOpCodeTflFullyConnected>(
                           fc_in_pattern(in, coord), w_matcher),
                       m_Op<kLiteRtOpCodeTflFullyConnected>(
                           fc_in_pattern(in, coord), w_matcher, m_Any())));
  };

  auto reshape_x_pattern = m_CaptureOrSameAs(
      &reshape_x_op,
      m_Op<kLiteRtOpCodeTflReshape>(
          fc_pattern(&fc_x_op, &fc_in_x, &coord_x, &w_x), m_Any()));

  auto reshape_y_pattern = m_CaptureOrSameAs(
      &reshape_y_op,
      m_Op<kLiteRtOpCodeTflReshape>(
          fc_pattern(&fc_y_op, &fc_in_y, &coord_y, &w_y), m_Any()));

  auto concat_pattern = m_CaptureOrSameAs(
      &concat_op, m_AnyOf(m_Op<kLiteRtOpCodeTflConcatenation>(
                              reshape_x_pattern, reshape_y_pattern),
                          m_Op<kLiteRtOpCodeTflConcatenation>(
                              reshape_y_pattern, reshape_x_pattern)));

  if (!Match(root_op, m_Op<kLiteRtOpCodeTflSum>(concat_pattern, m_Any()))) {
    return kLiteRtStatusPatternNoMatch;
  }

  if (root_op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }

  Tensor sum_out = root_op.Outputs()[0];
  if (!Match(sum_out, m_AllOf(m_ElementType(kLiteRtElementTypeFloat32),
                              m_Shape({kBatchSize, kSeqLen, kEmbeddingDim})))) {
    return kLiteRtStatusPatternNoMatch;
  }

  // Resolve coord_x and coord_y: captured declaratively or fallback.
  if (!coord_x.Get()) {
    coord_x = FindCoordTensorFallback(fc_in_x);
  }
  if (!coord_y.Get()) {
    coord_y = FindCoordTensorFallback(fc_in_y);
  }
  if (!coord_x.Get() || !coord_y.Get()) {
    return kLiteRtStatusPatternNoMatch;
  }
  if (!Match(coord_x, m_Shape({kBatchSize, kSeqLen})) ||
      !Match(coord_y, m_Shape({kBatchSize, kSeqLen}))) {
    return kLiteRtStatusPatternNoMatch;
  }

  // 5. Transpose weights [768, 10240] -> [10240, 768]
  constexpr size_t num_rows = kEmbeddingDim;
  constexpr size_t num_cols = kCoordDepth;
  auto w_x_bytes = w_x.Weights().Bytes();
  auto w_y_bytes = w_y.Weights().Bytes();
  if (w_x_bytes.size() != num_rows * num_cols * sizeof(float) ||
      w_y_bytes.size() != num_rows * num_cols * sizeof(float)) {
    return kLiteRtStatusPatternNoMatch;
  }

  const float* w_x_data = reinterpret_cast<const float*>(w_x_bytes.data());
  const float* w_y_data = reinterpret_cast<const float*>(w_y_bytes.data());

  std::vector<float> w_x_t(num_rows * num_cols);
  std::vector<float> w_y_t(num_rows * num_cols);
  for (size_t r = 0; r < num_rows; ++r) {
    for (size_t c = 0; c < num_cols; ++c) {
      w_x_t[c * num_rows + r] = w_x_data[r * num_cols + c];
      w_y_t[c * num_rows + r] = w_y_data[r * num_cols + c];
    }
  }

  // 6. Build transposed weight tensors
  const int32_t w_t_dims[2] = {kCoordDepth, kEmbeddingDim};
  RankedTensorType w_t_type(ElementType::Float32,
                            Layout(BuildLayout(w_t_dims, w_t_dims + 2)));
  auto w_x_t_tensor_res =
      builder.BuildTensor(RankedTensorSpecBuilder(w_t_type).Build());
  auto w_y_t_tensor_res =
      builder.BuildTensor(RankedTensorSpecBuilder(w_t_type).Build());
  if (!w_x_t_tensor_res || !w_y_t_tensor_res) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  Tensor w_x_t_tensor = *w_x_t_tensor_res;
  Tensor w_y_t_tensor = *w_y_t_tensor_res;

  auto w_x_build_res =
      builder.BuildWeights<float>(absl::MakeConstSpan(w_x_t), w_x_t_tensor);
  auto w_y_build_res =
      builder.BuildWeights<float>(absl::MakeConstSpan(w_y_t), w_y_t_tensor);
  if (!w_x_build_res || !w_y_build_res) {
    return kLiteRtStatusErrorRuntimeFailure;
  }

  // 7. Build Gather output tensors [1, 630, 768]
  const int32_t out_dims[3] = {kBatchSize, kSeqLen, kEmbeddingDim};
  RankedTensorType out_type(ElementType::Float32,
                            Layout(BuildLayout(out_dims, out_dims + 3)));
  auto out_x_res =
      builder.BuildTensor(RankedTensorSpecBuilder(out_type).Build());
  auto out_y_res =
      builder.BuildTensor(RankedTensorSpecBuilder(out_type).Build());
  if (!out_x_res || !out_y_res) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  Tensor out_x = *out_x_res;
  Tensor out_y = *out_y_res;

  // 8. Build Gather ops
  auto gather_x_res =
      builder.BuildOp(kLiteRtOpCodeTflGather, {w_x_t_tensor, coord_x}, {out_x});
  auto gather_y_res =
      builder.BuildOp(kLiteRtOpCodeTflGather, {w_y_t_tensor, coord_y}, {out_y});
  if (!gather_x_res || !gather_y_res) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  Op gather_x = *gather_x_res;
  Op gather_y = *gather_y_res;

  GatherOptions gather_options_x;
  gather_options_x.axis = 0;
  gather_options_x.batch_dims = 0;
  auto opt_status_x =
      builder.SetOpOptions(gather_x, std::move(gather_options_x));
  if (!opt_status_x) {
    return opt_status_x.Error().Status();
  }

  GatherOptions gather_options_y;
  gather_options_y.axis = 0;
  gather_options_y.batch_dims = 0;
  auto opt_status_y =
      builder.SetOpOptions(gather_y, std::move(gather_options_y));
  if (!opt_status_y) {
    return opt_status_y.Error().Status();
  }

  // 9. Build Add op writing directly to sum_out
  auto add_op_res =
      builder.BuildOp(kLiteRtOpCodeTflAdd, {out_x, out_y}, {sum_out});
  if (!add_op_res) {
    return add_op_res.Error().Status();
  }
  Op add_op = *add_op_res;
  AddOptions add_options;
  add_options.fused_activation_function = 0;
  builder.SetOpOptions(add_op, std::move(add_options));

  // 10. Clean up obsolete ops
  builder.EraseOp(root_op);
  builder.EraseOp(concat_op);
  builder.EraseOp(reshape_x_op);
  builder.EraseOp(reshape_y_op);
  builder.EraseOp(fc_x_op);
  builder.EraseOp(fc_y_op);

  return kLiteRtStatusOk;
}

}  // extern "C"
