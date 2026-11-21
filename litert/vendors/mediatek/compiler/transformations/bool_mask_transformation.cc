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

#include "litert/vendors/mediatek/compiler/transformations/bool_mask_transformation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/compiler/cc/litert_op_options.h"

namespace {

using litert::BuildLayout;
using litert::ElementType;
using litert::Expected;
using litert::Layout;
using litert::RankedTensorType;
using litert::Unexpected;
using litert::compiler::AddOptions;
using litert::compiler::BatchMatmulOptions;
using litert::compiler::Builder;
using litert::compiler::ConcatenationOptions;
using litert::compiler::Conv2dOptions;
using litert::compiler::DepthwiseConv2dOptions;
using litert::compiler::DivOptions;
using litert::compiler::FullyConnectedOptions;
using litert::compiler::GetOptionsAs;
using litert::compiler::MeanOptions;
using litert::compiler::MulOptions;
using litert::compiler::Op;
using litert::compiler::RankedTensorSpecBuilder;
using litert::compiler::ReduceMaxOptions;
using litert::compiler::ReduceMinOptions;
using litert::compiler::ReshapeOptions;
using litert::compiler::SoftmaxOptions;
using litert::compiler::SplitOptions;
using litert::compiler::SplitVOptions;
using litert::compiler::SqueezeOptions;
using litert::compiler::StridedSliceOptions;
using litert::compiler::SubOptions;
using litert::compiler::SumOptions;
using litert::compiler::Tensor;

// Limits constant tensor synthesis (e.g. same-shape NotEqual zero buffers) so
// unexpectedly large masks are not materialized as dense constants.
constexpr size_t kMaxMaterializedConstElements = 1u << 20;
constexpr int kMaxZeroOneRecursionDepth = 24;

bool IsUnquantizedOfType(const Tensor& tensor, ElementType type) {
  return tensor.Get() != nullptr && tensor.ElementType() == type &&
         !tensor.HasQuantization();
}

// Returns static positive dimensions for ranked tensors (empty vector for
// rank-0 scalars). Rejects unranked, dynamic (dim < 0), and zero-sized
// (dim == 0) tensors, as NeuroPilot operands require strictly positive dims.
std::optional<std::vector<int32_t>> GetStaticDims(const Tensor& tensor) {
  if (tensor.Get() == nullptr || tensor.TypeId() != kLiteRtRankedTensorType) {
    return std::nullopt;
  }
  auto rtt = tensor.RankedTensorType();
  if (!rtt) {
    return std::nullopt;
  }
  const auto dims = rtt->Layout().Dimensions();
  for (int32_t dim : dims) {
    if (dim <= 0) {
      return std::nullopt;
    }
  }
  return std::vector<int32_t>(dims.begin(), dims.end());
}

std::optional<size_t> NumElements(absl::Span<const int32_t> dims) {
  size_t count = 1;
  for (int32_t dim : dims) {
    if (dim <= 0) {
      return std::nullopt;
    }
    const size_t udim = static_cast<size_t>(dim);
    if (count > (std::numeric_limits<size_t>::max)() / udim) {
      return std::nullopt;
    }
    count *= udim;
  }
  return count;
}

std::optional<std::vector<int32_t>> BroadcastShapes(
    absl::Span<const int32_t> a, absl::Span<const int32_t> b) {
  for (int32_t d : a) {
    if (d <= 0) {
      return std::nullopt;
    }
  }
  for (int32_t d : b) {
    if (d <= 0) {
      return std::nullopt;
    }
  }
  const size_t rank = (std::max)(a.size(), b.size());
  std::vector<int32_t> out(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    const int32_t da = (i < a.size()) ? a[a.size() - 1 - i] : 1;
    const int32_t db = (i < b.size()) ? b[b.size() - 1 - i] : 1;
    if (da != db && da != 1 && db != 1) {
      return std::nullopt;
    }
    out[rank - 1 - i] = (da == 1) ? db : da;
  }
  return out;
}

Expected<Tensor> BuildTensorOfType(const Builder& builder, ElementType type,
                                   absl::Span<const int32_t> dims) {
  return builder.BuildTensor(
      RankedTensorSpecBuilder(
          RankedTensorType(type, Layout(BuildLayout(
                                     dims.data(), dims.data() + dims.size()))))
          .Build());
}

template <typename T>
Expected<Tensor> BuildConst(const Builder& builder,
                            absl::Span<const int32_t> dims,
                            absl::Span<const T> values) {
  LITERT_ASSIGN_OR_RETURN(
      Tensor tensor,
      BuildTensorOfType(builder, litert::GetElementType<T>(), dims));
  LITERT_RETURN_IF_ERROR(builder.BuildWeights<T>(values, tensor));
  return tensor;
}

Expected<Tensor> BuildScalarFloatConst(const Builder& builder, float value) {
  return BuildConst<float>(builder, {}, absl::Span<const float>(&value, 1));
}

bool IsUniformFloatConst(const Tensor& tensor, float target,
                         bool allow_negative_zero = false) {
  if (!IsUnquantizedOfType(tensor, ElementType::Float32) ||
      !tensor.IsConstant()) {
    return false;
  }
  auto weights = tensor.WeightsData<float>();
  if (!weights || weights->empty()) {
    return false;
  }
  for (float v : *weights) {
    // Rejects NaN as well as any inexact value (NaN == target is false).
    if (!(v == target)) {
      return false;
    }
    if (target == 0.0f && !allow_negative_zero && std::signbit(v)) {
      return false;
    }
  }
  return true;
}

bool IsZeroFloatConst(const Tensor& tensor) {
  return IsUniformFloatConst(tensor, 0.0f, /*allow_negative_zero=*/false);
}

bool IsAnyZeroFloatConst(const Tensor& tensor) {
  return IsUniformFloatConst(tensor, 0.0f, /*allow_negative_zero=*/true);
}

bool IsOneFloatConst(const Tensor& tensor) {
  return IsUniformFloatConst(tensor, 1.0f);
}

bool IsBinaryFloatConst(const Tensor& tensor) {
  if (!IsUnquantizedOfType(tensor, ElementType::Float32) ||
      !tensor.IsConstant()) {
    return false;
  }
  auto weights = tensor.WeightsData<float>();
  if (!weights || weights->empty()) {
    return false;
  }
  for (float v : *weights) {
    if (!((v == 0.0f && !std::signbit(v)) || v == 1.0f)) {
      return false;
    }
  }
  return true;
}

bool HasNonFiniteFloatConst(const Tensor& tensor) {
  if (tensor.Get() == nullptr || tensor.ElementType() != ElementType::Float32 ||
      !tensor.IsConstant()) {
    return false;
  }
  auto weights = tensor.WeightsData<float>();
  if (!weights || weights->empty()) {
    return true;
  }
  for (float v : *weights) {
    if (!std::isfinite(v)) {
      return true;
    }
  }
  return false;
}

bool MayPropagateNonFiniteFloatConstImpl(
    const Tensor& tensor, int depth,
    absl::flat_hash_map<LiteRtTensor, int>& memo) {
  if (depth < 0 || tensor.Get() == nullptr ||
      tensor.ElementType() != ElementType::Float32) {
    return false;
  }
  if (tensor.IsConstant()) {
    return HasNonFiniteFloatConst(tensor);
  }
  if (auto it = memo.find(tensor.Get()); it != memo.end()) {
    if (it->second == (std::numeric_limits<int>::max)() ||
        it->second >= depth) {
      return it->second == (std::numeric_limits<int>::max)();
    }
  }
  bool result = false;
  if (auto def = tensor.DefiningOp(); def.has_value()) {
    Op op(tensor.Context(), def->op);
    switch (op.Code()) {
      case kLiteRtOpCodeTflSelect:
      case kLiteRtOpCodeTflSelectV2:
        result =
            op.Inputs().size() == 3 && (MayPropagateNonFiniteFloatConstImpl(
                                            op.Inputs()[1], depth - 1, memo) ||
                                        MayPropagateNonFiniteFloatConstImpl(
                                            op.Inputs()[2], depth - 1, memo));
        break;
      case kLiteRtOpCodeTflReshape:
      case kLiteRtOpCodeTflSqueeze:
      case kLiteRtOpCodeTflExpandDims:
      case kLiteRtOpCodeTflSlice:
      case kLiteRtOpCodeTflStridedSlice:
      case kLiteRtOpCodeTflTile:
      case kLiteRtOpCodeTflTranspose:
      case kLiteRtOpCodeTflPad:
        result = !op.Inputs().empty() && MayPropagateNonFiniteFloatConstImpl(
                                             op.Inputs()[0], depth - 1, memo);
        break;
      case kLiteRtOpCodeTflPadv2:
        result =
            op.Inputs().size() == 3 && (MayPropagateNonFiniteFloatConstImpl(
                                            op.Inputs()[0], depth - 1, memo) ||
                                        MayPropagateNonFiniteFloatConstImpl(
                                            op.Inputs()[2], depth - 1, memo));
        break;
      case kLiteRtOpCodeTflConcatenation:
        for (const Tensor& in : op.Inputs()) {
          if (MayPropagateNonFiniteFloatConstImpl(in, depth - 1, memo)) {
            result = true;
            break;
          }
        }
        break;
      default:
        break;
    }
  }
  memo[tensor.Get()] = result ? (std::numeric_limits<int>::max)() : depth;
  return result;
}

bool MayPropagateNonFiniteFloatConst(const Tensor& tensor) {
  absl::flat_hash_map<LiteRtTensor, int> memo;
  return MayPropagateNonFiniteFloatConstImpl(tensor, kMaxZeroOneRecursionDepth,
                                             memo);
}

// Per-query memo for IsKnownZeroOneFloat. Maps a tensor to
// kZeroOneMemoKnownTrue once it is proven {0.0, 1.0}, or otherwise to the
// largest remaining depth at which it was found not provable. Without it,
// multi-input ops whose inputs share producers (e.g. `Mul(t, t)` chains) would
// be re-explored exponentially in the recursion depth.
using ZeroOneMemo = absl::flat_hash_map<LiteRtTensor, int>;
constexpr int kZeroOneMemoKnownTrue = (std::numeric_limits<int>::max)();

bool IsKnownZeroOneFloatImpl(const Tensor& tensor, int depth,
                             ZeroOneMemo& memo);

// Returns true when a DepthwiseConv2d selects at most one spatial {0.0, 1.0}
// sample per output channel with zero bias and non-negative/identity
// activation, preserving the {0.0, 1.0} value invariant.
bool IsZeroOneDepthwiseConv2d(const Op& op, int depth, ZeroOneMemo& memo) {
  if (op.Inputs().size() < 2 || op.Inputs().size() > 3) {
    return false;
  }
  if (!IsKnownZeroOneFloatImpl(op.Inputs()[0], depth - 1, memo)) {
    return false;
  }
  if (op.Inputs().size() == 3 && !IsZeroFloatConst(op.Inputs()[2])) {
    return false;
  }
  const Tensor filter = op.Inputs()[1];
  if (!IsUnquantizedOfType(filter, ElementType::Float32) ||
      !filter.IsConstant()) {
    return false;
  }
  const auto filter_dims = GetStaticDims(filter);
  if (!filter_dims || filter_dims->size() != 4 || (*filter_dims)[0] != 1) {
    return false;
  }
  const auto expected_elems = NumElements(*filter_dims);
  auto weights = filter.WeightsData<float>();
  if (!expected_elems || !weights || weights->size() != *expected_elems) {
    return false;
  }
  const size_t c_out = static_cast<size_t>((*filter_dims)[3]);
  std::vector<int> ones_per_channel(c_out, 0);
  for (size_t i = 0; i < weights->size(); ++i) {
    const float w = (*weights)[i];
    if (w == 0.0f && !std::signbit(w)) {
      continue;
    }
    if (w == 1.0f) {
      if (++ones_per_channel[i % c_out] > 1) {
        return false;
      }
      continue;
    }
    return false;
  }
  auto opts = GetOptionsAs<DepthwiseConv2dOptions>(op.Context(), op.Get());
  if (!opts || opts->stride_w <= 0 || opts->stride_h <= 0 ||
      opts->dilation_w_factor <= 0 || opts->dilation_h_factor <= 0 ||
      opts->depth_multiplier <= 0) {
    return false;
  }
  // Allow None (0), Relu (1), or Relu6 (3), which all leave {0.0, 1.0} intact.
  return opts->fused_activation_function == 0 ||
         opts->fused_activation_function == 1 ||
         opts->fused_activation_function == 3;
}

bool IsIdentityActivationOnZeroOne(uint32_t activation) {
  // None (0), Relu (1), and Relu6 (3) preserve {+0.0f, 1.0f}.
  return activation == 0 || activation == 1 || activation == 3;
}

// Uncached body of IsKnownZeroOneFloatImpl; recurses through the memo.
bool ComputeIsKnownZeroOneFloat(const Tensor& tensor, int depth,
                                ZeroOneMemo& memo) {
  if (tensor.IsConstant()) {
    return IsBinaryFloatConst(tensor);
  }
  auto def_op_info = tensor.DefiningOp();
  if (!def_op_info) {
    return false;
  }
  Op op(tensor.Context(), def_op_info->op);
  switch (op.Code()) {
    case kLiteRtOpCodeTflCast:
      return op.Inputs().size() == 1 &&
             IsUnquantizedOfType(op.Inputs()[0], ElementType::Bool) &&
             GetStaticDims(op.Inputs()[0]).has_value();
    case kLiteRtOpCodeTflReshape:
    case kLiteRtOpCodeTflSqueeze:
    case kLiteRtOpCodeTflExpandDims:
    case kLiteRtOpCodeTflSlice:
    case kLiteRtOpCodeTflStridedSlice:
    case kLiteRtOpCodeTflTile:
    case kLiteRtOpCodeTflTranspose:
    case kLiteRtOpCodeTflPad:
      return !op.Inputs().empty() &&
             IsKnownZeroOneFloatImpl(op.Inputs()[0], depth - 1, memo);
    case kLiteRtOpCodeTflPadv2:
      return op.Inputs().size() == 3 &&
             IsKnownZeroOneFloatImpl(op.Inputs()[0], depth - 1, memo) &&
             (IsZeroFloatConst(op.Inputs()[2]) ||
              IsOneFloatConst(op.Inputs()[2]));
    case kLiteRtOpCodeTflConcatenation: {
      if (op.Inputs().empty()) {
        return false;
      }
      auto opts = GetOptionsAs<ConcatenationOptions>(op.Context(), op.Get());
      if (!opts ||
          !IsIdentityActivationOnZeroOne(opts->fused_activation_function)) {
        return false;
      }
      for (const Tensor& in : op.Inputs()) {
        if (!IsKnownZeroOneFloatImpl(in, depth - 1, memo)) {
          return false;
        }
      }
      return true;
    }
    case kLiteRtOpCodeTflSub: {
      if (op.Inputs().size() != 2 || !IsOneFloatConst(op.Inputs()[0]) ||
          !IsKnownZeroOneFloatImpl(op.Inputs()[1], depth - 1, memo)) {
        return false;
      }
      auto opts = GetOptionsAs<SubOptions>(op.Context(), op.Get());
      return opts && opts->fused_activation_function == 0;
    }
    case kLiteRtOpCodeTflAbs: {
      if (op.Inputs().size() != 1) {
        return false;
      }
      const Tensor arg = op.Inputs()[0];
      if (IsKnownZeroOneFloatImpl(arg, depth - 1, memo)) {
        return true;
      }
      if (depth < 2 || !IsUnquantizedOfType(arg, ElementType::Float32) ||
          !GetStaticDims(arg)) {
        return false;
      }
      auto sub_def = arg.DefiningOp();
      if (!sub_def) {
        return false;
      }
      Op sub_op(op.Context(), sub_def->op);
      if (sub_op.Code() != kLiteRtOpCodeTflSub || sub_op.Inputs().size() != 2) {
        return false;
      }
      auto sub_opts = GetOptionsAs<SubOptions>(op.Context(), sub_op.Get());
      return sub_opts && sub_opts->fused_activation_function == 0 &&
             IsKnownZeroOneFloatImpl(sub_op.Inputs()[0], depth - 2, memo) &&
             IsKnownZeroOneFloatImpl(sub_op.Inputs()[1], depth - 2, memo);
    }
    case kLiteRtOpCodeTflMul: {
      if (op.Inputs().size() != 2 ||
          !IsKnownZeroOneFloatImpl(op.Inputs()[0], depth - 1, memo) ||
          !IsKnownZeroOneFloatImpl(op.Inputs()[1], depth - 1, memo)) {
        return false;
      }
      auto opts = GetOptionsAs<MulOptions>(op.Context(), op.Get());
      return opts &&
             IsIdentityActivationOnZeroOne(opts->fused_activation_function);
    }
    case kLiteRtOpCodeTflMinimum:
    case kLiteRtOpCodeTflMaximum:
      return op.Inputs().size() == 2 &&
             IsKnownZeroOneFloatImpl(op.Inputs()[0], depth - 1, memo) &&
             IsKnownZeroOneFloatImpl(op.Inputs()[1], depth - 1, memo);
    case kLiteRtOpCodeTflDepthwiseConv2d:
      return IsZeroOneDepthwiseConv2d(op, depth, memo);
    default:
      return false;
  }
}

bool IsKnownZeroOneFloatImpl(const Tensor& tensor, int depth,
                             ZeroOneMemo& memo) {
  if (depth < 0 || !IsUnquantizedOfType(tensor, ElementType::Float32) ||
      !GetStaticDims(tensor)) {
    return false;
  }
  // A proven result holds at any depth; a negative result is only reusable
  // when it was computed with at least as much remaining depth. Each tensor is
  // therefore evaluated at most kMaxZeroOneRecursionDepth + 1 times.
  if (auto it = memo.find(tensor.Get()); it != memo.end()) {
    if (it->second == kZeroOneMemoKnownTrue || it->second >= depth) {
      return it->second == kZeroOneMemoKnownTrue;
    }
  }
  const bool result = ComputeIsKnownZeroOneFloat(tensor, depth, memo);
  memo[tensor.Get()] = result ? kZeroOneMemoKnownTrue : depth;
  return result;
}

// Returns true when `tensor` is an unquantized Float32 tensor with static
// positive dimensions provably holding only {+0.0f, 1.0f} values.
bool IsKnownZeroOneFloat(const Tensor& tensor) {
  ZeroOneMemo memo;
  return IsKnownZeroOneFloatImpl(tensor, kMaxZeroOneRecursionDepth, memo);
}

Expected<Tensor> BuildBinaryFloatOp(const Builder& builder, LiteRtOpCode code,
                                    const Tensor& lhs, const Tensor& rhs,
                                    const Tensor& out) {
  LITERT_ASSIGN_OR_RETURN(Op new_op, builder.BuildOp(code, {lhs, rhs}, {out}));
  if (code == kLiteRtOpCodeTflAdd) {
    AddOptions opts;
    opts.fused_activation_function = 0;
    LITERT_RETURN_IF_ERROR(builder.SetOpOptions(new_op, std::move(opts)));
  } else if (code == kLiteRtOpCodeTflSub) {
    SubOptions opts;
    opts.fused_activation_function = 0;
    LITERT_RETURN_IF_ERROR(builder.SetOpOptions(new_op, std::move(opts)));
  } else if (code == kLiteRtOpCodeTflMul) {
    MulOptions opts;
    opts.fused_activation_function = 0;
    LITERT_RETURN_IF_ERROR(builder.SetOpOptions(new_op, std::move(opts)));
  }
  return out;
}

Expected<Tensor> BuildBinaryFloatOpWithDims(const Builder& builder,
                                            LiteRtOpCode code,
                                            const Tensor& lhs,
                                            const Tensor& rhs,
                                            absl::Span<const int32_t> dims) {
  LITERT_ASSIGN_OR_RETURN(
      Tensor out, BuildTensorOfType(builder, ElementType::Float32, dims));
  return BuildBinaryFloatOp(builder, code, lhs, rhs, out);
}

Expected<Tensor> ConvertBoolConstToFloat(const Builder& builder,
                                         const Tensor& bool_const,
                                         absl::Span<const int32_t> dims) {
  if (!IsUnquantizedOfType(bool_const, ElementType::Bool)) {
    return Unexpected(kLiteRtStatusPatternNoMatch);
  }
  const auto num_elems = NumElements(dims);
  if (!num_elems || !bool_const.HasWeights()) {
    return Unexpected(kLiteRtStatusPatternNoMatch);
  }
  const absl::Span<const uint8_t> bytes = bool_const.Weights().Bytes();
  if (bytes.size() != *num_elems) {
    return Unexpected(kLiteRtStatusPatternNoMatch);
  }
  std::vector<float> floats;
  floats.reserve(bytes.size());
  for (uint8_t b : bytes) {
    floats.push_back(b != 0 ? 1.0f : 0.0f);
  }
  return BuildConst<float>(builder, dims, floats);
}

bool CanResolveBoolAsZeroOneFloat(const Tensor& tensor,
                                  absl::Span<const int32_t> dims) {
  const auto tensor_dims = GetStaticDims(tensor);
  if (!tensor_dims || absl::MakeConstSpan(*tensor_dims) != dims) {
    return false;
  }
  if (IsUnquantizedOfType(tensor, ElementType::Float32)) {
    return IsKnownZeroOneFloat(tensor);
  }
  if (!IsUnquantizedOfType(tensor, ElementType::Bool)) {
    return false;
  }
  if (tensor.IsConstant()) {
    const auto num_elems = NumElements(dims);
    return num_elems && tensor.HasWeights() &&
           tensor.Weights().Bytes().size() == *num_elems;
  }
  auto def = tensor.DefiningOp();
  if (!def.has_value()) {
    return true;
  }
  Op def_op(tensor.Context(), def->op);
  if (def_op.Code() == kLiteRtOpCodeTflCast && def_op.Inputs().size() == 1) {
    const Tensor src = def_op.Inputs()[0];
    const auto src_dims = GetStaticDims(src);
    return src_dims && absl::MakeConstSpan(*src_dims) == dims &&
           IsKnownZeroOneFloat(src);
  }
  // Never synthesize a Cast(Bool->F32) after an unrewritten producer op (such
  // as integer Less/GreaterEqual/NotEqual, ReduceAll, or arbitrary-float
  // NotEqual): doing so prematurely breaks downstream multi-op patterns
  // (EntryEmbedding, FloorDiv, AttentionMask) and creates NPU-CPU ping-pong
  // islands.
  return false;
}

// Resolves an unquantized Bool tensor (or proven {0.0, 1.0} Float32 tensor)
// into a Float32 {0.0, 1.0} tensor positioned safely at the current op's
// insertion point.
Expected<Tensor> ResolveBoolAsZeroOneFloat(const Builder& builder,
                                           const Tensor& tensor,
                                           absl::Span<const int32_t> dims) {
  if (!CanResolveBoolAsZeroOneFloat(tensor, dims)) {
    return Unexpected(kLiteRtStatusPatternNoMatch);
  }
  if (IsUnquantizedOfType(tensor, ElementType::Float32)) {
    return tensor;
  }
  if (tensor.IsConstant()) {
    return ConvertBoolConstToFloat(builder, tensor, dims);
  }
  if (auto def = tensor.DefiningOp(); def.has_value()) {
    Op def_op(tensor.Context(), def->op);
    return def_op.Inputs()[0];
  }
  // Materialize a local Cast for Bool subgraph inputs rather than sharing
  // another consumer's Cast, which could reorder defs across ApplyChanges
  // splice points.
  LITERT_ASSIGN_OR_RETURN(
      Tensor f32_out, BuildTensorOfType(builder, ElementType::Float32, dims));
  LITERT_RETURN_IF_ERROR(
      builder.BuildOp(kLiteRtOpCodeTflCast, {tensor}, {f32_out}));
  return f32_out;
}

std::optional<Tensor> TryBypassDoubleBoolCast(const Builder& builder,
                                              const Tensor& input,
                                              absl::Span<const int32_t> dims) {
  if (!IsUnquantizedOfType(input, ElementType::Float32)) {
    return std::nullopt;
  }
  auto def = input.DefiningOp();
  if (!def.has_value()) {
    return std::nullopt;
  }
  Op cast_op(input.Context(), def->op);
  if (cast_op.Code() != kLiteRtOpCodeTflCast || cast_op.Inputs().size() != 1) {
    return std::nullopt;
  }
  const Tensor bool_src = cast_op.Inputs()[0];
  const auto bool_dims = GetStaticDims(bool_src);
  if (!IsUnquantizedOfType(bool_src, ElementType::Bool) || !bool_dims ||
      absl::MakeConstSpan(*bool_dims) != dims) {
    return std::nullopt;
  }
  if (bool_src.IsConstant()) {
    auto res = ConvertBoolConstToFloat(builder, bool_src, dims);
    return res.HasValue() ? std::optional<Tensor>(*res) : std::nullopt;
  }
  auto bool_def = bool_src.DefiningOp();
  if (!bool_def.has_value()) {
    return std::nullopt;
  }
  Op inner_cast(input.Context(), bool_def->op);
  if (inner_cast.Code() != kLiteRtOpCodeTflCast ||
      inner_cast.Inputs().size() != 1) {
    return std::nullopt;
  }
  const Tensor orig_f32 = inner_cast.Inputs()[0];
  const auto orig_dims = GetStaticDims(orig_f32);
  if (!orig_dims || absl::MakeConstSpan(*orig_dims) != dims ||
      !IsKnownZeroOneFloat(orig_f32)) {
    return std::nullopt;
  }
  return orig_f32;
}

template <typename OptionsT>
Expected<void> CopyOptions(const LiteRtCompilerContext* context,
                           const Builder& builder, const Op& src,
                           const Op& dst) {
  LITERT_ASSIGN_OR_RETURN(OptionsT opts,
                          GetOptionsAs<OptionsT>(context, src.Get()));
  return builder.SetOpOptions(dst, std::move(opts));
}

bool CanCloneOpWithNewInputs(LiteRtOpCode code) {
  switch (code) {
    case kLiteRtOpCodeTflAbs:
    case kLiteRtOpCodeTflAdd:
    case kLiteRtOpCodeTflBatchMatmul:
    case kLiteRtOpCodeTflCast:
    case kLiteRtOpCodeTflConcatenation:
    case kLiteRtOpCodeTflConv2d:
    case kLiteRtOpCodeTflDepthwiseConv2d:
    case kLiteRtOpCodeTflDequantize:
    case kLiteRtOpCodeTflDiv:
    case kLiteRtOpCodeTflEqual:
    case kLiteRtOpCodeTflExpandDims:
    case kLiteRtOpCodeTflFullyConnected:
    case kLiteRtOpCodeTflGreater:
    case kLiteRtOpCodeTflGreaterEqual:
    case kLiteRtOpCodeTflLess:
    case kLiteRtOpCodeTflLessEqual:
    case kLiteRtOpCodeTflLogicalAnd:
    case kLiteRtOpCodeTflLogicalNot:
    case kLiteRtOpCodeTflLogicalOr:
    case kLiteRtOpCodeTflLogistic:
    case kLiteRtOpCodeTflMaximum:
    case kLiteRtOpCodeTflMean:
    case kLiteRtOpCodeTflMinimum:
    case kLiteRtOpCodeTflMul:
    case kLiteRtOpCodeTflNeg:
    case kLiteRtOpCodeTflNotEqual:
    case kLiteRtOpCodeTflPad:
    case kLiteRtOpCodeTflPadv2:
    case kLiteRtOpCodeTflQuantize:
    case kLiteRtOpCodeTflReduceMax:
    case kLiteRtOpCodeTflReduceMin:
    case kLiteRtOpCodeTflRelu:
    case kLiteRtOpCodeTflReshape:
    case kLiteRtOpCodeTflRsqrt:
    case kLiteRtOpCodeTflSelect:
    case kLiteRtOpCodeTflSelectV2:
    case kLiteRtOpCodeTflSlice:
    case kLiteRtOpCodeTflSoftmax:
    case kLiteRtOpCodeTflSplit:
    case kLiteRtOpCodeTflSplitV:
    case kLiteRtOpCodeTflSqrt:
    case kLiteRtOpCodeTflSquaredDifference:
    case kLiteRtOpCodeTflSqueeze:
    case kLiteRtOpCodeTflStridedSlice:
    case kLiteRtOpCodeTflSub:
    case kLiteRtOpCodeTflSum:
    case kLiteRtOpCodeTflTanh:
    case kLiteRtOpCodeTflTile:
    case kLiteRtOpCodeTflTranspose:
      return true;
    default:
      return false;
  }
}

bool CanReadOpOptions(const LiteRtCompilerContext* context, const Op& src) {
  switch (src.Code()) {
    case kLiteRtOpCodeTflAdd:
      return GetOptionsAs<AddOptions>(context, src.Get()).HasValue();
    case kLiteRtOpCodeTflBatchMatmul:
      return GetOptionsAs<BatchMatmulOptions>(context, src.Get()).HasValue();
    case kLiteRtOpCodeTflConcatenation:
      return GetOptionsAs<ConcatenationOptions>(context, src.Get()).HasValue();
    case kLiteRtOpCodeTflConv2d:
      return GetOptionsAs<Conv2dOptions>(context, src.Get()).HasValue();
    case kLiteRtOpCodeTflDepthwiseConv2d:
      return GetOptionsAs<DepthwiseConv2dOptions>(context, src.Get())
          .HasValue();
    case kLiteRtOpCodeTflDiv:
      return GetOptionsAs<DivOptions>(context, src.Get()).HasValue();
    case kLiteRtOpCodeTflFullyConnected:
      return GetOptionsAs<FullyConnectedOptions>(context, src.Get()).HasValue();
    case kLiteRtOpCodeTflMean:
      return GetOptionsAs<MeanOptions>(context, src.Get()).HasValue();
    case kLiteRtOpCodeTflMul:
      return GetOptionsAs<MulOptions>(context, src.Get()).HasValue();
    case kLiteRtOpCodeTflReduceMax:
      return GetOptionsAs<ReduceMaxOptions>(context, src.Get()).HasValue();
    case kLiteRtOpCodeTflReduceMin:
      return GetOptionsAs<ReduceMinOptions>(context, src.Get()).HasValue();
    case kLiteRtOpCodeTflReshape: {
      const auto out_dims = src.Outputs().size() == 1
                                ? GetStaticDims(src.Outputs()[0])
                                : std::nullopt;
      if (!out_dims || out_dims->empty()) {
        return false;
      }
      return GetOptionsAs<ReshapeOptions>(context, src.Get()).HasValue();
    }
    case kLiteRtOpCodeTflSoftmax:
      return GetOptionsAs<SoftmaxOptions>(context, src.Get()).HasValue();
    case kLiteRtOpCodeTflSplit:
      return GetOptionsAs<SplitOptions>(context, src.Get()).HasValue();
    case kLiteRtOpCodeTflSplitV:
      return GetOptionsAs<SplitVOptions>(context, src.Get()).HasValue();
    case kLiteRtOpCodeTflSqueeze:
      return GetOptionsAs<SqueezeOptions>(context, src.Get()).HasValue();
    case kLiteRtOpCodeTflStridedSlice:
      return GetOptionsAs<StridedSliceOptions>(context, src.Get()).HasValue();
    case kLiteRtOpCodeTflSub:
      return GetOptionsAs<SubOptions>(context, src.Get()).HasValue();
    case kLiteRtOpCodeTflSum:
      return GetOptionsAs<SumOptions>(context, src.Get()).HasValue();
    default:
      return CanCloneOpWithNewInputs(src.Code());
  }
}

Expected<void> CopyOpOptions(const LiteRtCompilerContext* context,
                             const Builder& builder, const Op& src,
                             const Op& dst) {
  switch (src.Code()) {
    case kLiteRtOpCodeTflAdd:
      return CopyOptions<AddOptions>(context, builder, src, dst);
    case kLiteRtOpCodeTflBatchMatmul:
      return CopyOptions<BatchMatmulOptions>(context, builder, src, dst);
    case kLiteRtOpCodeTflConcatenation:
      return CopyOptions<ConcatenationOptions>(context, builder, src, dst);
    case kLiteRtOpCodeTflConv2d:
      return CopyOptions<Conv2dOptions>(context, builder, src, dst);
    case kLiteRtOpCodeTflDepthwiseConv2d:
      return CopyOptions<DepthwiseConv2dOptions>(context, builder, src, dst);
    case kLiteRtOpCodeTflDiv:
      return CopyOptions<DivOptions>(context, builder, src, dst);
    case kLiteRtOpCodeTflFullyConnected:
      return CopyOptions<FullyConnectedOptions>(context, builder, src, dst);
    case kLiteRtOpCodeTflMean:
      return CopyOptions<MeanOptions>(context, builder, src, dst);
    case kLiteRtOpCodeTflMul:
      return CopyOptions<MulOptions>(context, builder, src, dst);
    case kLiteRtOpCodeTflReduceMax:
      return CopyOptions<ReduceMaxOptions>(context, builder, src, dst);
    case kLiteRtOpCodeTflReduceMin:
      return CopyOptions<ReduceMinOptions>(context, builder, src, dst);
    case kLiteRtOpCodeTflReshape:
      return CopyOptions<ReshapeOptions>(context, builder, src, dst);
    case kLiteRtOpCodeTflSoftmax:
      return CopyOptions<SoftmaxOptions>(context, builder, src, dst);
    case kLiteRtOpCodeTflSplit:
      return CopyOptions<SplitOptions>(context, builder, src, dst);
    case kLiteRtOpCodeTflSplitV:
      return CopyOptions<SplitVOptions>(context, builder, src, dst);
    case kLiteRtOpCodeTflSqueeze:
      return CopyOptions<SqueezeOptions>(context, builder, src, dst);
    case kLiteRtOpCodeTflStridedSlice:
      return CopyOptions<StridedSliceOptions>(context, builder, src, dst);
    case kLiteRtOpCodeTflSub:
      return CopyOptions<SubOptions>(context, builder, src, dst);
    case kLiteRtOpCodeTflSum:
      return CopyOptions<SumOptions>(context, builder, src, dst);
    default:
      if (CanCloneOpWithNewInputs(src.Code())) {
        return {};
      }
      return Unexpected(kLiteRtStatusErrorUnsupported);
  }
}

std::optional<std::vector<int64_t>> ReadIntConstValues(const Tensor& tensor) {
  if (!tensor.IsConstant() || !GetStaticDims(tensor)) {
    return std::nullopt;
  }
  if (IsUnquantizedOfType(tensor, ElementType::Int32)) {
    auto vals32 = tensor.WeightsData<int32_t>();
    if (!vals32 || vals32->empty()) {
      return std::nullopt;
    }
    return std::vector<int64_t>(vals32->begin(), vals32->end());
  }
  if (IsUnquantizedOfType(tensor, ElementType::Int64)) {
    auto vals64 = tensor.WeightsData<int64_t>();
    if (!vals64 || vals64->empty()) {
      return std::nullopt;
    }
    return std::vector<int64_t>(vals64->begin(), vals64->end());
  }
  return std::nullopt;
}

Expected<Tensor> EnsureInt32Const(const Builder& builder, const Tensor& tensor,
                                  absl::Span<const int64_t> values) {
  const auto dims = GetStaticDims(tensor);
  if (!dims) {
    return Unexpected(kLiteRtStatusPatternNoMatch);
  }
  for (int64_t v : values) {
    if (v < (std::numeric_limits<int32_t>::min)() ||
        v > (std::numeric_limits<int32_t>::max)()) {
      return Unexpected(kLiteRtStatusPatternNoMatch);
    }
  }
  if (IsUnquantizedOfType(tensor, ElementType::Int32)) {
    return tensor;
  }
  std::vector<int32_t> vals32;
  vals32.reserve(values.size());
  for (int64_t v : values) {
    vals32.push_back(static_cast<int32_t>(v));
  }
  return BuildConst<int32_t>(builder, *dims, vals32);
}

Expected<Tensor> ValidateAndConvertTileReps(
    const Builder& builder, const Tensor& reps,
    absl::Span<const int32_t> in_dims, absl::Span<const int32_t> out_dims) {
  if (in_dims.empty() || in_dims.size() != out_dims.size()) {
    return Unexpected(kLiteRtStatusPatternNoMatch);
  }
  const auto reps_dims = GetStaticDims(reps);
  const auto reps_vals = ReadIntConstValues(reps);
  if (!reps_dims || reps_dims->size() != 1 ||
      static_cast<size_t>((*reps_dims)[0]) != in_dims.size() || !reps_vals ||
      reps_vals->size() != in_dims.size()) {
    return Unexpected(kLiteRtStatusPatternNoMatch);
  }
  for (size_t i = 0; i < in_dims.size(); ++i) {
    const int64_t rep = (*reps_vals)[i];
    if (rep <= 0 || rep > (std::numeric_limits<int32_t>::max)() ||
        static_cast<int64_t>(in_dims[i]) * rep != out_dims[i]) {
      return Unexpected(kLiteRtStatusPatternNoMatch);
    }
  }
  return EnsureInt32Const(builder, reps, *reps_vals);
}

LiteRtStatus BuildBoolCastBridge(const Builder& builder, const Op& old_op,
                                 const Tensor& f_out, const Tensor& bool_out) {
  LITERT_RETURN_IF_ERROR(
      builder.BuildOp(kLiteRtOpCodeTflCast, {f_out}, {bool_out}));
  LITERT_RETURN_IF_ERROR(builder.EraseOp(old_op));
  return kLiteRtStatusOk;
}

// Returns true for a NotEqual that the MediaTek legalization already delegates
// natively (unquantized Float32 inputs and Bool output sharing the same
// non-empty, static, positive dims) and that only feeds a subgraph output.
// Lowering such an op to Float32 arithmetic would just be re-emitted as
// NotEqual by BoolMaskOutputNotEqualTransformation.
bool IsNativeSameShapeOutputNotEqual(const Builder& builder, const Op& op) {
  if (op.Code() != kLiteRtOpCodeTflNotEqual || op.Inputs().size() != 2 ||
      op.Outputs().size() != 1) {
    return false;
  }
  const Tensor out = op.Outputs()[0];
  if (!IsUnquantizedOfType(out, ElementType::Bool) || !out.Uses().empty()) {
    return false;
  }
  auto is_sg_out = builder.IsSubgraphOutput(out);
  if (!is_sg_out || !*is_sg_out) {
    return false;
  }
  const Tensor lhs = op.Inputs()[0];
  const Tensor rhs = op.Inputs()[1];
  if (!IsUnquantizedOfType(lhs, ElementType::Float32) ||
      !IsUnquantizedOfType(rhs, ElementType::Float32)) {
    return false;
  }
  const auto lhs_dims = GetStaticDims(lhs);
  const auto rhs_dims = GetStaticDims(rhs);
  const auto out_dims = GetStaticDims(out);
  return lhs_dims && rhs_dims && out_dims && !lhs_dims->empty() &&
         *lhs_dims == *rhs_dims && *lhs_dims == *out_dims;
}

LiteRtStatus LowerLogicalNot(const Builder& builder, const Op& op) {
  if (op.Inputs().size() != 1 || op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  const Tensor in = op.Inputs()[0];
  const Tensor out = op.Outputs()[0];
  if (!IsUnquantizedOfType(in, ElementType::Bool) ||
      !IsUnquantizedOfType(out, ElementType::Bool)) {
    return kLiteRtStatusPatternNoMatch;
  }
  const auto dims = GetStaticDims(in);
  const auto out_dims = GetStaticDims(out);
  if (!dims || !out_dims || *dims != *out_dims) {
    return kLiteRtStatusPatternNoMatch;
  }
  LITERT_ASSIGN_OR_RETURN(Tensor x_f,
                          ResolveBoolAsZeroOneFloat(builder, in, *dims));
  LITERT_ASSIGN_OR_RETURN(Tensor one, BuildScalarFloatConst(builder, 1.0f));
  LITERT_ASSIGN_OR_RETURN(
      Tensor f_out, BuildBinaryFloatOpWithDims(builder, kLiteRtOpCodeTflSub,
                                               one, x_f, *out_dims));
  return BuildBoolCastBridge(builder, op, f_out, out);
}

LiteRtStatus LowerBinaryBoolOp(const Builder& builder, const Op& op,
                               LiteRtOpCode float_code) {
  if (op.Inputs().size() != 2 || op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  const Tensor lhs = op.Inputs()[0];
  const Tensor rhs = op.Inputs()[1];
  const Tensor out = op.Outputs()[0];
  if (!IsUnquantizedOfType(lhs, ElementType::Bool) ||
      !IsUnquantizedOfType(rhs, ElementType::Bool) ||
      !IsUnquantizedOfType(out, ElementType::Bool)) {
    return kLiteRtStatusPatternNoMatch;
  }
  const auto lhs_dims = GetStaticDims(lhs);
  const auto rhs_dims = GetStaticDims(rhs);
  const auto out_dims = GetStaticDims(out);
  if (!lhs_dims || !rhs_dims || !out_dims) {
    return kLiteRtStatusPatternNoMatch;
  }
  const auto bcast = BroadcastShapes(*lhs_dims, *rhs_dims);
  if (!bcast || *bcast != *out_dims ||
      !CanResolveBoolAsZeroOneFloat(lhs, *lhs_dims) ||
      !CanResolveBoolAsZeroOneFloat(rhs, *rhs_dims)) {
    return kLiteRtStatusPatternNoMatch;
  }
  LITERT_ASSIGN_OR_RETURN(Tensor lhs_f,
                          ResolveBoolAsZeroOneFloat(builder, lhs, *lhs_dims));
  Tensor rhs_f = lhs_f;
  if (rhs != lhs) {
    LITERT_ASSIGN_OR_RETURN(rhs_f,
                            ResolveBoolAsZeroOneFloat(builder, rhs, *rhs_dims));
  }
  LITERT_ASSIGN_OR_RETURN(
      Tensor f_out,
      BuildBinaryFloatOpWithDims(builder, float_code, lhs_f, rhs_f, *out_dims));
  return BuildBoolCastBridge(builder, op, f_out, out);
}

bool PreservesNonUnitDimsOrder(absl::Span<const int32_t> in_dims,
                               absl::Span<const int32_t> out_dims) {
  std::vector<int32_t> in_non_unit;
  std::vector<int32_t> out_non_unit;
  for (int32_t d : in_dims) {
    if (d != 1) {
      in_non_unit.push_back(d);
    }
  }
  for (int32_t d : out_dims) {
    if (d != 1) {
      out_non_unit.push_back(d);
    }
  }
  return in_non_unit == out_non_unit;
}

bool ValidateSqueezeDims(const LiteRtCompilerContext* context, const Op& op,
                         absl::Span<const int32_t> in_dims,
                         absl::Span<const int32_t> out_dims) {
  if (out_dims.size() > in_dims.size() ||
      !PreservesNonUnitDimsOrder(in_dims, out_dims)) {
    return false;
  }
  auto sq_opts = GetOptionsAs<SqueezeOptions>(context, op.Get());
  if (!sq_opts.HasValue()) {
    return true;
  }
  if (sq_opts->squeeze_dims.empty()) {
    // Empty squeeze_dims squeezes every size-1 dimension.
    for (int32_t d : out_dims) {
      if (d == 1) {
        return false;
      }
    }
    return true;
  }
  const int64_t rank = static_cast<int64_t>(in_dims.size());
  std::vector<bool> drop(in_dims.size(), false);
  for (int32_t raw_axis : sq_opts->squeeze_dims) {
    int64_t axis = raw_axis;
    if (axis < 0) {
      axis += rank;
    }
    if (axis < 0 || axis >= rank || in_dims[static_cast<size_t>(axis)] != 1) {
      return false;
    }
    drop[static_cast<size_t>(axis)] = true;
  }
  std::vector<int32_t> expected;
  for (size_t i = 0; i < in_dims.size(); ++i) {
    if (!drop[i]) {
      expected.push_back(in_dims[i]);
    }
  }
  return absl::MakeConstSpan(expected) == out_dims;
}

bool ValidateExpandDimsAxis(const Tensor& axis_tensor,
                            absl::Span<const int32_t> in_dims,
                            absl::Span<const int32_t> out_dims) {
  if (out_dims.size() != in_dims.size() + 1) {
    return false;
  }
  const auto axis_vals = ReadIntConstValues(axis_tensor);
  if (!axis_vals || axis_vals->size() != 1) {
    return false;
  }
  const int64_t out_rank = static_cast<int64_t>(out_dims.size());
  int64_t axis = (*axis_vals)[0];
  if (axis < 0) {
    axis += out_rank;
  }
  if (axis < 0 || axis >= out_rank ||
      out_dims[static_cast<size_t>(axis)] != 1) {
    return false;
  }
  size_t in_idx = 0;
  for (size_t out_idx = 0; out_idx < out_dims.size(); ++out_idx) {
    if (out_idx == static_cast<size_t>(axis)) {
      continue;
    }
    if (out_dims[out_idx] != in_dims[in_idx++]) {
      return false;
    }
  }
  return true;
}

// Lowers Bool Reshape, Squeeze, and ExpandDims into a 2-input Float32 Reshape
// (since NeuroPilot supports NEURON_RESHAPE natively, whereas Squeeze and
// ExpandDims are not in kSupportedOps).
LiteRtStatus LowerBoolShapeChangeToReshape(const LiteRtCompilerContext* context,
                                           const Builder& builder,
                                           const Op& op) {
  if (op.Inputs().empty() || op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  const Tensor in = op.Inputs()[0];
  const Tensor out = op.Outputs()[0];
  if (!IsUnquantizedOfType(in, ElementType::Bool) ||
      !IsUnquantizedOfType(out, ElementType::Bool)) {
    return kLiteRtStatusPatternNoMatch;
  }
  const auto in_dims = GetStaticDims(in);
  const auto out_dims = GetStaticDims(out);
  if (!in_dims || !out_dims || in_dims->empty() || out_dims->empty() ||
      !CanResolveBoolAsZeroOneFloat(in, *in_dims)) {
    return kLiteRtStatusPatternNoMatch;
  }
  const auto in_elems = NumElements(*in_dims);
  const auto out_elems = NumElements(*out_dims);
  if (!in_elems || !out_elems || *in_elems != *out_elems) {
    return kLiteRtStatusPatternNoMatch;
  }

  if (op.Code() == kLiteRtOpCodeTflReshape) {
    if (op.Inputs().size() > 2 ||
        (op.Inputs().size() == 2 &&
         (!op.Inputs()[1].IsConstant() || op.Inputs()[1].HasQuantization()))) {
      return kLiteRtStatusPatternNoMatch;
    }
  } else if (op.Code() == kLiteRtOpCodeTflSqueeze) {
    if (op.Inputs().size() != 1 ||
        !ValidateSqueezeDims(context, op, *in_dims, *out_dims)) {
      return kLiteRtStatusPatternNoMatch;
    }
  } else if (op.Code() == kLiteRtOpCodeTflExpandDims) {
    if (op.Inputs().size() != 2 ||
        !ValidateExpandDimsAxis(op.Inputs()[1], *in_dims, *out_dims)) {
      return kLiteRtStatusPatternNoMatch;
    }
  } else {
    return kLiteRtStatusPatternNoMatch;
  }

  LITERT_ASSIGN_OR_RETURN(Tensor in_f,
                          ResolveBoolAsZeroOneFloat(builder, in, *in_dims));
  LITERT_ASSIGN_OR_RETURN(
      Tensor shape_const,
      BuildConst<int32_t>(builder, {static_cast<int32_t>(out_dims->size())},
                          *out_dims));
  LITERT_ASSIGN_OR_RETURN(
      Tensor f_out,
      BuildTensorOfType(builder, ElementType::Float32, *out_dims));
  LITERT_ASSIGN_OR_RETURN(
      Op new_op,
      builder.BuildOp(kLiteRtOpCodeTflReshape, {in_f, shape_const}, {f_out}));
  ReshapeOptions reshape_opts;
  reshape_opts.new_shape = *out_dims;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(new_op, std::move(reshape_opts)));
  return BuildBoolCastBridge(builder, op, f_out, out);
}

LiteRtStatus LowerBoolDataMovementOp(const LiteRtCompilerContext* context,
                                     const Builder& builder, const Op& op) {
  if (op.Inputs().empty() || op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  const Tensor in = op.Inputs()[0];
  const Tensor out = op.Outputs()[0];
  if (!IsUnquantizedOfType(in, ElementType::Bool) ||
      !IsUnquantizedOfType(out, ElementType::Bool)) {
    return kLiteRtStatusPatternNoMatch;
  }
  const auto in_dims = GetStaticDims(in);
  const auto out_dims = GetStaticDims(out);
  if (!in_dims || !out_dims || in_dims->empty() || out_dims->empty() ||
      !CanResolveBoolAsZeroOneFloat(in, *in_dims)) {
    return kLiteRtStatusPatternNoMatch;
  }

  std::vector<Tensor> extra_inputs;
  if (op.Code() == kLiteRtOpCodeTflSlice) {
    if (op.Inputs().size() != 3 || in_dims->size() != out_dims->size()) {
      return kLiteRtStatusPatternNoMatch;
    }
    const auto begin_dims = GetStaticDims(op.Inputs()[1]);
    const auto size_dims = GetStaticDims(op.Inputs()[2]);
    const auto begin_vals = ReadIntConstValues(op.Inputs()[1]);
    const auto size_vals = ReadIntConstValues(op.Inputs()[2]);
    if (!begin_dims || !size_dims || begin_dims->size() != 1 ||
        size_dims->size() != 1 || !begin_vals || !size_vals ||
        begin_vals->size() != in_dims->size() ||
        size_vals->size() != in_dims->size()) {
      return kLiteRtStatusPatternNoMatch;
    }
    for (size_t i = 0; i < in_dims->size(); ++i) {
      const int64_t b = (*begin_vals)[i];
      const int64_t s = (*size_vals)[i];
      if (b < 0 || b >= (*in_dims)[i]) {
        return kLiteRtStatusPatternNoMatch;
      }
      const int64_t actual = (s == -1) ? ((*in_dims)[i] - b) : s;
      if (actual != (*out_dims)[i] || b + actual > (*in_dims)[i]) {
        return kLiteRtStatusPatternNoMatch;
      }
    }
    auto begin_i32 = EnsureInt32Const(builder, op.Inputs()[1], *begin_vals);
    auto size_i32 = EnsureInt32Const(builder, op.Inputs()[2], *size_vals);
    if (!begin_i32 || !size_i32) {
      return kLiteRtStatusPatternNoMatch;
    }
    extra_inputs.push_back(*begin_i32);
    extra_inputs.push_back(*size_i32);
  } else if (op.Code() == kLiteRtOpCodeTflStridedSlice) {
    if (op.Inputs().size() != 4) {
      return kLiteRtStatusPatternNoMatch;
    }
    auto ss_opts = GetOptionsAs<StridedSliceOptions>(context, op.Get());
    // LegalizeStridedSliceOp only forwards begin_mask, end_mask, and
    // shrink_axis_mask to NEURON_STRIDED_SLICE.
    if (!ss_opts || ss_opts->ellipsis_mask != 0 ||
        ss_opts->new_axis_mask != 0 || ss_opts->offset) {
      return kLiteRtStatusPatternNoMatch;
    }
    const auto begin_dims = GetStaticDims(op.Inputs()[1]);
    const auto end_dims = GetStaticDims(op.Inputs()[2]);
    const auto stride_dims = GetStaticDims(op.Inputs()[3]);
    const auto begin_vals = ReadIntConstValues(op.Inputs()[1]);
    const auto end_vals = ReadIntConstValues(op.Inputs()[2]);
    const auto stride_vals = ReadIntConstValues(op.Inputs()[3]);
    if (!begin_dims || !end_dims || !stride_dims || begin_dims->size() != 1 ||
        *begin_dims != *end_dims || *begin_dims != *stride_dims ||
        !begin_vals || !end_vals || !stride_vals) {
      return kLiteRtStatusPatternNoMatch;
    }
    for (int64_t stride : *stride_vals) {
      if (stride == 0) {
        return kLiteRtStatusPatternNoMatch;
      }
    }
    auto begin_i32 = EnsureInt32Const(builder, op.Inputs()[1], *begin_vals);
    auto end_i32 = EnsureInt32Const(builder, op.Inputs()[2], *end_vals);
    auto stride_i32 = EnsureInt32Const(builder, op.Inputs()[3], *stride_vals);
    if (!begin_i32 || !end_i32 || !stride_i32) {
      return kLiteRtStatusPatternNoMatch;
    }
    extra_inputs.push_back(*begin_i32);
    extra_inputs.push_back(*end_i32);
    extra_inputs.push_back(*stride_i32);
  } else if (op.Code() == kLiteRtOpCodeTflTile) {
    if (op.Inputs().size() != 2) {
      return kLiteRtStatusPatternNoMatch;
    }
    auto reps_i32 = ValidateAndConvertTileReps(builder, op.Inputs()[1],
                                               *in_dims, *out_dims);
    if (!reps_i32) {
      return kLiteRtStatusPatternNoMatch;
    }
    extra_inputs.push_back(*reps_i32);
  } else if (op.Code() == kLiteRtOpCodeTflTranspose) {
    if (op.Inputs().size() != 2 || in_dims->size() != out_dims->size()) {
      return kLiteRtStatusPatternNoMatch;
    }
    const auto perm_dims = GetStaticDims(op.Inputs()[1]);
    const auto perm_vals = ReadIntConstValues(op.Inputs()[1]);
    const size_t rank = in_dims->size();
    if (!perm_dims || perm_dims->size() != 1 || !perm_vals ||
        perm_vals->size() != rank) {
      return kLiteRtStatusPatternNoMatch;
    }
    std::vector<bool> seen(rank, false);
    for (size_t i = 0; i < rank; ++i) {
      const int64_t p = (*perm_vals)[i];
      if (p < 0 || static_cast<size_t>(p) >= rank ||
          seen[static_cast<size_t>(p)] ||
          (*in_dims)[static_cast<size_t>(p)] != (*out_dims)[i]) {
        return kLiteRtStatusPatternNoMatch;
      }
      seen[static_cast<size_t>(p)] = true;
    }
    auto perm_i32 = EnsureInt32Const(builder, op.Inputs()[1], *perm_vals);
    if (!perm_i32) {
      return kLiteRtStatusPatternNoMatch;
    }
    extra_inputs.push_back(*perm_i32);
  } else {
    return kLiteRtStatusPatternNoMatch;
  }

  LITERT_ASSIGN_OR_RETURN(Tensor in_f,
                          ResolveBoolAsZeroOneFloat(builder, in, *in_dims));
  std::vector<Tensor> new_inputs;
  new_inputs.reserve(1 + extra_inputs.size());
  new_inputs.push_back(in_f);
  for (const Tensor& extra : extra_inputs) {
    new_inputs.push_back(extra);
  }
  LITERT_ASSIGN_OR_RETURN(
      Tensor f_out,
      BuildTensorOfType(builder, ElementType::Float32, *out_dims));
  LITERT_ASSIGN_OR_RETURN(Op new_op,
                          builder.BuildOp(op.Code(), new_inputs, {f_out}));
  LITERT_RETURN_IF_ERROR(CopyOpOptions(context, builder, op, new_op));
  return BuildBoolCastBridge(builder, op, f_out, out);
}

LiteRtStatus LowerBoolConcatenation(const LiteRtCompilerContext* context,
                                    const Builder& builder, const Op& op) {
  if (op.Inputs().empty() || op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  const Tensor out = op.Outputs()[0];
  if (!IsUnquantizedOfType(out, ElementType::Bool)) {
    return kLiteRtStatusPatternNoMatch;
  }
  auto opts = GetOptionsAs<ConcatenationOptions>(context, op.Get());
  if (!opts || opts->fused_activation_function != 0) {
    return kLiteRtStatusPatternNoMatch;
  }
  const auto out_dims = GetStaticDims(out);
  if (!out_dims || out_dims->empty()) {
    return kLiteRtStatusPatternNoMatch;
  }
  const int64_t rank = static_cast<int64_t>(out_dims->size());
  int64_t norm_axis = opts->axis;
  if (norm_axis < 0) {
    norm_axis += rank;
  }
  if (norm_axis < 0 || norm_axis >= rank) {
    return kLiteRtStatusPatternNoMatch;
  }
  int64_t axis_sum = 0;
  for (const Tensor& in : op.Inputs()) {
    const auto dims = GetStaticDims(in);
    if (!dims || dims->size() != out_dims->size() ||
        !CanResolveBoolAsZeroOneFloat(in, *dims)) {
      return kLiteRtStatusPatternNoMatch;
    }
    for (size_t d = 0; d < out_dims->size(); ++d) {
      if (d == static_cast<size_t>(norm_axis)) {
        axis_sum += (*dims)[d];
      } else if ((*dims)[d] != (*out_dims)[d]) {
        return kLiteRtStatusPatternNoMatch;
      }
    }
  }
  if (axis_sum != (*out_dims)[static_cast<size_t>(norm_axis)]) {
    return kLiteRtStatusPatternNoMatch;
  }

  std::vector<Tensor> f_inputs;
  f_inputs.reserve(op.Inputs().size());
  absl::flat_hash_map<LiteRtTensor, Tensor> resolved_cache;
  for (const Tensor& in : op.Inputs()) {
    if (auto it = resolved_cache.find(in.Get()); it != resolved_cache.end()) {
      f_inputs.push_back(it->second);
      continue;
    }
    const auto dims = GetStaticDims(in);
    LITERT_ASSIGN_OR_RETURN(Tensor f_in,
                            ResolveBoolAsZeroOneFloat(builder, in, *dims));
    resolved_cache.emplace(in.Get(), f_in);
    f_inputs.push_back(f_in);
  }
  LITERT_ASSIGN_OR_RETURN(
      Tensor f_out,
      BuildTensorOfType(builder, ElementType::Float32, *out_dims));
  LITERT_ASSIGN_OR_RETURN(
      Op new_op,
      builder.BuildOp(kLiteRtOpCodeTflConcatenation, f_inputs, {f_out}));
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(new_op, std::move(*opts)));
  return BuildBoolCastBridge(builder, op, f_out, out);
}

LiteRtStatus LowerSelectToFloat(const Builder& builder, const Op& op) {
  if (op.Inputs().size() != 3 || op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  const Tensor cond = op.Inputs()[0];
  const Tensor t_val = op.Inputs()[1];
  const Tensor f_val = op.Inputs()[2];
  const Tensor out = op.Outputs()[0];
  if (!IsUnquantizedOfType(cond, ElementType::Bool)) {
    return kLiteRtStatusPatternNoMatch;
  }
  const bool bool_out = IsUnquantizedOfType(out, ElementType::Bool);
  if (!bool_out && !IsUnquantizedOfType(out, ElementType::Float32)) {
    return kLiteRtStatusPatternNoMatch;
  }
  const ElementType branch_type =
      bool_out ? ElementType::Bool : ElementType::Float32;
  if (!IsUnquantizedOfType(t_val, branch_type) ||
      !IsUnquantizedOfType(f_val, branch_type)) {
    return kLiteRtStatusPatternNoMatch;
  }
  // Never lower Select/SelectV2 when a Float32 branch (or an upstream
  // Select/data-movement chain feeding it) introduces a NaN or +/-Inf constant,
  // because IEEE-754 `0.0f * (+/-Inf)` and `0.0f * NaN` evaluate to NaN and
  // corrupt the selected finite branch.
  if (!bool_out && (MayPropagateNonFiniteFloatConst(t_val) ||
                    MayPropagateNonFiniteFloatConst(f_val))) {
    return kLiteRtStatusPatternNoMatch;
  }

  const auto cond_dims_opt = GetStaticDims(cond);
  const auto t_dims_opt = GetStaticDims(t_val);
  const auto f_dims_opt = GetStaticDims(f_val);
  const auto out_dims_opt = GetStaticDims(out);
  if (!cond_dims_opt || !t_dims_opt || !f_dims_opt || !out_dims_opt) {
    return kLiteRtStatusPatternNoMatch;
  }
  std::vector<int32_t> cond_dims = *cond_dims_opt;
  const std::vector<int32_t>& t_dims = *t_dims_opt;
  const std::vector<int32_t>& f_dims = *f_dims_opt;
  const std::vector<int32_t>& out_dims = *out_dims_opt;

  // Unlike SelectV2 (trailing NumPy broadcast), TFLite Select requires x and y
  // to have the exact output shape, and cond to either match out_dims, be a
  // rank-0 scalar, or be a rank-1 [B] vector selecting leading batch slices.
  const bool needs_select_1d_reshape =
      op.Code() == kLiteRtOpCodeTflSelect && cond_dims.size() == 1 &&
      out_dims.size() > 1 && cond_dims[0] == out_dims[0];
  if (op.Code() == kLiteRtOpCodeTflSelect) {
    if (t_dims != out_dims || f_dims != out_dims) {
      return kLiteRtStatusPatternNoMatch;
    }
    if (cond_dims != out_dims && !cond_dims.empty() &&
        !needs_select_1d_reshape) {
      return kLiteRtStatusPatternNoMatch;
    }
  }
  if (!CanResolveBoolAsZeroOneFloat(cond, cond_dims) ||
      (bool_out && (!CanResolveBoolAsZeroOneFloat(t_val, t_dims) ||
                    !CanResolveBoolAsZeroOneFloat(f_val, f_dims)))) {
    return kLiteRtStatusPatternNoMatch;
  }

  LITERT_ASSIGN_OR_RETURN(Tensor orig_cond_f,
                          ResolveBoolAsZeroOneFloat(builder, cond, cond_dims));
  Tensor cond_f = orig_cond_f;
  if (needs_select_1d_reshape) {
    std::vector<int32_t> reshaped_dims(out_dims.size(), 1);
    reshaped_dims[0] = cond_dims[0];
    LITERT_ASSIGN_OR_RETURN(
        Tensor shape_t,
        BuildConst<int32_t>(builder,
                            {static_cast<int32_t>(reshaped_dims.size())},
                            reshaped_dims));
    LITERT_ASSIGN_OR_RETURN(
        Tensor cond_reshaped,
        BuildTensorOfType(builder, ElementType::Float32, reshaped_dims));
    LITERT_ASSIGN_OR_RETURN(
        Op r_op, builder.BuildOp(kLiteRtOpCodeTflReshape, {cond_f, shape_t},
                                 {cond_reshaped}));
    ReshapeOptions r_opts;
    r_opts.new_shape = reshaped_dims;
    LITERT_RETURN_IF_ERROR(builder.SetOpOptions(r_op, std::move(r_opts)));
    cond_f = cond_reshaped;
    cond_dims = std::move(reshaped_dims);
  }

  const auto true_dims = BroadcastShapes(cond_dims, t_dims);
  const auto false_dims = BroadcastShapes(cond_dims, f_dims);
  if (!true_dims || !false_dims) {
    return kLiteRtStatusPatternNoMatch;
  }
  const auto combined_dims = BroadcastShapes(*true_dims, *false_dims);
  if (!combined_dims || *combined_dims != out_dims) {
    return kLiteRtStatusPatternNoMatch;
  }

  Tensor t_f = t_val;
  Tensor f_f = f_val;
  if (bool_out) {
    if (t_val == cond) {
      t_f = orig_cond_f;
    } else {
      LITERT_ASSIGN_OR_RETURN(
          t_f, ResolveBoolAsZeroOneFloat(builder, t_val, t_dims));
    }
    if (f_val == t_val) {
      f_f = t_f;
    } else if (f_val == cond) {
      f_f = orig_cond_f;
    } else {
      LITERT_ASSIGN_OR_RETURN(
          f_f, ResolveBoolAsZeroOneFloat(builder, f_val, f_dims));
    }
  }

  Tensor f_out = out;
  if (bool_out) {
    LITERT_ASSIGN_OR_RETURN(
        f_out, BuildTensorOfType(builder, ElementType::Float32, out_dims));
  }

  if (IsZeroFloatConst(f_f) && *true_dims == out_dims) {
    LITERT_RETURN_IF_ERROR(
        BuildBinaryFloatOp(builder, kLiteRtOpCodeTflMul, t_f, cond_f, f_out));
  } else if (IsZeroFloatConst(t_f) && *false_dims == out_dims) {
    LITERT_ASSIGN_OR_RETURN(Tensor one, BuildScalarFloatConst(builder, 1.0f));
    LITERT_ASSIGN_OR_RETURN(Tensor not_cond, BuildBinaryFloatOpWithDims(
                                                 builder, kLiteRtOpCodeTflSub,
                                                 one, cond_f, cond_dims));
    LITERT_RETURN_IF_ERROR(
        BuildBinaryFloatOp(builder, kLiteRtOpCodeTflMul, f_f, not_cond, f_out));
  } else {
    LITERT_ASSIGN_OR_RETURN(Tensor one, BuildScalarFloatConst(builder, 1.0f));
    LITERT_ASSIGN_OR_RETURN(Tensor not_cond, BuildBinaryFloatOpWithDims(
                                                 builder, kLiteRtOpCodeTflSub,
                                                 one, cond_f, cond_dims));
    LITERT_ASSIGN_OR_RETURN(Tensor true_part, BuildBinaryFloatOpWithDims(
                                                  builder, kLiteRtOpCodeTflMul,
                                                  cond_f, t_f, *true_dims));
    LITERT_ASSIGN_OR_RETURN(Tensor false_part, BuildBinaryFloatOpWithDims(
                                                   builder, kLiteRtOpCodeTflMul,
                                                   f_f, not_cond, *false_dims));
    LITERT_RETURN_IF_ERROR(BuildBinaryFloatOp(builder, kLiteRtOpCodeTflAdd,
                                              false_part, true_part, f_out));
  }

  if (bool_out) {
    return BuildBoolCastBridge(builder, op, f_out, out);
  }
  LITERT_RETURN_IF_ERROR(builder.EraseOp(op));
  return kLiteRtStatusOk;
}

LiteRtStatus LowerZeroOneEqualityComparison(const Builder& builder,
                                            const Op& current_op,
                                            bool is_equal) {
  if (current_op.Inputs().size() != 2 || current_op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  // Do not rewrite a natively delegable same-shape NotEqual -> [SG_OUT],
  // including the form produced by BoolMaskOutputNotEqualTransformation.
  if (!is_equal && IsNativeSameShapeOutputNotEqual(builder, current_op)) {
    return kLiteRtStatusPatternNoMatch;
  }

  const Tensor lhs = current_op.Inputs()[0];
  const Tensor rhs = current_op.Inputs()[1];
  const Tensor out = current_op.Outputs()[0];
  if (!IsUnquantizedOfType(out, ElementType::Bool) || lhs.Get() == nullptr ||
      rhs.Get() == nullptr || lhs.ElementType() != rhs.ElementType()) {
    return kLiteRtStatusPatternNoMatch;
  }
  const auto lhs_dims = GetStaticDims(lhs);
  const auto rhs_dims = GetStaticDims(rhs);
  const auto out_dims = GetStaticDims(out);
  if (!lhs_dims || !rhs_dims || !out_dims) {
    return kLiteRtStatusPatternNoMatch;
  }
  const auto bcast = BroadcastShapes(*lhs_dims, *rhs_dims);
  if (!bcast || *bcast != *out_dims) {
    return kLiteRtStatusPatternNoMatch;
  }

  const bool lhs_bool = IsUnquantizedOfType(lhs, ElementType::Bool);
  const bool rhs_bool = IsUnquantizedOfType(rhs, ElementType::Bool);
  const bool lhs_zero_one_f32 = !lhs_bool && IsKnownZeroOneFloat(lhs);
  const bool rhs_zero_one_f32 = !rhs_bool && IsKnownZeroOneFloat(rhs);

  auto emit_single_operand = [&](const Tensor& x, bool invert) -> LiteRtStatus {
    if (!invert) {
      return BuildBoolCastBridge(builder, current_op, x, out);
    }
    LITERT_ASSIGN_OR_RETURN(Tensor one, BuildScalarFloatConst(builder, 1.0f));
    LITERT_ASSIGN_OR_RETURN(
        Tensor inv, BuildBinaryFloatOpWithDims(builder, kLiteRtOpCodeTflSub,
                                               one, x, *out_dims));
    return BuildBoolCastBridge(builder, current_op, inv, out);
  };

  // Fast paths for zero/one checks on proven {0.0, 1.0} Float32 tensors.
  // Note: +/-0.0f have identical Equal/NotEqual truth tables on {0.0, 1.0}.
  if (lhs_zero_one_f32 && *lhs_dims == *out_dims) {
    if (IsAnyZeroFloatConst(rhs)) {
      return emit_single_operand(lhs, /*invert=*/is_equal);
    }
    if (IsOneFloatConst(rhs)) {
      return emit_single_operand(lhs, /*invert=*/!is_equal);
    }
  }
  if (rhs_zero_one_f32 && *rhs_dims == *out_dims) {
    if (IsAnyZeroFloatConst(lhs)) {
      return emit_single_operand(rhs, /*invert=*/is_equal);
    }
    if (IsOneFloatConst(lhs)) {
      return emit_single_operand(rhs, /*invert=*/!is_equal);
    }
  }

  // Only lower when both operands are exact {0.0, 1.0} / Bool values (allowing
  // +/-0.0f constants as zero). Arbitrary Float32 comparisons (e.g. 1e-12f,
  // +/-Inf, NaN) are left intact so semantics remain exact on NPU or CPU.
  const bool lhs_exact_zero_const =
      !lhs_bool && !lhs_zero_one_f32 && IsAnyZeroFloatConst(lhs);
  const bool rhs_exact_zero_const =
      !rhs_bool && !rhs_zero_one_f32 && IsAnyZeroFloatConst(rhs);
  if ((!lhs_bool && !lhs_zero_one_f32 && !lhs_exact_zero_const) ||
      (!rhs_bool && !rhs_zero_one_f32 && !rhs_exact_zero_const) ||
      (lhs_bool && !CanResolveBoolAsZeroOneFloat(lhs, *lhs_dims)) ||
      (rhs_bool && !CanResolveBoolAsZeroOneFloat(rhs, *rhs_dims))) {
    return kLiteRtStatusPatternNoMatch;
  }

  auto resolve_exact_operand =
      [&](const Tensor& operand, absl::Span<const int32_t> dims, bool is_bool,
          bool is_exact_zero_const) -> Expected<Tensor> {
    if (is_bool) {
      return ResolveBoolAsZeroOneFloat(builder, operand, dims);
    }
    if (is_exact_zero_const) {
      const auto num_elems = NumElements(dims);
      if (!num_elems || *num_elems > kMaxMaterializedConstElements) {
        return Unexpected(kLiteRtStatusPatternNoMatch);
      }
      std::vector<float> zeros(*num_elems, 0.0f);
      return BuildConst<float>(builder, dims, zeros);
    }
    return operand;
  };

  LITERT_ASSIGN_OR_RETURN(
      Tensor lhs_f,
      resolve_exact_operand(lhs, *lhs_dims, lhs_bool, lhs_exact_zero_const));
  Tensor rhs_f = lhs_f;
  if (rhs != lhs) {
    LITERT_ASSIGN_OR_RETURN(
        rhs_f,
        resolve_exact_operand(rhs, *rhs_dims, rhs_bool, rhs_exact_zero_const));
  }

  // On {0.0, 1.0} operands, |lhs_f - rhs_f| is exact XOR in {0.0f, 1.0f}, and
  // 1.0f - |lhs_f - rhs_f| is exact XNOR (Equal).
  LITERT_ASSIGN_OR_RETURN(
      Tensor diff, BuildBinaryFloatOpWithDims(builder, kLiteRtOpCodeTflSub,
                                              lhs_f, rhs_f, *out_dims));
  LITERT_ASSIGN_OR_RETURN(
      Tensor xor_f,
      BuildTensorOfType(builder, ElementType::Float32, *out_dims));
  LITERT_RETURN_IF_ERROR(builder.BuildOp(kLiteRtOpCodeTflAbs, {diff}, {xor_f}));
  Tensor result_f = xor_f;
  if (is_equal) {
    LITERT_ASSIGN_OR_RETURN(Tensor one, BuildScalarFloatConst(builder, 1.0f));
    LITERT_ASSIGN_OR_RETURN(
        result_f, BuildBinaryFloatOpWithDims(builder, kLiteRtOpCodeTflSub, one,
                                             xor_f, *out_dims));
  }
  return BuildBoolCastBridge(builder, current_op, result_f, out);
}

}  // namespace

extern "C" {

LiteRtStatus BoolMaskBypassCastTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op) {
  Builder builder(context, builder_ptr);
  Op current_op(context, op);
  const LiteRtOpCode op_code = current_op.Code();
  if (current_op.Outputs().empty()) {
    return kLiteRtStatusPatternNoMatch;
  }

  if (CanCloneOpWithNewInputs(op_code) &&
      CanReadOpOptions(context, current_op)) {
    bool has_null_tensor = false;
    for (const Tensor& in : current_op.Inputs()) {
      if (in.Get() == nullptr) {
        has_null_tensor = true;
        break;
      }
    }
    for (const Tensor& out : current_op.Outputs()) {
      if (out.Get() == nullptr) {
        has_null_tensor = true;
        break;
      }
    }
    if (!has_null_tensor) {
      std::vector<Tensor> new_inputs;
      new_inputs.reserve(current_op.Inputs().size());
      absl::flat_hash_map<LiteRtTensor, Tensor> bypassed_cache;
      bool replaced = false;
      for (const Tensor& in : current_op.Inputs()) {
        if (auto it = bypassed_cache.find(in.Get());
            it != bypassed_cache.end()) {
          new_inputs.push_back(it->second);
          replaced = true;
          continue;
        }
        if (auto dims = GetStaticDims(in); dims.has_value()) {
          if (auto bypassed = TryBypassDoubleBoolCast(builder, in, *dims);
              bypassed.has_value()) {
            bypassed_cache.emplace(in.Get(), *bypassed);
            new_inputs.push_back(*bypassed);
            replaced = true;
            continue;
          }
        }
        new_inputs.push_back(in);
      }
      if (replaced) {
        std::vector<Tensor> outs;
        outs.reserve(current_op.Outputs().size());
        for (const Tensor& out : current_op.Outputs()) {
          outs.push_back(out);
        }
        LITERT_ASSIGN_OR_RETURN(Op new_op,
                                builder.BuildOp(op_code, new_inputs, outs));
        LITERT_RETURN_IF_ERROR(
            CopyOpOptions(context, builder, current_op, new_op));
        LITERT_RETURN_IF_ERROR(builder.EraseOp(current_op));
        LITERT_LOG(
            LITERT_INFO,
            "BoolMaskBypassCast: bypassed Cast(Bool<->Float32) into op %d",
            op_code);
        return kLiteRtStatusOk;
      }
    }
  }

  // Fallback for non-scalar subgraph outputs fed directly by
  // Cast(bool_src -> out_f32) when bool_src is constant or cast from a proven
  // {0.0, 1.0} Float32 tensor.
  if (op_code == kLiteRtOpCodeTflCast && current_op.Inputs().size() == 1 &&
      current_op.Outputs().size() == 1) {
    const Tensor in = current_op.Inputs()[0];
    const Tensor out = current_op.Outputs()[0];
    if (IsUnquantizedOfType(in, ElementType::Bool) &&
        IsUnquantizedOfType(out, ElementType::Float32)) {
      LITERT_ASSIGN_OR_RETURN(bool is_sg_out, builder.IsSubgraphOutput(out));
      if (!is_sg_out) {
        return kLiteRtStatusPatternNoMatch;
      }
      const auto dims = GetStaticDims(out);
      const auto in_dims = GetStaticDims(in);
      if (!dims || dims->empty() || !in_dims || *in_dims != *dims) {
        return kLiteRtStatusPatternNoMatch;
      }
      bool has_zero_one_float_source = in.IsConstant();
      if (!has_zero_one_float_source) {
        if (auto def = in.DefiningOp(); def.has_value()) {
          Op def_op(context, def->op);
          if (def_op.Code() == kLiteRtOpCodeTflCast &&
              def_op.Inputs().size() == 1) {
            const Tensor src = def_op.Inputs()[0];
            const auto src_dims = GetStaticDims(src);
            has_zero_one_float_source =
                src_dims && *src_dims == *dims && IsKnownZeroOneFloat(src);
          }
        }
      }
      if (!has_zero_one_float_source) {
        return kLiteRtStatusPatternNoMatch;
      }
      LITERT_ASSIGN_OR_RETURN(Tensor resolved,
                              ResolveBoolAsZeroOneFloat(builder, in, *dims));
      LITERT_ASSIGN_OR_RETURN(Tensor zero,
                              BuildScalarFloatConst(builder, 0.0f));
      LITERT_RETURN_IF_ERROR(BuildBinaryFloatOp(builder, kLiteRtOpCodeTflAdd,
                                                resolved, zero, out));
      LITERT_RETURN_IF_ERROR(builder.EraseOp(current_op));
      LITERT_LOG(LITERT_INFO,
                 "BoolMaskBypassCast: lowered output Cast(Bool->F32) to Add");
      return kLiteRtStatusOk;
    }
  }

  return kLiteRtStatusPatternNoMatch;
}

LiteRtStatus BoolMaskCleanupCastTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op) {
  Builder builder(context, builder_ptr);
  Op current_op(context, op);
  if (current_op.Code() != kLiteRtOpCodeTflCast ||
      current_op.Inputs().size() != 1 || current_op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  const Tensor in = current_op.Inputs()[0];
  const Tensor out = current_op.Outputs()[0];
  if (in.ElementType() != ElementType::Bool &&
      out.ElementType() != ElementType::Bool) {
    return kLiteRtStatusPatternNoMatch;
  }
  if (!out.Uses().empty()) {
    return kLiteRtStatusPatternNoMatch;
  }
  LITERT_ASSIGN_OR_RETURN(bool is_sg_out, builder.IsSubgraphOutput(out));
  if (is_sg_out) {
    return kLiteRtStatusPatternNoMatch;
  }
  LITERT_RETURN_IF_ERROR(builder.EraseOp(current_op));
  LITERT_LOG(LITERT_INFO, "BoolMaskCleanupCast: erased unused Bool Cast op");
  return kLiteRtStatusOk;
}

LiteRtStatus BoolMaskNotEqualTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op) {
  Builder builder(context, builder_ptr);
  Op current_op(context, op);
  if (current_op.Code() != kLiteRtOpCodeTflNotEqual) {
    return kLiteRtStatusPatternNoMatch;
  }
  return LowerZeroOneEqualityComparison(builder, current_op,
                                        /*is_equal=*/false);
}

LiteRtStatus BoolMaskTileWrapTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op) {
  Builder builder(context, builder_ptr);
  Op out_cast(context, op);
  if (out_cast.Code() != kLiteRtOpCodeTflCast ||
      out_cast.Inputs().size() != 1 || out_cast.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  const Tensor tile_out = out_cast.Inputs()[0];
  const Tensor bool_out = out_cast.Outputs()[0];
  if (!IsUnquantizedOfType(tile_out, ElementType::Int32) ||
      !IsUnquantizedOfType(bool_out, ElementType::Bool)) {
    return kLiteRtStatusPatternNoMatch;
  }
  auto tile_def = tile_out.DefiningOp();
  if (!tile_def.has_value()) {
    return kLiteRtStatusPatternNoMatch;
  }
  Op tile_op(context, tile_def->op);
  if (tile_op.Code() != kLiteRtOpCodeTflTile || tile_op.Inputs().size() != 2 ||
      tile_op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  const Tensor tile_in = tile_op.Inputs()[0];
  const Tensor reps = tile_op.Inputs()[1];
  if (!IsUnquantizedOfType(tile_in, ElementType::Int32)) {
    return kLiteRtStatusPatternNoMatch;
  }
  auto in_def = tile_in.DefiningOp();
  if (!in_def.has_value()) {
    return kLiteRtStatusPatternNoMatch;
  }
  Op in_cast(context, in_def->op);
  if (in_cast.Code() != kLiteRtOpCodeTflCast || in_cast.Inputs().size() != 1 ||
      in_cast.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  const Tensor bool_in = in_cast.Inputs()[0];
  if (!IsUnquantizedOfType(bool_in, ElementType::Bool) &&
      !IsKnownZeroOneFloat(bool_in)) {
    return kLiteRtStatusPatternNoMatch;
  }
  const auto in_dims = GetStaticDims(bool_in);
  const auto tile_in_dims = GetStaticDims(tile_in);
  const auto tile_out_dims = GetStaticDims(tile_out);
  const auto out_dims = GetStaticDims(bool_out);
  if (!in_dims || !tile_in_dims || !tile_out_dims || !out_dims ||
      *in_dims != *tile_in_dims || *out_dims != *tile_out_dims ||
      !CanResolveBoolAsZeroOneFloat(bool_in, *in_dims)) {
    return kLiteRtStatusPatternNoMatch;
  }
  auto reps_i32 =
      ValidateAndConvertTileReps(builder, reps, *in_dims, *out_dims);
  if (!reps_i32) {
    return kLiteRtStatusPatternNoMatch;
  }

  LITERT_ASSIGN_OR_RETURN(bool tile_out_is_sg,
                          builder.IsSubgraphOutput(tile_out));
  LITERT_ASSIGN_OR_RETURN(bool tile_in_is_sg,
                          builder.IsSubgraphOutput(tile_in));
  const bool erase_tile = !tile_out_is_sg && tile_out.Uses().size() == 1;
  const bool erase_in_cast =
      erase_tile && !tile_in_is_sg && tile_in.Uses().size() == 1;

  LITERT_ASSIGN_OR_RETURN(
      Tensor f_in, ResolveBoolAsZeroOneFloat(builder, bool_in, *in_dims));
  LITERT_ASSIGN_OR_RETURN(
      Tensor f_out,
      BuildTensorOfType(builder, ElementType::Float32, *out_dims));
  LITERT_RETURN_IF_ERROR(
      builder.BuildOp(kLiteRtOpCodeTflTile, {f_in, *reps_i32}, {f_out}));
  LITERT_RETURN_IF_ERROR(
      builder.BuildOp(kLiteRtOpCodeTflCast, {f_out}, {bool_out}));

  LITERT_RETURN_IF_ERROR(builder.EraseOp(out_cast));
  if (erase_tile) {
    LITERT_RETURN_IF_ERROR(builder.EraseOp(tile_op));
  }
  if (erase_in_cast) {
    LITERT_RETURN_IF_ERROR(builder.EraseOp(in_cast));
  }
  LITERT_LOG(
      LITERT_INFO,
      "BoolMaskTileWrap: lowered Cast->Int32 Tile->Cast to Float32 Tile");
  return kLiteRtStatusOk;
}

LiteRtStatus BoolMaskOpToFloatTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op) {
  Builder builder(context, builder_ptr);
  Op current_op(context, op);
  switch (current_op.Code()) {
    case kLiteRtOpCodeTflLogicalNot:
      return LowerLogicalNot(builder, current_op);
    case kLiteRtOpCodeTflLogicalAnd:
      return LowerBinaryBoolOp(builder, current_op, kLiteRtOpCodeTflMul);
    case kLiteRtOpCodeTflLogicalOr:
      return LowerBinaryBoolOp(builder, current_op, kLiteRtOpCodeTflMaximum);
    case kLiteRtOpCodeTflEqual:
      return LowerZeroOneEqualityComparison(builder, current_op,
                                            /*is_equal=*/true);
    case kLiteRtOpCodeTflReshape:
    case kLiteRtOpCodeTflSqueeze:
    case kLiteRtOpCodeTflExpandDims:
      return LowerBoolShapeChangeToReshape(context, builder, current_op);
    case kLiteRtOpCodeTflSlice:
    case kLiteRtOpCodeTflStridedSlice:
    case kLiteRtOpCodeTflTile:
    case kLiteRtOpCodeTflTranspose:
      return LowerBoolDataMovementOp(context, builder, current_op);
    case kLiteRtOpCodeTflConcatenation:
      return LowerBoolConcatenation(context, builder, current_op);
    case kLiteRtOpCodeTflSelect:
    case kLiteRtOpCodeTflSelectV2:
      return LowerSelectToFloat(builder, current_op);
    default:
      return kLiteRtStatusPatternNoMatch;
  }
}

LiteRtStatus BoolMaskOutputNotEqualTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op) {
  Builder builder(context, builder_ptr);
  Op current_op(context, op);
  if (current_op.Code() != kLiteRtOpCodeTflCast ||
      current_op.Inputs().size() != 1 || current_op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  const Tensor in = current_op.Inputs()[0];
  const Tensor out = current_op.Outputs()[0];
  if (!IsUnquantizedOfType(in, ElementType::Float32) ||
      !IsUnquantizedOfType(out, ElementType::Bool) || !out.Uses().empty()) {
    return kLiteRtStatusPatternNoMatch;
  }
  LITERT_ASSIGN_OR_RETURN(bool is_sg_out, builder.IsSubgraphOutput(out));
  if (!is_sg_out) {
    return kLiteRtStatusPatternNoMatch;
  }
  const auto in_dims = GetStaticDims(in);
  const auto out_dims = GetStaticDims(out);
  // Rank-0 NotEqual is not delegable (see VerifyCommonOp), so keep the Cast.
  if (!in_dims || !out_dims || in_dims->empty() || *in_dims != *out_dims) {
    return kLiteRtStatusPatternNoMatch;
  }
  const auto num_elems = NumElements(*in_dims);
  if (!num_elems || *num_elems > kMaxMaterializedConstElements) {
    return kLiteRtStatusPatternNoMatch;
  }
  std::vector<float> zero_vals(*num_elems, 0.0f);
  LITERT_ASSIGN_OR_RETURN(Tensor zeros,
                          BuildConst<float>(builder, *in_dims, zero_vals));
  LITERT_RETURN_IF_ERROR(
      builder.BuildOp(kLiteRtOpCodeTflNotEqual, {in, zeros}, {out}));
  LITERT_RETURN_IF_ERROR(builder.EraseOp(current_op));
  LITERT_LOG(LITERT_INFO,
             "BoolMaskOutputNotEqual: replaced output Cast with NotEqual");
  return kLiteRtStatusOk;
}

}  // extern "C"
