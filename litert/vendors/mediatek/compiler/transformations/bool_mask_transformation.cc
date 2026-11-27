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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_matchers.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/compiler/cc/litert_op_options.h"
#include "litert/compiler/cc/transformation_utils.h"

namespace {

using ::litert::ElementType;
using ::litert::Expected;
using ::litert::Unexpected;
using ::litert::compiler::AddOptions;
using ::litert::compiler::BroadcastShapes;
using ::litert::compiler::BuildBinaryOp;
using ::litert::compiler::BuildBinaryOpInto;
using ::litert::compiler::BuildCast;
using ::litert::compiler::BuildCastInto;
using ::litert::compiler::BuildConcat;
using ::litert::compiler::BuildConst;
using ::litert::compiler::Builder;
using ::litert::compiler::BuildReshape;
using ::litert::compiler::BuildScalarConst;
using ::litert::compiler::BuildSlice;
using ::litert::compiler::BuildTypedTensor;
using ::litert::compiler::CheckedNumElements;
using ::litert::compiler::ConcatenationOptions;
using ::litert::compiler::ConstI32;
using ::litert::compiler::DepthwiseConv2dOptions;
using ::litert::compiler::GetOptionsAs;
using ::litert::compiler::GetStaticPositiveDims;
using ::litert::compiler::HasDims;
using ::litert::compiler::m_AllOf;
using ::litert::compiler::m_Capture;
using ::litert::compiler::m_ConstantElements;
using ::litert::compiler::m_ElementType;
using ::litert::compiler::m_HasNoUses;
using ::litert::compiler::m_HasStaticPositiveShape;
using ::litert::compiler::m_IsConstant;
using ::litert::compiler::m_IsSubgraphInput;
using ::litert::compiler::m_IsSubgraphOutput;
using ::litert::compiler::m_IsUnquantized;
using ::litert::compiler::m_Not;
using ::litert::compiler::m_Op;
using ::litert::compiler::m_OpCodeAnyOf;
using ::litert::compiler::m_Options;
using ::litert::compiler::m_Rank;
using ::litert::compiler::m_RankAtLeast;
using ::litert::compiler::m_SameShapeAs;
using ::litert::compiler::m_Shape;
using ::litert::compiler::m_SingleOutput;
using ::litert::compiler::m_UnquantizedElementType;
using ::litert::compiler::Match;
using ::litert::compiler::MulOptions;
using ::litert::compiler::Op;
using ::litert::compiler::ReadIntConstValues;
using ::litert::compiler::ReshapeOptions;
using ::litert::compiler::SubOptions;
using ::litert::compiler::Tensor;

// Lightweight Expected unwrap helpers that avoid instantiating
// ErrorStatusBuilder and SourceLocation formatting at every lowering site.
template <typename T>
LiteRtStatus UnwrapStatus(const Expected<T>& res) {
  return res.HasValue() ? kLiteRtStatusOk : res.Error().Status();
}

template <typename T>
LiteRtStatus AssignOrStatus(Expected<T>&& res, T* out) {
  if (!res.HasValue()) {
    return res.Error().Status();
  }
  *out = std::move(*res);
  return kLiteRtStatusOk;
}

constexpr size_t kMaxMaterializedConstElements = 1u << 20;
constexpr int kMaxZeroOneRecursionDepth = 24;
constexpr int kMemoKnownTrue = (std::numeric_limits<int>::max)();

using TensorDepthMemo = std::vector<std::pair<LiteRtTensor, int>>;

std::optional<bool> LookupMemo(const TensorDepthMemo& memo, LiteRtTensor t,
                               int depth) {
  for (const auto& [cached_t, cached_depth] : memo) {
    if (cached_t == t) {
      if (cached_depth == kMemoKnownTrue || cached_depth >= depth) {
        return cached_depth == kMemoKnownTrue;
      }
      return std::nullopt;
    }
  }
  return std::nullopt;
}

void StoreMemo(TensorDepthMemo& memo, LiteRtTensor t, bool result, int depth) {
  const int encoded = result ? kMemoKnownTrue : depth;
  for (auto& [cached_t, cached_depth] : memo) {
    if (cached_t == t) {
      cached_depth = encoded;
      return;
    }
  }
  memo.emplace_back(t, encoded);
}

auto m_StaticType(LiteRtElementType type) {
  return m_AllOf(m_UnquantizedElementType(type), m_HasStaticPositiveShape());
}

bool MatchStaticType(const Tensor& tensor, LiteRtElementType type,
                     size_t min_rank = 0) {
  return Match(tensor, m_StaticType(type)) &&
         (min_rank == 0 || Match(tensor, m_RankAtLeast(min_rank)));
}

bool MatchSingleStaticOutput(const Op& op, Tensor* out, LiteRtElementType type,
                             size_t min_rank = 0) {
  return Match(op, m_SingleOutput(m_Capture(out))) &&
         MatchStaticType(*out, type, min_rank);
}

bool MatchCastInput(const Tensor& tensor, Tensor* src, Op* cast_op = nullptr) {
  Op def_op;
  if (!Match(tensor, m_Capture(&def_op)) ||
      !Match(def_op, m_Op<kLiteRtOpCodeTflCast>(m_Capture(src)))) {
    return false;
  }
  if (cast_op != nullptr) {
    *cast_op = def_op;
  }
  return true;
}

bool MatchFloatConstPred(const Tensor& tensor, bool (*pred)(float)) {
  return Match(tensor,
               m_AllOf(m_UnquantizedElementType(kLiteRtElementTypeFloat32),
                       m_ConstantElements<float>(pred)));
}

bool IsPositiveZeroFloat(float v) { return v == 0.0f && !std::signbit(v); }
bool IsAnyZeroFloat(float v) { return v == 0.0f; }
bool IsOneFloat(float v) { return v == 1.0f; }
bool IsBinaryZeroOneFloat(float v) {
  return (v == 0.0f && !std::signbit(v)) || v == 1.0f;
}
bool IsFiniteFloat(float v) { return std::isfinite(v); }

bool IsIdentityActivationOnZeroOne(uint32_t activation) {
  // None (0), Relu (1), and Relu6 (3) preserve {+0.0f, 1.0f}.
  return activation == 0 || activation == 1 || activation == 3;
}

template <typename OptionsT>
bool HasActivationPred(const Op& op, bool (*pred)(uint32_t)) {
  return Match(op, m_Options<OptionsT>([pred](const OptionsT& o) {
                 return pred(o.fused_activation_function);
               }));
}

bool IsNoActivation(uint32_t activation) { return activation == 0; }

bool MayPropagateNonFiniteFloatConstImpl(const Tensor& tensor, int depth,
                                         TensorDepthMemo& memo) {
  if (tensor.Get() == nullptr || tensor.ElementType() != ElementType::Float32) {
    return false;
  }
  if (Match(tensor, m_IsConstant())) {
    return !Match(tensor, m_ConstantElements<float>(&IsFiniteFloat));
  }
  Op def_op;
  if (!Match(tensor, m_Capture(&def_op))) {
    return false;
  }
  if (depth < 0) {
    return true;
  }
  // Memoize `!result` ("proven free of propagated non-finite constants"), which
  // is monotonic in `depth` just like `IsKnownZeroOneFloatImpl`.
  if (std::optional<bool> proven_safe = LookupMemo(memo, tensor.Get(), depth)) {
    return !*proven_safe;
  }
  bool result = false;
  const std::vector<Tensor> inputs = def_op.Inputs();
  switch (def_op.Code()) {
    case kLiteRtOpCodeTflSelect:
    case kLiteRtOpCodeTflSelectV2:
      result =
          inputs.size() == 3 &&
          (MayPropagateNonFiniteFloatConstImpl(inputs[1], depth - 1, memo) ||
           MayPropagateNonFiniteFloatConstImpl(inputs[2], depth - 1, memo));
      break;
    case kLiteRtOpCodeTflReshape:
    case kLiteRtOpCodeTflSlice:
    case kLiteRtOpCodeTflTile:
    case kLiteRtOpCodeTflPad:
      result = !inputs.empty() &&
               MayPropagateNonFiniteFloatConstImpl(inputs[0], depth - 1, memo);
      break;
    case kLiteRtOpCodeTflConcatenation:
      for (const Tensor& in : inputs) {
        if (MayPropagateNonFiniteFloatConstImpl(in, depth - 1, memo)) {
          result = true;
          break;
        }
      }
      break;
    default:
      break;
  }
  StoreMemo(memo, tensor.Get(), !result, depth);
  return result;
}

bool MayPropagateNonFiniteFloatConst(const Tensor& tensor) {
  TensorDepthMemo memo;
  return MayPropagateNonFiniteFloatConstImpl(tensor, kMaxZeroOneRecursionDepth,
                                             memo);
}

bool IsKnownZeroOneFloatImpl(const Tensor& tensor, int depth,
                             TensorDepthMemo& memo);

// Returns true when a DepthwiseConv2d selects at most one spatial {0.0, 1.0}
// sample per output channel with zero bias and identity-on-{0,1} activation.
bool IsZeroOneDepthwiseConv2d(const Op& op, int depth, TensorDepthMemo& memo) {
  const std::vector<Tensor> inputs = op.Inputs();
  if (inputs.size() < 2 || inputs.size() > 3 ||
      !Match(op, m_Options<DepthwiseConv2dOptions>(
                     [](const DepthwiseConv2dOptions& o) {
                       return o.stride_w > 0 && o.stride_h > 0 &&
                              o.dilation_w_factor > 0 &&
                              o.dilation_h_factor > 0 &&
                              o.depth_multiplier > 0 &&
                              IsIdentityActivationOnZeroOne(
                                  o.fused_activation_function);
                     })) ||
      !IsKnownZeroOneFloatImpl(inputs[0], depth - 1, memo) ||
      !MatchStaticType(inputs[1], kLiteRtElementTypeFloat32) ||
      !Match(inputs[1], m_AllOf(m_IsConstant(), m_Shape({1, -1, -1, -1}))) ||
      (inputs.size() == 3 &&
       !MatchFloatConstPred(inputs[2], &IsPositiveZeroFloat))) {
    return false;
  }
  const std::optional<std::vector<int32_t>> filter_dims =
      GetStaticPositiveDims(inputs[1]);
  const std::optional<size_t> expected_elems =
      filter_dims ? CheckedNumElements(*filter_dims) : std::nullopt;
  const Expected<absl::Span<const float>> weights =
      inputs[1].WeightsData<float>();
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
    if (w == 1.0f && ++ones_per_channel[i % c_out] <= 1) {
      continue;
    }
    return false;
  }
  return true;
}

bool ComputeIsKnownZeroOneFloat(const Tensor& tensor, int depth,
                                TensorDepthMemo& memo) {
  if (MatchFloatConstPred(tensor, &IsBinaryZeroOneFloat)) {
    return true;
  }
  Op def_op;
  if (!Match(tensor, m_Capture(&def_op))) {
    return false;
  }
  const std::vector<Tensor> inputs = def_op.Inputs();
  switch (def_op.Code()) {
    case kLiteRtOpCodeTflCast:
      return inputs.size() == 1 &&
             MatchStaticType(inputs[0], kLiteRtElementTypeBool);
    case kLiteRtOpCodeTflSub:
      return inputs.size() == 2 &&
             HasActivationPred<SubOptions>(def_op, &IsNoActivation) &&
             MatchFloatConstPred(inputs[0], &IsOneFloat) &&
             IsKnownZeroOneFloatImpl(inputs[1], depth - 1, memo);
    case kLiteRtOpCodeTflAbs: {
      if (inputs.size() != 1) {
        return false;
      }
      if (IsKnownZeroOneFloatImpl(inputs[0], depth - 1, memo)) {
        return true;
      }
      Op sub_op;
      if (!MatchStaticType(inputs[0], kLiteRtElementTypeFloat32) ||
          !Match(inputs[0], m_Capture(&sub_op)) ||
          sub_op.Code() != kLiteRtOpCodeTflSub ||
          !HasActivationPred<SubOptions>(sub_op, &IsNoActivation)) {
        return false;
      }
      const std::vector<Tensor> sub_in = sub_op.Inputs();
      return sub_in.size() == 2 &&
             IsKnownZeroOneFloatImpl(sub_in[0], depth - 2, memo) &&
             IsKnownZeroOneFloatImpl(sub_in[1], depth - 2, memo);
    }
    case kLiteRtOpCodeTflMul:
      if (!HasActivationPred<MulOptions>(def_op,
                                         &IsIdentityActivationOnZeroOne)) {
        return false;
      }
      [[fallthrough]];
    case kLiteRtOpCodeTflMinimum:
    case kLiteRtOpCodeTflMaximum:
      return inputs.size() == 2 &&
             IsKnownZeroOneFloatImpl(inputs[0], depth - 1, memo) &&
             IsKnownZeroOneFloatImpl(inputs[1], depth - 1, memo);
    case kLiteRtOpCodeTflReshape:
    case kLiteRtOpCodeTflSlice:
    case kLiteRtOpCodeTflTile:
    case kLiteRtOpCodeTflPad:
      return !inputs.empty() &&
             IsKnownZeroOneFloatImpl(inputs[0], depth - 1, memo);
    case kLiteRtOpCodeTflConcatenation:
      if (inputs.empty() || !HasActivationPred<ConcatenationOptions>(
                                def_op, &IsIdentityActivationOnZeroOne)) {
        return false;
      }
      for (const Tensor& in : inputs) {
        if (!IsKnownZeroOneFloatImpl(in, depth - 1, memo)) {
          return false;
        }
      }
      return true;
    case kLiteRtOpCodeTflDepthwiseConv2d:
      return IsZeroOneDepthwiseConv2d(def_op, depth, memo);
    default:
      return false;
  }
}

bool IsKnownZeroOneFloatImpl(const Tensor& tensor, int depth,
                             TensorDepthMemo& memo) {
  if (depth < 0 || !MatchStaticType(tensor, kLiteRtElementTypeFloat32)) {
    return false;
  }
  if (std::optional<bool> cached = LookupMemo(memo, tensor.Get(), depth)) {
    return *cached;
  }
  const bool result = ComputeIsKnownZeroOneFloat(tensor, depth, memo);
  StoreMemo(memo, tensor.Get(), result, depth);
  return result;
}

bool IsKnownZeroOneFloat(const Tensor& tensor) {
  TensorDepthMemo memo;
  return IsKnownZeroOneFloatImpl(tensor, kMaxZeroOneRecursionDepth, memo);
}

Expected<Tensor> BuildFloatInvert(const Builder& builder, const Tensor& x,
                                  absl::Span<const int32_t> dims) {
  Expected<Tensor> one = BuildScalarConst<float>(builder, 1.0f);
  if (!one) {
    return Unexpected(one.Error().Status());
  }
  return BuildBinaryOp(builder, kLiteRtOpCodeTflSub, *one, x,
                       ElementType::Float32, dims);
}

bool CanMaterializeZeroFloatConst(absl::Span<const int32_t> dims) {
  const std::optional<size_t> num_elems = CheckedNumElements(dims);
  return num_elems && *num_elems <= kMaxMaterializedConstElements;
}

Expected<Tensor> BuildZeroFloatConst(const Builder& builder,
                                     absl::Span<const int32_t> dims) {
  const std::optional<size_t> num_elems = CheckedNumElements(dims);
  if (!num_elems || *num_elems > kMaxMaterializedConstElements) {
    return Unexpected(kLiteRtStatusPatternNoMatch);
  }
  std::vector<float> zeros(*num_elems, 0.0f);
  return BuildConst<float>(builder, dims, zeros);
}

Expected<Tensor> ConvertBoolConstToFloat(const Builder& builder,
                                         const Tensor& bool_const,
                                         absl::Span<const int32_t> dims) {
  if (!MatchStaticType(bool_const, kLiteRtElementTypeBool) ||
      !Match(bool_const, m_IsConstant())) {
    return Unexpected(kLiteRtStatusPatternNoMatch);
  }
  const std::optional<size_t> num_elems = CheckedNumElements(dims);
  const absl::Span<const uint8_t> bytes = bool_const.Weights().Bytes();
  if (!num_elems || bytes.size() != *num_elems) {
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
  if (!Match(tensor, m_HasStaticPositiveShape()) || !HasDims(tensor, dims)) {
    return false;
  }
  if (MatchStaticType(tensor, kLiteRtElementTypeFloat32)) {
    return IsKnownZeroOneFloat(tensor);
  }
  if (!MatchStaticType(tensor, kLiteRtElementTypeBool)) {
    return false;
  }
  if (Match(tensor, m_IsConstant())) {
    const std::optional<size_t> num_elems = CheckedNumElements(dims);
    return num_elems && tensor.Weights().Bytes().size() == *num_elems;
  }
  if (Match(tensor, m_IsSubgraphInput())) {
    return true;
  }
  // Never synthesize a Cast(Bool->F32) after an unrewritten producer op: doing
  // so prematurely breaks downstream multi-op patterns (EntryEmbedding,
  // FloorDiv, AttentionMask) and creates NPU-CPU ping-pong islands.
  Tensor src;
  return MatchCastInput(tensor, &src) &&
         MatchStaticType(src, kLiteRtElementTypeFloat32) &&
         Match(src, m_SameShapeAs(&tensor)) && IsKnownZeroOneFloat(src);
}

Expected<Tensor> ResolveBoolAsZeroOneFloat(const Builder& builder,
                                           const Tensor& tensor,
                                           absl::Span<const int32_t> dims) {
  if (!CanResolveBoolAsZeroOneFloat(tensor, dims)) {
    return Unexpected(kLiteRtStatusPatternNoMatch);
  }
  if (MatchStaticType(tensor, kLiteRtElementTypeFloat32)) {
    return tensor;
  }
  if (Match(tensor, m_IsConstant())) {
    return ConvertBoolConstToFloat(builder, tensor, dims);
  }
  Tensor src;
  if (MatchCastInput(tensor, &src)) {
    return src;
  }
  return BuildCast(builder, tensor, ElementType::Float32, dims);
}

std::optional<Tensor> TryBypassDoubleBoolCast(const Builder& builder,
                                              const Tensor& input) {
  Tensor bool_src;
  if (!MatchStaticType(input, kLiteRtElementTypeFloat32) ||
      !MatchCastInput(input, &bool_src) ||
      !MatchStaticType(bool_src, kLiteRtElementTypeBool) ||
      !Match(bool_src, m_SameShapeAs(&input))) {
    return std::nullopt;
  }
  if (Match(bool_src, m_IsConstant())) {
    const std::optional<std::vector<int32_t>> dims =
        GetStaticPositiveDims(bool_src);
    if (!dims) {
      return std::nullopt;
    }
    Expected<Tensor> res = ConvertBoolConstToFloat(builder, bool_src, *dims);
    return res.HasValue() ? std::optional<Tensor>(*res) : std::nullopt;
  }
  Tensor orig_f32;
  if (MatchCastInput(bool_src, &orig_f32) &&
      MatchStaticType(orig_f32, kLiteRtElementTypeFloat32) &&
      Match(orig_f32, m_SameShapeAs(&input)) && IsKnownZeroOneFloat(orig_f32)) {
    return orig_f32;
  }
  return std::nullopt;
}

auto m_ZeroOptionBypassConsumer() {
  return m_OpCodeAnyOf<
      kLiteRtOpCodeTflAbs, kLiteRtOpCodeTflCast, kLiteRtOpCodeTflDequantize,
      kLiteRtOpCodeTflEqual, kLiteRtOpCodeTflMaximum, kLiteRtOpCodeTflMinimum,
      kLiteRtOpCodeTflNotEqual, kLiteRtOpCodeTflPad, kLiteRtOpCodeTflQuantize,
      kLiteRtOpCodeTflRelu, kLiteRtOpCodeTflSelect, kLiteRtOpCodeTflSelectV2,
      kLiteRtOpCodeTflSlice, kLiteRtOpCodeTflTile>();
}

template <typename OptionsT>
LiteRtStatus BuildOptionedBypassOp(const Builder& builder, const Op& src,
                                   const std::vector<Tensor>& inputs,
                                   const std::vector<Tensor>& outputs) {
  OptionsT opts;
  if (LiteRtStatus s =
          AssignOrStatus(GetOptionsAs<OptionsT>(src.ctx(), src.Get()), &opts);
      s != kLiteRtStatusOk) {
    return s;
  }
  Op dst;
  if (LiteRtStatus s =
          AssignOrStatus(builder.BuildOp(src.Code(), inputs, outputs), &dst);
      s != kLiteRtStatusOk) {
    return s;
  }
  return UnwrapStatus(builder.SetOpOptions(dst, std::move(opts)));
}

LiteRtStatus BuildBypassConsumerOp(const Builder& builder, const Op& src,
                                   const std::vector<Tensor>& inputs,
                                   const std::vector<Tensor>& outputs) {
  switch (src.Code()) {
    case kLiteRtOpCodeTflAdd:
      return BuildOptionedBypassOp<AddOptions>(builder, src, inputs, outputs);
    case kLiteRtOpCodeTflConcatenation:
      return BuildOptionedBypassOp<ConcatenationOptions>(builder, src, inputs,
                                                         outputs);
    case kLiteRtOpCodeTflMul:
      return BuildOptionedBypassOp<MulOptions>(builder, src, inputs, outputs);
    case kLiteRtOpCodeTflReshape:
      return BuildOptionedBypassOp<ReshapeOptions>(builder, src, inputs,
                                                   outputs);
    case kLiteRtOpCodeTflSub:
      return BuildOptionedBypassOp<SubOptions>(builder, src, inputs, outputs);
    default:
      if (!Match(src, m_ZeroOptionBypassConsumer())) {
        return kLiteRtStatusPatternNoMatch;
      }
      return UnwrapStatus(builder.BuildOp(src.Code(), inputs, outputs));
  }
}

bool IsSupportedBypassConsumer(const Op& op) {
  switch (op.Code()) {
    case kLiteRtOpCodeTflAdd:
      return GetOptionsAs<AddOptions>(op.ctx(), op.Get()).HasValue();
    case kLiteRtOpCodeTflConcatenation:
      return GetOptionsAs<ConcatenationOptions>(op.ctx(), op.Get()).HasValue();
    case kLiteRtOpCodeTflMul:
      return GetOptionsAs<MulOptions>(op.ctx(), op.Get()).HasValue();
    case kLiteRtOpCodeTflReshape:
      return Match(op, m_SingleOutput(m_AllOf(m_HasStaticPositiveShape(),
                                              m_RankAtLeast(1)))) &&
             GetOptionsAs<ReshapeOptions>(op.ctx(), op.Get()).HasValue();
    case kLiteRtOpCodeTflSub:
      return GetOptionsAs<SubOptions>(op.ctx(), op.Get()).HasValue();
    default:
      return Match(op, m_ZeroOptionBypassConsumer());
  }
}

Expected<Tensor> ValidateAndConvertTileReps(
    const Builder& builder, const Tensor& reps,
    absl::Span<const int32_t> in_dims, absl::Span<const int32_t> out_dims) {
  if (in_dims.empty() || in_dims.size() != out_dims.size() ||
      !Match(reps,
             m_AllOf(m_IsConstant(), m_HasStaticPositiveShape(), m_Rank(1)))) {
    return Unexpected(kLiteRtStatusPatternNoMatch);
  }
  const std::optional<std::vector<int64_t>> reps_vals =
      ReadIntConstValues(reps);
  if (!reps_vals || reps_vals->size() != in_dims.size()) {
    return Unexpected(kLiteRtStatusPatternNoMatch);
  }
  std::vector<int32_t> reps_i32;
  reps_i32.reserve(in_dims.size());
  for (size_t i = 0; i < in_dims.size(); ++i) {
    const int64_t rep = (*reps_vals)[i];
    if (rep <= 0 || rep > (std::numeric_limits<int32_t>::max)() ||
        static_cast<int64_t>(in_dims[i]) * rep != out_dims[i]) {
      return Unexpected(kLiteRtStatusPatternNoMatch);
    }
    reps_i32.push_back(static_cast<int32_t>(rep));
  }
  if (MatchStaticType(reps, kLiteRtElementTypeInt32)) {
    return reps;
  }
  return ConstI32(builder, reps_i32);
}

LiteRtStatus BuildBoolCastBridge(const Builder& builder, const Op& old_op,
                                 const Tensor& f_out, const Tensor& bool_out) {
  if (LiteRtStatus s = UnwrapStatus(BuildCastInto(builder, f_out, bool_out));
      s != kLiteRtStatusOk) {
    return s;
  }
  return UnwrapStatus(builder.EraseOp(old_op));
}

bool IsNativeSameShapeOutputNotEqual(const Op& op) {
  Tensor out;
  const std::vector<Tensor> inputs = op.Inputs();
  return op.Code() == kLiteRtOpCodeTflNotEqual && inputs.size() == 2 &&
         MatchStaticType(inputs[0], kLiteRtElementTypeFloat32, 1) &&
         MatchStaticType(inputs[1], kLiteRtElementTypeFloat32) &&
         Match(inputs[1], m_SameShapeAs(&inputs[0])) &&
         MatchSingleStaticOutput(op, &out, kLiteRtElementTypeBool) &&
         Match(out, m_AllOf(m_SameShapeAs(&inputs[0]), m_HasNoUses(),
                            m_IsSubgraphOutput()));
}

LiteRtStatus LowerLogicalNot(const Builder& builder, const Op& op) {
  Tensor in;
  Tensor out;
  if (!Match(op, m_Op<kLiteRtOpCodeTflLogicalNot>(m_Capture(&in))) ||
      !MatchStaticType(in, kLiteRtElementTypeBool) ||
      !MatchSingleStaticOutput(op, &out, kLiteRtElementTypeBool) ||
      !Match(out, m_SameShapeAs(&in))) {
    return kLiteRtStatusPatternNoMatch;
  }
  const std::optional<std::vector<int32_t>> dims = GetStaticPositiveDims(out);
  if (!dims || !CanResolveBoolAsZeroOneFloat(in, *dims)) {
    return kLiteRtStatusPatternNoMatch;
  }
  Tensor x_f;
  if (LiteRtStatus s =
          AssignOrStatus(ResolveBoolAsZeroOneFloat(builder, in, *dims), &x_f);
      s != kLiteRtStatusOk) {
    return s;
  }
  Tensor f_out;
  if (LiteRtStatus s =
          AssignOrStatus(BuildFloatInvert(builder, x_f, *dims), &f_out);
      s != kLiteRtStatusOk) {
    return s;
  }
  return BuildBoolCastBridge(builder, op, f_out, out);
}

LiteRtStatus LowerBinaryBoolOp(const Builder& builder, const Op& op,
                               LiteRtOpCode float_code) {
  Tensor out;
  const std::vector<Tensor> inputs = op.Inputs();
  if (inputs.size() != 2 ||
      !MatchStaticType(inputs[0], kLiteRtElementTypeBool) ||
      !MatchStaticType(inputs[1], kLiteRtElementTypeBool) ||
      !MatchSingleStaticOutput(op, &out, kLiteRtElementTypeBool)) {
    return kLiteRtStatusPatternNoMatch;
  }
  const std::optional<std::vector<int32_t>> lhs_dims =
      GetStaticPositiveDims(inputs[0]);
  const std::optional<std::vector<int32_t>> rhs_dims =
      GetStaticPositiveDims(inputs[1]);
  const std::optional<std::vector<int32_t>> out_dims =
      GetStaticPositiveDims(out);
  if (!lhs_dims || !rhs_dims || !out_dims) {
    return kLiteRtStatusPatternNoMatch;
  }
  const std::optional<std::vector<int32_t>> bcast =
      BroadcastShapes(*lhs_dims, *rhs_dims);
  if (!bcast || *bcast != *out_dims ||
      !CanResolveBoolAsZeroOneFloat(inputs[0], *lhs_dims) ||
      !CanResolveBoolAsZeroOneFloat(inputs[1], *rhs_dims)) {
    return kLiteRtStatusPatternNoMatch;
  }
  Tensor lhs_f;
  if (LiteRtStatus s = AssignOrStatus(
          ResolveBoolAsZeroOneFloat(builder, inputs[0], *lhs_dims), &lhs_f);
      s != kLiteRtStatusOk) {
    return s;
  }
  Tensor rhs_f = lhs_f;
  if (inputs[1] != inputs[0]) {
    if (LiteRtStatus s = AssignOrStatus(
            ResolveBoolAsZeroOneFloat(builder, inputs[1], *rhs_dims), &rhs_f);
        s != kLiteRtStatusOk) {
      return s;
    }
  }
  Tensor f_out;
  if (LiteRtStatus s =
          AssignOrStatus(BuildBinaryOp(builder, float_code, lhs_f, rhs_f,
                                       ElementType::Float32, *out_dims),
                         &f_out);
      s != kLiteRtStatusOk) {
    return s;
  }
  return BuildBoolCastBridge(builder, op, f_out, out);
}

LiteRtStatus LowerBoolReshape(const Builder& builder, const Op& op) {
  Tensor out;
  const std::vector<Tensor> inputs = op.Inputs();
  if (inputs.empty() || inputs.size() > 2 ||
      !MatchStaticType(inputs[0], kLiteRtElementTypeBool, 1) ||
      (inputs.size() == 2 &&
       !Match(inputs[1], m_AllOf(m_IsConstant(), m_IsUnquantized()))) ||
      !MatchSingleStaticOutput(op, &out, kLiteRtElementTypeBool, 1)) {
    return kLiteRtStatusPatternNoMatch;
  }
  const std::optional<std::vector<int32_t>> in_dims =
      GetStaticPositiveDims(inputs[0]);
  const std::optional<std::vector<int32_t>> out_dims =
      GetStaticPositiveDims(out);
  if (!in_dims || !out_dims ||
      !CanResolveBoolAsZeroOneFloat(inputs[0], *in_dims)) {
    return kLiteRtStatusPatternNoMatch;
  }
  const std::optional<size_t> in_elems = CheckedNumElements(*in_dims);
  const std::optional<size_t> out_elems = CheckedNumElements(*out_dims);
  if (!in_elems || !out_elems || *in_elems != *out_elems) {
    return kLiteRtStatusPatternNoMatch;
  }
  Tensor in_f;
  if (LiteRtStatus s = AssignOrStatus(
          ResolveBoolAsZeroOneFloat(builder, inputs[0], *in_dims), &in_f);
      s != kLiteRtStatusOk) {
    return s;
  }
  Tensor f_out;
  if (LiteRtStatus s = AssignOrStatus(
          BuildReshape(builder, in_f, in_f, *out_dims, ""), &f_out);
      s != kLiteRtStatusOk) {
    return s;
  }
  return BuildBoolCastBridge(builder, op, f_out, out);
}

LiteRtStatus LowerBoolSlice(const Builder& builder, const Op& op) {
  Tensor in;
  Tensor begin_t;
  Tensor size_t_arg;
  Tensor out;
  if (!Match(op,
             m_Op<kLiteRtOpCodeTflSlice>(m_Capture(&in), m_Capture(&begin_t),
                                         m_Capture(&size_t_arg))) ||
      !MatchStaticType(in, kLiteRtElementTypeBool, 1) ||
      !Match(begin_t, m_AllOf(m_IsConstant(), m_Rank(1))) ||
      !Match(size_t_arg, m_AllOf(m_IsConstant(), m_Rank(1))) ||
      !MatchSingleStaticOutput(op, &out, kLiteRtElementTypeBool, 1)) {
    return kLiteRtStatusPatternNoMatch;
  }
  const std::optional<std::vector<int32_t>> in_dims = GetStaticPositiveDims(in);
  const std::optional<std::vector<int32_t>> out_dims =
      GetStaticPositiveDims(out);
  const std::optional<std::vector<int64_t>> begin_vals =
      ReadIntConstValues(begin_t);
  const std::optional<std::vector<int64_t>> size_vals =
      ReadIntConstValues(size_t_arg);
  if (!in_dims || !out_dims || in_dims->size() != out_dims->size() ||
      !begin_vals || !size_vals || begin_vals->size() != in_dims->size() ||
      size_vals->size() != in_dims->size() ||
      !CanResolveBoolAsZeroOneFloat(in, *in_dims)) {
    return kLiteRtStatusPatternNoMatch;
  }
  std::vector<int32_t> begin_i32(in_dims->size());
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
    begin_i32[i] = static_cast<int32_t>(b);
  }
  Tensor in_f;
  if (LiteRtStatus s = AssignOrStatus(
          ResolveBoolAsZeroOneFloat(builder, in, *in_dims), &in_f);
      s != kLiteRtStatusOk) {
    return s;
  }
  Tensor f_out;
  if (LiteRtStatus s = AssignOrStatus(
          BuildSlice(builder, in_f, begin_i32, *out_dims, in_f, ""), &f_out);
      s != kLiteRtStatusOk) {
    return s;
  }
  return BuildBoolCastBridge(builder, op, f_out, out);
}

LiteRtStatus LowerBoolConcatenation(const LiteRtCompilerContext* context,
                                    const Builder& builder, const Op& op) {
  Tensor out;
  const std::vector<Tensor> inputs = op.Inputs();
  if (inputs.empty() ||
      !HasActivationPred<ConcatenationOptions>(op, &IsNoActivation) ||
      !MatchSingleStaticOutput(op, &out, kLiteRtElementTypeBool, 1)) {
    return kLiteRtStatusPatternNoMatch;
  }
  Expected<ConcatenationOptions> opts =
      GetOptionsAs<ConcatenationOptions>(context, op.Get());
  const std::optional<std::vector<int32_t>> out_dims =
      GetStaticPositiveDims(out);
  if (!opts || !out_dims) {
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
  for (const Tensor& in : inputs) {
    const std::optional<std::vector<int32_t>> dims = GetStaticPositiveDims(in);
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
  f_inputs.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    const Tensor& in = inputs[i];
    bool reused = false;
    for (size_t j = 0; j < i; ++j) {
      if (inputs[j] == in) {
        f_inputs.push_back(f_inputs[j]);
        reused = true;
        break;
      }
    }
    if (!reused) {
      Tensor f_in;
      if (LiteRtStatus s =
              AssignOrStatus(ResolveBoolAsZeroOneFloat(
                                 builder, in, *GetStaticPositiveDims(in)),
                             &f_in);
          s != kLiteRtStatusOk) {
        return s;
      }
      f_inputs.push_back(f_in);
    }
  }
  Tensor f_out;
  if (LiteRtStatus s = AssignOrStatus(
          BuildConcat(builder, f_inputs, static_cast<int32_t>(norm_axis),
                      f_inputs[0], *out_dims, ""),
          &f_out);
      s != kLiteRtStatusOk) {
    return s;
  }
  return BuildBoolCastBridge(builder, op, f_out, out);
}

LiteRtStatus LowerSelectToFloat(const Builder& builder, const Op& op) {
  Tensor out;
  const std::vector<Tensor> inputs = op.Inputs();
  if (inputs.size() != 3 || !Match(op, m_SingleOutput(m_Capture(&out))) ||
      !MatchStaticType(inputs[0], kLiteRtElementTypeBool)) {
    return kLiteRtStatusPatternNoMatch;
  }
  const bool bool_out = MatchStaticType(out, kLiteRtElementTypeBool);
  if (!bool_out && !MatchStaticType(out, kLiteRtElementTypeFloat32)) {
    return kLiteRtStatusPatternNoMatch;
  }
  const LiteRtElementType branch_type =
      bool_out ? kLiteRtElementTypeBool : kLiteRtElementTypeFloat32;
  const Tensor& cond = inputs[0];
  const Tensor& t_val = inputs[1];
  const Tensor& f_val = inputs[2];
  if (!MatchStaticType(t_val, branch_type) ||
      !MatchStaticType(f_val, branch_type)) {
    return kLiteRtStatusPatternNoMatch;
  }
  // Never lower Select/SelectV2 when a Float32 branch (or an upstream chain)
  // may propagate a NaN or +/-Inf constant (`0.0f * NaN/Inf` produces NaN).
  if (!bool_out && (MayPropagateNonFiniteFloatConst(t_val) ||
                    MayPropagateNonFiniteFloatConst(f_val))) {
    return kLiteRtStatusPatternNoMatch;
  }

  const std::optional<std::vector<int32_t>> cond_dims_opt =
      GetStaticPositiveDims(cond);
  const std::optional<std::vector<int32_t>> t_dims_opt =
      GetStaticPositiveDims(t_val);
  const std::optional<std::vector<int32_t>> f_dims_opt =
      GetStaticPositiveDims(f_val);
  const std::optional<std::vector<int32_t>> out_dims_opt =
      GetStaticPositiveDims(out);
  if (!cond_dims_opt || !t_dims_opt || !f_dims_opt || !out_dims_opt) {
    return kLiteRtStatusPatternNoMatch;
  }
  const std::vector<int32_t>& raw_cond_dims = *cond_dims_opt;
  const std::vector<int32_t>& t_dims = *t_dims_opt;
  const std::vector<int32_t>& f_dims = *f_dims_opt;
  const std::vector<int32_t>& out_dims = *out_dims_opt;

  // TFLite Select (unlike SelectV2) requires exact x/y output shapes and allows
  // a 1D [B] condition selecting leading batch slices.
  const bool is_select_v1 = op.Code() == kLiteRtOpCodeTflSelect;
  const bool needs_select_1d_reshape =
      is_select_v1 && raw_cond_dims.size() == 1 && out_dims.size() > 1 &&
      raw_cond_dims[0] == out_dims[0];
  if (is_select_v1 && (t_dims != out_dims || f_dims != out_dims ||
                       (raw_cond_dims != out_dims && !raw_cond_dims.empty() &&
                        !needs_select_1d_reshape))) {
    return kLiteRtStatusPatternNoMatch;
  }
  if (!CanResolveBoolAsZeroOneFloat(cond, raw_cond_dims) ||
      (bool_out && (!CanResolveBoolAsZeroOneFloat(t_val, t_dims) ||
                    !CanResolveBoolAsZeroOneFloat(f_val, f_dims)))) {
    return kLiteRtStatusPatternNoMatch;
  }

  std::vector<int32_t> cond_dims = raw_cond_dims;
  if (needs_select_1d_reshape) {
    cond_dims.assign(out_dims.size(), 1);
    cond_dims[0] = raw_cond_dims[0];
  }
  const std::optional<std::vector<int32_t>> true_dims =
      BroadcastShapes(cond_dims, t_dims);
  const std::optional<std::vector<int32_t>> false_dims =
      BroadcastShapes(cond_dims, f_dims);
  if (!true_dims || !false_dims) {
    return kLiteRtStatusPatternNoMatch;
  }
  const std::optional<std::vector<int32_t>> combined_dims =
      BroadcastShapes(*true_dims, *false_dims);
  if (!combined_dims || *combined_dims != out_dims) {
    return kLiteRtStatusPatternNoMatch;
  }

  Tensor orig_cond_f;
  if (LiteRtStatus s = AssignOrStatus(
          ResolveBoolAsZeroOneFloat(builder, cond, raw_cond_dims),
          &orig_cond_f);
      s != kLiteRtStatusOk) {
    return s;
  }
  Tensor cond_f = orig_cond_f;
  if (needs_select_1d_reshape) {
    if (LiteRtStatus s = AssignOrStatus(
            BuildReshape(builder, cond_f, cond_f, cond_dims, ""), &cond_f);
        s != kLiteRtStatusOk) {
      return s;
    }
  }

  Tensor t_f = t_val;
  Tensor f_f = f_val;
  if (bool_out) {
    auto resolve_branch =
        [&](const Tensor& branch,
            absl::Span<const int32_t> dims) -> Expected<Tensor> {
      if (branch == cond) {
        return orig_cond_f;
      }
      return ResolveBoolAsZeroOneFloat(builder, branch, dims);
    };
    if (LiteRtStatus s = AssignOrStatus(resolve_branch(t_val, t_dims), &t_f);
        s != kLiteRtStatusOk) {
      return s;
    }
    if (f_val == t_val) {
      f_f = t_f;
    } else if (LiteRtStatus s =
                   AssignOrStatus(resolve_branch(f_val, f_dims), &f_f);
               s != kLiteRtStatusOk) {
      return s;
    }
  }

  Tensor f_out = out;
  if (bool_out) {
    if (LiteRtStatus s = AssignOrStatus(
            BuildTypedTensor(builder, ElementType::Float32, out_dims), &f_out);
        s != kLiteRtStatusOk) {
      return s;
    }
  }

  if (MatchFloatConstPred(f_f, &IsPositiveZeroFloat) &&
      *true_dims == out_dims) {
    if (LiteRtStatus s = UnwrapStatus(BuildBinaryOpInto(
            builder, kLiteRtOpCodeTflMul, t_f, cond_f, f_out));
        s != kLiteRtStatusOk) {
      return s;
    }
  } else if (MatchFloatConstPred(t_f, &IsPositiveZeroFloat) &&
             *false_dims == out_dims) {
    Tensor not_cond;
    if (LiteRtStatus s = AssignOrStatus(
            BuildFloatInvert(builder, cond_f, cond_dims), &not_cond);
        s != kLiteRtStatusOk) {
      return s;
    }
    if (LiteRtStatus s = UnwrapStatus(BuildBinaryOpInto(
            builder, kLiteRtOpCodeTflMul, f_f, not_cond, f_out));
        s != kLiteRtStatusOk) {
      return s;
    }
  } else {
    Tensor not_cond;
    if (LiteRtStatus s = AssignOrStatus(
            BuildFloatInvert(builder, cond_f, cond_dims), &not_cond);
        s != kLiteRtStatusOk) {
      return s;
    }
    Tensor true_part;
    if (LiteRtStatus s =
            AssignOrStatus(BuildBinaryOp(builder, kLiteRtOpCodeTflMul, cond_f,
                                         t_f, ElementType::Float32, *true_dims),
                           &true_part);
        s != kLiteRtStatusOk) {
      return s;
    }
    Tensor false_part;
    if (LiteRtStatus s = AssignOrStatus(
            BuildBinaryOp(builder, kLiteRtOpCodeTflMul, f_f, not_cond,
                          ElementType::Float32, *false_dims),
            &false_part);
        s != kLiteRtStatusOk) {
      return s;
    }
    if (LiteRtStatus s = UnwrapStatus(BuildBinaryOpInto(
            builder, kLiteRtOpCodeTflAdd, false_part, true_part, f_out));
        s != kLiteRtStatusOk) {
      return s;
    }
  }

  if (bool_out) {
    return BuildBoolCastBridge(builder, op, f_out, out);
  }
  return UnwrapStatus(builder.EraseOp(op));
}

LiteRtStatus LowerZeroOneEqualityComparison(const Builder& builder,
                                            const Op& current_op,
                                            bool is_equal) {
  if (!is_equal && IsNativeSameShapeOutputNotEqual(current_op)) {
    return kLiteRtStatusPatternNoMatch;
  }
  Tensor out;
  const std::vector<Tensor> inputs = current_op.Inputs();
  if (inputs.size() != 2 ||
      !MatchSingleStaticOutput(current_op, &out, kLiteRtElementTypeBool) ||
      !Match(inputs[0], m_HasStaticPositiveShape()) ||
      !Match(inputs[1], m_HasStaticPositiveShape()) ||
      inputs[0].ElementType() != inputs[1].ElementType()) {
    return kLiteRtStatusPatternNoMatch;
  }
  const Tensor& lhs = inputs[0];
  const Tensor& rhs = inputs[1];
  const std::optional<std::vector<int32_t>> lhs_dims =
      GetStaticPositiveDims(lhs);
  const std::optional<std::vector<int32_t>> rhs_dims =
      GetStaticPositiveDims(rhs);
  const std::optional<std::vector<int32_t>> out_dims =
      GetStaticPositiveDims(out);
  if (!lhs_dims || !rhs_dims || !out_dims) {
    return kLiteRtStatusPatternNoMatch;
  }
  const std::optional<std::vector<int32_t>> bcast =
      BroadcastShapes(*lhs_dims, *rhs_dims);
  if (!bcast || *bcast != *out_dims) {
    return kLiteRtStatusPatternNoMatch;
  }

  const bool lhs_bool = MatchStaticType(lhs, kLiteRtElementTypeBool);
  const bool rhs_bool = MatchStaticType(rhs, kLiteRtElementTypeBool);
  const bool lhs_zero_one_f32 = !lhs_bool && IsKnownZeroOneFloat(lhs);
  const bool rhs_zero_one_f32 = !rhs_bool && IsKnownZeroOneFloat(rhs);

  auto try_const_fast_path =
      [&](const Tensor& x, bool x_zero_one, absl::Span<const int32_t> x_dims,
          const Tensor& other) -> std::optional<LiteRtStatus> {
    if (!x_zero_one || x_dims != absl::MakeConstSpan(*out_dims)) {
      return std::nullopt;
    }
    const bool other_zero = MatchFloatConstPred(other, &IsAnyZeroFloat);
    const bool other_one =
        !other_zero && MatchFloatConstPred(other, &IsOneFloat);
    if (!other_zero && !other_one) {
      return std::nullopt;
    }
    const bool invert = other_zero ? is_equal : !is_equal;
    if (!invert) {
      return BuildBoolCastBridge(builder, current_op, x, out);
    }
    Expected<Tensor> inv = BuildFloatInvert(builder, x, *out_dims);
    if (!inv) {
      return inv.Error().Status();
    }
    return BuildBoolCastBridge(builder, current_op, *inv, out);
  };

  if (std::optional<LiteRtStatus> res =
          try_const_fast_path(lhs, lhs_zero_one_f32, *lhs_dims, rhs)) {
    return *res;
  }
  if (std::optional<LiteRtStatus> res =
          try_const_fast_path(rhs, rhs_zero_one_f32, *rhs_dims, lhs)) {
    return *res;
  }

  const bool lhs_exact_zero_const = !lhs_bool && !lhs_zero_one_f32 &&
                                    MatchFloatConstPred(lhs, &IsAnyZeroFloat);
  const bool rhs_exact_zero_const = !rhs_bool && !rhs_zero_one_f32 &&
                                    MatchFloatConstPred(rhs, &IsAnyZeroFloat);
  if ((!lhs_bool && !lhs_zero_one_f32 && !lhs_exact_zero_const) ||
      (!rhs_bool && !rhs_zero_one_f32 && !rhs_exact_zero_const) ||
      (lhs_bool && !CanResolveBoolAsZeroOneFloat(lhs, *lhs_dims)) ||
      (rhs_bool && !CanResolveBoolAsZeroOneFloat(rhs, *rhs_dims)) ||
      (lhs_exact_zero_const && !CanMaterializeZeroFloatConst(*lhs_dims)) ||
      (rhs_exact_zero_const && !CanMaterializeZeroFloatConst(*rhs_dims))) {
    return kLiteRtStatusPatternNoMatch;
  }

  auto resolve_exact_operand =
      [&](const Tensor& operand, absl::Span<const int32_t> dims, bool is_bool,
          bool is_exact_zero_const) -> Expected<Tensor> {
    if (is_bool) {
      return ResolveBoolAsZeroOneFloat(builder, operand, dims);
    }
    if (is_exact_zero_const) {
      return BuildZeroFloatConst(builder, dims);
    }
    return operand;
  };

  Tensor lhs_f;
  if (LiteRtStatus s = AssignOrStatus(
          resolve_exact_operand(lhs, *lhs_dims, lhs_bool, lhs_exact_zero_const),
          &lhs_f);
      s != kLiteRtStatusOk) {
    return s;
  }
  Tensor rhs_f = lhs_f;
  if (rhs != lhs) {
    if (LiteRtStatus s =
            AssignOrStatus(resolve_exact_operand(rhs, *rhs_dims, rhs_bool,
                                                 rhs_exact_zero_const),
                           &rhs_f);
        s != kLiteRtStatusOk) {
      return s;
    }
  }

  Tensor diff;
  if (LiteRtStatus s =
          AssignOrStatus(BuildBinaryOp(builder, kLiteRtOpCodeTflSub, lhs_f,
                                       rhs_f, ElementType::Float32, *out_dims),
                         &diff);
      s != kLiteRtStatusOk) {
    return s;
  }
  Tensor xor_f;
  if (LiteRtStatus s = AssignOrStatus(
          BuildTypedTensor(builder, ElementType::Float32, *out_dims), &xor_f);
      s != kLiteRtStatusOk) {
    return s;
  }
  if (LiteRtStatus s =
          UnwrapStatus(builder.BuildOp(kLiteRtOpCodeTflAbs, {diff}, {xor_f}));
      s != kLiteRtStatusOk) {
    return s;
  }
  Tensor result_f = xor_f;
  if (is_equal) {
    if (LiteRtStatus s = AssignOrStatus(
            BuildFloatInvert(builder, xor_f, *out_dims), &result_f);
        s != kLiteRtStatusOk) {
      return s;
    }
  }
  return BuildBoolCastBridge(builder, current_op, result_f, out);
}

}  // namespace

extern "C" {

LiteRtStatus BoolMaskBypassCastTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op) {
  if (context == nullptr || builder_ptr == nullptr || op == nullptr) {
    return kLiteRtStatusPatternNoMatch;
  }
  Builder builder(context, builder_ptr);
  Op current_op(context, op);
  const std::vector<Tensor> inputs = current_op.Inputs();
  const std::vector<Tensor> outputs = current_op.Outputs();
  if (outputs.empty() || !IsSupportedBypassConsumer(current_op)) {
    return kLiteRtStatusPatternNoMatch;
  }
  for (const Tensor& in : inputs) {
    if (in.Get() == nullptr) {
      return kLiteRtStatusPatternNoMatch;
    }
  }
  for (const Tensor& out : outputs) {
    if (out.Get() == nullptr) {
      return kLiteRtStatusPatternNoMatch;
    }
  }

  std::vector<Tensor> new_inputs;
  new_inputs.reserve(inputs.size());
  bool replaced = false;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const Tensor& in = inputs[i];
    bool reused = false;
    for (size_t j = 0; j < i; ++j) {
      if (inputs[j] == in) {
        new_inputs.push_back(new_inputs[j]);
        if (new_inputs[j] != in) {
          replaced = true;
        }
        reused = true;
        break;
      }
    }
    if (reused) {
      continue;
    }
    if (std::optional<Tensor> bypassed = TryBypassDoubleBoolCast(builder, in);
        bypassed.has_value()) {
      new_inputs.push_back(*bypassed);
      replaced = true;
      continue;
    }
    new_inputs.push_back(in);
  }
  if (!replaced) {
    return kLiteRtStatusPatternNoMatch;
  }

  const LiteRtOpCode op_code = current_op.Code();
  if (LiteRtStatus s =
          BuildBypassConsumerOp(builder, current_op, new_inputs, outputs);
      s != kLiteRtStatusOk) {
    return s;
  }
  if (LiteRtStatus s = UnwrapStatus(builder.EraseOp(current_op));
      s != kLiteRtStatusOk) {
    return s;
  }
  LITERT_LOG(LITERT_INFO,
             "BoolMaskBypassCast: bypassed Cast(Bool<->Float32) into op %d",
             op_code);
  return kLiteRtStatusOk;
}

LiteRtStatus BoolMaskCleanupCastTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op) {
  if (context == nullptr || builder_ptr == nullptr || op == nullptr) {
    return kLiteRtStatusPatternNoMatch;
  }
  Builder builder(context, builder_ptr);
  Op current_op(context, op);
  Tensor in;
  Tensor out;
  if (!Match(current_op, m_Op<kLiteRtOpCodeTflCast>(m_Capture(&in))) ||
      !Match(current_op, m_SingleOutput(m_Capture(&out))) ||
      !Match(out, m_AllOf(m_HasNoUses(), m_Not(m_IsSubgraphOutput()))) ||
      !out.GetSubgraph().HasValue() ||
      (!Match(in, m_ElementType(kLiteRtElementTypeBool)) &&
       !Match(out, m_ElementType(kLiteRtElementTypeBool)))) {
    return kLiteRtStatusPatternNoMatch;
  }
  if (LiteRtStatus s = UnwrapStatus(builder.EraseOp(current_op));
      s != kLiteRtStatusOk) {
    return s;
  }
  LITERT_LOG(LITERT_INFO, "BoolMaskCleanupCast: erased unused Bool Cast op");
  return kLiteRtStatusOk;
}

LiteRtStatus BoolMaskNotEqualTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op) {
  if (context == nullptr || builder_ptr == nullptr || op == nullptr) {
    return kLiteRtStatusPatternNoMatch;
  }
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
  if (context == nullptr || builder_ptr == nullptr || op == nullptr) {
    return kLiteRtStatusPatternNoMatch;
  }
  Builder builder(context, builder_ptr);
  Op out_cast(context, op);
  Tensor bool_in;
  Tensor tile_in;
  Tensor reps;
  Tensor tile_out;
  Tensor bool_out;
  Op in_cast;
  Op tile_op;
  if (!Match(out_cast, m_Op<kLiteRtOpCodeTflCast>(m_Capture(&tile_out))) ||
      !MatchStaticType(tile_out, kLiteRtElementTypeInt32) ||
      !MatchSingleStaticOutput(out_cast, &bool_out, kLiteRtElementTypeBool) ||
      !Match(bool_out, m_SameShapeAs(&tile_out)) ||
      !Match(tile_out, m_Capture(&tile_op)) ||
      !Match(tile_op, m_Op<kLiteRtOpCodeTflTile>(m_Capture(&tile_in),
                                                 m_Capture(&reps))) ||
      !MatchStaticType(tile_in, kLiteRtElementTypeInt32) ||
      !MatchCastInput(tile_in, &bool_in, &in_cast) ||
      (!MatchStaticType(bool_in, kLiteRtElementTypeBool) &&
       !MatchStaticType(bool_in, kLiteRtElementTypeFloat32)) ||
      !Match(bool_in, m_SameShapeAs(&tile_in))) {
    return kLiteRtStatusPatternNoMatch;
  }
  const std::optional<std::vector<int32_t>> in_dims =
      GetStaticPositiveDims(bool_in);
  const std::optional<std::vector<int32_t>> out_dims =
      GetStaticPositiveDims(bool_out);
  if (!in_dims || !out_dims ||
      !CanResolveBoolAsZeroOneFloat(bool_in, *in_dims)) {
    return kLiteRtStatusPatternNoMatch;
  }
  Expected<Tensor> reps_i32 =
      ValidateAndConvertTileReps(builder, reps, *in_dims, *out_dims);
  if (!reps_i32) {
    return kLiteRtStatusPatternNoMatch;
  }

  const bool erase_tile = tile_out.GetSubgraph().HasValue() &&
                          !tile_out.IsSubgraphOutput() &&
                          tile_out.Uses().size() == 1;
  const bool erase_in_cast = erase_tile && tile_in.GetSubgraph().HasValue() &&
                             !tile_in.IsSubgraphOutput() &&
                             tile_in.Uses().size() == 1;

  Tensor f_in;
  if (LiteRtStatus s = AssignOrStatus(
          ResolveBoolAsZeroOneFloat(builder, bool_in, *in_dims), &f_in);
      s != kLiteRtStatusOk) {
    return s;
  }
  Tensor f_out;
  if (LiteRtStatus s = AssignOrStatus(
          BuildTypedTensor(builder, ElementType::Float32, *out_dims), &f_out);
      s != kLiteRtStatusOk) {
    return s;
  }
  if (LiteRtStatus s = UnwrapStatus(
          builder.BuildOp(kLiteRtOpCodeTflTile, {f_in, *reps_i32}, {f_out}));
      s != kLiteRtStatusOk) {
    return s;
  }
  if (LiteRtStatus s = UnwrapStatus(BuildCastInto(builder, f_out, bool_out));
      s != kLiteRtStatusOk) {
    return s;
  }

  if (LiteRtStatus s = UnwrapStatus(builder.EraseOp(out_cast));
      s != kLiteRtStatusOk) {
    return s;
  }
  if (erase_tile) {
    if (LiteRtStatus s = UnwrapStatus(builder.EraseOp(tile_op));
        s != kLiteRtStatusOk) {
      return s;
    }
  }
  if (erase_in_cast) {
    if (LiteRtStatus s = UnwrapStatus(builder.EraseOp(in_cast));
        s != kLiteRtStatusOk) {
      return s;
    }
  }
  LITERT_LOG(
      LITERT_INFO,
      "BoolMaskTileWrap: lowered Cast->Int32 Tile->Cast to Float32 Tile");
  return kLiteRtStatusOk;
}

LiteRtStatus BoolMaskOpToFloatTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op) {
  if (context == nullptr || builder_ptr == nullptr || op == nullptr) {
    return kLiteRtStatusPatternNoMatch;
  }
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
      return LowerBoolReshape(builder, current_op);
    case kLiteRtOpCodeTflSlice:
      return LowerBoolSlice(builder, current_op);
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
  if (context == nullptr || builder_ptr == nullptr || op == nullptr) {
    return kLiteRtStatusPatternNoMatch;
  }
  Builder builder(context, builder_ptr);
  Op current_op(context, op);
  Tensor in;
  Tensor out;
  if (!Match(current_op, m_Op<kLiteRtOpCodeTflCast>(m_Capture(&in))) ||
      !MatchStaticType(in, kLiteRtElementTypeFloat32, 1) ||
      !MatchSingleStaticOutput(current_op, &out, kLiteRtElementTypeBool) ||
      !Match(out, m_AllOf(m_SameShapeAs(&in), m_HasNoUses(),
                          m_IsSubgraphOutput()))) {
    return kLiteRtStatusPatternNoMatch;
  }
  const std::optional<std::vector<int32_t>> in_dims = GetStaticPositiveDims(in);
  if (!in_dims) {
    return kLiteRtStatusPatternNoMatch;
  }
  Tensor zeros;
  if (LiteRtStatus s =
          AssignOrStatus(BuildZeroFloatConst(builder, *in_dims), &zeros);
      s != kLiteRtStatusOk) {
    return s;
  }
  if (LiteRtStatus s = UnwrapStatus(
          BuildBinaryOpInto(builder, kLiteRtOpCodeTflNotEqual, in, zeros, out));
      s != kLiteRtStatusOk) {
    return s;
  }
  if (LiteRtStatus s = UnwrapStatus(builder.EraseOp(current_op));
      s != kLiteRtStatusOk) {
    return s;
  }
  LITERT_LOG(LITERT_INFO,
             "BoolMaskOutputNotEqual: replaced output Cast with NotEqual");
  return kLiteRtStatusOk;
}

}  // extern "C"
