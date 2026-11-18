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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_SIMPLE_BINARY_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_SIMPLE_BINARY_H_

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/reference_numeric_utils.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/tensor_layout_util.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::internal {

inline LiteRtStatus InferElementwiseBinary(absl::Span<Dims> input_shapes,
                                           std::vector<Dims>& output_shapes) {
  if (input_shapes.size() != 2 || output_shapes.size() != 1) {
    LITERT_LOG(LITERT_ERROR,
               "Invalid number of input/output shapes for binary op.");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  const auto& s1 = input_shapes[0];
  const auto& s2 = input_shapes[1];

  size_t rank1 = s1.size();
  size_t rank2 = s2.size();
  size_t max_rank = std::max(rank1, rank2);
  Dims out_shape(max_rank);

  for (int i = 0; i < max_rank; ++i) {
    // Reverse index
    int idx1 = rank1 - 1 - i;
    int idx2 = rank2 - 1 - i;
    int32_t d1 = (idx1 >= 0) ? s1[idx1] : 1;
    int32_t d2 = (idx2 >= 0) ? s2[idx2] : 1;

    if (d1 == -1 || d2 == -1) {
      out_shape[max_rank - 1 - i] = -1;
    } else if (d1 == d2) {
      out_shape[max_rank - 1 - i] = d1;
    } else if (d1 == 1) {
      out_shape[max_rank - 1 - i] = d2;
    } else if (d2 == 1) {
      out_shape[max_rank - 1 - i] = d1;
    } else {
      LITERT_LOG(LITERT_ERROR, "Incompatible dims for binary op.");
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

#define DEFINE_SIMPLE_BINARY_INFER(name)                              \
  inline LiteRtStatus Infer##name(const LiteRtOpT& op,                \
                                  absl::Span<Dims> input_shapes,      \
                                  std::vector<Dims>& output_shapes) { \
    return InferElementwiseBinary(input_shapes, output_shapes);       \
  }

DEFINE_SIMPLE_BINARY_INFER(Add)
DEFINE_SIMPLE_BINARY_INFER(Atan2)
DEFINE_SIMPLE_BINARY_INFER(Div)
DEFINE_SIMPLE_BINARY_INFER(Equal)
DEFINE_SIMPLE_BINARY_INFER(FloorDiv)
DEFINE_SIMPLE_BINARY_INFER(FloorMod)
DEFINE_SIMPLE_BINARY_INFER(Greater)
DEFINE_SIMPLE_BINARY_INFER(GreaterEqual)
DEFINE_SIMPLE_BINARY_INFER(Less)
DEFINE_SIMPLE_BINARY_INFER(LessEqual)
DEFINE_SIMPLE_BINARY_INFER(LogicalAnd)
DEFINE_SIMPLE_BINARY_INFER(LogicalOr)
DEFINE_SIMPLE_BINARY_INFER(Maximum)
DEFINE_SIMPLE_BINARY_INFER(Minimum)
DEFINE_SIMPLE_BINARY_INFER(Mul)
DEFINE_SIMPLE_BINARY_INFER(NotEqual)
DEFINE_SIMPLE_BINARY_INFER(Pow)
DEFINE_SIMPLE_BINARY_INFER(Prelu)
DEFINE_SIMPLE_BINARY_INFER(SquaredDifference)
DEFINE_SIMPLE_BINARY_INFER(Sub)

#undef DEFINE_SIMPLE_BINARY_INFER

// Evaluates `op(a, b)` elementwise over two concrete tensors under
// trailing-dimension NumPy broadcasting (`output.dims == broadcast(a.dims,
// b.dims)`).
//
// Domain and preconditions:
//   - Shapes must be concrete and nonnegative, and `output.dims` must equal the
//     exact inferred broadcast shape of `a.dims` and `b.dims`.
//   - Span capacities must cover the checked element counts for all tensors.
//   - `output.data` must not alias `a.data` or `b.data` when broadcasting.
//   - Valid empty outputs return `kLiteRtStatusOk` without invoking `op` or
//     dereferencing any data pointer.
template <typename In1T, typename In2T, typename OutT, typename BinaryOp>
inline LiteRtStatus ReferenceBinaryGeneric(ConstTensorView<In1T> a,
                                           ConstTensorView<In2T> b,
                                           TensorView<OutT> output,
                                           BinaryOp op) {
  auto plan = PrepareBroadcastPlan<2>(
      {a.dims, b.dims}, {sizeof(In1T), sizeof(In2T)}, output.dims, sizeof(OutT),
      BroadcastOutputMode::kInferredExact);
  if (!plan) {
    return plan.Error().Status();
  }
  if (!ValidateBufferCapacity(a.data, plan->input_num_elements[0]) ||
      !ValidateBufferCapacity(b.data, plan->input_num_elements[1]) ||
      !ValidateBufferCapacity(output.data, plan->output_num_elements)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  for (const auto& span : plan->Spans()) {
    const size_t a_base = span.input_offsets[0];
    const size_t b_base = span.input_offsets[1];
    const size_t a_step = span.input_steps[0];
    const size_t b_step = span.input_steps[1];
    OutT* out_ptr = output.data.data() + span.output_offset;
    for (size_t i = 0; i < span.count; ++i) {
      out_ptr[i] = op(a.data[a_base + i * a_step], b.data[b_base + i * b_step]);
    }
  }
  return kLiteRtStatusOk;
}

namespace detail {

template <typename In1T, typename In2T, typename OutT, typename Fn>
inline LiteRtStatus InvokeBinaryFromRawPointers(
    const In1T* a_data, const int32_t* a_dims, int a_rank, const In2T* b_data,
    const int32_t* b_dims, int b_rank, OutT* output_data,
    const int32_t* output_dims, int rank, Fn&& fn) {
  auto a_span = MakeDimsSpan(a_dims, a_rank);
  auto b_span = MakeDimsSpan(b_dims, b_rank);
  auto out_span = MakeDimsSpan(output_dims, rank);
  if (!a_span || !b_span || !out_span) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto plan = PrepareBroadcastPlan<2>(
      {*a_span, *b_span}, {sizeof(In1T), sizeof(In2T)}, *out_span, sizeof(OutT),
      BroadcastOutputMode::kInferredExact);
  if (!plan) {
    return plan.Error().Status();
  }
  if (plan->IsEmpty()) {
    return kLiteRtStatusOk;
  }
  if (a_data == nullptr || b_data == nullptr || output_data == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return fn(
      ConstTensorView<In1T>{
          absl::MakeConstSpan(a_data, plan->input_num_elements[0]), *a_span},
      ConstTensorView<In2T>{
          absl::MakeConstSpan(b_data, plan->input_num_elements[1]), *b_span},
      TensorView<OutT>{absl::MakeSpan(output_data, plan->output_num_elements),
                       *out_span});
}

}  // namespace detail

template <typename InT, typename OutT, typename BinaryOp>
inline LiteRtStatus ReferenceBinaryGeneric(
    const InT* input1_data, const int32_t* input1_dims, int input1_rank,
    const InT* input2_data, const int32_t* input2_dims, int input2_rank,
    OutT* output_data, const int32_t* output_dims, int rank, BinaryOp op) {
  return detail::InvokeBinaryFromRawPointers<InT, InT, OutT>(
      input1_data, input1_dims, input1_rank, input2_data, input2_dims,
      input2_rank, output_data, output_dims, rank,
      [&](ConstTensorView<InT> a, ConstTensorView<InT> b,
          TensorView<OutT> out) {
        return ReferenceBinaryGeneric<InT, InT, OutT>(a, b, out, op);
      });
}

inline LiteRtStatus ReferenceAdd(
    const float* a_data, const int32_t* a_dims, int a_rank, const float* b_data,
    const int32_t* b_dims, int b_rank, float* output_data,
    const int32_t* output_dims, int rank,
    tflite::ActivationFunctionType faf = tflite::ActivationFunctionType_NONE) {
  if (LiteRtStatus status = ReferenceBinaryGeneric(
          a_data, a_dims, a_rank, b_data, b_dims, b_rank, output_data,
          output_dims, rank, std::plus<float>());
      status != kLiteRtStatusOk) {
    return status;
  }
  ApplyActivation(output_data, output_dims, rank, faf);
  return kLiteRtStatusOk;
}

inline LiteRtStatus ReferenceSub(
    const float* a_data, const int32_t* a_dims, int a_rank, const float* b_data,
    const int32_t* b_dims, int b_rank, float* output_data,
    const int32_t* output_dims, int rank,
    tflite::ActivationFunctionType faf = tflite::ActivationFunctionType_NONE) {
  if (LiteRtStatus status = ReferenceBinaryGeneric(
          a_data, a_dims, a_rank, b_data, b_dims, b_rank, output_data,
          output_dims, rank, std::minus<float>());
      status != kLiteRtStatusOk) {
    return status;
  }
  ApplyActivation(output_data, output_dims, rank, faf);
  return kLiteRtStatusOk;
}

inline LiteRtStatus ReferenceMul(
    const float* a_data, const int32_t* a_dims, int a_rank, const float* b_data,
    const int32_t* b_dims, int b_rank, float* output_data,
    const int32_t* output_dims, int rank,
    tflite::ActivationFunctionType faf = tflite::ActivationFunctionType_NONE) {
  if (LiteRtStatus status = ReferenceBinaryGeneric(
          a_data, a_dims, a_rank, b_data, b_dims, b_rank, output_data,
          output_dims, rank, std::multiplies<float>());
      status != kLiteRtStatusOk) {
    return status;
  }
  ApplyActivation(output_data, output_dims, rank, faf);
  return kLiteRtStatusOk;
}

inline LiteRtStatus ReferenceDiv(
    const float* a_data, const int32_t* a_dims, int a_rank, const float* b_data,
    const int32_t* b_dims, int b_rank, float* output_data,
    const int32_t* output_dims, int rank,
    tflite::ActivationFunctionType faf = tflite::ActivationFunctionType_NONE) {
  if (LiteRtStatus status = ReferenceBinaryGeneric(
          a_data, a_dims, a_rank, b_data, b_dims, b_rank, output_data,
          output_dims, rank, std::divides<float>());
      status != kLiteRtStatusOk) {
    return status;
  }
  ApplyActivation(output_data, output_dims, rank, faf);
  return kLiteRtStatusOk;
}

inline LiteRtStatus ReferencePow(const float* a_data, const int32_t* a_dims,
                                 int a_rank, const float* b_data,
                                 const int32_t* b_dims, int b_rank,
                                 float* output_data, const int32_t* output_dims,
                                 int rank) {
  return ReferenceBinaryGeneric(
      a_data, a_dims, a_rank, b_data, b_dims, b_rank, output_data, output_dims,
      rank, [](float a, float b) { return std::pow(a, b); });
}

template <typename T, typename OutT = bool>
inline LiteRtStatus ReferenceEqual(const T* a_data, const int32_t* a_dims,
                                   int a_rank, const T* b_data,
                                   const int32_t* b_dims, int b_rank,
                                   OutT* output_data,
                                   const int32_t* output_dims, int rank) {
  return ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                                output_data, output_dims, rank,
                                std::equal_to<T>());
}

template <typename T, typename OutT = bool>
inline LiteRtStatus ReferenceNotEqual(const T* a_data, const int32_t* a_dims,
                                      int a_rank, const T* b_data,
                                      const int32_t* b_dims, int b_rank,
                                      OutT* output_data,
                                      const int32_t* output_dims, int rank) {
  return ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                                output_data, output_dims, rank,
                                std::not_equal_to<T>());
}

template <typename T, typename OutT = bool>
inline LiteRtStatus ReferenceGreater(const T* a_data, const int32_t* a_dims,
                                     int a_rank, const T* b_data,
                                     const int32_t* b_dims, int b_rank,
                                     OutT* output_data,
                                     const int32_t* output_dims, int rank) {
  return ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                                output_data, output_dims, rank,
                                std::greater<T>());
}

template <typename T, typename OutT = bool>
inline LiteRtStatus ReferenceGreaterEqual(
    const T* a_data, const int32_t* a_dims, int a_rank, const T* b_data,
    const int32_t* b_dims, int b_rank, OutT* output_data,
    const int32_t* output_dims, int rank) {
  return ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                                output_data, output_dims, rank,
                                std::greater_equal<T>());
}

template <typename T, typename OutT = bool>
inline LiteRtStatus ReferenceLess(const T* a_data, const int32_t* a_dims,
                                  int a_rank, const T* b_data,
                                  const int32_t* b_dims, int b_rank,
                                  OutT* output_data, const int32_t* output_dims,
                                  int rank) {
  return ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                                output_data, output_dims, rank, std::less<T>());
}

template <typename T, typename OutT = bool>
inline LiteRtStatus ReferenceLessEqual(const T* a_data, const int32_t* a_dims,
                                       int a_rank, const T* b_data,
                                       const int32_t* b_dims, int b_rank,
                                       OutT* output_data,
                                       const int32_t* output_dims, int rank) {
  return ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                                output_data, output_dims, rank,
                                std::less_equal<T>());
}

// Computes elementwise floor division under trailing-dimension NumPy
// broadcasting (`output.dims == broadcast(a.dims, b.dims)`).
//
// Domain policy:
//   - For integer `T`, any zero element in `b` when the output is nonempty
//     returns `kLiteRtStatusErrorInvalidArgument` without modifying `output`
//     (matching TFLite's built-in `floor_div.cc`).
//   - For signed integer `T`, `min<T> / -1` saturates to `max<T>` to avoid
//     signed overflow UB.
//   - For floating-point `T`, evaluates `std::floor(a / b)` under IEEE-754.
template <typename T>
inline LiteRtStatus ReferenceFloorDiv(ConstTensorView<T> a,
                                      ConstTensorView<T> b,
                                      TensorView<T> output) {
  auto plan = PrepareBroadcastPlan<2>({a.dims, b.dims}, {sizeof(T), sizeof(T)},
                                      output.dims, sizeof(T),
                                      BroadcastOutputMode::kInferredExact);
  if (!plan) {
    return plan.Error().Status();
  }
  if (!ValidateBufferCapacity(a.data, plan->input_num_elements[0]) ||
      !ValidateBufferCapacity(b.data, plan->input_num_elements[1]) ||
      !ValidateBufferCapacity(output.data, plan->output_num_elements)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (plan->IsEmpty()) {
    return kLiteRtStatusOk;
  }
  if constexpr (std::is_integral_v<T>) {
    for (size_t i = 0; i < plan->input_num_elements[1]; ++i) {
      if (b.data[i] == T{0}) {
        return kLiteRtStatusErrorInvalidArgument;
      }
    }
  }
  for (const auto& span : plan->Spans()) {
    const size_t a_base = span.input_offsets[0];
    const size_t b_base = span.input_offsets[1];
    const size_t a_step = span.input_steps[0];
    const size_t b_step = span.input_steps[1];
    T* out_ptr = output.data.data() + span.output_offset;
    for (size_t i = 0; i < span.count; ++i) {
      const T av = a.data[a_base + i * a_step];
      const T bv = b.data[b_base + i * b_step];
      if constexpr (std::is_floating_point_v<T>) {
        out_ptr[i] = static_cast<T>(std::floor(av / bv));
      } else {
        if constexpr (std::is_signed_v<T>) {
          if (av == std::numeric_limits<T>::min() && bv == T{-1}) {
            out_ptr[i] = std::numeric_limits<T>::max();
            continue;
          }
        }
        T q = av / bv;
        T r = av % bv;
        if (r != 0 && ((av < 0) != (bv < 0))) {
          --q;
        }
        out_ptr[i] = q;
      }
    }
  }
  return kLiteRtStatusOk;
}

template <typename T>
inline LiteRtStatus ReferenceFloorDiv(const T* a_data, const int32_t* a_dims,
                                      int a_rank, const T* b_data,
                                      const int32_t* b_dims, int b_rank,
                                      T* output_data,
                                      const int32_t* output_dims, int rank) {
  return detail::InvokeBinaryFromRawPointers<T, T, T>(
      a_data, a_dims, a_rank, b_data, b_dims, b_rank, output_data, output_dims,
      rank, [](ConstTensorView<T> a, ConstTensorView<T> b, TensorView<T> out) {
        return ReferenceFloorDiv<T>(a, b, out);
      });
}

// Computes elementwise floor modulo under trailing-dimension NumPy
// broadcasting (`output.dims == broadcast(a.dims, b.dims)`).
//
// Domain policy:
//   - For integer `T`, any zero element in `b` when the output is nonempty
//     returns `kLiteRtStatusErrorInvalidArgument` without modifying `output`
//     (matching TFLite's built-in `floor_mod.cc`).
//   - For signed integer `T`, `min<T> % -1` evaluates to `0` without UB.
//   - For floating-point `T`, evaluates `a - std::floor(a / b) * b`.
template <typename T>
inline LiteRtStatus ReferenceFloorMod(ConstTensorView<T> a,
                                      ConstTensorView<T> b,
                                      TensorView<T> output) {
  auto plan = PrepareBroadcastPlan<2>({a.dims, b.dims}, {sizeof(T), sizeof(T)},
                                      output.dims, sizeof(T),
                                      BroadcastOutputMode::kInferredExact);
  if (!plan) {
    return plan.Error().Status();
  }
  if (!ValidateBufferCapacity(a.data, plan->input_num_elements[0]) ||
      !ValidateBufferCapacity(b.data, plan->input_num_elements[1]) ||
      !ValidateBufferCapacity(output.data, plan->output_num_elements)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (plan->IsEmpty()) {
    return kLiteRtStatusOk;
  }
  if constexpr (std::is_integral_v<T>) {
    for (size_t i = 0; i < plan->input_num_elements[1]; ++i) {
      if (b.data[i] == T{0}) {
        return kLiteRtStatusErrorInvalidArgument;
      }
    }
  }
  for (const auto& span : plan->Spans()) {
    const size_t a_base = span.input_offsets[0];
    const size_t b_base = span.input_offsets[1];
    const size_t a_step = span.input_steps[0];
    const size_t b_step = span.input_steps[1];
    T* out_ptr = output.data.data() + span.output_offset;
    for (size_t i = 0; i < span.count; ++i) {
      const T av = a.data[a_base + i * a_step];
      const T bv = b.data[b_base + i * b_step];
      if constexpr (std::is_floating_point_v<T>) {
        out_ptr[i] = static_cast<T>(av - std::floor(av / bv) * bv);
      } else {
        if constexpr (std::is_signed_v<T>) {
          if (av == std::numeric_limits<T>::min() && bv == T{-1}) {
            out_ptr[i] = T{0};
            continue;
          }
        }
        T r = av % bv;
        if (r != 0 && ((av < 0) != (bv < 0))) {
          r += bv;
        }
        out_ptr[i] = r;
      }
    }
  }
  return kLiteRtStatusOk;
}

template <typename T>
inline LiteRtStatus ReferenceFloorMod(const T* a_data, const int32_t* a_dims,
                                      int a_rank, const T* b_data,
                                      const int32_t* b_dims, int b_rank,
                                      T* output_data,
                                      const int32_t* output_dims, int rank) {
  return detail::InvokeBinaryFromRawPointers<T, T, T>(
      a_data, a_dims, a_rank, b_data, b_dims, b_rank, output_data, output_dims,
      rank, [](ConstTensorView<T> a, ConstTensorView<T> b, TensorView<T> out) {
        return ReferenceFloorMod<T>(a, b, out);
      });
}

inline LiteRtStatus ReferenceLogicalAnd(ConstTensorView<bool> a,
                                        ConstTensorView<bool> b,
                                        TensorView<bool> output) {
  return ReferenceBinaryGeneric<bool, bool, bool>(a, b, output,
                                                  std::logical_and<>());
}

inline LiteRtStatus ReferenceLogicalAnd(const bool* a_data,
                                        const int32_t* a_dims, int a_rank,
                                        const bool* b_data,
                                        const int32_t* b_dims, int b_rank,
                                        bool* output_data,
                                        const int32_t* output_dims, int rank) {
  return ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                                output_data, output_dims, rank,
                                std::logical_and<>());
}

inline LiteRtStatus ReferenceLogicalOr(ConstTensorView<bool> a,
                                       ConstTensorView<bool> b,
                                       TensorView<bool> output) {
  return ReferenceBinaryGeneric<bool, bool, bool>(a, b, output,
                                                  std::logical_or<>());
}

inline LiteRtStatus ReferenceLogicalOr(const bool* a_data,
                                       const int32_t* a_dims, int a_rank,
                                       const bool* b_data,
                                       const int32_t* b_dims, int b_rank,
                                       bool* output_data,
                                       const int32_t* output_dims, int rank) {
  return ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                                output_data, output_dims, rank,
                                std::logical_or<>());
}

template <typename T>
inline LiteRtStatus ReferenceBitwiseXor(ConstTensorView<T> a,
                                        ConstTensorView<T> b,
                                        TensorView<T> output) {
  return ReferenceBinaryGeneric<T, T, T>(a, b, output, std::bit_xor<T>());
}

template <typename T>
inline LiteRtStatus ReferenceBitwiseXor(const T* a_data, const int32_t* a_dims,
                                        int a_rank, const T* b_data,
                                        const int32_t* b_dims, int b_rank,
                                        T* output_data,
                                        const int32_t* output_dims, int rank) {
  return ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                                output_data, output_dims, rank,
                                std::bit_xor<T>());
}

// Matches TFLite's built-in RIGHT_SHIFT kernel (tflite/kernels/right_shift.cc):
// clamps the shift count to [0, sizeof(T) * CHAR_BIT - 1] before shifting.
template <typename T>
inline LiteRtStatus ReferenceRightShift(ConstTensorView<T> a,
                                        ConstTensorView<T> b,
                                        TensorView<T> output) {
  static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool>,
                "ReferenceRightShift requires a non-bool integer type.");
  return ReferenceBinaryGeneric<T, T, T>(a, b, output, [](T lhs, T rhs) -> T {
    constexpr T kMaxShift = static_cast<T>(sizeof(T) * CHAR_BIT - 1);
    T clamped = rhs;
    if constexpr (std::is_signed_v<T>) {
      if (clamped < 0) {
        clamped = 0;
      }
    }
    if (clamped > kMaxShift) {
      clamped = kMaxShift;
    }
    return static_cast<T>(lhs >> clamped);
  });
}

template <typename T>
inline LiteRtStatus ReferenceRightShift(const T* a_data, const int32_t* a_dims,
                                        int a_rank, const T* b_data,
                                        const int32_t* b_dims, int b_rank,
                                        T* output_data,
                                        const int32_t* output_dims, int rank) {
  return detail::InvokeBinaryFromRawPointers<T, T, T>(
      a_data, a_dims, a_rank, b_data, b_dims, b_rank, output_data, output_dims,
      rank, [](ConstTensorView<T> a, ConstTensorView<T> b, TensorView<T> out) {
        return ReferenceRightShift<T>(a, b, out);
      });
}

// Evaluates PReLU (`x >= 0 ? x : alpha * x`) with explicit dequantization,
// `float` arithmetic, and saturating requantization.
//
// Domain and preconditions:
//   - Matches TFLite PReLU (`tflite/kernels/activations.cc`): `output.dims`
//     must equal `input.dims`, and `alpha.dims` must broadcast into
//     `input.dims` under trailing-dimension NumPy broadcasting.
//   - `output.data` must not alias `alpha.data` when broadcasting.
//   - Valid empty outputs return `kLiteRtStatusOk` without dereferencing data.
template <typename T>
inline LiteRtStatus ReferencePrelu(ConstTensorView<T> input,
                                   ConstTensorView<T> alpha,
                                   TensorView<T> output,
                                   const AffineQuantSpec& input_quant = {},
                                   const AffineQuantSpec& alpha_quant = {},
                                   const AffineQuantSpec& output_quant = {}) {
  if (output.dims.size() != input.dims.size() ||
      !std::equal(input.dims.begin(), input.dims.end(), output.dims.begin())) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return ReferenceBinaryGeneric<T, T, T>(
      input, alpha, output,
      [input_quant, alpha_quant, output_quant](T in_val, T alpha_val) -> T {
        const float x = DequantizeValue(in_val, input_quant);
        const float a = DequantizeValue(alpha_val, alpha_quant);
        const float y = x >= 0.0f ? x : a * x;
        return QuantizeValue<T>(y, output_quant);
      });
}

// Elementwise sum of equal-shape tensors in `float` accumulation precision
// with per-tensor affine quantization support. Valid empty tensors
// (`num_elements == 0`) return `kLiteRtStatusOk` without dereferencing data.
template <typename T>
inline LiteRtStatus ReferenceAddN(absl::Span<const T* const> inputs,
                                  T* output_data, size_t num_elements,
                                  const AffineQuantSpec& input_quant = {},
                                  const AffineQuantSpec& output_quant = {}) {
  if (!CheckedByteCount(num_elements, sizeof(T))) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (num_elements == 0) {
    return kLiteRtStatusOk;
  }
  if (output_data == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  for (const T* in_ptr : inputs) {
    if (in_ptr == nullptr) {
      return kLiteRtStatusErrorInvalidArgument;
    }
  }
  for (size_t i = 0; i < num_elements; ++i) {
    float sum = 0.0f;
    for (const T* in_ptr : inputs) {
      sum += DequantizeValue(in_ptr[i], input_quant);
    }
    output_data[i] = QuantizeValue<T>(sum, output_quant);
  }
  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_SIMPLE_BINARY_H_
