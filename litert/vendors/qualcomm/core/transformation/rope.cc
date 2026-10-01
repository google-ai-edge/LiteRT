// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/rope.h"

#include <cstddef>
#include <utility>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/builders/rotary_embedding_op_builder.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {
namespace {

void CloneNamespace(const OpWrapper& source, OpWrapper& destination) {
  const absl::string_view source_name = source.GetName();
  const size_t slash = source_name.rfind('/');
  if (slash != absl::string_view::npos) {
    destination.AddPrefixToName(
        absl::StrCat(source_name.substr(0, slash), "/"));
  }
}

bool IsRope(const OpWrapper& first_slice, const OpWrapper& second_slice,
            const OpWrapper& first_mul, const OpWrapper& second_mul,
            const OpWrapper& subtract, const OpWrapper& third_mul,
            const OpWrapper& fourth_mul, const OpWrapper& add,
            const OpWrapper& concat) {
  const auto& token_embedding = first_slice.GetInputTensor(0);
  const auto& cos = first_mul.GetInputTensor(1);
  const auto& sin = second_mul.GetInputTensor(1);
  const auto& output = concat.GetOutputTensor(0);
  return first_slice.GetInputTensor(0) == second_slice.GetInputTensor(0) &&
         first_slice.GetOutputTensor(0) == first_mul.GetInputTensor(0) &&
         second_slice.GetOutputTensor(0) == second_mul.GetInputTensor(0) &&
         first_mul.GetOutputTensor(0) == subtract.GetInputTensor(0) &&
         second_mul.GetOutputTensor(0) == subtract.GetInputTensor(1) &&
         second_slice.GetOutputTensor(0) == third_mul.GetInputTensor(0) &&
         first_slice.GetOutputTensor(0) == fourth_mul.GetInputTensor(0) &&
         cos == third_mul.GetInputTensor(1) &&
         sin == fourth_mul.GetInputTensor(1) &&
         third_mul.GetOutputTensor(0) == add.GetInputTensor(0) &&
         fourth_mul.GetOutputTensor(0) == add.GetInputTensor(1) &&
         subtract.GetOutputTensor(0) == concat.GetInputTensor(0) &&
         add.GetOutputTensor(0) == concat.GetInputTensor(1) &&
         token_embedding.IsF32() && cos.IsF32() && sin.IsF32() &&
         output.IsF32() &&
         token_embedding.GetDimensions() == output.GetDimensions() &&
         IsElementWiseMultiply(first_mul) && IsElementWiseMultiply(second_mul) &&
         subtract.IsOpCode(QnnOpCode::kElementWiseSubtract) &&
         IsElementWiseMultiply(third_mul) &&
         IsElementWiseMultiply(fourth_mul) && IsElementWiseAdd(add);
}

}  // namespace

size_t TransformRope(std::function<bool(OpWrapper&)> validate_op_config,
                     std::vector<OpWrapper>& ops, size_t start_index,
                     TensorPool& tensor_pool, size_t pattern_size) {
  constexpr size_t kFirstSlice = 0;
  constexpr size_t kSecondSlice = 1;
  constexpr size_t kFirstMul = 2;
  constexpr size_t kSecondMul = 3;
  constexpr size_t kSubtract = 4;
  constexpr size_t kThirdMul = 5;
  constexpr size_t kFourthMul = 6;
  constexpr size_t kAdd = 7;
  constexpr size_t kConcat = 8;
  constexpr size_t kPatternSize = 9;

  if (pattern_size != kPatternSize) return 1;
  const auto& first_slice = ops[start_index + kFirstSlice];
  const auto& second_slice = ops[start_index + kSecondSlice];
  const auto& first_mul = ops[start_index + kFirstMul];
  const auto& second_mul = ops[start_index + kSecondMul];
  const auto& subtract = ops[start_index + kSubtract];
  const auto& third_mul = ops[start_index + kThirdMul];
  const auto& fourth_mul = ops[start_index + kFourthMul];
  const auto& add = ops[start_index + kAdd];
  const auto& concat = ops[start_index + kConcat];
  if (!IsRope(first_slice, second_slice, first_mul, second_mul, subtract,
              third_mul, fourth_mul, add, concat)) {
    return 1;
  }

  QNN_LOG_INFO("[G2G] RoPE pattern matched at op %zu.", start_index);
  auto rotary_embedding = CreateRotaryEmbeddingOp(
      first_slice.GetInputTensor(0), first_mul.GetInputTensor(1),
      second_mul.GetInputTensor(1), concat.GetOutputTensor(0),
      /*interleaved=*/true);
  if (!validate_op_config(rotary_embedding)) {
    QNN_LOG_WARNING("[G2G] RoPE: RotaryEmbedding validation failed.");
    return 1;
  }
  CloneNamespace(concat, rotary_embedding);
  ops.erase(ops.begin() + start_index,
            ops.begin() + start_index + kPatternSize);
  ops.insert(ops.begin() + start_index, std::move(rotary_embedding));
  QNN_LOG_INFO("[G2G] RoPE transformed at op %zu.", start_index);
  return 1;
}

}  // namespace qnn
