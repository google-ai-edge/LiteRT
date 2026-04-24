// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_CC_LITERT_MODEL_PREDICATES_H_
#define ODML_LITERT_LITERT_CC_LITERT_MODEL_PREDICATES_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_ranked_tensor_type.h"

/// @file
/// @brief Defines predicates for matching patterns in a LiteRT model graph.
///
/// These predicates are used to identify specific nodes (ops) and edges
/// (tensors) in the graph based on their properties, such as type, shape, and
/// connections.
/// @note All `std::optional` arguments in matcher functions are considered a
/// vacuous match if they are `std::nullopt`.

namespace litert {

/// @brief Holds information about a tensor's type and dimensions for matching.
struct TensorTypeInfo {
  std::optional<ElementType> element_type = std::nullopt;
  std::optional<absl::InlinedVector<int32_t, 4>> dims = std::nullopt;

  explicit TensorTypeInfo(ElementType element_type)
      : element_type(element_type) {}
  explicit TensorTypeInfo(absl::InlinedVector<int32_t, 4> dims) : dims(dims) {}
  TensorTypeInfo(ElementType element_type, absl::InlinedVector<int32_t, 4> dims)
      : element_type(element_type), dims(dims) {}
};

/// @brief Holds information about a tensor's usage for matching.
struct UseInfo {
  std::optional<LiteRtOpCode> op_code = std::nullopt;
  std::optional<LiteRtParamIndex> user_param_ind = std::nullopt;
};

/// @brief Checks if a tensor has the given type and shape information.
/// @param tensor_type The tensor type to inspect.
/// @param expected The expected tensor type information.
/// @return `true` if the tensor type matches the expected information.
bool MatchRankedTensorType(const RankedTensorType& tensor_type,
                           const TensorTypeInfo& expected);

/// @brief Checks if an op has a signature matching the given types.
/// @param op The op to inspect.
/// @param expected_inputs The expected input tensor types.
/// @param expected_outputs The expected output tensor types.
/// @return `true` if the op's signature matches the expected types.
bool MatchOpType(
    const Op& op,
    const std::vector<std::optional<TensorTypeInfo>>& expected_inputs,
    const std::vector<std::optional<TensorTypeInfo>>& expected_outputs);

/// @brief Checks if a tensor contains weights whose values match the expected
/// data.
/// @tparam T The data type of the weights.
/// @param tensor The tensor to inspect.
/// @param expected_data The expected weight data.
/// @return `true` if the tensor's weights match the expected data.
template <typename T>
bool MatchWeights(const Tensor& tensor, absl::Span<const T> expected_data) {
  auto weights = tensor.WeightsData<T>();
  return weights.HasValue() && *weights == expected_data;
}

/// @brief Checks if a tensor has a user with the given information.
/// @param tensor The tensor to inspect.
/// @param expected_use The expected usage information.
/// @return `true` if the tensor has a user matching the expected information.
bool MatchUse(const Tensor& tensor, const UseInfo& expected_use);

/// @brief Checks if a tensor has matching users.
/// @param tensor The tensor to inspect.
/// @param expected_uses The expected usage information for the tensor's users.
/// @param strict If `true`, the number of `expected_uses` must equal the
/// number of actual uses. Otherwise, it checks if each `expected_use` matches
/// at least one actual use.
/// @return `true` if the tensor's users match the expected information.
bool MatchUses(const Tensor& tensor, const std::vector<UseInfo>& expected_uses,
               bool strict = true);

namespace internal::model_predicates_detail {

template <typename T>
bool Any(absl::Span<const T> vals,
         const std::function<bool(const T&)>& unary_pred) {
  for (const auto& val : vals) {
    if (unary_pred(val)) {
      return true;
    }
  }
  return false;
}

inline bool UseSoftEqual(const Tensor::TensorUse& actual_use,
                         const UseInfo& expected_use) {
  if (expected_use.user_param_ind.has_value() &&
      actual_use.user_arg_ind != expected_use.user_param_ind.value()) {
    return false;
  }
  if (expected_use.op_code.has_value() &&
      actual_use.user.Code() != expected_use.op_code.value()) {
    return false;
  }
  return true;
}

}  // namespace internal::model_predicates_detail

inline bool MatchRankedTensorType(const RankedTensorType& tensor_type,
                                  const TensorTypeInfo& expected) {
  if (expected.element_type.has_value() &&
      (tensor_type.ElementType() != expected.element_type.value())) {
    return false;
  }

  if (expected.dims.has_value()) {
    auto actual_dims = tensor_type.Layout().Dimensions();
    auto expected_dims = absl::MakeConstSpan(expected.dims.value());
    return AllZip(actual_dims, expected_dims,
                  [](auto l, auto r) -> bool { return l == r; });
  }
  return true;
}

inline bool MatchOpType(
    const Op& op,
    const std::vector<std::optional<TensorTypeInfo>>& expected_inputs,
    const std::vector<std::optional<TensorTypeInfo>>& expected_outputs) {
  auto match = [](const Tensor& actual,
                  const std::optional<TensorTypeInfo>& expected) -> bool {
    if (!expected.has_value()) {
      return true;
    }
    auto actual_ranked_tensor_type = actual.RankedTensorType();
    if (!actual_ranked_tensor_type) {
      return false;
    }
    return MatchRankedTensorType(*actual_ranked_tensor_type, expected.value());
  };

  const bool inputs_match = AllZip(absl::MakeConstSpan(op.Inputs()),
                                   absl::MakeConstSpan(expected_inputs), match);
  const bool outputs_match =
      AllZip(absl::MakeConstSpan(op.Outputs()),
             absl::MakeConstSpan(expected_outputs), match);
  return inputs_match && outputs_match;
}

inline bool MatchUse(const Tensor& tensor, const UseInfo& expected_use) {
  auto soft_equal = [&expected_use = std::as_const(expected_use)](
                        const Tensor::TensorUse& actual_use) {
    return internal::model_predicates_detail::UseSoftEqual(actual_use,
                                                           expected_use);
  };
  return internal::model_predicates_detail::Any<Tensor::TensorUse>(
      tensor.Uses(), soft_equal);
}

inline bool MatchUses(const Tensor& tensor,
                      const std::vector<UseInfo>& expected_uses, bool strict) {
  const auto uses = tensor.Uses();
  if (strict && uses.size() != expected_uses.size()) {
    return false;
  }
  auto not_use = [&tensor =
                      std::as_const(tensor)](const UseInfo& expected_use) {
    return !MatchUse(tensor, expected_use);
  };
  return !internal::model_predicates_detail::Any<UseInfo>(expected_uses,
                                                          not_use);
}

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_MODEL_PREDICATES_H_
