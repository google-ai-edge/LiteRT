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
#include <optional>
#include <vector>

#include "absl/container/inlined_vector.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
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

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_MODEL_PREDICATES_H_
